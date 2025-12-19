from typing import Tuple, Optional
import einops
import torch
from torch import nn
import torch.nn.functional as F
import math

# try:
#    from flash_attn_interface import flash_attn_func  # type: ignore[import]
# except ImportError:
#    # Fallback to FlashAttention 2
#    from flash_attn import flash_attn_func  # type: ignore[import]
from torch.nn.functional import scaled_dot_product_attention

from models.common import trunc_normal_init_
from utils.metric import get_metric_offsets, HAS_TORCHVISION

if HAS_TORCHVISION:
    import torchvision.ops

CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((out_features, in_features)), std=1.0 / (in_features**0.5)
            )
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )


class CastedEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float,
        cast_to: torch.dtype,
    ):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(
                torch.empty((num_embeddings, embedding_dim)), std=init_std
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        head_dim,
        num_heads,
        num_key_value_heads,
        causal=False,
        dropout=0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.dropout = dropout

        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(
            batch_size,
            seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(
            lambda t: einops.rearrange(t, "B S H D -> B H S D"), (query, key, value)
        )  # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            is_causal=self.causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = einops.rearrange(attn_output, "B H S D -> B S H D")
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class ConvSwiGLU(nn.Module):
    """
    SwiGLU with 1D Depthwise Convolution for local mixing.
    Adapted from URM implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        conv_kernel: int = 2,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()

        inter = (
            intermediate_size
            if intermediate_size is not None
            else _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        )
        self.inter = inter
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        
        # Depthwise convolution
        self.dwconv = nn.Conv1d(
            in_channels=inter,
            out_channels=inter,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=inter,
            bias=True,
        )

        self.act = nn.SiLU()
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x_ffn = F.silu(gate) * up
        
        # Conv1d expects [B, C, L]
        x_conv = self.dwconv(x_ffn.transpose(1, 2).to(self.dwconv.weight.dtype))
        
        # Fix shape after potential padding mismatch
        x_conv = x_conv[..., :up.size(1)]
        
        x_conv = self.act(x_conv)
        x_conv = x_conv.transpose(1, 2).contiguous()
        x_out = self.down_proj(x_conv)

        return x_out


class MetricSwiGLU(nn.Module):
    """
    SwiGLU with 2D Metric-based Deformable Convolution.
    Adapts the local mixing kernel based on the input features using Finsler-Randers metric.
    
    The grid portion of the sequence (after puzzle_emb_len tokens) must be a perfect square.
    Puzzle embedding tokens are processed with standard SwiGLU (no conv).
    """

    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        kernel_size: int = 3,
        puzzle_emb_len: int = 0,
    ):
        super().__init__()
        if not HAS_TORCHVISION:
            raise ImportError(
                "torchvision is required for MetricSwiGLU (deform_conv2d)"
            )

        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.inter = inter
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.puzzle_emb_len = puzzle_emb_len

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        
        # Metric prediction network: predicts (M, w) parameters from hidden states
        # 7 channels: 4 for M (2x2 sym), 2 for w, 1 for scale/confidence
        self.metric_net = nn.Conv2d(inter, 7, kernel_size=3, padding=1)
        
        # Depthwise convolution weight
        self.conv_weight = nn.Parameter(
            torch.randn(inter, 1, kernel_size, kernel_size, dtype=torch.bfloat16)
            / kernel_size
        )
        self.conv_bias = nn.Parameter(torch.zeros(inter, dtype=torch.bfloat16))
        
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        
        # Linear projection
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x_ffn = F.silu(gate) * up  # [B, L, inter]
        
        # Split puzzle embedding tokens from grid tokens
        if self.puzzle_emb_len > 0:
            prefix_ffn = x_ffn[:, : self.puzzle_emb_len]  # [B, puzzle_emb_len, inter]
            grid_ffn = x_ffn[:, self.puzzle_emb_len :]  # [B, L - puzzle_emb_len, inter]
            grid_len = L - self.puzzle_emb_len
        else:
            prefix_ffn = None
            grid_ffn = x_ffn
            grid_len = L
        
        # Check if grid is square
        S = int(math.sqrt(grid_len))
        if S * S != grid_len:
            raise ValueError(
                f"MetricSwiGLU requires square grid length. "
                f"Got L={L}, puzzle_emb_len={self.puzzle_emb_len}, grid_len={grid_len} "
                f"(√{grid_len} ≈ {math.sqrt(grid_len):.2f})"
            )

        # [B, inter, H, W]
        x_2d = grid_ffn.transpose(1, 2).reshape(B, self.inter, S, S)
        
        # Compute metric offsets
        offsets = get_metric_offsets(x_2d, self.metric_net, self.kernel_size)
        
        # Apply Deformable Convolution
        out_2d = torchvision.ops.deform_conv2d(
            input=x_2d,
            offset=offsets,
            weight=self.conv_weight,
            bias=self.conv_bias,
            padding=self.pad,
            groups=self.inter,
        )
        
        # Flatten back
        grid_out = out_2d.reshape(B, self.inter, grid_len).transpose(1, 2)
        
        # Concatenate prefix back if present
        if prefix_ffn is not None:
            out = torch.cat([prefix_ffn, grid_out], dim=1)
        else:
            out = grid_out
        
        return self.down_proj(out)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
