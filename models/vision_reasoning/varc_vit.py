"""
Vision-based models for ARC reasoning.

VARCViT - ViT baseline with optional MetricConvolutions patch embedding.
Reuses LoopTransformerBlock and patterns from loop_transformer.py.
"""

from __future__ import annotations

from typing import Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

try:
    import torchvision.ops

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from models.common import trunc_normal_init_
from models.layers import RotaryEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.recursive_reasoning.loop_transformer import LoopTransformerBlock


# =============================================================================
# Metric Convolutions - Patch Embedding with Finsler-Randers Geometry
# =============================================================================


def _finsler_randers(v, M, w, eps=1e-6):
    """F(v) = sqrt(v^T M v) + w^T v"""
    M_2x2 = M.reshape(M.shape[0], 2, 2, *M.shape[-2:])
    norm = torch.sqrt(torch.einsum("bjrc,bijrc,birc->brc", v, M_2x2, v) + eps)
    drift = torch.einsum("birc,birc->brc", w, v)
    return norm + drift


def _sample_tangent_ball(batch_size, M, w, kh, kw, device, eps=1e-6):
    """Sample points on unit tangent ball using onion peeling."""
    out_shape = M.shape[-2:]
    u_list, s_list = [], []

    for k in range(kh // 2 + 1):
        if k == 0:
            theta = torch.zeros(1, device=device)
            u_list.append(torch.stack([torch.cos(theta), torch.sin(theta)], dim=1))
            s_list.append(torch.zeros(1, device=device))
        else:
            n = 8 * k
            theta = torch.linspace(0, 2 * math.pi * (1 - 1 / n), n, device=device)
            u_list.append(torch.stack([torch.cos(theta), torch.sin(theta)], dim=1))
            s_list.append(torch.full((n,), k / (kh // 2), device=device))

    u = torch.cat(u_list, dim=0)  # (N, 2)
    s = torch.cat(s_list)  # (N,)
    N = u.shape[0]

    # Expand for batch and spatial dims
    u_exp = u.view(1, N, 2, 1, 1).expand(batch_size, -1, -1, *out_shape)

    # Compute F(u) for each direction
    u_flat = u_exp.reshape(batch_size * N, 2, *out_shape)
    M_rep = M.repeat(N, 1, 1, 1)
    w_rep = w.repeat(N, 1, 1, 1)
    F_u = _finsler_randers(u_flat, M_rep, w_rep, eps).view(batch_size, N, *out_shape)

    # y = u / F(u) * s
    y = u_exp / (F_u.unsqueeze(2) + eps) * s.view(1, N, 1, 1, 1)
    return y.view(batch_size, kh, kw, 2, *out_shape)


class MetricPatchEmbed(nn.Module):
    """Patch embedding with learnable Finsler-Randers metric."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim, eps_w=0.5):
        super().__init__()
        assert HAS_TORCHVISION, "torchvision required for MetricPatchEmbed"

        self.patch_size = patch_size
        self.eps_w = eps_w

        # Predict metric: 4 (M) + 2 (w) + 1 (scale) = 7 channels
        self.metric_net = nn.Conv2d(in_chans, 7, patch_size, stride=patch_size)
        self.proj_weight = nn.Parameter(
            torch.randn(embed_dim, in_chans, patch_size, patch_size)
            / (patch_size * math.sqrt(in_chans))
        )
        self.proj_bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        # Predict metric parameters
        params = self.metric_net(x)  # (B, 7, H', W')

        # Build positive-definite M from eigendecomposition
        v = F.normalize(params[:, :2].permute(0, 2, 3, 1), dim=-1)
        eigvals = 2 * torch.sigmoid(params[:, 2:4].permute(0, 2, 3, 1))
        scale = 0.5 + 1.5 * torch.sigmoid(params[:, 4:5].permute(0, 2, 3, 1))
        eigvals = eigvals * scale

        R = torch.stack([v, torch.stack([-v[..., 1], v[..., 0]], -1)], -1)
        M = R @ torch.diag_embed(eigvals) @ R.transpose(-1, -2)
        M = M.permute(0, 3, 4, 1, 2).reshape(B, 4, *params.shape[-2:])

        # Build drift w with constraint
        w = params[:, 5:7]
        if self.eps_w < 1.0:
            w = w * (1 - self.eps_w) * torch.sigmoid(w.norm(dim=1, keepdim=True))
        else:
            w = torch.zeros_like(w)

        # Sample tangent ball and compute offsets
        y = _sample_tangent_ball(B, M, w, P, P, x.device)

        # Standard kernel grid
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(-(P // 2), P // 2 + 1, device=x.device),
                torch.arange(-(P // 2), P // 2 + 1, device=x.device),
                indexing="ij",
            ),
            -1,
        ).float()

        offsets = (y.permute(0, 4, 5, 1, 2, 3).flip(-1) - grid).permute(
            0, 3, 4, 5, 1, 2
        )
        offsets = offsets.reshape(B, -1, *params.shape[-2:])

        # Deformable convolution
        out = torchvision.ops.deform_conv2d(
            x, offsets, self.proj_weight, self.proj_bias, stride=P
        )
        return out.flatten(2).transpose(1, 2)


class StandardPatchEmbed(nn.Module):
    """Standard patch embedding for discrete tokens."""

    def __init__(self, img_size, patch_size, vocab_size, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Conv2d(embed_dim, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        B = x.shape[0]
        if x.dim() == 2:
            x = x.view(B, self.img_size, self.img_size)
        x = self.token_embed(x.long()).permute(0, 3, 1, 2)
        return self.proj(x).flatten(2).transpose(1, 2)


# =============================================================================
# VARCViT Model
# =============================================================================


@dataclass
class VARCViTCarry:
    hidden_states: torch.Tensor
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class VARCViTConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int

    # Auto-computed from seq_len if None
    image_size: Optional[int] = None
    patch_size: int = 3

    # Transformer
    hidden_size: int = 256
    num_heads: int = 8
    expansion: float = 2.0
    depth: int = 6
    dropout: float = 0.0

    # Puzzle embedding
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: Optional[int] = None

    # ACT
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0
    act_enabled: bool = False
    act_inference: bool = False

    # Metric patch
    use_metric_patch: bool = False
    metric_eps_w: float = 0.5

    # Loop reasoning
    outer_cycles: int = 1

    forward_dtype: str = "bfloat16"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def model_post_init(self, __context):
        if self.image_size is None:
            size = int(math.sqrt(self.seq_len))
            assert size * size == self.seq_len, f"seq_len={self.seq_len} not square"
            object.__setattr__(self, "image_size", size)


class VARCViTInner(nn.Module):
    def __init__(self, config: VARCViTConfig):
        super().__init__()
        self.config = config
        self.dtype = getattr(torch, config.forward_dtype)

        num_patches = (config.image_size // config.patch_size) ** 2

        # Patch embedding
        if config.use_metric_patch:
            self.patch_embed = MetricPatchEmbed(
                config.image_size,
                config.patch_size,
                config.vocab_size,
                config.hidden_size,
                config.metric_eps_w,
            )
            self._use_onehot = True
        else:
            self.patch_embed = StandardPatchEmbed(
                config.image_size,
                config.patch_size,
                config.vocab_size,
                config.hidden_size,
            )
            self._use_onehot = False

        # Puzzle embedding
        self.puzzle_emb_len = (
            -(config.puzzle_emb_ndim // -config.hidden_size)
            if config.puzzle_emb_ndim > 0 and not config.puzzle_emb_len
            else (config.puzzle_emb_len or 0)
        )
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=self.dtype,
            )

        # Task token
        self.task_token = nn.Parameter(
            trunc_normal_init_(
                torch.empty(1, config.hidden_size, dtype=self.dtype), std=0.02
            )
        )

        # Sequence length and RoPE
        self.seq_len = num_patches + 1 + self.puzzle_emb_len
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_heads, self.seq_len, config.rope_theta
        )

        # Reuse LoopTransformerBlock from loop_transformer.py
        # Create a minimal config-like object for compatibility
        class _BlockConfig:
            hidden_size = config.hidden_size
            num_heads = config.num_heads
            expansion = config.expansion
            rms_norm_eps = config.rms_norm_eps
            dropout = config.dropout

        self.blocks = nn.ModuleList(
            [LoopTransformerBlock(_BlockConfig()) for _ in range(config.depth)]
        )

        # Output heads
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def forward(self, batch, prev_h=None):
        B = batch["inputs"].shape[0]
        x = batch["inputs"]

        # Patch embedding
        if self._use_onehot:
            x = x.view(B, self.config.image_size, self.config.image_size)
            x = (
                F.one_hot(x.long(), self.config.vocab_size)
                .permute(0, 3, 1, 2)
                .to(self.dtype)
            )
        h = self.patch_embed(x).to(self.dtype)

        # Prepend task token
        h = torch.cat([self.task_token.expand(B, -1, -1), h], dim=1)

        # Prepend puzzle embedding
        if self.config.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(batch["puzzle_identifiers"])
            pad = self.puzzle_emb_len * self.config.hidden_size - pe.shape[-1]
            if pad > 0:
                pe = F.pad(pe, (0, pad))
            h = torch.cat([pe.view(B, self.puzzle_emb_len, -1), h], dim=1)

        # Add previous hidden state for loop reasoning
        if prev_h is not None:
            h = h + prev_h

        # Transformer blocks
        cos_sin = self.rotary_emb()
        for block in self.blocks:
            h = block(cos_sin=cos_sin, hidden_states=h)

        # Output: upsample patches to full resolution
        start = self.puzzle_emb_len + 1
        patch_h = h[:, start:]
        ps = self.config.image_size // self.config.patch_size
        patch_h = patch_h.view(B, ps, ps, -1).permute(0, 3, 1, 2)
        upsampled = F.interpolate(patch_h, self.config.image_size, mode="nearest")
        upsampled = upsampled.permute(0, 2, 3, 1).reshape(
            B, -1, self.config.hidden_size
        )

        logits = self.lm_head(upsampled)
        q = self.q_head(h[:, self.puzzle_emb_len]).float()

        return h, logits, (q[..., 0], q[..., 1])

    def empty_carry(self, batch_size):
        return torch.zeros(
            batch_size,
            self.seq_len,
            self.config.hidden_size,
            dtype=self.dtype,
            device=self.task_token.device,
        )


class VARCViT(nn.Module):
    """Vision Transformer for ARC with optional MetricConvolutions and loop reasoning."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = VARCViTConfig(**config_dict)
        self.inner = VARCViTInner(self.config)

    @property
    def puzzle_emb(self):
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch):
        B = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return VARCViTCarry(
            self.inner.empty_carry(B),
            torch.zeros(B, dtype=torch.int32, device=device),
            torch.ones(B, dtype=torch.bool, device=device),
            {k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry, batch, compute_target_q=False):
        # Update halted sequences
        data = {
            k: torch.where(carry.halted.view(-1, *([1] * (v.ndim - 1))), batch[k], v)
            for k, v in carry.current_data.items()
        }

        mask = carry.halted.view(-1, 1, 1)
        h = torch.where(
            mask, torch.zeros_like(carry.hidden_states), carry.hidden_states
        )
        steps = torch.where(carry.halted, 0, carry.steps)

        # Forward with optional loop cycles
        for _ in range(self.config.outer_cycles):
            h, logits, (q_halt, q_cont) = self.inner(data, h)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_cont,
        }

        with torch.no_grad():
            steps = steps + 1
            halted = steps >= self.config.halt_max_steps
            if self.config.act_enabled and self.config.halt_max_steps > 1:
                halted = halted | (q_halt > q_cont)

        return VARCViTCarry(h.detach(), steps, halted, data), outputs


class VARCMetricViT(VARCViT):
    """VARCViT with MetricConvolutions patch embedding."""

    def __init__(self, config_dict):
        config_dict = {**config_dict, "use_metric_patch": True}
        super().__init__(config_dict)


class VARCLoopViT(VARCViT):
    """VARCViT with MetricConvolutions and loop reasoning."""

    def __init__(self, config_dict):
        config_dict = {**config_dict, "use_metric_patch": True}
        if "outer_cycles" not in config_dict:
            config_dict["outer_cycles"] = 3
        super().__init__(config_dict)
