"""Transformer blocks and reasoning modules."""

from typing import List, Optional

import torch
from torch import nn

from models.layers import rms_norm, SwiGLU, ConvSwiGLU, MetricSwiGLU, Attention, CosSin
from models.looper.config import LoopTransformerConfig


class TransformerBlock(nn.Module):
    """Single transformer block with post-norm architecture."""

    def __init__(self, config: LoopTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
            dropout=config.dropout,
        )
        
        mlp_type = getattr(config, "mlp_type", "swiglu")
        if mlp_type == "metric_conv":
            # Get puzzle_emb_len from config for proper grid handling
            # Use same logic as core.py for consistency
            puzzle_emb_len = config.puzzle_emb_len
            if puzzle_emb_len is None or puzzle_emb_len == 0:
                if config.puzzle_emb_ndim > 0:
                    # Auto-compute: ceil(puzzle_emb_ndim / hidden_size)
                    puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)
                else:
                    puzzle_emb_len = 0
            self.mlp = MetricSwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
                puzzle_emb_len=puzzle_emb_len,
            )
        elif mlp_type == "conv_swiglu":
            self.mlp = ConvSwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )
        else:
            self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
            
        self.norm_eps = config.rms_norm_eps
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Post-Norm structure for stability in recursive loops.

        In recursive architectures, Post-Norm ensures outputs are always normalized,
        preventing numerical explosion across multiple cycles.
        """
        hidden_states = rms_norm(
            hidden_states
            + self.dropout(
                self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
            ),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.dropout(self.mlp(hidden_states)),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


class ReasoningModule(nn.Module):
    """Stack of transformer blocks with optional input injection."""

    def __init__(self, layers: List[TransformerBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with optional input injection.

        Args:
            hidden_states: Current hidden states
            input_injection: Optional tensor to add to hidden states before processing
            **kwargs: Additional arguments (e.g., cos_sin for rotary embeddings)
        """
        if input_injection is not None:
            hidden_states = hidden_states + input_injection

        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states
