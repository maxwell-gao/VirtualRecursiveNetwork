"""Configuration classes for Loop Transformer."""

from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
from pydantic import BaseModel


@dataclass
class CoreCarry:
    """Carry state for the core transformer (inner loop states)."""

    states: Dict[str, torch.Tensor]


@dataclass
class ModelCarry:
    """Carry state for the full model (including ACT/halt logic)."""

    core_carry: CoreCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class LoopStateConfig(BaseModel):
    """Configuration for a single loop state (e.g., z_H, z_L)."""

    name: str
    layers: int
    share_weights_with: Optional[str] = None


class LoopStageConfig(BaseModel):
    """Configuration for a processing stage in the loop schedule."""

    target: str
    sources: List[str] = []
    include_inputs: bool = False
    repeat: int = 1
    repeat_key: Optional[str] = None


class LoopTransformerConfig(BaseModel):
    """Main configuration for Loop Transformer."""

    # Core architecture
    batch_size: int
    seq_len: int
    vocab_size: int
    num_puzzle_identifiers: int

    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    forward_dtype: str = "bfloat16"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    mlp_type: str = "swiglu"  # "swiglu", "metric_conv"
    image_size: Optional[int] = None

    # Puzzle embeddings
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: Optional[int] = None

    # Adaptive Computation Time (ACT)
    halt_max_steps: int
    halt_exploration_prob: float
    act_enabled: bool = True
    act_inference: bool = False
    no_ACT_continue: bool = True

    # Denoising Implicit Scheduling (DIS)
    dis_enabled: bool = False
    dis_max_steps: int = 16
    dis_schedule: str = "linear"
    dis_loss_method: str = "mask"

    # Loop scheduling
    outer_cycles: int
    no_grad_cycles: Optional[int] = None

    # State and stage definitions
    states: List[LoopStateConfig]
    stages: List[LoopStageConfig]
    readout_state: str
    halt_state: Optional[str] = None

    # Training options
    gradient_checkpointing: bool = False
