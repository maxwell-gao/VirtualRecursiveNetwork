"""
Looper: Recursive reasoning model with adaptive computation.

This is a new implementation featuring:
- Configurable multi-state loop architecture
- Adaptive Computation Time (ACT) support
- Denoising Implicit Scheduling (DIS) support
"""

from models.looper.config import (
    LoopStateConfig,
    LoopStageConfig,
    LoopTransformerConfig,
    CoreCarry,
    ModelCarry,
)
from models.looper.blocks import (
    TransformerBlock,
    ReasoningModule,
)
from models.looper.core import LoopTransformerCore
from models.looper.model import LoopTransformer

__all__ = [
    # Config
    "LoopStateConfig",
    "LoopStageConfig",
    "LoopTransformerConfig",
    "CoreCarry",
    "ModelCarry",
    # Blocks
    "TransformerBlock",
    "ReasoningModule",
    # Core
    "LoopTransformerCore",
    # Model
    "LoopTransformer",
]
