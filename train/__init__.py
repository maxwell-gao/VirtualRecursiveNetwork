"""
Training module for recursive reasoning models.
Provides Fabric-based distributed training utilities.
"""

from train.config import (
    LossConfig,
    ArchConfig,
    EvaluatorConfig,
    PretrainConfig,
)
from train.state import TrainState, init_train_state, save_train_state, load_checkpoint
from train.early_stopping import EarlyStoppingWrapper
from train.data import create_dataloader, create_evaluators
from train.loops import train_batch, evaluate
from train.schedulers import cosine_schedule_with_warmup, compute_lr

__all__ = [
    # Config
    "LossConfig",
    "ArchConfig",
    "EvaluatorConfig",
    "PretrainConfig",
    # State
    "TrainState",
    "init_train_state",
    "save_train_state",
    "load_checkpoint",
    # Early stopping
    "EarlyStoppingWrapper",
    # Data
    "create_dataloader",
    "create_evaluators",
    # Loops
    "train_batch",
    "evaluate",
    # Schedulers
    "cosine_schedule_with_warmup",
    "compute_lr",
]
