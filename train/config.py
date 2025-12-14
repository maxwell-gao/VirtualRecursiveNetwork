"""
Configuration classes for pretraining.
"""

from typing import Optional, List
import pydantic


class LossConfig(pydantic.BaseModel):
    """Configuration for loss function."""

    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    """Configuration for model architecture."""

    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    """Configuration for evaluators."""

    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    """Main configuration for pretraining."""

    # Config
    arch: ArchConfig

    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []

    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float
    grad_clip_norm: float = 1.0
    optimizer: str = "adam_atan2"

    # Muon
    muon_lr: float = 0.002
    muon_weight_decay: float = 0.01
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0  # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False  # use Exponential-Moving-Average
    ema_rate: float = 0.999  # EMA-rate
    freeze_weights: bool = (
        False  # If True, freeze weights and only learn the embeddings
    )

    # Early stopping
    early_stopping: bool = False  # Enable early stopping
    early_stopping_monitor: str = "exact_accuracy"  # Metric to monitor
    early_stopping_patience: int = 3  # Number of checks with no improvement
    early_stopping_mode: str = "max"  # "min" or "max"
    early_stopping_min_delta: float = 0.0  # Minimum change to qualify as improvement
