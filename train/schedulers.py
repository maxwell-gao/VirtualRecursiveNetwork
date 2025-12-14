"""
Learning rate schedulers.
"""

import math


def cosine_schedule_with_warmup(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
) -> float:
    """
    Cosine learning rate schedule with warmup.

    Args:
        current_step: Current training step
        base_lr: Base learning rate
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_ratio: Minimum learning rate ratio (relative to base_lr)
        num_cycles: Number of cosine cycles

    Returns:
        Learning rate for current step
    """
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )


def compute_lr(base_lr: float, config, train_state) -> float:
    """
    Compute learning rate with cosine schedule for current training state.

    Args:
        base_lr: Base learning rate
        config: PretrainConfig instance
        train_state: TrainState instance

    Returns:
        Learning rate for current step
    """
    return cosine_schedule_with_warmup(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )
