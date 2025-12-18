"""Loop Transformer model with ACT/DIS support."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from models.looper.config import (
    LoopTransformerConfig,
    ModelCarry,
    CoreCarry,
)
from models.looper.core import LoopTransformerCore


class LoopTransformer(nn.Module):
    """
    Loop Transformer with Adaptive Computation Time (ACT) support.

    This model implements recursive reasoning through loop scheduling,
    with optional adaptive halting and denoising implicit scheduling.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LoopTransformerConfig(**config_dict)
        self.core = LoopTransformerCore(self.config)

    @property
    def puzzle_emb(self):
        """Access to puzzle embeddings (for optimizer setup)."""
        return getattr(self.core, "puzzle_emb", None)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> ModelCarry:
        """
        Initialize carry state for a new batch.

        All sequences start in "halted" state and will be activated
        on the first forward pass.
        """
        batch_size = batch["inputs"].shape[0]
        core_carry = self.core.empty_carry(batch_size)
        zeros = torch.zeros((batch_size,), dtype=torch.int32)
        halted = torch.ones((batch_size,), dtype=torch.bool)
        current_data = {k: torch.empty_like(v) for k, v in batch.items()}
        return ModelCarry(core_carry, zeros, halted, current_data)

    def forward(
        self,
        carry: ModelCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
        step: Optional[torch.Tensor] = None,
    ) -> Tuple[ModelCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive computation.

        Args:
            carry: Current model carry state
            batch: Input batch
            compute_target_q: Whether to compute target Q values (for training)
            step: Optional step ID for DIS

        Returns:
            Tuple of (new_carry, outputs_dict)
        """
        # DIS mode: simple forward without ACT logic
        if self.config.dis_enabled and step is not None:
            new_core_carry, logits, _ = self.core(
                carry.core_carry, batch, step_ids=step
            )
            outputs = {"logits": logits}
            new_steps = carry.steps + 1
            halted = new_steps >= self.config.dis_max_steps
            return (
                ModelCarry(new_core_carry, new_steps, halted, batch),
                outputs,
            )

        # ACT mode: adaptive halting logic

        # Reset core carry for halted sequences
        core_carry = self.core.reset_carry(carry.halted, carry.core_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)

        # Update current data for halted sequences
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (v.ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward through core
        new_core_carry, logits, (q_halt_logits, q_continue_logits) = self.core(
            core_carry, new_current_data
        )

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        # Compute halting decisions (no gradient)
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            # Use adaptive halting if enabled
            use_adaptive = (self.config.halt_max_steps > 1) and (
                (self.training and self.config.act_enabled)
                or (not self.training and self.config.act_inference)
            )

            if use_adaptive:
                # Halting criterion
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration during training
                if self.training:
                    explore = (
                        torch.rand_like(q_halt_logits)
                        < self.config.halt_exploration_prob
                    )
                    min_halt = explore * torch.randint_like(
                        new_steps, low=2, high=self.config.halt_max_steps + 1
                    )
                    halted = halted & (new_steps >= min_halt)

                # Compute target Q for continue (if requested)
                if (
                    self.training
                    and compute_target_q
                    and not self.config.no_ACT_continue
                ):
                    next_logits = self.core(core_carry, new_current_data)[-1]
                    next_halt, next_continue = next_logits
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_halt,
                            torch.maximum(next_halt, next_continue),
                        )
                    )

        return ModelCarry(
            new_core_carry,
            new_steps,
            halted,
            new_current_data,
        ), outputs
