"""Core transformer logic with loop scheduling."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.common import trunc_normal_init_
from models.layers import RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.embed.sparse import CastedSparseEmbedding
from models.looper.config import (
    LoopTransformerConfig,
    LoopStageConfig,
    CoreCarry,
)
from models.looper.blocks import TransformerBlock, ReasoningModule


class LoopTransformerCore(nn.Module):
    """
    Core transformer with recursive loop scheduling.

    Manages embedding, multiple loop states, and scheduled execution.
    """

    def __init__(self, config: LoopTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        # Embedding setup
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            config.vocab_size,
            config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # DIS step embedding
        if config.dis_enabled:
            self.step_emb = nn.Embedding(config.dis_max_steps, config.hidden_size)

        # Puzzle embedding
        puzzle_len = (
            -(config.puzzle_emb_ndim // -config.hidden_size)
            if (config.puzzle_emb_ndim > 0 and not config.puzzle_emb_len)
            else (config.puzzle_emb_len or 0)
        )
        self.puzzle_emb_len = puzzle_len

        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                config.num_puzzle_identifiers,
                config.puzzle_emb_ndim,
                batch_size=config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # Position encodings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.seq_len + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                config.seq_len + self.puzzle_emb_len,
                config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError(f"Position encoding {config.pos_encodings} not supported")

        # State modules
        self.state_names = [state.name for state in config.states]
        self.state_modules = nn.ModuleDict()
        self.state_module_refs: Dict[str, ReasoningModule] = {}

        for state_cfg in config.states:
            if state_cfg.share_weights_with is not None:
                if state_cfg.share_weights_with not in self.state_module_refs:
                    raise ValueError(
                        f"State '{state_cfg.name}' references undefined share_weights_with="
                        f"'{state_cfg.share_weights_with}'"
                    )
                module = self.state_module_refs[state_cfg.share_weights_with]
            else:
                blocks = [TransformerBlock(config) for _ in range(state_cfg.layers)]
                module = ReasoningModule(blocks)
                self.state_modules[state_cfg.name] = module

            self.state_module_refs[state_cfg.name] = module

            # State initialization buffer
            init = trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1
            )
            self.register_buffer(
                f"{state_cfg.name}_init",
                init,
                persistent=True,
            )

        # Initialize Q heads with bias towards not halting
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    @property
    def readout_state(self) -> str:
        """State used for final output logits."""
        return self.config.readout_state

    @property
    def halt_state(self) -> str:
        """State used for halt decision."""
        return self.config.halt_state or self.config.readout_state

    def compute_input_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute input embeddings with optional puzzle embeddings."""
        embedding = self.embed_tokens(batch["inputs"].to(torch.int32))

        # Add puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(batch["puzzle_identifiers"])
            pad_count = (
                self.puzzle_emb_len * self.config.hidden_size
                - puzzle_embedding.shape[-1]
            )
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (
                    puzzle_embedding.view(
                        -1, self.puzzle_emb_len, self.config.hidden_size
                    ),
                    embedding,
                ),
                dim=-2,
            )

        # Add positional encodings (learned)
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> CoreCarry:
        """Create empty carry state for a new sequence."""
        device = self.embed_tokens.embedding_weight.device
        states = {}
        shape = (
            batch_size,
            self.config.seq_len + self.puzzle_emb_len,
            self.config.hidden_size,
        )
        for name in self.state_names:
            states[name] = torch.empty(shape, dtype=self.forward_dtype, device=device)
        return CoreCarry(states=states)

    def reset_carry(self, reset_flag: torch.Tensor, carry: CoreCarry) -> CoreCarry:
        """Reset carry states based on reset flag."""
        new_states = {}
        mask = reset_flag.view(-1, 1, 1)
        for name in self.state_names:
            init = getattr(self, f"{name}_init")
            new_states[name] = torch.where(mask, init, carry.states[name])
        return CoreCarry(states=new_states)

    def aggregate_sources(
        self,
        stage: LoopStageConfig,
        states: Dict[str, torch.Tensor],
        input_embeddings: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Aggregate source states for a processing stage."""
        components: List[torch.Tensor] = []

        if stage.include_inputs:
            components.append(input_embeddings)

        for src in stage.sources:
            components.append(states[src])

        if not components:
            return None

        # Sum all components
        out = torch.zeros_like(states[stage.target])
        for comp in components:
            out = out + comp
        return out

    def resolve_repeat(self, stage: LoopStageConfig) -> int:
        """Resolve repeat count for a stage (supports dynamic config keys)."""
        if stage.repeat_key is not None:
            if not hasattr(self.config, stage.repeat_key):
                raise AttributeError(
                    f"Loop stage repeat_key '{stage.repeat_key}' missing in config"
                )
            return int(getattr(self.config, stage.repeat_key))
        return stage.repeat

    def run_schedule(
        self,
        states: Dict[str, torch.Tensor],
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Optional[CosSin]],
    ) -> None:
        """Execute one full loop schedule (all stages)."""
        for stage in self.config.stages:
            repeats = self.resolve_repeat(stage)
            if repeats <= 0:
                continue

            for _ in range(repeats):
                injection = self.aggregate_sources(stage, states, input_embeddings)
                module = self.state_module_refs[stage.target]

                # Use gradient checkpointing if enabled
                if (
                    self.config.gradient_checkpointing
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    states[stage.target] = checkpoint(
                        module,
                        states[stage.target],
                        injection,
                        use_reentrant=False,
                        **seq_info,
                    )
                else:
                    states[stage.target] = module(
                        hidden_states=states[stage.target],
                        input_injection=injection,
                        **seq_info,
                    )

    def forward(
        self,
        carry: CoreCarry,
        batch: Dict[str, torch.Tensor],
        step_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[CoreCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the loop transformer.

        Args:
            carry: Current carry state
            batch: Input batch
            step_ids: Optional step IDs for DIS

        Returns:
            Tuple of (new_carry, logits, (q_halt_logits, q_continue_logits))
        """
        # Prepare sequence info (positional encodings)
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Compute input embeddings
        input_embeddings = self.compute_input_embeddings(batch)

        # Add DIS step embeddings if enabled
        if self.config.dis_enabled and step_ids is not None:
            if not isinstance(step_ids, torch.Tensor):
                step_ids = torch.tensor(
                    step_ids, device=input_embeddings.device, dtype=torch.long
                )
            if step_ids.ndim == 0:
                step_ids = step_ids.expand(input_embeddings.shape[0])

            input_embeddings = input_embeddings + self.step_emb(step_ids).unsqueeze(1)

        # Copy states from carry
        states = {name: tensor for name, tensor in carry.states.items()}

        # Determine gradient vs no-gradient cycles
        if self.config.dis_enabled:
            no_grad_cycles = 0
            grad_cycles = self.config.outer_cycles
        else:
            no_grad_cycles = (
                self.config.outer_cycles - 1
                if self.config.no_grad_cycles is None
                else self.config.no_grad_cycles
            )
            no_grad_cycles = max(0, min(no_grad_cycles, self.config.outer_cycles - 1))
            grad_cycles = max(1, self.config.outer_cycles - no_grad_cycles)

        # Run no-gradient cycles
        if no_grad_cycles > 0:
            with torch.no_grad():
                for _ in range(no_grad_cycles):
                    self.run_schedule(states, input_embeddings, seq_info)

        # Run gradient cycles
        with nullcontext():
            for _ in range(grad_cycles):
                self.run_schedule(states, input_embeddings, seq_info)

        # Prepare outputs
        new_states = {name: tensor.detach() for name, tensor in states.items()}
        readout = states[self.readout_state]
        logits = self.lm_head(readout)[:, self.puzzle_emb_len :]

        halt_source = states[self.halt_state]
        q_halt_logits, q_continue_logits = (
            self.q_head(halt_source[:, 0]).to(torch.float32).chunk(2, dim=-1)
        )

        return (
            CoreCarry(states=new_states),
            logits,
            (
                q_halt_logits.squeeze(-1),
                q_continue_logits.squeeze(-1),
            ),
        )

