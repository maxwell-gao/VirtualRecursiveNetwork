from __future__ import annotations

from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class LoopTransformerInnerCarry:
    states: Dict[str, torch.Tensor]


@dataclass
class LoopTransformerCarry:
    inner_carry: LoopTransformerInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class LoopStateConfig(BaseModel):
    name: str
    layers: int
    share_weights_with: Optional[str] = None


class LoopStageConfig(BaseModel):
    target: str
    sources: List[str] = []
    include_inputs: bool = False
    repeat: int = 1
    repeat_key: Optional[str] = None


class LoopTransformerConfig(BaseModel):
    # Core
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

    puzzle_emb_ndim: int = 0
    puzzle_emb_len: Optional[int] = None

    halt_max_steps: int
    halt_exploration_prob: float
    act_enabled: bool = True
    act_inference: bool = False
    no_ACT_continue: bool = True
    gradient_checkpointing: bool = False

    dis_enabled: bool = False
    dis_max_steps: int = 16
    dis_schedule: str = "linear"
    dis_loss_method: str = "mask"

    outer_cycles: int
    warmup_cycles: Optional[int] = None

    states: List[LoopStateConfig]
    stages: List[LoopStageConfig]
    readout_state: str
    halt_state: Optional[str] = None


class LoopTransformerBlock(nn.Module):
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
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # Pre-Norm structure for better stability
        normed_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        hidden_states = hidden_states + self.dropout(
            self.self_attn(cos_sin=cos_sin, hidden_states=normed_states)
        )

        normed_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        hidden_states = hidden_states + self.dropout(self.mlp(normed_states))

        return hidden_states


class LoopReasoningModule(nn.Module):
    def __init__(self, layers: List[LoopTransformerBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_injection is not None:
            hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class LoopTransformerInner(nn.Module):
    def __init__(self, config: LoopTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

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

        if config.dis_enabled:
            self.step_emb = nn.Embedding(config.dis_max_steps, config.hidden_size)

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
            raise NotImplementedError()

        self.state_names = [state.name for state in config.states]
        self.state_modules = nn.ModuleDict()
        self._state_module_refs: Dict[str, LoopReasoningModule] = {}
        for state_cfg in config.states:
            if state_cfg.share_weights_with is not None:
                if state_cfg.share_weights_with not in self._state_module_refs:
                    raise ValueError(
                        f"State '{state_cfg.name}' references undefined share_weights_with="
                        f"'{state_cfg.share_weights_with}'"
                    )
                module = self._state_module_refs[state_cfg.share_weights_with]
            else:
                blocks = [LoopTransformerBlock(config) for _ in range(state_cfg.layers)]
                module = LoopReasoningModule(blocks)
                self.state_modules[state_cfg.name] = module
            self._state_module_refs[state_cfg.name] = module

            init = trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1
            )
            self.register_buffer(
                f"{state_cfg.name}_init",
                init,
                persistent=True,
            )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    @property
    def readout_state(self) -> str:
        return self.config.readout_state

    @property
    def halt_state(self) -> str:
        return self.config.halt_state or self.config.readout_state

    def _input_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = self.embed_tokens(batch["inputs"].to(torch.int32))

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

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> LoopTransformerInnerCarry:
        device = self.embed_tokens.embedding_weight.device
        states = {}
        shape = (
            batch_size,
            self.config.seq_len + self.puzzle_emb_len,
            self.config.hidden_size,
        )
        for name in self.state_names:
            states[name] = torch.empty(shape, dtype=self.forward_dtype, device=device)
        return LoopTransformerInnerCarry(states=states)

    def reset_carry(
        self, reset_flag: torch.Tensor, carry: LoopTransformerInnerCarry
    ) -> LoopTransformerInnerCarry:
        new_states = {}
        mask = reset_flag.view(-1, 1, 1)
        for name in self.state_names:
            init = getattr(self, f"{name}_init")
            new_states[name] = torch.where(mask, init, carry.states[name])
        return LoopTransformerInnerCarry(states=new_states)

    def _aggregate_sources(
        self,
        stage: LoopStageConfig,
        states: Dict[str, torch.Tensor],
        input_embeddings: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        components: List[torch.Tensor] = []
        if stage.include_inputs:
            components.append(input_embeddings)
        for src in stage.sources:
            components.append(states[src])
        if not components:
            return None
        out = torch.zeros_like(states[stage.target])
        for comp in components:
            out = out + comp
        return out

    def _resolve_repeat(self, stage: LoopStageConfig) -> int:
        if stage.repeat_key is not None:
            if not hasattr(self.config, stage.repeat_key):
                raise AttributeError(
                    f"Loop stage repeat_key '{stage.repeat_key}' missing in config"
                )
            return int(getattr(self.config, stage.repeat_key))
        return stage.repeat

    def _run_schedule(
        self,
        states: Dict[str, torch.Tensor],
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Optional[CosSin]],
    ) -> None:
        for stage in self.config.stages:
            repeats = self._resolve_repeat(stage)
            if repeats <= 0:
                continue
            for _ in range(repeats):
                injection = self._aggregate_sources(stage, states, input_embeddings)
                module = self._state_module_refs[stage.target]

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
        carry: LoopTransformerInnerCarry,
        batch: Dict[str, torch.Tensor],
        step_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        LoopTransformerInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(batch)

        if self.config.dis_enabled and step_ids is not None:
            # Ensure step_ids is a tensor on the correct device
            if not isinstance(step_ids, torch.Tensor):
                step_ids = torch.tensor(
                    step_ids, device=input_embeddings.device, dtype=torch.long
                )
            # If scalar, expand to batch size
            if step_ids.ndim == 0:
                step_ids = step_ids.expand(input_embeddings.shape[0])

            input_embeddings = input_embeddings + self.step_emb(step_ids).unsqueeze(1)

        states = {name: tensor for name, tensor in carry.states.items()}

        if self.config.dis_enabled:
            warmup_cycles = 0
            grad_cycles = self.config.outer_cycles
        else:
            warmup_cycles = (
                self.config.outer_cycles - 1
                if self.config.warmup_cycles is None
                else self.config.warmup_cycles
            )
            warmup_cycles = max(0, min(warmup_cycles, self.config.outer_cycles - 1))
            grad_cycles = max(1, self.config.outer_cycles - warmup_cycles)

        if warmup_cycles > 0:
            with torch.no_grad():
                for _ in range(warmup_cycles):
                    self._run_schedule(states, input_embeddings, seq_info)

        with nullcontext():
            for _ in range(grad_cycles):
                self._run_schedule(states, input_embeddings, seq_info)

        new_states = {name: tensor.detach() for name, tensor in states.items()}
        readout = states[self.readout_state]
        logits = self.lm_head(readout)[:, self.puzzle_emb_len :]
        halt_source = states[self.halt_state]
        q_halt_logits, q_continue_logits = (
            self.q_head(halt_source[:, 0]).to(torch.float32).chunk(2, dim=-1)
        )

        inner_carry = LoopTransformerInnerCarry(states=new_states)
        return (
            inner_carry,
            logits,
            (
                q_halt_logits.squeeze(-1),
                q_continue_logits.squeeze(-1),
            ),
        )


class LoopTransformerModel_ACT(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LoopTransformerConfig(**config_dict)
        self.inner = LoopTransformerInner(self.config)

    @property
    def puzzle_emb(self):
        return getattr(self.inner, "puzzle_emb", None)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> LoopTransformerCarry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        inner = self.inner.empty_carry(batch_size)
        zeros = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        halted = torch.ones((batch_size,), dtype=torch.bool, device=device)

        # Initialize inner states
        inner = self.inner.reset_carry(halted, inner)

        current_data = {k: v for k, v in batch.items()}
        return LoopTransformerCarry(inner, zeros, halted, current_data)

    def forward(
        self,
        carry: LoopTransformerCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
        step: Optional[torch.Tensor] = None,
    ) -> Tuple[LoopTransformerCarry, Dict[str, torch.Tensor]]:
        if self.config.dis_enabled and step is not None:
            new_inner_carry, logits, _ = self.inner(
                carry.inner_carry, batch, step_ids=step
            )
            outputs = {"logits": logits}
            new_steps = carry.steps + 1
            halted = new_steps >= self.config.dis_max_steps
            return (
                LoopTransformerCarry(new_inner_carry, new_steps, halted, batch),
                outputs,
            )

        inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (v.ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            inner_carry, new_current_data
        )

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            use_adaptive = (self.config.halt_max_steps > 1) and (
                (self.training and self.config.act_enabled)
                or (not self.training and self.config.act_inference)
            )

            if use_adaptive:
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                if self.training:
                    explore = (
                        torch.rand_like(q_halt_logits)
                        < self.config.halt_exploration_prob
                    )
                    min_halt = explore * torch.randint_like(
                        new_steps, low=2, high=self.config.halt_max_steps + 1
                    )
                    halted = halted & (new_steps >= min_halt)

                if (
                    self.training
                    and compute_target_q
                    and not self.config.no_ACT_continue
                ):
                    next_logits = self.inner(inner_carry, new_current_data)[-1]
                    next_halt, next_continue = next_logits
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_halt,
                            torch.maximum(next_halt, next_continue),
                        )
                    )

        return LoopTransformerCarry(
            new_inner_carry,
            new_steps,
            halted,
            new_current_data,
        ), outputs
