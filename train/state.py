"""
Training state management: model creation, checkpointing, state initialization.
"""

from typing import Any, Sequence, Union
from dataclasses import dataclass
import os

import torch
from torch import nn
from lightning.fabric import Fabric
from adam_atan2_pytorch import AdamAtan2
from muon import MuonWithAuxAdam

from train.config import PretrainConfig
from puzzle_dataset import PuzzleDatasetMetadata
from utils.functions import load_model_class
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


@dataclass
class TrainState:
    """Container for training state."""

    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[Union[float, list[float]]]
    carry: Any

    step: int
    total_steps: int


def load_checkpoint(model: nn.Module, config: PretrainConfig) -> None:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        config: PretrainConfig with checkpoint path
    """
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(
                    f"Resetting puzzle embedding as shape is different. "
                    f"Found {puzzle_emb.shape}, Expected {expected_shape}"
                )
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True)
                    .expand(expected_shape)
                    .contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


def save_train_state(config: PretrainConfig, train_state: TrainState) -> None:
    """
    Save model checkpoint.

    Args:
        config: PretrainConfig with checkpoint path
        train_state: TrainState to save
    """
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(
        train_state.model.state_dict(),
        os.path.join(config.checkpoint_path, f"step_{train_state.step}"),
    )


def get_muon_param_groups(model: nn.Module, config: PretrainConfig):
    # model is ACTLossHead -> model.model -> model.model.inner
    inner = model.model.inner

    embed_params = set()
    head_params = set()

    # Embeddings
    if hasattr(inner, "embed_tokens"):
        embed_params.update(inner.embed_tokens.parameters())
    if hasattr(inner, "embed_pos"):
        embed_params.update(inner.embed_pos.parameters())

    # Heads
    if hasattr(inner, "lm_head"):
        head_params.update(inner.lm_head.parameters())
    if hasattr(inner, "q_head"):
        head_params.update(inner.q_head.parameters())

    hidden_weights = []
    other_params = []

    for p in model.parameters():
        if p in embed_params or p in head_params:
            other_params.append(p)
        elif p.ndim >= 2:
            hidden_weights.append(p)
        else:
            other_params.append(p)

    return [
        dict(
            params=hidden_weights,
            use_muon=True,
            lr=config.muon_lr,
            weight_decay=config.muon_weight_decay,
            momentum=config.muon_momentum,
        ),
        dict(
            params=other_params,
            use_muon=False,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        ),
    ]


def create_model(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    fabric: Fabric,
) -> tuple[nn.Module, list, list]:
    """
    Create model and optimizers, broadcast weights using Fabric.

    Args:
        config: PretrainConfig
        train_metadata: Dataset metadata
        fabric: Fabric instance

    Returns:
        Tuple of (model, optimizers, optimizer_lrs)
    """
    vocab_size = train_metadata.vocab_size
    if config.arch.__pydantic_extra__ and config.arch.__pydantic_extra__.get(
        "dis_enabled", False
    ):
        vocab_size += 1

    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // fabric.world_size,
        vocab_size=vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False,  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    # Use torch.device("cuda") context like original code
    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore

        # Compile model (inside cuda context like original)
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint on rank 0
        if fabric.global_rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0 using Fabric
        if fabric.world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    # Fabric broadcast is not in-place, unlike torch.distributed.broadcast
                    broadcasted = fabric.broadcast(param, src=0)
                    param.data.copy_(broadcasted)

    # Get puzzle_emb_ndim from arch extra config (default to non-zero if not specified)
    puzzle_emb_ndim = config.arch.__pydantic_extra__.get("puzzle_emb_ndim", 1)  # type: ignore

    optimizers = []
    optimizer_lrs = []

    # 1. Puzzle Embedding Optimizer
    if puzzle_emb_ndim > 0:
        optimizers.append(
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0.0001,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=fabric.world_size,
            )
        )
        optimizer_lrs.append(config.puzzle_emb_lr)

    # 2. Main Optimizer (if not frozen)
    if not config.freeze_weights:
        if config.optimizer == "muon":
            param_groups = get_muon_param_groups(model, config)
            optimizers.append(MuonWithAuxAdam(param_groups))
            optimizer_lrs.append([config.muon_lr, config.lr])
        else:
            optimizers.append(
                AdamAtan2(
                    model.parameters(),
                    lr=0.0001,  # Needs to be set by scheduler
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2),
                )
            )
            optimizer_lrs.append(config.lr)

    return model, optimizers, optimizer_lrs


def init_train_state(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    fabric: Fabric,
) -> TrainState:
    """
    Initialize training state with Fabric.

    Args:
        config: PretrainConfig
        train_metadata: Dataset metadata
        fabric: Fabric instance

    Returns:
        Initialized TrainState
    """
    # Estimated total training steps
    total_steps = int(
        config.epochs
        * train_metadata.total_groups
        * train_metadata.mean_puzzle_examples
        / config.global_batch_size
    )

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, fabric)

    step = 0
    if config.load_checkpoint:
        try:
            basename = os.path.basename(config.load_checkpoint)
            if basename.startswith("step_"):
                step = int(basename.split("_")[1])
                if fabric.global_rank == 0:
                    print(f"Resuming from step {step}")
        except Exception:
            pass

    return TrainState(
        step=step,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )
