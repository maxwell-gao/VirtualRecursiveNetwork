"""
Training and evaluation loops.
"""

from typing import Any, List, Optional, Dict, Tuple
import os

import torch
from lightning.fabric import Fabric

from train.config import PretrainConfig
from train.state import TrainState
from train.schedulers import compute_lr
from puzzle_dataset import PuzzleDatasetMetadata
from utils.dis import get_dis_target
from models.losses import IGNORE_LABEL_ID


def _train_dis_mask_method(
    train_state: TrainState,
    batch: Dict[str, torch.Tensor],
    dis_max_steps: int,
    dis_schedule: str,
    vocab_size: int,
    global_batch_size: int,
    config: "PretrainConfig",
    fabric: "Fabric",
) -> Dict[str, torch.Tensor]:
    """
    Train using DIS with mask method.
    
    Following the paper (Figure 3), we optimize after EACH step,
    not accumulate gradients across all steps.
    """
    # Generate fixed noise for monotonic masking
    B, L = batch["labels"].shape
    mask_noise = torch.rand(B, L, device="cuda")

    accumulated_metrics = {}

    for step in range(dis_max_steps):
        # Generate target with decreasing mask rate
        y_true = batch["labels"]
        y_target = get_dis_target(
            y_true,
            step,
            dis_max_steps,
            vocab_size,
            vocab_size - 1,  # mask_token_id is the last token
            IGNORE_LABEL_ID,
            dis_schedule,
            noise=mask_noise,
        )

        # Protect input clues from being masked
        if "inputs" in batch and batch["inputs"].shape == batch["labels"].shape:
            is_clue = batch["inputs"] == batch["labels"]
            y_target = torch.where(is_clue, y_true, y_target)

        batch_step = batch.copy()
        batch_step["labels"] = y_target

        # Forward
        step_tensor = torch.tensor(step, device="cuda", dtype=torch.long)
        train_state.carry, loss, metrics, _, _ = train_state.model(
            carry=train_state.carry,
            batch=batch_step,
            return_keys=[],
            step=step_tensor,
        )

        # Accumulate metrics for logging
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            if k not in accumulated_metrics:
                accumulated_metrics[k] = v
            else:
                accumulated_metrics[k] += v

        # === Paper-style: optimize after EACH step ===
        # Scale loss by batch size only (not by dis_max_steps)
        ((1 / global_batch_size) * loss).backward()
        
        # Allreduce gradients
        _allreduce_gradients(train_state, fabric)
        
        # Optimizer step (following paper Figure 3)
        _apply_optimizers(config, train_state, fabric)

    return {k: v / dis_max_steps for k, v in accumulated_metrics.items()}


def _train_dis_loss_method(
    train_state: TrainState,
    batch: Dict[str, torch.Tensor],
    dis_max_steps: int,
    global_batch_size: int,
    config: "PretrainConfig",
    fabric: "Fabric",
) -> Dict[str, torch.Tensor]:
    """
    Train using DIS with loss method (enforce improvement).
    
    This is a variant that uses improvement penalty instead of masked targets.
    Unlike mask method, this requires all steps' losses before computing the final loss,
    so we use single optimizer step at the end (not per-step).
    """
    losses = []
    accumulated_metrics = {}

    for step in range(dis_max_steps):
        batch_step = batch.copy()

        # Forward
        step_tensor = torch.tensor(step, device="cuda", dtype=torch.long)
        train_state.carry, loss, metrics, _, _ = train_state.model(
            carry=train_state.carry,
            batch=batch_step,
            return_keys=[],
            step=step_tensor,
        )

        losses.append(loss)

        # Accumulate metrics
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            if k not in accumulated_metrics:
                accumulated_metrics[k] = v
            else:
                accumulated_metrics[k] += v

    # Compute total loss: L_final + sum(max(0, L_k - L_{k-1}.detach()))
    total_loss = losses[-1]
    for k in range(1, len(losses)):
        improvement_penalty = torch.relu(losses[k] - losses[k - 1].detach())
        total_loss = total_loss + improvement_penalty

    # Backward and optimize (single step for loss method)
    ((1 / global_batch_size) * total_loss).backward()
    _allreduce_gradients(train_state, fabric)
    _apply_optimizers(config, train_state, fabric)

    return {k: v / dis_max_steps for k, v in accumulated_metrics.items()}


def _train_standard_act(
    train_state: TrainState,
    batch: Dict[str, torch.Tensor],
    global_batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Train using standard ACT method."""
    # Init carry if needed
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    # Backward
    ((1 / global_batch_size) * loss).backward()

    return metrics


def _apply_optimizers(
    config: PretrainConfig,
    train_state: TrainState,
    fabric: Fabric,
) -> float:
    """Apply optimizer step and return learning rate."""
    lr_this_step = None

    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        if isinstance(base_lr, list):
            for param_group, group_base_lr in zip(optim.param_groups, base_lr):
                lr_this_step = compute_lr(group_base_lr, config, train_state)
                param_group["lr"] = lr_this_step
        else:
            lr_this_step = compute_lr(base_lr, config, train_state)
            for param_group in optim.param_groups:
                param_group["lr"] = lr_this_step

        if config.grad_clip_norm > 0.0:
            fabric.clip_gradients(
                train_state.model, optim, max_norm=config.grad_clip_norm
            )

        optim.step()
        optim.zero_grad()

    return lr_this_step


def _allreduce_gradients(train_state: TrainState, fabric: Fabric) -> None:
    """All-reduce gradients across ranks."""
    if fabric.world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                reduced_grad = fabric.all_reduce(param.grad, reduce_op="sum")
                param.grad.data.copy_(reduced_grad)


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    fabric: Fabric,
) -> Optional[dict]:
    """
    Train on a single batch using Fabric for gradient synchronization.

    Args:
        config: PretrainConfig
        train_state: TrainState
        batch: Input batch
        global_batch_size: Global batch size across all ranks
        fabric: Fabric instance

    Returns:
        Metrics dictionary (only on rank 0), or None
    """
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return None

    # To device (use .cuda() like original code for consistency)
    batch = {k: v.cuda() for k, v in batch.items()}

    # Check for DIS
    model_ref = train_state.model
    if hasattr(model_ref, "model"):  # ACTLossHead
        model_ref = model_ref.model
    if hasattr(model_ref, "_orig_mod"):  # torch.compile
        model_ref = model_ref._orig_mod

    dis_enabled = getattr(model_ref.config, "dis_enabled", False)

    # Zero grads before training
    for optim in train_state.optimizers:
        optim.zero_grad()

    # Execute training method based on configuration
    if dis_enabled:
        # Initialize carry for DIS (always start fresh)
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

        # Choose DIS training method
        # Note: DIS methods handle optimizer steps internally (per-step optimization)
        dis_loss_method = getattr(model_ref.config, "dis_loss_method", "mask")
        if dis_loss_method == "mask":
            metrics = _train_dis_mask_method(
                train_state,
                batch,
                model_ref.config.dis_max_steps,
                model_ref.config.dis_schedule,
                model_ref.config.vocab_size,
                global_batch_size,
                config,
                fabric,
            )
        else:
            metrics = _train_dis_loss_method(
                train_state,
                batch,
                model_ref.config.dis_max_steps,
                global_batch_size,
                config,
                fabric,
            )
        # DIS already handled optimizer steps internally
        lr_this_step = train_state.optimizer_lrs[0]
        if isinstance(lr_this_step, list):
            lr_this_step = lr_this_step[0]
        lr_this_step = compute_lr(lr_this_step, config, train_state)
    else:
        metrics = _train_standard_act(train_state, batch, global_batch_size)

        # Synchronize gradients across ranks
        _allreduce_gradients(train_state, fabric)

        # Apply optimizer step
        lr_this_step = _apply_optimizers(config, train_state, fabric)

    # Reduce metrics (use reduce to dst=0, not all_reduce)
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(
            sorted(metrics.keys())
        )  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if fabric.world_size > 1:
            # Use reduce to rank 0 only (like original dist.reduce)
            import torch.distributed as dist

            dist.reduce(metric_values, dst=0)

        if fabric.global_rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {
                f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
                for k, v in reduced_metrics.items()
            }

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics

    return None


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    fabric: Fabric,
    cpu_group: Optional[Any],
) -> Optional[dict]:
    """
    Evaluate model using Fabric for distributed communication.

    Args:
        config: PretrainConfig
        train_state: TrainState
        eval_loader: Evaluation data loader
        eval_metadata: Evaluation dataset metadata
        evaluators: List of evaluators
        fabric: Fabric instance
        cpu_group: CPU process group for evaluators

    Returns:
        Metrics dictionary (only on rank 0), or None
    """
    import torch.distributed as dist

    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if fabric.global_rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            # To device (use .cuda() like original code)
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if fabric.global_rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(
                            v.cpu()
                        )  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())),
                    dtype=torch.float32,
                    device="cuda",
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds,
                os.path.join(
                    config.checkpoint_path,
                    f"step_{train_state.step}_all_preds.{fabric.global_rank}",
                ),
            )

        del save_preds

        # Reduce to rank 0 (use dist.reduce like original, not all_reduce)
        if metric_values is not None:
            if fabric.world_size > 1:
                dist.reduce(metric_values, dst=0)

            if fabric.global_rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if fabric.global_rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")

        for i, evaluator in enumerate(evaluators):
            if fabric.global_rank == 0:
                print(
                    f"Running evaluator {i + 1}/{len(evaluators)}: {evaluator.__class__.__name__}"
                )

            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(
                evaluator_save_path,
                rank=fabric.global_rank,
                world_size=fabric.world_size,
                group=cpu_group,
            )
            if fabric.global_rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")

        if fabric.global_rank == 0:
            print("All evaluators completed!")

    return reduced_metrics
