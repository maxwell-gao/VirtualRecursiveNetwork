"""
Fabric-based pretraining script for recursive reasoning models.
Modular version using train/ package.

Compatible with both:
- Single GPU: python pretrain_fabric.py
- Multi-GPU via torchrun: torchrun --nproc-per-node 4 pretrain_fabric.py
"""

import os
import copy
import shutil
import yaml
import subprocess

import torch
import torch.distributed as dist
import tqdm
import wandb
import coolname
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric

from train.config import PretrainConfig
from train.state import init_train_state, save_train_state
from train.data import create_dataloader, create_evaluators
from train.loops import train_batch, evaluate
from train.early_stopping import EarlyStoppingWrapper
from utils.functions import get_model_source_path
from models.ema import EMAHelper


def save_code_and_config(config: PretrainConfig) -> None:
    """Save code and config for reproducibility."""
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name),
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, fabric: Fabric) -> PretrainConfig:
    """Load and synchronize config across all ranks using Fabric."""
    config = None

    if fabric.global_rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = (
                f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
            )
        if config.run_name is None:
            config.run_name = (
                f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
            )
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(
                "checkpoints", config.project_name, config.run_name
            )

    # Broadcast config from rank 0 using Fabric
    config = fabric.broadcast(config, src=0)

    return config  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig) -> None:
    """
    Main training entry point using Fabric.

    Compatible with:
    - Single GPU: python pretrain_fabric.py
    - Multi-GPU via torchrun: torchrun --nproc-per-node 4 pretrain_fabric.py
    """
    # Set matmul precision for better performance on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")

    # Determine if launched via torchrun (sets LOCAL_RANK env var)
    is_torchrun = "LOCAL_RANK" in os.environ

    if is_torchrun:
        # When using torchrun, let Fabric detect the environment
        # torchrun already sets up WORLD_SIZE, RANK, LOCAL_RANK
        fabric = Fabric(
            accelerator="cuda",
            strategy="ddp",  # Use DDP strategy for torchrun
            precision="32-true",
        )
    else:
        # Single GPU or Fabric-managed multi-GPU
        fabric = Fabric(
            accelerator="cuda",
            devices="auto",
            strategy="auto",
            precision="32-true",
        )

    fabric.launch()

    # Create CPU process group for evaluators
    cpu_process_group = None
    if fabric.world_size > 1 and dist.is_initialized():
        cpu_process_group = dist.new_group(backend="gloo")

    # Load sync'ed config
    config = load_synced_config(hydra_config, fabric)

    # Seed RNGs
    fabric.seed_everything(config.seed + fabric.global_rank)

    # Dataset
    train_epochs_per_iter = (
        config.eval_interval if config.eval_interval is not None else config.epochs
    )
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, (
        "Eval interval must be a divisor of total epochs."
    )

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        fabric=fabric,
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
    )

    try:
        eval_loader, eval_metadata = create_dataloader(
            config,
            "test",
            fabric=fabric,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
        )
    except Exception:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, fabric)

    # Progress bar and logger (rank 0 only)
    progress_bar = None
    ema_helper = None

    if fabric.global_rank == 0:
        progress_bar = tqdm.tqdm(
            total=train_state.total_steps, initial=train_state.step
        )
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True, init_timeout=300),
        )
        wandb.log(
            {"num_params": sum(x.numel() for x in train_state.model.parameters())},
            step=0,
        )
        save_code_and_config(config)

    if config.ema:
        print("Setup EMA")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Early stopping
    early_stopping = EarlyStoppingWrapper(config)
    if config.early_stopping and fabric.global_rank == 0:
        print(
            f"Early stopping enabled: monitoring '{config.early_stopping_monitor}' "
            f"with patience={config.early_stopping_patience}, mode='{config.early_stopping_mode}'"
        )

    # Training Loop
    steps_per_epoch = train_state.total_steps / config.epochs
    start_iter = int(train_state.step / steps_per_epoch / train_epochs_per_iter)

    for iter_id in range(start_iter, total_iters):
        fabric.print(
            f"[Rank {fabric.global_rank}, World Size {fabric.world_size}]: "
            f"Epoch {iter_id * train_epochs_per_iter}"
        )

        # Train
        if fabric.global_rank == 0:
            print("TRAIN")
        train_state.model.train()

        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(
                config, train_state, batch, global_batch_size, fabric=fabric
            )

            if fabric.global_rank == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)

            if config.ema:
                ema_helper.update(train_state.model)

        # Evaluation
        if iter_id >= config.min_eval_interval:
            if fabric.global_rank == 0:
                print("EVALUATE")

            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state

            train_state_eval.model.eval()
            metrics = evaluate(
                config,
                train_state_eval,
                eval_loader,
                eval_metadata,
                evaluators,
                fabric=fabric,
                cpu_group=cpu_process_group,
            )

            if fabric.global_rank == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)

            # Early stopping check
            should_stop = early_stopping.check(metrics, fabric=fabric)

            # Checkpointing
            if fabric.global_rank == 0:
                print("SAVE CHECKPOINT")
            if fabric.global_rank == 0 and (
                config.checkpoint_every_eval
                or (iter_id == total_iters - 1)
                or should_stop
            ):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

            # Early stopping exit
            if should_stop:
                if fabric.global_rank == 0:
                    print(
                        f"Early stopping: training stopped at epoch "
                        f"{(iter_id + 1) * train_epochs_per_iter}"
                    )
                break

    # Finalize
    wandb.finish()


if __name__ == "__main__":
    launch()
