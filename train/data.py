"""
Data loading utilities.
"""

from typing import Any, List

from torch.utils.data import DataLoader
from lightning.fabric import Fabric

from train.config import PretrainConfig
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class


def create_dataloader(
    config: PretrainConfig,
    split: str,
    fabric: Fabric,
    **kwargs,
) -> tuple[DataLoader, PuzzleDatasetMetadata]:
    """
    Create dataloader with Fabric-aware rank and world_size.

    Args:
        config: PretrainConfig
        split: "train" or "test"
        fabric: Fabric instance
        **kwargs: Additional arguments for PuzzleDatasetConfig

    Returns:
        Tuple of (dataloader, metadata)
    """
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths_test
            if len(config.data_paths_test) > 0 and split == "test"
            else config.data_paths,
            rank=fabric.global_rank,
            num_replicas=fabric.world_size,
            **kwargs,
        ),
        split=split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader, dataset.metadata


def create_evaluators(
    config: PretrainConfig,
    eval_metadata: PuzzleDatasetMetadata,
) -> List[Any]:
    """
    Create evaluators for validation.

    Args:
        config: PretrainConfig with evaluator configs
        eval_metadata: Evaluation dataset metadata

    Returns:
        List of evaluator instances
    """
    data_paths = (
        config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    )
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path,
                eval_metadata=eval_metadata,
                **cfg.__pydantic_extra__,
            )  # type: ignore
            evaluators.append(cls)

    return evaluators
