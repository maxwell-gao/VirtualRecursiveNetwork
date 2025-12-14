"""
Early stopping wrapper using Lightning's EarlyStopping callback.
"""

import torch
from lightning.fabric import Fabric
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from train.config import PretrainConfig


class EarlyStoppingWrapper:
    """Wrapper to use Lightning's EarlyStopping in a non-Lightning training loop."""

    def __init__(self, config: PretrainConfig):
        self.enabled = config.early_stopping
        if self.enabled:
            self.callback = EarlyStopping(
                monitor=config.early_stopping_monitor,
                patience=config.early_stopping_patience,
                mode=config.early_stopping_mode,
                min_delta=config.early_stopping_min_delta,
                verbose=True,
                check_finite=True,
            )
            self.should_stop = False
        else:
            self.callback = None
            self.should_stop = False

    def check(self, metrics: dict, fabric: Fabric) -> bool:
        """
        Check if training should stop based on the monitored metric.
        Uses Fabric for broadcasting the decision across all ranks.

        Args:
            metrics: Dictionary of metrics from evaluation
            fabric: Fabric instance for distributed communication

        Returns:
            True if training should stop, False otherwise
        """
        if not self.enabled or self.callback is None:
            return False

        should_stop = False

        # Only rank 0 evaluates the stopping criteria
        if fabric.global_rank == 0:
            # Find the monitored metric in the metrics dict
            monitor = self.callback.monitor
            current = None

            # Search for the metric in the nested metrics structure
            if metrics is not None:
                for set_name, set_metrics in metrics.items():
                    if isinstance(set_metrics, dict):
                        if monitor in set_metrics:
                            current = set_metrics[monitor]
                            break
                    elif monitor in str(set_name):
                        current = set_metrics
                        break

            if current is None:
                print(f"Early stopping: metric '{monitor}' not found in metrics")
            else:
                # Convert to tensor if needed
                if not isinstance(current, torch.Tensor):
                    current = torch.tensor(current)

                # Use _evaluate_stopping_criteria which returns (should_stop, reason)
                should_stop, reason = self.callback._evaluate_stopping_criteria(current)

                if should_stop:
                    self.should_stop = True
                    print(
                        f"Early stopping triggered! "
                        f"Best {monitor}: {self.callback.best_score:.6f}, "
                        f"Reason: {reason}"
                    )

        # Broadcast decision to all ranks using Fabric
        should_stop = fabric.broadcast(should_stop, src=0)
        self.should_stop = should_stop

        return should_stop
