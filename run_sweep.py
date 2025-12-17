"""
Helper script to launch wandb sweep for hyperparameter search.
Optimized for 4x A100 GPUs using torchrun for distributed training.

Usage:
    python run_sweep.py --config config/wandb/sweep_stages_random.yaml
    python run_sweep.py --config config/wandb/sweep_stages_bayes.yaml
    python run_sweep.py --config config/wandb/sweep_stages.yaml
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Launch wandb sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="config/wandb/sweep_stages_random.yaml",
        help="Path to wandb sweep configuration file",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Limit number of runs (optional)",
    )

    args = parser.parse_args()

    # Initialize sweep
    cmd = ["wandb", "sweep", args.config]
    if args.count:
        cmd.extend(["--count", str(args.count)])

    print(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    if result.returncode == 0:
        print("\nSweep created successfully!")
        print("Use the following command to start an agent:")
        print("  wandb agent <sweep_id>")
        print("\nOr run locally with limited count:")
        print("  wandb agent <sweep_id> --count 1")
        print("\nNote: The sweep is configured for 4x A100 GPUs using torchrun.")
    else:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
