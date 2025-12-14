"""
Wrapper script for wandb sweep to handle Hydra argument format.
Strips '--' prefix from wandb arguments before passing to pretrain.py.

When called by torchrun, this script receives all arguments and cleans them
before passing to pretrain.py.
"""

import sys
import os
import subprocess


def main():
    # Get all arguments passed to this script
    # These will be: arch=loop_transformer-3stage, --arch.stages.0.include_inputs=True, etc.
    args = sys.argv[1:]
    
    # Strip '--' prefix from arguments that start with '--'
    # Hydra expects arguments without '--' prefix
    cleaned_args = []
    for arg in args:
        if arg.startswith("--"):
            # Remove '--' prefix for Hydra
            cleaned_args.append(arg[2:])
        else:
            # Keep as is (e.g., arch=loop_transformer-3stage)
            cleaned_args.append(arg)
    
    # Call pretrain.py with cleaned arguments
    # Use the same Python interpreter and preserve environment
    cmd = [sys.executable, "pretrain.py"] + cleaned_args
    
    # Execute pretrain.py with cleaned arguments
    # This will be called by torchrun on each process
    # Use subprocess.run to maintain compatibility with torchrun's process management
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()

