# WandB Sweep Configuration Guide

This directory contains WandB Sweep configuration files for searching `stages` parameters in `loop_transformer-3stage.yaml`.

## Configuration Files

All sweep configuration files are located in `config/wandb/`:

### 1. `config/wandb/sweep_stages.yaml` - Grid Search

- **Method**: Grid Search
- **Characteristics**: Exhaustive search over all parameter combinations
- **Use Case**: Small parameter space, need comprehensive search
- **Combinations**: 4 × 2 × 4 × 2 × 4 = 256 combinations

### 2. `config/wandb/sweep_stages_random.yaml` - Random Search (Recommended)

- **Method**: Random Search
- **Characteristics**: Random sampling, more efficient
- **Use Case**: Large parameter space, need quick exploration
- **Recommendation**: Recommended for most cases

### 3. `config/wandb/sweep_stages_bayes.yaml` - Bayesian Optimization

- **Method**: Bayesian Optimization
- **Characteristics**: Intelligent parameter selection based on historical results
- **Use Case**: Need efficient optimal parameter finding
- **Recommendation**: Recommended when run count is limited

## Parameters Being Searched

All configurations search the following stages parameters:

- **Stage 0 (z_L)**:
  - `repeat`: 1-4
  - `include_inputs`: true/false

- **Stage 1 (z_M)**:
  - `repeat`: 1-4
  - `include_inputs`: true/false

- **Stage 2 (z_H)**:
  - `repeat`: 1-4
  - (Note: This stage does not have include_inputs by default)

## Hardware Configuration

**Optimized for 4x A100 GPUs**: All sweep configurations use `torchrun --nproc-per-node=4` for distributed training across 4 GPUs.

## Usage

### Method 1: Using Helper Script (Recommended)

```bash
# Use random search (recommended)
python run_sweep.py --config config/wandb/sweep_stages_random.yaml

# Use Bayesian optimization
python run_sweep.py --config config/wandb/sweep_stages_bayes.yaml

# Use grid search (large parameter space, use with caution)
python run_sweep.py --config config/wandb/sweep_stages.yaml
```

### Method 2: Direct wandb Command

```bash
# 1. Initialize sweep
wandb sweep config/wandb/sweep_stages_random.yaml

# 2. Start agent (will display sweep_id)
# Output example: wandb agent <entity>/<project>/<sweep_id>
wandb agent <sweep_id>

# 3. Run on multiple machines (optional)
# Run the same agent command on each machine
```

### Method 3: Limit Run Count

```bash
# Run only 10 trials
python run_sweep.py --config config/wandb/sweep_stages_random.yaml --count 10

# Or use wandb command
wandb sweep config/wandb/sweep_stages_random.yaml --count 10
```

## Configuration Details

### Optimization Target

- **Metric**: `exact_accuracy`
- **Goal**: `maximize` (maximize accuracy)

### Early Termination

Hyperband early stopping is configured:

- `min_iter`: 3 (minimum 3 evaluation runs)
- `max_iter`: 10 (maximum 10 evaluation runs)
- `s`: 2
- `eta`: 3

### Distributed Training

- **GPUs**: 4x A100
- **Method**: torchrun with DDP (Distributed Data Parallel)
- **Command**: `torchrun --nproc-per-node=4 pretrain_sweep_wrapper.py`
- **Wrapper Script**: `pretrain_sweep_wrapper.py` is used to convert wandb's `--arg=value` format to Hydra's `arg=value` format

## Customizing Search Space

To modify search ranges, edit the corresponding YAML file:

```yaml
parameters:
  arch.stages.0.repeat:
    distribution: int_uniform
    min: 1  # Modify minimum value
    max: 8  # Modify maximum value
```

## Notes

1. **Install wandb**: `pip install wandb` or `uv pip install wandb`
2. **Login to wandb**: First-time use requires `wandb login`
3. **Hydra parameter format**: wandb sweep automatically converts parameters to Hydra format (e.g., `arch.stages.0.repeat=3`)
4. **Project name**: Modify the `project` field in the config file to specify wandb project name
5. **Resource consumption**: Grid Search runs all combinations, ensure sufficient computational resources
6. **Multi-GPU**: The configuration is optimized for 4x A100 GPUs. For different GPU counts, modify `--nproc-per-node` in the command section
7. **Wrapper script**: The sweep configs use `pretrain_sweep_wrapper.py` to convert wandb's `--arg=value` format to Hydra's `arg=value` format. This script is automatically used by the sweep configurations.

## Viewing Results

1. Visit [wandb.ai](https://wandb.ai) to view sweep progress and results
2. Compare all runs on the project page
3. Use wandb's parallel coordinates plot to view parameter-performance relationships

## Troubleshooting

### GPU Issues

- Ensure all 4 GPUs are available: `nvidia-smi`
- Check CUDA visibility: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Verify torchrun installation: `python -c "import torch; print(torch.__version__)"`

### WandB Issues

- Check wandb login: `wandb login`
- Verify project access permissions
- Check network connectivity for wandb cloud
