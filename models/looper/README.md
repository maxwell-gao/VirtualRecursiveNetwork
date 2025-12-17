# Looper

A modular recursive reasoning architecture with adaptive computation time (ACT) and denoising implicit scheduling (DIS).

## Architecture Overview

```
models/looper/
├── __init__.py          # Public API exports
├── config.py            # Configuration dataclasses
├── blocks.py            # Transformer blocks
├── core.py              # Core loop logic
├── model.py             # Top-level model
└── README.md            # This file
```

## Key Components

### Configuration (`config.py`)

- **`LoopStateConfig`**: Defines a single loop state (e.g., `z_H`, `z_L`)
- **`LoopStageConfig`**: Defines a processing stage in the loop schedule
- **`LoopTransformerConfig`**: Main model configuration
- **`CoreCarry`**: Internal state carrier for the core
- **`ModelCarry`**: Full state carrier including ACT logic

### Blocks (`blocks.py`)

- **`TransformerBlock`**: Post-norm transformer block for stable recursive computation
- **`ReasoningModule`**: Stack of blocks with optional input injection

### Core (`core.py`)

- **`LoopTransformerCore`**: Core transformer with:
  - Embedding layer (token + puzzle + positional)
  - Multiple loop states with configurable scheduling
  - Gradient/no-gradient cycle control
  - DIS step embeddings

### Model (`model.py`)

- **`LoopTransformer`**: Full model with:
  - ACT adaptive halting logic
  - DIS progressive denoising
  - Automatic carry state management

## Usage

### Basic Import

```python
from models.looper import LoopTransformer

model = LoopTransformer(config_dict)
```

### Configuration Example

```yaml
name: looper@LoopTransformer
hidden_size: 512
num_heads: 8
outer_cycles: 3
no_grad_cycles: 2

states:
  - name: z_H
    layers: 1
  - name: z_L
    layers: 1
    share_weights_with: z_H

stages:
  - target: z_L
    sources: [z_H]
    include_inputs: true
    repeat: 6
  - target: z_H
    sources: [z_L]
    repeat: 3

readout_state: z_H
halt_state: z_H
```

## Design Principles

1. **No underscore prefixes**: Use clear, PEP8-compliant names
   - ~~`_run_schedule`~~ → `run_schedule`
   - ~~`LoopTransformerModel_ACT`~~ → `LoopTransformer`

2. **Separation of concerns**:
   - Config: Data structures only
   - Blocks: Reusable components
   - Core: Loop scheduling logic
   - Model: High-level interface

3. **Clear naming**:
   - `CoreCarry` instead of `LoopTransformerInnerCarry`
   - `ModelCarry` instead of `LoopTransformerCarry`
   - `TransformerBlock` instead of `LoopTransformerBlock`

## Migration from Old Code

### Import Changes

```python
# Old
from models.recursive_reasoning.loop_transformer import LoopTransformerModel_ACT

# New
from models.looper import LoopTransformer
```

### Config Changes

```yaml
# Old
name: recursive_reasoning.loop_transformer@LoopTransformerModel_ACT

# New
name: looper@LoopTransformer
```

### Carry State Names

```python
# Old
LoopTransformerInnerCarry(states=...)
LoopTransformerCarry(inner_carry=..., ...)

# New
CoreCarry(states=...)
ModelCarry(core_carry=..., ...)
```

## Features

- ✅ **Modular design**: Each file < 300 lines
- ✅ **Clean naming**: No underscores in public API
- ✅ **Type hints**: Full type annotations
- ✅ **Documentation**: Docstrings for all public methods
- ✅ **Separated from legacy code**: Independent of `recursive_reasoning/`

