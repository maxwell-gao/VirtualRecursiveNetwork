# Migration Guide: loop_transformer → looper

## What Changed?

The module has been renamed from `loop_transformer` to `looper` for cleaner naming (no underscores).

## Quick Update

### Python Imports

```python
# Before
from models.loop_transformer import LoopTransformer

# After
from models.looper import LoopTransformer
```

### YAML Configs

```yaml
# Before
name: loop_transformer@LoopTransformer

# After
name: looper@LoopTransformer
```

### Config File Names

**Note**: Config file names remain unchanged (e.g., `loop_transformer.yaml`).
Only the module name in the `name:` field changes to `looper@LoopTransformer`.

## Status

✅ All internal imports updated  
✅ All 10 YAML configs updated  
✅ Documentation updated  
✅ No breaking changes to functionality

## Why?

- **Cleaner**: `looper` instead of `loop_transformer` 
- **No underscores**: Looks better
- **Shorter**: Easier to type
- **Memorable**: Distinctive name

