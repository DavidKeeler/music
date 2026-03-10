# Scratchpad: Fix Autoregressive Training Bug

## Understanding

The issue is in `src/music_generation/train.py` line 179. The code uses a Python `if` statement to choose between pure teacher forcing and autoregressive training:

```python
if self.tf_ratio >= 1.0 - 1e-6:
    # Pure teacher forcing (lines 180-195)
else:
    # Autoregressive training (lines 197-246)
```

This causes `OperatorNotAllowedInGraphError` because TensorFlow's graph mode doesn't allow Python control flow on tensor values. The solution is to use `tf.cond` instead.

## Plan

1. Extract pure teacher forcing logic (lines 180-195) into a nested function `pure_teacher_forcing()`
2. Extract autoregressive logic (lines 197-246) into a nested function `autoregressive_training()`
3. Both functions must return identical structure: `{"loss": ..., "grad_norm": ..., "tf_ratio": ...}`
4. Replace Python `if` with `tf.cond(self.tf_ratio >= 1.0 - 1e-6, pure_teacher_forcing, autoregressive_training)`

## Implementation Notes

- Minimal changes only - preserve all existing logic
- Both branches must return the same dictionary structure
- The functions will be closures that capture `self`, `x`, `y` from the outer scope
