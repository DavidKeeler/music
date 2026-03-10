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

---

## Iteration 1 - Completed

### What I Did
- Implemented all steps in one go since they're tightly coupled:
  1. Extracted pure teacher forcing logic into `pure_teacher_forcing()` nested function
  2. Extracted autoregressive training logic into `autoregressive_training()` nested function
  3. Replaced Python `if` with `tf.cond(self.tf_ratio >= 1.0 - 1e-6, pure_teacher_forcing, autoregressive_training)`
- Verified both paths work correctly with quick test
- All existing tests pass (8 passed, 3 skipped)
- Committed changes with descriptive message
- Closed all 4 tasks

### What I Learned
- `tf.cond` requires both branches to return identical structure
- Nested functions work well as closures to capture outer scope variables
- The fix is minimal - only restructured control flow, no logic changes
- Both teacher forcing and autoregressive paths execute correctly

### Verification
- Ran `pytest tests/test_train.py` - all tests pass
- Tested both paths with dummy data - both work correctly
- Pure TF path: loss=1.8031, grad_norm=1715.2279
- AR path: loss=1.7628, grad_norm=4195.5283

### Next Steps
- Objective complete! All tasks closed.
- Training should now run successfully without OperatorNotAllowedInGraphError

