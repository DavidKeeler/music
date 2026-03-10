# Scratchpad: Fix Autoregressive Training Bug

## Understanding

The code already has `tf.cond` implemented for the main conditional (line 249-251), but there's a **type mismatch bug** in the NaN/Inf detection code that appears in BOTH nested functions:

**Problem locations:**
- Line 185 in `pure_teacher_forcing()`
- Line 237 in `autoregressive_training()`

**Current buggy code:**
```python
tf.cond(
    tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)),
    lambda: tf.print("⚠️  WARNING: NaN/Inf loss detected!"),  # Returns bool
    lambda: tf.constant(0)  # Returns int32
)
```

**Issue:** The two branches return different types (bool vs int32), causing `TypeError`.

**Fix:** Replace with simple `tf.print` that always logs (no conditional needed):
```python
tf.print("Loss:", loss, "IsNaN:", tf.math.is_nan(loss), "IsInf:", tf.math.is_inf(loss))
```

This is simpler, always provides diagnostic info, and avoids the type mismatch entirely.

## Plan

1. Replace the problematic `tf.cond` NaN/Inf detection in `pure_teacher_forcing()` (line 184-188)
2. Replace the problematic `tf.cond` NaN/Inf detection in `autoregressive_training()` (line 236-240)
3. Test that training runs without errors

## Implementation Notes

- Minimal change: just replace 5 lines with 1 line in each function
- No logic changes, just fixing the type mismatch
- The simple `tf.print` provides better diagnostics anyway (shows actual values)

---

## Iteration 1 - Completed

### What I Did
- Fixed both NaN/Inf detection issues in one commit
- Replaced nested `tf.cond` with simple `tf.print` in both functions
- Tested with dummy data - train_step executes successfully
- Committed with descriptive message explaining the fix

### Verification
- Ran quick test with dummy data
- Train step successful: loss=1.8498, grad_norm=1708.6211, tf_ratio=1.0000
- No TypeError, no OperatorNotAllowedInGraphError
- Loss diagnostics now print on every step

### What I Learned
- `tf.print` returns a bool, `tf.constant(0)` returns int32 - type mismatch
- Simple unconditional logging is better than conditional warning
- Always provides diagnostic info, easier to debug

### Next Steps
- All tasks closed
- Objective complete - training should now run without errors

