# Requirements

## Questions and Answers

### Q1: What is the exact error message or stack trace when running `python3 -m src.music_generation.train`?

**A1:** Two errors occur:

1. **Type mismatch error (initial):**
   ```
   TypeError: Input 'y' of 'Maximum' Op has type int64 that does not match type int32 of argument 'x'.
   ```

2. **Main error (during training):**
   ```
   tensorflow.python.framework.errors_impl.OperatorNotAllowedInGraphError: Using a symbolic `tf.Tensor` as a Python `bool` is not allowed.
   ```
   
   Location: `train.py`, line 179 in `train_step`:
   ```python
   if self.tf_ratio >= 1.0 - 1e-6:
   ```

The model builds successfully but fails during the first training step when trying to use `self.tf_ratio` (a TensorFlow variable) in a Python conditional statement.

### Q2: What was the teacher forcing ratio (`tf_ratio`) implemented as - a Python variable, tf.Variable, or something else?

**A2:** Looking at the code:

```python
# Line 119: tf_ratio is a tf.Variable
self.tf_ratio = tf.Variable(initial_tf_ratio, trainable=False, dtype=tf.float32, name="tf_ratio")

# Line 179: Used in Python if statement (causes the error)
if self.tf_ratio >= 1.0 - 1e-6:
```

The `tf_ratio` is a `tf.Variable`, which is a symbolic tensor. It cannot be used directly in Python conditionals during graph execution.

### Q3: Are there any other memory efficiency concerns with the new autoregressive implementation beyond fixing the conditional?

**A3:** Yes - the old implementation was very memory intensive. A memory-efficient fix was already implemented in `/Users/davidkeeler/code/conducting` (the previous project). The current bug appeared while trying to port that memory fix to this codebase.

### Q4: Should we research the memory-efficient implementation from the previous project before designing the fix?

**A4:** Let's fix the immediate bug first (the `tf.Variable` in Python conditional), then look at the memory efficiency implementation from the previous project.

### Q5: What is the expected behavior when `tf_ratio >= 1.0` - should it use pure teacher forcing (parallel training) or something else?

**A5:** Pure teacher forcing (parallel training) - this is more memory efficient since it processes the entire sequence in one forward pass instead of autoregressively.

### Q6: Are there any other requirements or constraints for the fix (e.g., must maintain backward compatibility, specific TensorFlow patterns to follow)?

**A6:** Minimal changes - keep the fix as small and focused as possible.

## Summary

The bug is clear:
- **Root cause:** Using `tf.Variable` (`self.tf_ratio`) in a Python `if` statement during graph execution
- **Location:** Line 179 in `train_step()` method
- **Fix needed:** Replace Python conditional with TensorFlow conditional (`tf.cond`)
- **Constraint:** Minimal changes only

Requirements clarification is complete.

