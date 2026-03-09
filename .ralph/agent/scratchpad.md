# Scratchpad: Fix Teacher Forcing Type Mismatch and Memory Issue

## Understanding

The objective is to fix two critical bugs in `src/music_generation/train.py`:

1. **Line 125 - Type mismatch**: `self.training_step` is int64 but `self.warmup_steps` is int32, causing TypeError in subtraction
2. **Line 220 - Memory explosion**: GradientTape tracks the growing `ar_input` tensor through the autoregressive loop, causing 168 MB memory usage instead of 0.65 MB

## Analysis

**Fix 1 (Line 125):**
- Current: `step_after_warmup = tf.maximum(0, self.training_step - self.warmup_steps)`
- Issue: Type mismatch between int64 and int32
- Solution: Cast `self.warmup_steps` to int64
- Impact: Single line change, no side effects

**Fix 2 (Line 220):**
- Current: `ar_input = tf.concat([ar_input, next_input], axis=1)`
- Issue: GradientTape tracks ar_input growth, creating massive computation graph
- Solution: Wrap `next_input` with `tf.stop_gradient()` since ar_input is just a container
- Impact: Single line change, gradients still flow through `preds` list (which is what we want)

## Plan

Following the spec's 6-step plan:
1. Fix type mismatch in update_tf_ratio() - line 125
2. Fix memory leak in autoregressive loop - line 220
3. Add unit tests for type compatibility
4. Add unit tests for gradient flow
5. Add integration test for full training
6. Manual testing and verification

Each step is atomic and testable.

## Implementation Strategy

- Make both fixes first (they're independent)
- Create comprehensive test suite in tests/test_train.py
- Run tests to verify both fixes work
- Commit atomically

## Next Steps

Create tasks for each step in the plan.

---

## Iteration 1 - Completed

### What I Did
- Applied both fixes to `src/music_generation/train.py`:
  1. Line 125: Added `tf.cast(self.warmup_steps, tf.int64)` to fix type mismatch
  2. Line 223: Added `tf.stop_gradient(next_input)` to fix memory leak
- Verified both fixes work with quick test
- Committed changes with descriptive message
- Closed both fix tasks

### What I Learned
- The autoregressive training implementation was already in place from a previous iteration
- Both fixes are minimal one-line changes as specified
- Quick verification shows no TypeError and training step works correctly

### Next Steps
- Task task-1773015090-68b9 is now ready: Add unit tests for type compatibility
- After that: Add gradient flow tests and integration tests

---

## Iteration 2 - Completed

### What I Did
- Created comprehensive unit tests in `tests/test_train.py`:
  - TestTypeCompatibility: 3 tests verifying no TypeError, warmup behavior, and type consistency
  - TestGradientFlow: 3 tests verifying gradients flow correctly, loss consistency, and memory efficiency
- All 6 tests pass successfully
- Committed changes with descriptive message
- Closed both test tasks (task-1773015090-68b9 and task-1773015093-1d9d)

### What I Learned
- Tests use SimpleMelGenerator mock to avoid complex dependencies
- Type compatibility tests verify warmup behavior and int64 casting
- Gradient flow tests verify stop_gradient doesn't affect loss values (only backprop)
- Memory efficiency test confirms GradientTape doesn't track ar_input growth

### Next Steps
- Check remaining tasks - integration test may be next
- All unit tests are complete and passing

