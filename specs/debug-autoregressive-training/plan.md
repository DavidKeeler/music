# Implementation Plan

## Checklist
- [ ] Step 1: Extract pure teacher forcing logic into function
- [ ] Step 2: Extract autoregressive logic into function
- [ ] Step 3: Replace Python if with tf.cond
- [ ] Step 4: Verify training runs successfully
- [ ] Step 5: Test tf_ratio transition behavior

---

## Step 1: Extract pure teacher forcing logic into function

**Objective:** Isolate the pure teacher forcing code path (lines 180-195) into a callable function for use with `tf.cond`.

**Implementation:**
- Create nested function `pure_teacher_forcing()` inside `train_step()`
- Move lines 180-195 (the teacher forcing branch) into this function
- Ensure it returns `{"loss": loss, "grad_norm": grad_norm, "tf_ratio": self.tf_ratio_metric.result()}`
- Keep all variable references intact (x, y, self.base_model, etc.)

**Code location:** `src/music_generation/train.py`, `MelGeneratorTraining.train_step()` method

**Test requirements:**
- Function should be callable without errors
- Return structure matches expected dictionary format
- All tensor operations remain graph-compatible

**Integration notes:**
- Function is nested inside `train_step()` to access local variables (x, y)
- No changes to the actual teacher forcing logic
- Preserves existing gradient computation and metrics

**Demo:**
```python
# Can call directly for testing
result = pure_teacher_forcing()
assert "loss" in result and "grad_norm" in result and "tf_ratio" in result
```

---

## Step 2: Extract autoregressive logic into function

**Objective:** Isolate the autoregressive training code path (lines 197-246) into a callable function for use with `tf.cond`.

**Implementation:**
- Create nested function `autoregressive_training()` inside `train_step()`
- Move lines 197-246 (the autoregressive branch) into this function
- Ensure it returns `{"loss": loss, "grad_norm": grad_norm, "tf_ratio": self.tf_ratio_metric.result()}`
- Keep all variable references intact (x, y, self.base_model, batch_size, seq_len, etc.)

**Code location:** `src/music_generation/train.py`, `MelGeneratorTraining.train_step()` method

**Test requirements:**
- Function should be callable without errors
- Return structure matches expected dictionary format (same as Step 1)
- Autoregressive loop and scheduled sampling logic preserved

**Integration notes:**
- Function is nested inside `train_step()` to access local variables
- No changes to the actual autoregressive logic
- Preserves existing gradient computation and metrics

**Demo:**
```python
# Can call directly for testing
result = autoregressive_training()
assert "loss" in result and "grad_norm" in result and "tf_ratio" in result
```

---

## Step 3: Replace Python if with tf.cond

**Objective:** Replace the problematic Python `if` statement with TensorFlow's graph-compatible `tf.cond`.

**Implementation:**
- Remove the Python `if self.tf_ratio >= 1.0 - 1e-6:` statement (line 179)
- Replace with: `return tf.cond(self.tf_ratio >= 1.0 - 1e-6, pure_teacher_forcing, autoregressive_training)`
- Ensure both functions are defined before the `tf.cond` call
- No other changes to `train_step()` logic

**Code location:** `src/music_generation/train.py`, line 179 in `MelGeneratorTraining.train_step()`

**Test requirements:**
- No `OperatorNotAllowedInGraphError` when training starts
- Correct branch executes based on `tf_ratio` value
- Return value structure is consistent regardless of branch

**Integration notes:**
- `tf.cond` evaluates predicate at execution time (not graph construction)
- Both branches must return identical structure (already ensured in Steps 1-2)
- The predicate `self.tf_ratio >= 1.0 - 1e-6` is a symbolic tensor comparison (graph-compatible)

**Demo:**
```python
# Training should start without errors
model.fit(dataset, epochs=1)
# First batch should complete successfully
```

---

## Step 4: Verify training runs successfully

**Objective:** Confirm the fix resolves the bug and training completes at least one epoch.

**Implementation:**
- Run: `python3 -m src.music_generation.train --data_dir ~/data/music/musicnet/train_data --checkpoint_dir ~/models/conducting/mel_checkpoints --epochs 1 --batch_size 4 --lr 0.00005`
- Monitor console output for errors
- Verify metrics are logged (loss, grad_norm, tf_ratio)
- Confirm epoch completes successfully

**Test requirements:**
- Command exits with status 0 (success)
- No `OperatorNotAllowedInGraphError`
- No `TypeError` about int64/int32 mismatch
- Training progresses through batches
- Checkpoint is saved after epoch

**Integration notes:**
- Use small epoch count (1) for quick verification
- Check that tf_ratio starts at 1.0 and updates correctly
- Verify both GPU and CPU execution paths work

**Demo:**
```bash
# Should see output like:
# Epoch 1/1
# 100/100 [==============================] - 45s 450ms/step - loss: 0.1234 - grad_norm: 1.23 - tf_ratio: 1.0000
# ✓ Training completed successfully
```

---

## Step 5: Test tf_ratio transition behavior

**Objective:** Verify the conditional correctly switches between teacher forcing and autoregressive paths as tf_ratio decays.

**Implementation:**
- Run training for 3-5 epochs with aggressive decay: `--epochs 5 --batch_size 4`
- Monitor tf_ratio values in logs
- Verify smooth transition when tf_ratio crosses 1.0 threshold
- Check that both paths execute without errors

**Test requirements:**
- tf_ratio should start at 1.0 (pure teacher forcing)
- tf_ratio should decay over steps
- When tf_ratio < 1.0, autoregressive path should activate
- No errors during transition
- Loss should remain stable (no sudden spikes)

**Integration notes:**
- The transition happens automatically via `update_tf_ratio()` at each step
- Both paths should produce valid gradients
- Memory usage may increase when switching to autoregressive (expected)

**Demo:**
```bash
# Run with default decay settings
python3 -m src.music_generation.train --epochs 5 --batch_size 4

# Expected log output:
# Epoch 1: tf_ratio: 1.0000 (teacher forcing)
# Epoch 2: tf_ratio: 0.9950 (teacher forcing)
# Epoch 3: tf_ratio: 0.9900 (teacher forcing)
# ...eventually...
# Epoch N: tf_ratio: 0.9899 (autoregressive kicks in)
```

---

## Notes

- **Minimal changes:** Only modifying the `train_step()` method
- **No API changes:** External interface remains identical
- **Backward compatible:** Existing checkpoints and configs work unchanged
- **Future work:** After this fix, investigate memory-efficient autoregressive implementation from `/Users/davidkeeler/code/conducting`
