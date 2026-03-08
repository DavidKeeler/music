# Implementation Plan: Debug Training

## Checklist
- [ ] Step 1: Fix TensorFlow graph execution error in train_step
- [ ] Step 2: Add shape validation for dataset batches
- [ ] Step 3: Verify model builds correctly on first batch
- [ ] Step 4: Test training for one complete epoch
- [ ] Step 5: Validate checkpoint saving

---

## Step 1: Fix TensorFlow Graph Execution Error

**Objective**: Replace Python control flow with TensorFlow graph-compatible operations in the train_step method.

**Implementation**:
- Locate the NaN/Inf detection code in `src/music_generation/train.py` (line ~87)
- Replace `if tf.math.is_nan(loss) or tf.math.is_inf(loss):` with `tf.cond`
- Use `tf.math.logical_or` to combine conditions
- Ensure both branches of `tf.cond` return compatible types

**Code Change**:
```python
# Replace this:
if tf.math.is_nan(loss) or tf.math.is_inf(loss):
    tf.print("⚠️  WARNING: NaN/Inf loss detected!")

# With this:
tf.cond(
    tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)),
    lambda: tf.print("⚠️  WARNING: NaN/Inf loss detected!"),
    lambda: tf.constant(0)
)
```

**Tests**:
- Run training command with batch_size=4
- Verify no `OperatorNotAllowedInGraphError` occurs
- Confirm training step executes without graph errors

**Integration**:
- No changes to other components required
- Existing train_step signature remains unchanged

**Demo**:
Training starts and processes first batch without TensorFlow graph execution errors.

---

## Step 2: Add Shape Validation for Dataset Batches

**Objective**: Verify dataset produces correctly shaped tensors before training begins.

**Implementation**:
- In `train()` function, after dataset creation, take one sample batch
- Print shapes of x and y tensors
- Add assertions to verify expected dimensions: `[batch_size, seq_len, 80]`
- Verify batch_size matches configuration

**Code Addition** (in `train()` function):
```python
# After dataset creation, before model.fit()
print("Validating dataset shapes...")
for x, y in dataset.take(1):
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")
    tf.debugging.assert_equal(tf.shape(x)[0], batch_size, message="Batch size mismatch")
    tf.debugging.assert_equal(tf.shape(x)[2], 80, message="Mel channels should be 80")
    tf.debugging.assert_equal(tf.shape(x), tf.shape(y), message="Input and target shapes must match")
print("✓ Dataset validation passed")
```

**Tests**:
- Run training and check console output shows shape validation
- Verify shapes match expected dimensions
- Confirm validation passes before training starts

**Integration**:
- Runs once before model.fit() is called
- Does not affect training loop performance
- Provides early failure if dataset is misconfigured

**Demo**:
Console displays dataset shapes and validation confirmation before training begins.

---

## Step 3: Verify Model Builds Correctly on First Batch

**Objective**: Ensure model architecture initializes properly and produces expected output shapes.

**Implementation**:
- After model creation, call model on a sample batch to trigger build
- Print model summary after build
- Verify output shape matches input shape
- Check parameter count is reasonable (not 0 or unexpectedly large)

**Code Addition** (in `train()` function):
```python
# After model creation, before model.fit()
print("Building model with sample batch...")
for x, y in dataset.take(1):
    _ = model.base_model(x, training=False)
    break

print("\nModel architecture:")
model.base_model.summary()

# Verify model has trainable parameters
total_params = sum([tf.size(v).numpy() for v in model.base_model.trainable_variables])
print(f"\nTotal trainable parameters: {total_params:,}")
assert total_params > 0, "Model has no trainable parameters!"
```

**Tests**:
- Run training and verify model summary is displayed
- Check parameter count is greater than 0
- Confirm model builds without errors

**Integration**:
- Runs once before training starts
- Does not modify model architecture
- Provides diagnostic information for debugging

**Demo**:
Model summary and parameter count displayed before training begins.

---

## Step 4: Test Training for One Complete Epoch

**Objective**: Run training end-to-end and verify it completes one full epoch without crashes.

**Implementation**:
- Run training command with small configuration:
  - `--epochs 1`
  - `--batch_size 4`
  - `--lr 0.0001`
- Monitor console output for errors
- Verify loss values are logged each step
- Confirm epoch completes successfully

**Command**:
```bash
python3 -m src.music_generation.train \
    --data_dir ~/data/music/musicnet/train_data \
    --checkpoint_dir ~/models/conducting/mel_checkpoints \
    --epochs 1 \
    --batch_size 4 \
    --lr 0.0001
```

**Tests**:
- Training completes without exceptions
- Loss values are finite (not NaN/Inf)
- Progress bar shows all steps completed
- Final epoch summary is displayed

**Integration**:
- Full end-to-end test of training pipeline
- Validates all previous fixes work together
- Confirms dataset, model, and training loop are compatible

**Demo**:
Training runs from start to finish for one epoch, displaying loss values and completing successfully.

---

## Step 5: Validate Checkpoint Saving

**Objective**: Ensure model checkpoints are saved correctly after training.

**Implementation**:
- After training completes, check checkpoint directory exists
- Verify checkpoint files are created
- List checkpoint files and their sizes
- Optionally load checkpoint to verify it's valid

**Code Addition** (at end of `train()` function):
```python
# After model.fit() completes
print("\nValidating checkpoints...")
checkpoint_path = os.path.join(checkpoint_dir, "mel_generator.h5")
if os.path.exists(checkpoint_path):
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    print(f"✓ Checkpoint saved: {checkpoint_path} ({file_size:.2f} MB)")
else:
    print(f"⚠️  Warning: Checkpoint not found at {checkpoint_path}")
```

**Tests**:
- Run training for 1 epoch
- Check checkpoint directory for saved files
- Verify file size is reasonable (> 0 bytes)
- Confirm checkpoint can be loaded (optional)

**Integration**:
- Validates ModelCheckpoint callback is configured correctly
- Ensures training progress can be saved and resumed
- Provides confirmation that training artifacts are persisted

**Demo**:
After training completes, checkpoint file path and size are displayed, confirming successful save.
