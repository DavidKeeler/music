# Implementation Plan

## Checklist
- [ ] Step 1: Simplify MelGeneratorTraining class
- [ ] Step 2: Update train() function and remove teacher forcing parameters
- [ ] Step 3: Clean up config.py
- [ ] Step 4: Fix dataset prefetching
- [ ] Step 5: Update memory profiling script
- [ ] Step 6: Add shape validation and NaN detection
- [ ] Step 7: Test and validate memory reduction

---

## Step 1: Simplify MelGeneratorTraining Class

**Objective:** Replace autoregressive loop with single forward pass.

**Implementation:**
Rewrite `MelGeneratorTraining` in `src/music_generation/train.py`:

**Files to modify:**
- `src/music_generation/train.py`

**Code changes:**
```python
class MelGeneratorTraining(tf.keras.Model):
    """Simplified training wrapper with parallel processing."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            # Single forward pass - causal mask prevents future leakage
            preds = self.base_model(x, training=True)
            
            # Compare predictions at t with targets at t+1
            loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
        
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {"loss": loss}
```

**Delete:**
- `__init__` parameters: `initial_tf_ratio`, `decay_k`, `min_ratio`, `context_len`
- `compute_tf_ratio()` method
- Entire autoregressive loop in `train_step()`
- Teacher forcing logic
- `tf_ratio` return value

**Tests:**
1. Unit test: Verify train_step runs without errors
2. Unit test: Verify loss shape is scalar
3. Unit test: Verify gradients are computed

**Integration:**
Breaking change - requires updating all calls to `MelGeneratorTraining()`

**Demo:**
Run single training step, show it completes in <1 second (vs ~10 seconds before).

---

## Step 2: Update train() Function

**Objective:** Remove teacher forcing parameters from training function.

**Implementation:**
Update `train()` in `src/music_generation/train.py`:

**Files to modify:**
- `src/music_generation/train.py`

**Code changes:**
```python
def train(data_dir, cache_dir, checkpoint_dir, batch_size=BATCH_SIZE, 
          epochs=NUM_EPOCHS, lr=LEARNING_RATE, resume_from=None):
    """Train the mel generator model."""
    
    check_batch_size(batch_size)
    
    print(f"Loading dataset from {data_dir}")
    dataset = create_dataset(data_dir, cache_dir, batch_size)
    
    steps_per_epoch = 100
    total_steps = steps_per_epoch * epochs
    
    lr_schedule = WarmupCosineSchedule(lr, warmup_steps=5 * steps_per_epoch, total_steps=total_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5)
    
    base_model = MelGenerator()
    model = MelGeneratorTraining(base_model)  # Simplified - no extra params
    model.compile(optimizer=optimizer)
    
    # ... rest of function unchanged
```

**Delete from function signature:**
- Remove any teacher forcing parameters if present

**Delete from main():**
```python
# Remove these argument definitions:
parser.add_argument('--initial_tf_ratio', ...)
parser.add_argument('--tf_decay_k', ...)
parser.add_argument('--min_tf_ratio', ...)
```

**Tests:**
1. Verify train() runs without errors
2. Verify model creation succeeds
3. Verify training loop starts

**Integration:**
Update any scripts that call `train()` with teacher forcing parameters

**Demo:**
Run training for 1 epoch, show it completes successfully.

---

## Step 3: Clean Up config.py

**Objective:** Remove unused teacher forcing configuration.

**Implementation:**
Edit `src/music_generation/config.py`:

**Files to modify:**
- `src/music_generation/config.py`

**Code changes:**
```python
# DELETE these lines:
# Teacher forcing parameters
INITIAL_TF_RATIO = 1.0
TF_DECAY_K = 1e-5
MIN_TF_RATIO = 0.05
```

**Keep everything else:**
- Model parameters (D_MODEL, NUM_HEADS, etc.)
- Training parameters (BATCH_SIZE, LEARNING_RATE, etc.)
- Dataset parameters (SEQ_LEN, N_MELS, etc.)

**Tests:**
1. Verify config imports without errors
2. Verify no code references deleted constants

**Integration:**
Check for any imports of deleted constants and remove them

**Demo:**
Import config, verify no errors.

---

## Step 4: Fix Dataset Prefetching

**Objective:** Use bounded prefetch to prevent memory over-allocation.

**Implementation:**
Update `create_dataset()` in `src/music_generation/dataset.py`:

**Files to modify:**
- `src/music_generation/dataset.py`

**Code changes:**
```python
def create_dataset(data_dir: str, cache_dir: str, batch_size: int, shuffle: bool = True):
    """Create tf.data.Dataset for training."""
    # ... existing code ...
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)  # Changed from tf.data.AUTOTUNE
    
    return dataset
```

**Tests:**
1. Verify dataset creation succeeds
2. Verify iteration works
3. Profile memory during dataset iteration

**Integration:**
Non-breaking change, transparent to users

**Demo:**
Create dataset, iterate through batches, show memory stays bounded.

---

## Step 5: Update Memory Profiling Script

**Objective:** Remove teacher forcing parameters from profiling script.

**Implementation:**
Update `scripts/profile_memory.py`:

**Files to modify:**
- `scripts/profile_memory.py`

**Code changes:**
```python
def main():
    parser = argparse.ArgumentParser(description='Profile memory usage during training')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    # DELETE: initial_tf_ratio, tf_decay_k, min_tf_ratio arguments
    
    args = parser.parse_args()
    
    # ... tracemalloc setup ...
    
    train(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
        # DELETE: initial_tf_ratio, tf_decay_k, min_tf_ratio parameters
    )
```

**Tests:**
1. Verify script runs without errors
2. Verify memory profiling output is generated

**Integration:**
Update any documentation referencing old parameters

**Demo:**
Run profiling script, show memory usage is dramatically lower.

---

## Step 6: Add Shape Validation and NaN Detection

**Objective:** Add defensive checks for debugging.

**Implementation:**
Enhance `train_step()` in `src/music_generation/train.py`:

**Files to modify:**
- `src/music_generation/train.py`

**Code changes:**
```python
def train_step(self, data):
    x, y = data
    
    # Shape validation (can be removed after testing)
    tf.debugging.assert_equal(
        tf.shape(x), tf.shape(y),
        message="Input and target shapes must match"
    )
    
    with tf.GradientTape() as tape:
        preds = self.base_model(x, training=True)
        
        # Verify prediction shape
        tf.debugging.assert_equal(
            tf.shape(preds), tf.shape(x),
            message="Prediction shape must match input shape"
        )
        
        loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
        
        # NaN/Inf detection
        if tf.math.is_nan(loss) or tf.math.is_inf(loss):
            tf.print("⚠️  WARNING: NaN/Inf loss detected!")
            return {"loss": loss}
    
    grads = tape.gradient(loss, self.base_model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
    
    return {"loss": loss}
```

**Tests:**
1. Test with valid data (assertions pass)
2. Test with mismatched shapes (assertion fails)
3. Test with NaN-inducing data (warning printed)

**Integration:**
Non-breaking, adds safety checks

**Demo:**
Run training, show no assertion errors with valid data.

---

## Step 7: Test and Validate Memory Reduction

**Objective:** Measure and validate the memory improvements.

**Implementation:**
Create comprehensive test script:

**Files to create:**
- `tests/test_parallel_training.py`

**Code:**
```python
"""Test parallel training implementation."""
import tensorflow as tf
import pytest
from src.music_generation.model import MelGenerator
from src.music_generation.train import MelGeneratorTraining


def test_train_step_shape():
    """Test train_step produces correct shapes."""
    model = MelGenerator()
    training_model = MelGeneratorTraining(model)
    training_model.compile(optimizer='adam')
    
    # Create dummy data
    batch_size, seq_len, n_mels = 2, 128, 80
    x = tf.random.normal([batch_size, seq_len, n_mels])
    y = tf.random.normal([batch_size, seq_len, n_mels])
    
    # Run train step
    result = training_model.train_step((x, y))
    
    # Verify loss is scalar
    assert result['loss'].shape == ()
    assert not tf.math.is_nan(result['loss'])


def test_memory_usage():
    """Test memory usage is reasonable."""
    import tracemalloc
    
    tracemalloc.start()
    
    model = MelGenerator()
    training_model = MelGeneratorTraining(model)
    training_model.compile(optimizer='adam')
    
    # Create data
    x = tf.random.normal([4, 128, 80])
    y = tf.random.normal([4, 128, 80])
    
    # Run multiple steps
    for _ in range(10):
        training_model.train_step((x, y))
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Peak should be < 500MB (was ~50GB before)
    assert peak < 500 * 1024 * 1024, f"Peak memory too high: {peak / 1024**2:.1f} MB"


def test_loss_decreases():
    """Test loss decreases over steps."""
    model = MelGenerator()
    training_model = MelGeneratorTraining(model)
    training_model.compile(optimizer='adam')
    
    # Create data
    x = tf.random.normal([4, 128, 80])
    y = x + tf.random.normal([4, 128, 80]) * 0.1  # Similar to input
    
    losses = []
    for _ in range(20):
        result = training_model.train_step((x, y))
        losses.append(result['loss'].numpy())
    
    # Loss should decrease
    assert losses[-1] < losses[0], "Loss should decrease during training"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

**Manual validation:**
```bash
# 1. Profile memory before/after
python scripts/profile_memory.py \
  --data_dir ~/data/music/musicnet/train_data \
  --batch_size 4 \
  --epochs 2

# 2. Run full training
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints \
  --epochs 50 \
  --batch_size 8  # Can use larger batch now!

# 3. Generate audio sample
python -m src.music_generation.inference \
  --checkpoint ./checkpoints/best_model.h5 \
  --seed_audio seed.wav \
  --output generated.wav
```

**Tests:**
1. Run pytest suite
2. Profile memory (should be ~100x lower)
3. Train for 50 epochs (should complete without OOM)
4. Generate audio (should sound good)

**Integration:**
Final validation of all changes

**Demo:**
Show side-by-side comparison:
- Memory: 50GB → 500MB
- Speed: 10 sec/step → 0.1 sec/step
- Quality: Equivalent or better

---

## Summary

This plan fixes the critical training loop bug that caused ~100x memory overhead. The fix is simple: replace the autoregressive loop with a single forward pass, relying on causal masking to prevent future information leakage.

**Key changes:**
1. Simplify `MelGeneratorTraining` (remove loop, teacher forcing)
2. Remove teacher forcing parameters from config and functions
3. Fix dataset prefetching (AUTOTUNE → 2)
4. Add validation and NaN detection
5. Comprehensive testing

**Expected impact:**
- Memory: ~100x reduction (50GB → 500MB)
- Speed: ~100x faster (10 sec/step → 0.1 sec/step)
- Quality: Equivalent or better (correct training method)
- Simplicity: Much simpler code

**Estimated time:** ~1 hour for implementation and testing
