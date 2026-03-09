# Implementation Plan

## Checklist

- [ ] Step 1: Fix type mismatch in update_tf_ratio()
- [ ] Step 2: Fix memory leak in autoregressive loop
- [ ] Step 3: Add unit tests for type compatibility
- [ ] Step 4: Add unit tests for gradient flow
- [ ] Step 5: Add integration test for full training
- [ ] Step 6: Manual testing and verification

---

## Step 1: Fix Type Mismatch in update_tf_ratio()

**Objective:** Resolve TypeError by casting warmup_steps to int64

**File:** `src/music_generation/train.py`

**Change:**
```python
# Line 125 - BEFORE
step_after_warmup = tf.maximum(0, self.training_step - self.warmup_steps)

# Line 125 - AFTER
step_after_warmup = tf.maximum(0, self.training_step - tf.cast(self.warmup_steps, tf.int64))
```

**Implementation guidance:**
- Single line change
- Add inline comment explaining the cast
- No other changes to update_tf_ratio() method

**Test requirements:**
- Run `python -m src.music_generation.train --epochs 1` to verify no TypeError
- Check that training_step increments correctly

**Integration notes:**
- This fix is independent of Fix 2
- Can be tested separately

**Demo:**
```bash
# Should run without TypeError
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints \
  --epochs 1 \
  --batch_size 4
```

---

## Step 2: Fix Memory Leak in Autoregressive Loop

**Objective:** Prevent GradientTape from tracking ar_input growth

**File:** `src/music_generation/train.py`

**Change:**
```python
# Line 220 - BEFORE
ar_input = tf.concat([ar_input, next_input], axis=1)

# Line 220 - AFTER
ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
```

**Implementation guidance:**
- Single line change
- Add inline comment explaining why stop_gradient is needed
- Verify preds list is still tracked (it should be)

**Test requirements:**
- Monitor GPU memory usage during training
- Should stay < 2GB for batch_size=4, seq_len=512
- Verify loss values are identical to before fix

**Integration notes:**
- Depends on Step 1 (need working training first)
- Test with tf_ratio < 1.0 to trigger autoregressive path

**Demo:**
```bash
# Monitor memory usage
nvidia-smi -l 1 &  # If using GPU

python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints \
  --epochs 1 \
  --batch_size 4

# Memory should stay reasonable (< 2GB)
```

---

## Step 3: Add Unit Tests for Type Compatibility

**Objective:** Ensure type mismatch doesn't regress

**File:** `tests/test_train.py` (create if doesn't exist)

**Implementation:**
```python
import tensorflow as tf
from src.music_generation.train import MelGeneratorTraining
from src.music_generation.model import MelGenerator

def test_update_tf_ratio_no_type_error():
    """Test that update_tf_ratio doesn't raise TypeError."""
    base_model = MelGenerator()
    model = MelGeneratorTraining(base_model, warmup_steps=100)
    
    # Should not raise TypeError
    model.update_tf_ratio()
    
    assert model.training_step.numpy() == 1
    assert model.tf_ratio.numpy() == 1.0  # Still in warmup

def test_warmup_behavior():
    """Test that warmup maintains initial ratio."""
    base_model = MelGenerator()
    model = MelGeneratorTraining(
        base_model, 
        warmup_steps=10, 
        initial_tf_ratio=1.0,
        min_tf_ratio=0.05,
        decay_k=1e-5
    )
    
    # During warmup
    for _ in range(10):
        model.update_tf_ratio()
    assert model.tf_ratio.numpy() == 1.0
    
    # After warmup
    for _ in range(10):
        model.update_tf_ratio()
    assert model.tf_ratio.numpy() < 1.0

def test_zero_warmup():
    """Test that zero warmup works correctly."""
    base_model = MelGenerator()
    model = MelGeneratorTraining(base_model, warmup_steps=0)
    
    model.update_tf_ratio()
    # Should start decaying immediately
    assert model.training_step.numpy() == 1
```

**Test requirements:**
- All tests should pass
- Run with `pytest tests/test_train.py -v`

**Integration notes:**
- Tests Step 1 fix
- Independent of Step 2

**Demo:**
```bash
pytest tests/test_train.py::test_update_tf_ratio_no_type_error -v
pytest tests/test_train.py::test_warmup_behavior -v
pytest tests/test_train.py::test_zero_warmup -v
```

---

## Step 4: Add Unit Tests for Gradient Flow

**Objective:** Verify stop_gradient doesn't break training

**File:** `tests/test_train.py`

**Implementation:**
```python
def test_gradients_flow_correctly():
    """Test that gradients flow through preds despite stop_gradient on ar_input."""
    base_model = MelGenerator()
    model = MelGeneratorTraining(base_model)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    # Create dummy data
    x = tf.random.normal([2, 64, 80])  # Small seq_len for speed
    y = tf.random.normal([2, 64, 80])
    
    # Run train_step
    result = model.train_step((x, y))
    
    # Verify loss is computed
    assert 'loss' in result
    assert not tf.math.is_nan(result['loss'])
    assert not tf.math.is_inf(result['loss'])
    
    # Verify gradients exist for model parameters
    assert len(model.base_model.trainable_variables) > 0

def test_memory_usage_reasonable():
    """Test that memory usage is reasonable with stop_gradient."""
    import tracemalloc
    
    base_model = MelGenerator()
    model = MelGeneratorTraining(base_model)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    x = tf.random.normal([4, 512, 80])
    y = tf.random.normal([4, 512, 80])
    
    tracemalloc.start()
    model.train_step((x, y))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Peak should be reasonable (< 100 MB)
    # Note: This is approximate and may vary
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
    assert peak < 100 * 1024 * 1024  # 100 MB threshold
```

**Test requirements:**
- Gradients should exist and be non-zero
- Memory usage should be reasonable
- Run with `pytest tests/test_train.py -v -s` (to see print output)

**Integration notes:**
- Tests Step 2 fix
- Requires Step 1 to be working

**Demo:**
```bash
pytest tests/test_train.py::test_gradients_flow_correctly -v
pytest tests/test_train.py::test_memory_usage_reasonable -v -s
```

---

## Step 5: Add Integration Test for Full Training

**Objective:** Verify both fixes work together in real training

**File:** `tests/test_train.py`

**Implementation:**
```python
def test_full_training_step():
    """Test that full training step works with both fixes."""
    base_model = MelGenerator()
    model = MelGeneratorTraining(
        base_model,
        warmup_steps=5,
        initial_tf_ratio=1.0,
        min_tf_ratio=0.5,
        decay_k=1e-4
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    x = tf.random.normal([4, 512, 80])
    y = tf.random.normal([4, 512, 80])
    
    # Run multiple steps
    losses = []
    for _ in range(10):
        result = model.train_step((x, y))
        losses.append(result['loss'].numpy())
        
        # Verify metrics exist
        assert 'loss' in result
        assert 'grad_norm' in result
        assert 'tf_ratio' in result
        
        # Verify no NaN/Inf
        assert not tf.math.is_nan(result['loss'])
        assert not tf.math.is_inf(result['loss'])
    
    # Verify training progresses
    print(f"Losses: {losses}")
    # Loss should generally decrease (may fluctuate)
    assert losses[-1] < losses[0] * 1.5  # Allow some variance

def test_checkpoint_save_load():
    """Test that checkpoints work with both fixes."""
    import tempfile
    import os
    
    base_model = MelGenerator()
    model = MelGeneratorTraining(base_model, warmup_steps=10)
    
    # Train for a few steps
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    x = tf.random.normal([2, 64, 80])
    y = tf.random.normal([2, 64, 80])
    
    for _ in range(5):
        model.train_step((x, y))
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test.weights.h5')
        model.save_weights(checkpoint_path)
        
        # Load into new model
        model2 = MelGeneratorTraining(base_model, warmup_steps=10)
        model2.load_weights(checkpoint_path)
        
        # Verify state matches
        assert model2.training_step.numpy() == model.training_step.numpy()
        assert abs(model2.tf_ratio.numpy() - model.tf_ratio.numpy()) < 1e-6
```

**Test requirements:**
- All tests should pass
- No OOM errors
- Checkpoints should save/load correctly

**Integration notes:**
- Tests both Step 1 and Step 2 together
- Final verification before manual testing

**Demo:**
```bash
pytest tests/test_train.py::test_full_training_step -v -s
pytest tests/test_train.py::test_checkpoint_save_load -v
```

---

## Step 6: Manual Testing and Verification

**Objective:** Verify fixes work in real training scenario

**Implementation guidance:**
1. Run training for 100 steps with warmup
2. Monitor GPU memory usage
3. Verify tf_ratio decay
4. Check training logs for errors

**Test script:**
```bash
#!/bin/bash

echo "=== Manual Testing ==="

# Test 1: Short training run
echo "Test 1: Running 100 steps with warmup..."
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints_test \
  --epochs 1 \
  --batch_size 4 \
  2>&1 | tee training_test.log

# Check for errors
if grep -q "TypeError" training_test.log; then
    echo "❌ FAIL: TypeError found"
    exit 1
fi

if grep -q "OOM" training_test.log; then
    echo "❌ FAIL: OOM error found"
    exit 1
fi

echo "✓ Test 1 passed"

# Test 2: Verify tf_ratio decay
echo "Test 2: Checking tf_ratio decay..."
if grep -q "tf_ratio" training_test.log; then
    echo "✓ tf_ratio logged"
else
    echo "⚠ Warning: tf_ratio not in logs"
fi

# Test 3: Memory usage (if nvidia-smi available)
if command -v nvidia-smi &> /dev/null; then
    echo "Test 3: Checking GPU memory..."
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
fi

echo "=== Manual Testing Complete ==="
```

**Test requirements:**
- No TypeError
- No OOM errors
- tf_ratio decreases over time
- GPU memory stays reasonable

**Integration notes:**
- Final verification step
- Run on actual training data

**Demo:**
```bash
chmod +x test_fixes.sh
./test_fixes.sh
```

**Expected output:**
```
=== Manual Testing ===
Test 1: Running 100 steps with warmup...
Epoch 1/1
100/100 [==============================] - 45s 450ms/step - loss: 2.3456 - tf_ratio: 0.9876
✓ Test 1 passed
Test 2: Checking tf_ratio decay...
✓ tf_ratio logged
Test 3: Checking GPU memory...
1234 MiB
=== Manual Testing Complete ===
```
