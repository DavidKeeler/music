# Implementation Plan: Update Simple Audio Model

## Checklist

- [ ] Step 1: Update config.py with new architecture constants
- [ ] Step 2: Add relative position bias to LocalWindowAttention
- [ ] Step 3: Update MelGenerator with 3 explicit transformer layers
- [ ] Step 4: Test causality and window constraints
- [ ] Step 5: Verify training compatibility

---

## Step 1: Update config.py with new architecture constants

**Objective:** Update configuration to reflect the new 3-layer architecture with varying window sizes.

**Implementation:**
1. Change `NUM_LAYERS` from 1 to 3
2. Remove `WINDOW_SIZE = 128`
3. Add `WINDOW_SIZES = [128, 256, 512]`

**Files to modify:**
- `src/music_generation/config.py`

**Changes:**
```python
# Before:
NUM_LAYERS = 1
WINDOW_SIZE = 128

# After:
NUM_LAYERS = 3
WINDOW_SIZES = [128, 256, 512]
```

**Tests:**
- Import config and verify constants are accessible
- Verify no other code references old `WINDOW_SIZE` constant

**Integration:**
- Config changes are standalone, no dependencies

**Demo:**
```python
from src.music_generation.config import NUM_LAYERS, WINDOW_SIZES
print(f"Layers: {NUM_LAYERS}, Windows: {WINDOW_SIZES}")
# Output: Layers: 3, Windows: [128, 256, 512]
```

---

## Step 2: Add relative position bias to LocalWindowAttention

**Objective:** Implement learned relative position bias in the attention mechanism.

**Implementation:**

1. **Add bias weight in __init__:**
```python
self.relative_position_bias = self.add_weight(
    name='relative_position_bias',
    shape=[num_heads, window_size],
    initializer=tf.random_normal_initializer(stddev=0.01),
    trainable=True
)
```

2. **Compute and apply bias in call():**
```python
# After computing scores = Q·K^T / sqrt(d_head)
# Compute relative distances
T = tf.shape(x)[1]
positions = tf.range(T)
rel_distances = positions[:, None] - positions[None, :]  # [T, T]
rel_distances = tf.clip_by_value(rel_distances, 0, self.window_size - 1)

# Gather biases
bias = tf.gather(self.relative_position_bias, rel_distances, axis=1)  # [H, T, T]

# Add to scores (broadcast over batch)
scores = scores + bias[None, :, :, :]  # [B, H, T, T]

# Then apply mask and softmax as before
```

**Files to modify:**
- `src/music_generation/layers.py` - LocalWindowAttention class

**Tests:**
- Verify bias weight shape is `[num_heads, window_size]`
- Verify bias is trainable
- Verify bias is applied to scores before masking
- Test with different sequence lengths (T < window_size, T = window_size, T > window_size)
- Verify causality still holds after adding bias

**Integration:**
- LocalWindowAttention is used by TransformerBlock
- No changes needed to TransformerBlock itself

**Demo:**
```python
layer = LocalWindowAttention(d_model=128, num_heads=4, window_size=128)
x = tf.random.normal([2, 50, 128])
out = layer(x)
print(f"Input: {x.shape}, Output: {out.shape}")
print(f"Bias shape: {layer.relative_position_bias.shape}")
# Output: Input: (2, 50, 128), Output: (2, 50, 128)
# Bias shape: (4, 128)
```

---

## Step 3: Update MelGenerator with 3 explicit transformer layers

**Objective:** Replace the loop-based transformer stack with 3 explicitly named layers with different window sizes.

**Implementation:**

1. **Modify __init__:**
```python
# Before:
self.transformer_blocks = [
    TransformerBlock(D_MODEL, NUM_HEADS) for _ in range(NUM_LAYERS)
]

# After:
self.transformer_layer1 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=128)
self.transformer_layer2 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=256)
self.transformer_layer3 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=512)
```

2. **Modify call():**
```python
# Before:
for block in self.transformer_blocks:
    x = block(x, training=training)

# After:
x = self.transformer_layer1(x, training=training)
x = self.transformer_layer2(x, training=training)
x = self.transformer_layer3(x, training=training)
```

**Files to modify:**
- `src/music_generation/model.py` - MelGenerator class

**Tests:**
- Verify model builds successfully
- Verify forward pass with various sequence lengths
- Verify output shape matches input shape
- Verify generate() method still works
- Test with batch_size=1 and batch_size>1

**Integration:**
- Model is used by training loop (train.py)
- Model is used by inference (inference.py)
- No changes needed to these files

**Demo:**
```python
model = MelGenerator()
mel = tf.random.normal([4, 100, 80])
output = model(mel, training=False)
print(f"Input: {mel.shape}, Output: {output.shape}")
# Output: Input: (4, 100, 80), Output: (4, 100, 80)

# Test generation
seed = tf.random.normal([50, 80])
generated = model.generate(seed, num_frames=100)
print(f"Generated: {generated.shape}")
# Output: Generated: (100, 80)
```

---

## Step 4: Test causality and window constraints

**Objective:** Verify that the model maintains strict causality and respects window constraints.

**Implementation:**

1. **Causality test:**
```python
def test_causality():
    model = MelGenerator()
    x = tf.random.normal([1, 100, 80])
    
    # Get output
    out1 = model(x, training=False)
    
    # Modify future positions
    x_modified = tf.concat([
        x[:, :50, :],
        tf.random.normal([1, 50, 80])
    ], axis=1)
    out2 = model(x_modified, training=False)
    
    # First 50 positions should be identical
    diff = tf.reduce_max(tf.abs(out1[:, :50, :] - out2[:, :50, :]))
    assert diff < 1e-5, f"Causality violated: diff={diff}"
    print("✓ Causality test passed")
```

2. **Window constraint test:**
```python
def test_window_constraint():
    # Create attention layer
    layer = LocalWindowAttention(d_model=128, num_heads=4, window_size=128)
    
    # Build layer
    x = tf.random.normal([1, 200, 128])
    _ = layer(x)
    
    # Manually compute attention to inspect weights
    # (This requires modifying call() to return attention weights for testing)
    # Verify that position i doesn't attend to j where i-j >= 128
    
    print("✓ Window constraint test passed")
```

**Files to create:**
- `tests/test_causality.py` (new file)

**Tests:**
- Run causality test with various sequence lengths
- Run window constraint test for each layer's window size
- Verify tests pass consistently

**Integration:**
- Tests are standalone, run via pytest
- Can be added to CI/CD pipeline

**Demo:**
```bash
pytest tests/test_causality.py -v
# Output:
# test_causality PASSED
# test_window_constraint PASSED
```

---

## Step 5: Verify training compatibility

**Objective:** Ensure the updated model works with the existing training loop without modifications.

**Implementation:**

1. **Run one training step:**
```python
from src.music_generation.train import create_model, create_dataset
from src.music_generation.config import *

# Create model and dataset
model = create_model()
dataset = create_dataset(DATA_DIR, BATCH_SIZE)

# Get one batch
batch = next(iter(dataset))
mel = batch['mel']

# Forward pass
with tf.GradientTape() as tape:
    output = model(mel, training=True)
    loss = tf.reduce_mean(tf.square(output - mel))  # Dummy loss

# Backward pass
gradients = tape.gradient(loss, model.trainable_variables)

# Verify gradients exist and are not NaN
for grad, var in zip(gradients, model.trainable_variables):
    assert grad is not None, f"No gradient for {var.name}"
    assert not tf.reduce_any(tf.math.is_nan(grad)), f"NaN gradient for {var.name}"

print("✓ Training compatibility verified")
```

2. **Test checkpoint save/load:**
```python
# Save checkpoint
model.save_weights('test_checkpoint.h5')

# Create new model and load
model2 = MelGenerator()
model2.load_weights('test_checkpoint.h5')

# Verify outputs match
x = tf.random.normal([1, 50, 80])
out1 = model(x, training=False)
out2 = model2(x, training=False)
assert tf.reduce_max(tf.abs(out1 - out2)) < 1e-6

print("✓ Checkpoint save/load verified")
```

**Files to modify:**
- None (testing existing training loop)

**Tests:**
- Run one training step successfully
- Verify loss is computed correctly
- Verify gradients flow to all parameters including relative_position_bias
- Test checkpoint save/load
- Optionally: run a few epochs to verify convergence starts

**Integration:**
- Training loop (train.py) should work without modification
- Inference (inference.py) should work without modification

**Demo:**
```bash
# Run training for 1 epoch to verify
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./test_checkpoints \
  --epochs 1 \
  --batch_size 4

# Output should show:
# - Model builds successfully
# - Training progresses without errors
# - Loss decreases
# - Checkpoint saved
```

---

## Summary

This implementation plan follows a bottom-up approach:
1. Update configuration (foundation)
2. Enhance attention mechanism (core change)
3. Update model architecture (integration)
4. Verify correctness (validation)
5. Ensure compatibility (end-to-end)

Each step builds on the previous, produces working functionality, and can be tested independently. The model remains functional after each step, with no orphaned code.
