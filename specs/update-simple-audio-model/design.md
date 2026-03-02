# Design Document: Update Simple Audio Model

## Overview

Upgrade the TensorFlow mel autoregressive model from a single-layer transformer to a 3-layer architecture with increasing window sizes and learned relative positional encoding.

## Detailed Requirements

### Architecture Changes

**Current Architecture:**
```
Input (mel) → Dense → CausalConv → CausalConv → 1×TransformerBlock(window=128) → CausalConv → Dense → Output
```

**New Architecture:**
```
Input (mel) → Dense → CausalConv → CausalConv → 
  TransformerBlock(window=128) → 
  TransformerBlock(window=256) → 
  TransformerBlock(window=512) → 
CausalConv → Dense → Output
```

### Key Requirements

1. **3 Transformer Layers** with increasing window sizes: 128, 256, 512
2. **Relative Positional Encoding** - learned bias per layer
3. **Strict Causality** - no future information leakage
4. **No Global Attention** - only local windowed attention
5. **Maintain Compatibility** - with existing training loop and generation API
6. **Stable Model Size** - no changes to D_MODEL, NUM_HEADS, or other hyperparameters

## Architecture Overview

### Model Pipeline

```
[B, T, 80] mel spectrogram
    ↓
[B, T, 128] input projection (Dense)
    ↓
[B, T, 128] causal conv block 1 (kernel=3, residual)
    ↓
[B, T, 128] causal conv block 2 (kernel=3, residual)
    ↓
[B, T, 128] transformer layer 1 (window=128, relative bias)
    ↓
[B, T, 128] transformer layer 2 (window=256, relative bias)
    ↓
[B, T, 128] transformer layer 3 (window=512, relative bias)
    ↓
[B, T, 128] causal conv head (kernel=3, residual)
    ↓
[B, T, 80] output projection (Dense)
```

### Attention Mechanism

Each transformer layer uses **Local Window Causal Attention with Relative Position Bias**:

```
Input: [B, T, D]
  ↓
Q, K, V projections: [B, H, T, d_head]
  ↓
Attention scores: Q·K^T / sqrt(d_head)  [B, H, T, T]
  ↓
Add relative position bias  [B, H, T, T]
  ↓
Apply causal + window mask  [B, H, T, T]
  ↓
Softmax (handle NaN)  [B, H, T, T]
  ↓
Apply to values: Attn·V  [B, H, T, d_head]
  ↓
Concatenate heads and project: [B, T, D]
```

## Components and Interfaces

### 1. LocalWindowAttention Layer

**Modified Class:**
```python
class LocalWindowAttention(tf.keras.layers.Layer):
    """Local window causal multi-head attention with relative position bias."""
    
    def __init__(self, d_model, num_heads, window_size=128, **kwargs):
        """
        Args:
            d_model: Model dimension (128)
            num_heads: Number of attention heads (4)
            window_size: Local attention window size (128, 256, or 512)
        """
```

**New Components:**
- `relative_position_bias`: Trainable weight `[num_heads, window_size]`
  - Initialized with small random values (e.g., `tf.random_normal_initializer(stddev=0.01)`)
  - Independent per layer (not shared)

**Modified call() method:**
1. Compute Q, K, V projections
2. Compute scaled dot-product attention scores
3. **NEW:** Add relative position bias based on distance i-j
4. Apply combined causal + window mask
5. Softmax with NaN handling
6. Apply attention to values
7. Project output

**Relative Position Bias Application:**
```python
# Compute relative distances
positions = tf.range(T)
rel_distances = positions[:, None] - positions[None, :]  # [T, T]
rel_distances = tf.clip_by_value(rel_distances, 0, window_size - 1)

# Gather biases: [num_heads, window_size] -> [num_heads, T, T]
bias = tf.gather(self.relative_position_bias, rel_distances, axis=1)

# Add to scores (broadcast over batch)
scores = scores + bias[None, :, :, :]  # [B, H, T, T]
```

**Masking Logic:**
```python
# Causal mask: position i cannot attend to j > i
causal_mask = 1.0 - tf.linalg.band_part(tf.ones([T, T]), -1, 0)

# Window mask: position i cannot attend to j where i-j >= window_size
window_mask = tf.cast(
    positions[:, None] - positions[None, :] >= window_size,
    tf.float32
)

# Combined mask (mask if EITHER condition fails)
mask = tf.maximum(causal_mask, window_mask)
scores = scores + (mask * -1e9)
```

### 2. TransformerBlock Layer

**No changes required** - already accepts `window_size` parameter and passes it to LocalWindowAttention.

**Usage:**
```python
TransformerBlock(d_model=128, num_heads=4, window_size=128)
TransformerBlock(d_model=128, num_heads=4, window_size=256)
TransformerBlock(d_model=128, num_heads=4, window_size=512)
```

### 3. MelGenerator Model

**Modified __init__:**
```python
def __init__(self):
    super().__init__()
    
    # Input projection
    self.input_proj = tf.keras.layers.Dense(D_MODEL)
    
    # Causal conv layers
    self.conv1 = CausalConvBlock(D_MODEL, kernel_size=3, residual=True)
    self.conv2 = CausalConvBlock(D_MODEL, kernel_size=3, residual=True)
    
    # Transformer stack - 3 explicit layers with increasing window sizes
    self.transformer_layer1 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=128)
    self.transformer_layer2 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=256)
    self.transformer_layer3 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=512)
    
    # Conv head
    self.conv_head = CausalConvBlock(D_MODEL, kernel_size=3, residual=True)
    
    # Output projection
    self.output_proj = tf.keras.layers.Dense(N_MELS)
```

**Modified call():**
```python
def call(self, mel, training=False):
    # Input projection
    x = self.input_proj(mel)
    
    # Causal convs
    x = self.conv1(x, training=training)
    x = self.conv2(x, training=training)
    
    # Transformer stack
    x = self.transformer_layer1(x, training=training)
    x = self.transformer_layer2(x, training=training)
    x = self.transformer_layer3(x, training=training)
    
    # Conv head
    x = self.conv_head(x, training=training)
    
    # Output projection
    x = self.output_proj(x)
    
    return x
```

**No changes to generate() method** - autoregressive generation logic remains identical.

### 4. Configuration Updates

**config.py changes:**
```python
# Model Architecture Parameters
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 3  # Updated from 1
WINDOW_SIZES = [128, 256, 512]  # New constant
```

**Deprecated constants:**
- `WINDOW_SIZE = 128` - removed (now using WINDOW_SIZES list)

## Data Models

No changes to data models. All tensor shapes remain the same:
- Input: `[batch, seq_len, N_MELS]`
- Hidden: `[batch, seq_len, D_MODEL]`
- Output: `[batch, seq_len, N_MELS]`

## Error Handling

### NaN in Attention Weights

**Problem:** When all positions in a row are masked, softmax produces NaN.

**Solution:**
```python
attn_weights = tf.nn.softmax(scores, axis=-1)
attn_weights = tf.where(tf.math.is_nan(attn_weights), 0.0, attn_weights)
```

### Shape Compatibility

**Ensure dynamic shapes work in graph mode:**
- Use `tf.shape(x)` instead of `x.shape`
- Avoid Python control flow in call() methods
- All operations must be TensorFlow ops

### Causality Verification

**Critical:** Verify that output at position t depends only on inputs ≤ t.

**Test approach:**
```python
# Modify future positions
x_modified = tf.concat([x[:, :t, :], random_noise], axis=1)

# Outputs at positions < t should be identical
assert outputs[:, :t, :] == outputs_modified[:, :t, :]
```

## Acceptance Criteria

### Given: A trained MelGenerator model with the new architecture
### When: Processing mel spectrogram sequences
### Then: The model should:

1. **Maintain causality**
   - Output at position t depends only on inputs at positions ≤ t
   - Verified by modifying future positions and checking past outputs remain unchanged

2. **Respect window constraints**
   - Position i attends only to positions j where 0 ≤ i-j < window_size
   - Verified by inspecting attention weight patterns

3. **Learn relative position patterns**
   - Relative position bias tensors are updated during training
   - Different layers learn different bias patterns

4. **Support autoregressive generation**
   - generate() method works without modification
   - Produces coherent audio output frame-by-frame

5. **Run in graph mode**
   - Model compiles successfully with tf.function
   - No Python control flow errors during training

6. **Maintain model size**
   - Total parameters increase only by relative position bias (~3.5K parameters)
   - D_MODEL, NUM_HEADS unchanged

### Given: Training loop with teacher forcing
### When: Training the model
### Then: The training should:

1. **Complete without errors**
   - No shape mismatches
   - No NaN/Inf in losses or gradients

2. **Show convergence**
   - Loss decreases over epochs
   - Generated samples improve in quality

## Testing Strategy

### Unit Tests

1. **LocalWindowAttention**
   - Test relative position bias shape: `[num_heads, window_size]`
   - Test bias is applied correctly to attention scores
   - Test causality: future positions don't affect past
   - Test window constraint: positions beyond window are masked
   - Test NaN handling in softmax

2. **MelGenerator**
   - Test forward pass with various sequence lengths
   - Test output shape matches input shape
   - Test generate() method produces expected output shape
   - Test causality in full model

### Integration Tests

1. **Training Loop**
   - Test one training step completes successfully
   - Test gradient flow through all layers
   - Test checkpoint save/load

2. **Generation**
   - Test autoregressive generation produces valid audio
   - Test generation with different seed lengths
   - Test temperature scaling works

### Manual Verification

1. **Attention Patterns**
   - Visualize attention weights for each layer
   - Verify causal + window structure
   - Check that relative bias is learned (not all zeros)

2. **Audio Quality**
   - Listen to generated samples
   - Compare to baseline (if available)

## Appendices

### A. Technology Choices

**TensorFlow 2.x with Keras:**
- Mature, well-documented framework
- Good graph mode support for efficient training
- Compatible with existing codebase

**Learned Relative Position Bias (T5-style):**
- More flexible than fixed biases (ALiBi)
- Proven effective in practice
- Simple to implement
- Minimal parameter overhead

**Full T×T Masking:**
- Simple and correct
- Acceptable memory overhead for our sequence lengths
- Easier to debug than sparse implementations

### B. Research Findings Summary

**Relative Positional Encoding:**
- T5 uses learned bias with bucketing for efficiency
- ALiBi uses fixed linear bias for length generalization
- Causal attention provides implicit position information
- Learned bias is more flexible and suitable for our use case

**TensorFlow Patterns:**
- Use `tf.linalg.band_part` for efficient causal masking
- Combine masks with `tf.maximum`
- Apply bias via `tf.gather` on relative distances
- Handle NaN with `tf.where` after softmax

### C. Alternative Approaches Considered

**1. Shared Bias Across Layers**
- Rejected: Different window sizes need different bias ranges
- Would require interpolation/truncation (added complexity)

**2. Fixed Bias (ALiBi-style)**
- Rejected: Less flexible than learned bias
- Fixed slopes may not be optimal for audio

**3. Banded/Sparse Attention**
- Rejected: Added complexity for marginal gains
- Full computation is acceptable for our sizes

**4. Global + Local Attention Mix**
- Rejected: Requirement specifies no global attention
- Purely local attention is simpler

### D. Parameter Count

**Relative Position Bias Parameters:**
- Layer 1: 4 heads × 128 window = 512 parameters
- Layer 2: 4 heads × 256 window = 1,024 parameters
- Layer 3: 4 heads × 512 window = 2,048 parameters
- **Total: 3,584 parameters** (negligible compared to model size)

**No other parameter changes:**
- D_MODEL unchanged (128)
- NUM_HEADS unchanged (4)
- FFN dimensions unchanged
- Conv layers unchanged

### E. Migration Notes

**Breaking Changes:**
- Old checkpoints (1 layer, window=128) are incompatible
- Must retrain from scratch with new architecture

**Config Changes:**
- `NUM_LAYERS`: 1 → 3
- `WINDOW_SIZE`: removed
- `WINDOW_SIZES`: new constant [128, 256, 512]

**Code Changes:**
- `LocalWindowAttention`: add relative_position_bias weight
- `LocalWindowAttention.call()`: add bias application logic
- `MelGenerator.__init__()`: define 3 explicit transformer layers
- `MelGenerator.call()`: call 3 layers sequentially
- `config.py`: update constants

**No Changes:**
- Training loop (train.py)
- Dataset pipeline (dataset.py)
- Inference API (inference.py)
- Vocoder (vocoder.py)
- Loss functions (losses.py)
