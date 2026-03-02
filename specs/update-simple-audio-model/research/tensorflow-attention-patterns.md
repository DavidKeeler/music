# Research: TensorFlow Attention Implementation Patterns

## Overview
Best practices and efficient patterns for implementing custom causal attention layers in TensorFlow 2.x with Keras.

## Key Implementation Patterns

### 1. Causal Masking in TensorFlow/Keras

**Built-in Support** (TensorFlow 2.10+):
- `tf.keras.layers.Attention` supports `use_causal_mask=True` parameter
- `tf.keras.layers.MultiHeadAttention` also supports causal masking
- Handles masking internally with optimized implementations

**Manual Implementation Pattern**:
```python
# Create causal mask: position i can only attend to j where j ≤ i
T = tf.shape(x)[1]
causal_mask = tf.linalg.band_part(tf.ones([T, T]), -1, 0)  # Lower triangular
causal_mask = 1.0 - causal_mask  # Invert: 1 = masked, 0 = allowed
scores = scores + (causal_mask * -1e9)  # Add large negative before softmax
```

**Key points**:
- Use large negative value (-1e9) rather than -inf to avoid NaN issues
- Apply mask BEFORE softmax
- `tf.linalg.band_part(x, -1, 0)` creates lower triangular matrix efficiently

### 2. Window + Causal Masking

**Combined Mask Pattern**:
```python
# Causal mask: j ≤ i
causal_mask = 1.0 - tf.linalg.band_part(tf.ones([T, T]), -1, 0)

# Window mask: i - j < window_size
indices = tf.range(T)
row_indices = tf.expand_dims(indices, 1)  # [T, 1]
col_indices = tf.expand_dims(indices, 0)  # [1, T]
window_mask = tf.cast(row_indices - col_indices >= window_size, tf.float32)

# Combine: mask if EITHER condition fails
combined_mask = tf.maximum(causal_mask, window_mask)
scores = scores + (combined_mask * -1e9)
```

**Efficiency consideration**:
- For training with fixed sequence lengths, masks can be precomputed once
- For variable lengths, compute on-the-fly (still fast for reasonable T)
- Memory: O(T²) for mask, but typically small compared to attention scores

### 3. Banded Matrix Optimization

**Concept**: For local window attention, only compute attention for positions within the window.

**Current approach** (compute full T×T, then mask):
```python
scores = tf.matmul(q, k, transpose_b=True)  # [B, H, T, T]
scores = scores + mask  # Apply mask
attn = tf.nn.softmax(scores, axis=-1)
```

**Optimized approach** (banded computation):
- More complex to implement in pure TensorFlow
- Requires custom CUDA kernels for significant speedup (like FlashAttention)
- For our use case (T ≤ 512, window ≤ 512), full computation + masking is acceptable

**Recommendation**: Start with full computation + masking. Profile first, optimize only if needed.

### 4. Relative Position Bias Application

**Efficient indexing pattern**:
```python
# Precompute relative distances
T = tf.shape(x)[1]
positions = tf.range(T)
rel_distances = tf.expand_dims(positions, 0) - tf.expand_dims(positions, 1)  # [T, T]

# Clip to valid range [0, window_size-1]
rel_distances = tf.clip_by_value(rel_distances, 0, window_size - 1)

# Gather biases: [num_heads, window_size] -> [num_heads, T, T]
# For each head h, position (i,j): bias[h, rel_distances[i,j]]
bias = tf.gather(self.relative_position_bias, rel_distances, axis=1)  # [H, T, T]

# Add to scores
scores = scores + tf.expand_dims(bias, 0)  # Broadcast over batch: [B, H, T, T]
```

**Alternative using tf.gather_nd**:
```python
# Create indices for gathering
indices = tf.stack([
    tf.tile(tf.range(num_heads)[:, None, None], [1, T, T]),  # head indices
    tf.tile(rel_distances[None, :, :], [num_heads, 1, 1])    # distance indices
], axis=-1)

bias = tf.gather_nd(self.relative_position_bias, indices)  # [H, T, T]
```

### 5. Handling NaN in Attention Weights

**Problem**: When entire row is masked, softmax produces NaN.

**Solution pattern**:
```python
attn_weights = tf.nn.softmax(scores, axis=-1)
attn_weights = tf.where(tf.math.is_nan(attn_weights), 0.0, attn_weights)
```

**Why this works**:
- Masked positions have scores of -1e9
- If all positions in a row are masked, softmax produces NaN
- Replace NaN with 0.0 (no attention to any position)
- This is safe because the output will be zero anyway

### 6. Graph Mode Compatibility

**Critical for TensorFlow 2.x**:
- Use `tf.shape(x)` for dynamic shapes, not `x.shape`
- Avoid Python control flow (if/for) inside `call()` - use TensorFlow ops
- Use `tf.function` decorator for performance (automatic in model training)

**Example - WRONG**:
```python
def call(self, x):
    T = x.shape[1]  # Static shape - fails with dynamic inputs
    for i in range(T):  # Python loop - not graph-compatible
        ...
```

**Example - CORRECT**:
```python
def call(self, x):
    T = tf.shape(x)[1]  # Dynamic shape
    # Use vectorized TensorFlow ops instead of loops
    ...
```

### 7. Memory-Efficient Attention

**For large sequences**, consider:

1. **Gradient checkpointing**: Trade compute for memory
   ```python
   @tf.recompute_grad
   def attention_block(x):
       return self.attention(x)
   ```

2. **Mixed precision**: Use float16 for computation, float32 for accumulation
   ```python
   policy = tf.keras.mixed_precision.Policy('mixed_float16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

3. **Chunked computation**: Process in smaller chunks (more complex)

**For our use case** (T ≤ 512, window ≤ 512):
- Standard implementation is sufficient
- Memory usage is manageable
- Focus on correctness over micro-optimizations

## Recommended Implementation Structure

```python
class LocalWindowAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, window_size=128, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads
        
        # Projections
        self.qkv = tf.keras.layers.Dense(3 * d_model)
        self.out_proj = tf.keras.layers.Dense(d_model)
        
        # Relative position bias
        self.relative_position_bias = self.add_weight(
            name='relative_position_bias',
            shape=[num_heads, window_size],
            initializer='zeros',
            trainable=True
        )
    
    def call(self, x):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, T, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.head_dim))
        
        # Add relative position bias
        positions = tf.range(T)
        rel_distances = tf.expand_dims(positions, 0) - tf.expand_dims(positions, 1)
        rel_distances = tf.clip_by_value(rel_distances, 0, self.window_size - 1)
        bias = tf.gather(self.relative_position_bias, rel_distances, axis=1)
        scores = scores + tf.expand_dims(bias, 0)
        
        # Create combined causal + window mask
        causal_mask = 1.0 - tf.linalg.band_part(tf.ones([T, T]), -1, 0)
        window_mask = tf.cast(
            tf.expand_dims(positions, 0) - tf.expand_dims(positions, 1) >= self.window_size,
            tf.float32
        )
        mask = tf.maximum(causal_mask, window_mask)
        scores = scores + (mask * -1e9)
        
        # Softmax and handle NaN
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = tf.where(tf.math.is_nan(attn_weights), 0.0, attn_weights)
        
        # Apply attention
        out = tf.matmul(attn_weights, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, T, self.d_model])
        
        return self.out_proj(out)
```

## Performance Considerations

### Profiling First
Before optimizing:
1. Profile with `tf.profiler` to identify bottlenecks
2. Check if attention is actually the bottleneck (often it's data loading or other layers)
3. Optimize only if needed

### When to Optimize
- If training is too slow
- If running out of memory
- If inference latency is critical

### What NOT to Optimize Prematurely
- Banded matrix computation (complex, marginal gains for our sizes)
- Custom CUDA kernels (maintenance burden)
- Exotic attention variants (stick to proven patterns)

## Testing Patterns

**Verify causality**:
```python
# Test that future positions don't affect past
x = tf.random.normal([1, 10, d_model])
out1 = layer(x)

# Modify future positions
x_modified = tf.concat([x[:, :5, :], tf.random.normal([1, 5, d_model])], axis=1)
out2 = layer(x_modified)

# First 5 positions should be identical
assert tf.reduce_max(tf.abs(out1[:, :5, :] - out2[:, :5, :])) < 1e-5
```

**Verify window constraint**:
```python
# Check that positions beyond window don't contribute
# (More complex - inspect attention weights)
```

## Summary

For our causal audio model with local window attention:
1. Use standard TensorFlow ops for masking (efficient enough)
2. Combine causal + window masks with `tf.maximum`
3. Apply relative position bias via `tf.gather`
4. Handle NaN in softmax with `tf.where`
5. Use `tf.shape` for dynamic shapes (graph mode compatibility)
6. Profile before optimizing - correctness first, speed second

Content was rephrased for compliance with licensing restrictions.
