# Research: Causal Masking and Parallel Autoregressive Training

## Overview

Research on how modern transformer models implement causal masking to enable parallel training while maintaining autoregressive semantics.

## Causal Attention Mask Pattern

### Standard Implementation

From GPT-style transformers, the causal mask is a **lower-triangular matrix** that prevents position `t` from attending to positions `> t`:

```python
def causal_attention_mask(n_dest, n_src, dtype):
    """Create lower-triangular causal mask."""
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest  # Lower triangle
    return tf.cast(m, dtype)
```

**Mask shape:** `[seq_len, seq_len]`

**Example for seq_len=5:**
```
[[1, 0, 0, 0, 0],   # Position 0 attends only to itself
 [1, 1, 0, 0, 0],   # Position 1 attends to 0,1
 [1, 1, 1, 0, 0],   # Position 2 attends to 0,1,2
 [1, 1, 1, 1, 0],   # Position 3 attends to 0,1,2,3
 [1, 1, 1, 1, 1]]   # Position 4 attends to 0,1,2,3,4
```

### Application in Attention

```python
# Compute attention scores
score = tf.matmul(query, key, transpose_b=True)  # [B, H, T, T]
scaled_score = score / tf.sqrt(dim_key)

# Apply causal mask
attention_mask = causal_attention_mask(seq_len, seq_len, dtype)
attention_mask = tf.reshape(attention_mask, [1, 1, seq_len, seq_len])

# Mask out future positions (set to large negative value)
scaled_score = scaled_score * attention_mask - 1e4 * (1 - attention_mask)

# Softmax (masked positions become ~0 probability)
weights = tf.nn.softmax(scaled_score, axis=-1)
output = tf.matmul(weights, value)
```

**Key insight:** The mask is applied **before softmax** by setting future positions to large negative values (-1e4 or -inf), which become ~0 after softmax.

## Parallel Training with Causal Masks

### The Core Principle

**Training:** Process entire sequence `[0, 1, 2, ..., T-1]` in a single forward pass with causal masking.

**Loss computation:** Compare predictions at each position with the next ground-truth token:
```python
# Input:  [0, 1, 2, ..., T-1]
# Output: [pred_1, pred_2, ..., pred_T]
# Target: [1, 2, 3, ..., T]

loss = loss_fn(preds[:, :-1, :], targets[:, 1:, :])
```

**Equivalence to autoregressive loop:**
- Position 0 predicts token 1 (sees only token 0)
- Position 1 predicts token 2 (sees only tokens 0,1)
- Position T-1 predicts token T (sees tokens 0..T-1)

This is **mathematically identical** to running T forward passes with growing prefixes, but computed in parallel.

### TensorFlow/Keras Built-in Support

**Modern Keras layers support causal masking natively:**

```python
# MultiHeadAttention with causal mask (TensorFlow 2.10+)
attention_layer = tf.keras.layers.MultiHeadAttention(
    num_heads=8,
    key_dim=64
)

# During call
output = attention_layer(
    query=x,
    value=x,
    key=x,
    use_causal_mask=True  # ← Enables causal masking
)
```

**Alternative: Manual mask creation**
```python
# Create causal mask
seq_len = tf.shape(x)[1]
mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # Lower triangle

# Apply to attention
output = attention_layer(query=x, value=x, key=x, attention_mask=mask)
```

## Parallel Scheduled Sampling

### The Challenge

Traditional scheduled sampling mixes ground truth and model predictions **during** the forward pass:
```python
for t in range(seq_len):
    pred_t = model(input[:t])
    use_teacher = random() < tf_ratio
    input[t] = ground_truth[t] if use_teacher else pred_t
```

This requires sequential processing and breaks parallelization.

### Solution: Parallel Scheduled Sampling (PSS)

**Key papers:**
1. "Parallel Scheduled Sampling" (Duckworth et al., ICLR 2020)
2. "Improving Generalization of Transformer for Speech Recognition with Parallel Schedule Sampling" (Zhou et al., 2019)

**Core idea:** Pre-corrupt the input sequence before the forward pass, then train with causal masking.

### PSS Algorithm (Simplified)

```python
def parallel_scheduled_sampling(ground_truth, model, tf_ratio):
    """
    Apply scheduled sampling in parallel.
    
    Args:
        ground_truth: [batch, seq_len, features]
        model: Causal model
        tf_ratio: Teacher forcing ratio
    
    Returns:
        loss: Training loss
    """
    batch_size, seq_len, _ = ground_truth.shape
    
    # Step 1: Get model predictions for entire sequence
    preds = model(ground_truth, training=True)  # [batch, seq_len, features]
    
    # Step 2: Create sampling mask (which positions use ground truth)
    # Sample per-position, per-batch
    use_teacher = tf.random.uniform([batch_size, seq_len, 1]) < tf_ratio
    
    # Step 3: Mix ground truth and predictions
    # Shift predictions: pred[t] is used as input for position t+1
    preds_shifted = tf.concat([ground_truth[:, :1, :], preds[:, :-1, :]], axis=1)
    mixed_input = tf.where(use_teacher, ground_truth, preds_shifted)
    
    # Step 4: Forward pass with mixed input
    final_preds = model(mixed_input, training=True)
    
    # Step 5: Compute loss against ground truth
    loss = loss_fn(final_preds[:, :-1, :], ground_truth[:, 1:, :])
    
    return loss
```

**Key differences from sequential scheduled sampling:**
1. **Two forward passes** instead of T forward passes
2. First pass gets predictions for all positions
3. Second pass uses mixed input (ground truth + predictions)
4. Still much faster: O(2) vs O(T)

### Alternative: Input Corruption

**Simpler approach** that avoids the two-pass requirement:

```python
def input_corruption_training(ground_truth, model, corruption_prob):
    """
    Corrupt input by adding noise instead of mixing with predictions.
    
    This is a single-pass approximation of scheduled sampling.
    """
    # Add noise to simulate prediction errors
    noise = tf.random.normal(tf.shape(ground_truth)) * 0.1
    use_noise = tf.random.uniform(tf.shape(ground_truth)) < corruption_prob
    corrupted_input = tf.where(use_noise, ground_truth + noise, ground_truth)
    
    # Single forward pass with causal masking
    preds = model(corrupted_input, training=True)
    
    # Loss against clean ground truth
    loss = loss_fn(preds[:, :-1, :], ground_truth[:, 1:, :])
    
    return loss
```

**Trade-off:** Simpler and faster, but noise doesn't perfectly simulate model predictions.

## Causal Convolutions

### Implementation Pattern

For convolutional layers, causality is achieved through **left-padding**:

```python
class CausalConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation_rate
        self.conv = tf.keras.layers.Conv1D(
            filters, kernel_size, 
            dilation_rate=dilation_rate, 
            padding='valid'
        )
    
    def call(self, x):
        # Pad left only
        x = tf.pad(x, [[0, 0], [self.padding, 0], [0, 0]])
        return self.conv(x)
```

**Why this works:**
- Kernel size 3: needs 2 past values
- Padding of 2 on the left ensures output[t] depends only on input[t-2:t+1]
- No future information leaks

## Local Window Attention (Already Causal)

The current implementation uses **local window attention** which is inherently causal:

```python
# From layers.py
# Pad K and V on the left with (window_size - 1) zeros
k_padded = tf.pad(k, [[0, 0], [0, 0], [pad_width, 0], [0, 0]])

# Each position attends to its left window
indices = tf.range(T)[:, None] + tf.range(window_size)[None, :]
k_windows = tf.gather(k_padded, indices, axis=2)
```

**This is already a causal implementation** - no changes needed to the attention mechanism itself.

## Summary: What Enables Parallel Training

1. **Causal masking** (attention) or **causal padding** (convolution) in model architecture
2. **Single forward pass** with full sequence as input
3. **Shifted loss computation:** `preds[:, :-1, :] vs targets[:, 1:, :]`
4. **Optional:** Parallel scheduled sampling for exposure bias mitigation

## References

- [Text Generation with miniature GPT (Keras example)](https://apoorvnandan.github.io/2020/05/29/mini-gpt/) - Causal mask implementation
- [Parallel Scheduled Sampling (ICLR 2020)](https://openreview.net/forum?id=HkedQp4tPr) - Duckworth et al.
- [Improving Generalization of Transformer with PSS (arXiv:1911.00203)](https://arxiv.org/abs/1911.00203v2) - Zhou et al.
- TensorFlow 2.10+ MultiHeadAttention with `use_causal_mask` parameter
