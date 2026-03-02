# Research: Relative Positional Encoding in Transformers

## Overview
Survey of relative positional encoding approaches for causal transformer models, focusing on learned bias methods suitable for TensorFlow implementation.

## Key Approaches

### 1. T5 Relative Position Bias
**Source**: [Exploring the Limits of Transfer Learning (T5 paper)](https://arxiv.org/abs/1910.10683)

**Implementation**:
- Learned lookup table of shape `[num_buckets, num_heads]` (typically 32 buckets)
- Shared across all layers (for efficiency)
- Different bias per head (not shared across heads)
- Uses bucketing strategy for distances beyond a threshold

**Key characteristics**:
- Each head learns independent bias patterns (no enforced long-term decay)
- Both positive and negative biases are learned
- No predetermined pattern - network learns what works best
- Efficient: only 32 × num_heads parameters

**Bucketing formula** (from T5):
```
For relative distance (i-j):
- If 0 ≤ i-j < N/2: use b[i-j]
- If N/2 ≤ i-j < max_distance: use logarithmic bucketing
- If i-j ≥ max_distance: use b[N]
```

**Advantages**:
- Proven effective in practice (T5 models)
- Flexible - each head learns its own pattern
- Parameter efficient

**For our use case**:
- We want per-layer biases (not shared across layers) since we have only 3 layers with different window sizes
- Simpler than T5: direct learned bias per relative distance within window
- No bucketing needed since window sizes are fixed (128, 256, 512)

### 2. ALiBi (Attention with Linear Biases)
**Source**: [Train Short, Test Long paper](https://arxiv.org/abs/2108.12409)

**Implementation**:
- Fixed (not learned) linear bias: `-m × (i - j)` where m is head-specific slope
- Slope per head: `m = 2^(-8/n)` where n is number of heads
- Creates recency bias (recent tokens weighted more)

**Key characteristics**:
- No learned parameters
- Long-term decay property built-in
- Good length generalization

**Not suitable for our use case**:
- We want learned biases (more flexible)
- Fixed slopes may not be optimal for audio generation
- Our requirement specifies "learned relative position bias"

### 3. Causal Attention and Position Information
**Research finding**: Recent work shows that causal masking itself provides positional information in decoder-only models. Models without explicit positional encoding (NoPE) can still perform well because:
- Causal mask creates implicit position bias
- Earlier positions attend to more contextualized representations
- The constraint that position i can only attend to j ≤ i encodes ordering

**Implication**: Relative position bias in causal setting may be less critical than in bidirectional attention, but still beneficial for capturing fine-grained positional relationships.

## Recommended Approach for Our Model

### Design Decision: Per-Layer Learned Bias

**Rationale**:
1. **Independent per layer**: Each of our 3 layers has different window size (128, 256, 512), so sharing biases across layers doesn't make sense
2. **Learned, not fixed**: More flexible than ALiBi, allows network to discover optimal patterns
3. **Simple implementation**: Direct bias tensor of shape `[num_heads, window_size]` per layer

**Implementation pattern**:
```python
class LocalWindowAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, window_size=128, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Learned relative position bias: [num_heads, window_size]
        # Bias for relative distance d (where 0 ≤ d < window_size)
        self.relative_position_bias = self.add_weight(
            name='relative_position_bias',
            shape=[num_heads, window_size],
            initializer='zeros',  # or small random init
            trainable=True
        )
    
    def call(self, x):
        # Compute attention scores
        scores = ...  # [B, H, T, T]
        
        # Add relative position bias
        # For each query position i and key position j where j ≤ i:
        #   distance = i - j
        #   if distance < window_size: add bias[head, distance]
        
        # Apply causal + window mask, then softmax
        ...
```

**Bias application**:
- For query at position i attending to key at position j (where j ≤ i):
  - Compute relative distance: `d = i - j`
  - If `d < window_size`: add `bias[head, d]` to attention score
  - If `d ≥ window_size`: position is masked anyway (outside window)

**Initialization**: Start with zeros or small random values. The network will learn appropriate patterns during training.

## TensorFlow Implementation Considerations

### Efficient Bias Indexing
Instead of building full T×T bias matrices, use efficient indexing:
1. Precompute relative distance matrix for current sequence length T
2. Clip distances to `[0, window_size-1]`
3. Use `tf.gather` to extract biases for valid positions
4. Combine with causal + window mask

### Memory Efficiency
- Bias parameters: `num_heads × window_size` per layer
  - Layer 1: 4 × 128 = 512 parameters
  - Layer 2: 4 × 256 = 1,024 parameters  
  - Layer 3: 4 × 512 = 2,048 parameters
  - Total: 3,584 parameters (negligible)

## References

1. [T5 Paper - Relative Position Bias](https://arxiv.org/abs/1910.10683)
2. [ALiBi - Attention with Linear Biases](https://arxiv.org/abs/2108.12409)
3. [Reinforced Knowledge - Position Encoding Survey](https://reinforcedknowledge.com/position-information-in-transformer-based-models-exploring-the-main-methods-and-approaches/)
4. [TensorFlow Models - RelativePositionBias Layer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/RelativePositionBias)
5. [Impact of Positional Encoding on Length Generalization](https://arxiv.org/abs/2305.19466) - Shows causal attention provides implicit position info

## Summary

For our 3-layer causal transformer with varying window sizes (128, 256, 512):
- Use **learned relative position bias** per layer
- Shape: `[num_heads, window_size]` for each layer
- Initialize with zeros or small random values
- Apply bias based on relative distance `i - j` for causal positions
- Simple, efficient, and flexible approach that allows each layer to learn optimal positional patterns for its window size

Content was rephrased for compliance with licensing restrictions.
