# Research: Parallel Scheduled Sampling Mechanics

## Overview

Detailed analysis of the parallel scheduled sampling algorithm from the provided context, including correctness verification and edge case identification.

## The Algorithm

### Core Concept

**Traditional scheduled sampling** (current implementation):
```
for each timestep t:
    predict frame t
    randomly choose: use ground truth OR use prediction
    append chosen frame to input
    continue to t+1
```
- **Problem**: Sequential, O(seq_len) forward passes

**Parallel scheduled sampling**:
```
predict entire sequence (1 forward pass)
randomly mix predictions with ground truth
predict again with mixed input (1 forward pass)
compute loss
```
- **Benefit**: Parallel, O(1) forward passes (actually 2)

## Step-by-Step Breakdown

### Step 1: Initial Forward Pass
```python
preds = self.base_model(x, training=True)
```
- **Input**: Ground truth sequence `x` with shape `[batch, seq_len, mel_dim]`
- **Output**: Predictions `preds` with shape `[batch, seq_len, mel_dim]`
- **Purpose**: Get model's predictions for entire sequence

### Step 2: Create Scheduled Sampling Mask
```python
mask = tf.random.uniform(tf.shape(x)) < tf_ratio
```
- **Shape**: `[batch, seq_len, mel_dim]` (broadcasts to match input)
- **Values**: Boolean tensor
  - `True` where random value < tf_ratio → use ground truth
  - `False` where random value ≥ tf_ratio → use prediction
- **Behavior**: 
  - When tf_ratio = 1.0: all True (pure teacher forcing)
  - When tf_ratio = 0.5: ~50% True, ~50% False
  - When tf_ratio = 0.05: ~5% True, ~95% False

### Step 3: Mix Ground Truth and Predictions
```python
mixed = tf.where(mask, x, preds)
```
- **Operation**: Element-wise selection
  - Where mask is True: take from `x` (ground truth)
  - Where mask is False: take from `preds` (model prediction)
- **Result**: Mixed sequence containing both ground truth and predictions
- **Shape**: `[batch, seq_len, mel_dim]`

### Step 4: Shift for Next-Frame Prediction
```python
inputs = mixed[:, :-1]   # Remove last frame
targets = y[:, 1:]        # Remove first frame
```
- **Purpose**: Align inputs and targets for next-frame prediction
- **Example** (seq_len=5):
  - `mixed = [f0, f1, f2, f3, f4]`
  - `inputs = [f0, f1, f2, f3]` (predict next frame)
  - `targets = [f1, f2, f3, f4]` (ground truth next frames)

### Step 5: Second Forward Pass
```python
preds2 = self.base_model(inputs, training=True)
```
- **Input**: Mixed sequence (shifted)
- **Output**: Predictions for next frames
- **Shape**: `[batch, seq_len-1, mel_dim]`

### Step 6: Compute Loss
```python
loss = tf.reduce_mean(tf.abs(preds2 - targets))
```
- **Loss type**: Mean Absolute Error (MAE)
- **Matches**: Current implementation's loss function

## Correctness Verification

### Does This Maintain Scheduled Sampling Behavior?

**Yes.** The key property of scheduled sampling is:
> "The model sometimes sees its own predictions as input, not just ground truth"

In parallel scheduled sampling:
- First pass: Model predicts based on ground truth
- Mix: Some frames replaced with predictions
- Second pass: Model sees mixed input (including its own errors)
- Loss: Computed on predictions from mixed input

This achieves the same goal: **exposure to model's own errors during training**.

### Difference from Autoregressive Scheduled Sampling

**Autoregressive** (current):
- At timestep t, model has seen its own predictions for timesteps 1..t-1
- Predictions accumulate sequentially

**Parallel**:
- At timestep t, model sees a random mix of ground truth and predictions
- Mix is independent per timestep (not accumulated)

**Impact**: Slightly different error exposure pattern, but both achieve the core goal of preventing exposure bias.

## Edge Cases and Considerations

### Edge Case 1: tf_ratio = 1.0 (Pure Teacher Forcing)
```python
mask = tf.random.uniform(tf.shape(x)) < 1.0  # All True
mixed = tf.where(mask, x, preds)              # All ground truth
```
- **Result**: `mixed = x` (no predictions used)
- **Behavior**: Equivalent to pure teacher forcing
- **Optimization**: Could skip first forward pass entirely

### Edge Case 2: tf_ratio = 0.0 (Pure Prediction)
```python
mask = tf.random.uniform(tf.shape(x)) < 0.0  # All False
mixed = tf.where(mask, x, preds)              # All predictions
```
- **Result**: `mixed = preds` (no ground truth used)
- **Behavior**: Model trains on its own predictions
- **Risk**: Could be unstable early in training

### Edge Case 3: First Frame Handling
The first frame should always be ground truth (no previous prediction exists):
```python
# Option 1: Force first frame to be ground truth
mask = tf.concat([
    tf.ones([batch, 1, mel_dim], dtype=tf.bool),  # First frame always True
    tf.random.uniform([batch, seq_len-1, mel_dim]) < tf_ratio
], axis=1)

# Option 2: Accept that first frame might be prediction
# (simpler, probably fine in practice)
```

### Edge Case 4: Gradient Flow
- **First pass**: Gradients from `preds` flow through model
- **Second pass**: Gradients from `preds2` flow through model
- **Question**: Should we stop gradients on `preds` before mixing?

**Answer from context**:
```python
mixed = tf.where(mask, x, preds)  # No stop_gradient
```

This means gradients flow through both passes. The model learns:
1. To predict well from ground truth (first pass)
2. To predict well from mixed input (second pass)

This is **correct** - we want both gradient signals.

## Comparison to Current Implementation

### Current Autoregressive Approach
```python
ar_input = x[:, :1, :]
for t in range(1, SEQ_LEN):
    pred = self.base_model(ar_input[:, -MAX_CONTEXT_FRAMES:, :], training=True)
    next_frame = pred[:, -1:, :]
    use_teacher = tf.random.uniform([batch, 1, 1]) < self.tf_ratio
    next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame)
    ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
```

**Key differences**:
1. **Sequential vs parallel**: Loop vs vectorized
2. **Accumulation**: Predictions accumulate vs independent mixing
3. **Context window**: Uses MAX_CONTEXT_FRAMES vs full sequence
4. **Stop gradient**: Uses `stop_gradient` on input vs no stop_gradient

### Behavioral Equivalence

The approaches are **not exactly equivalent** but achieve the same training goal:

| Aspect | Autoregressive | Parallel |
|--------|---------------|----------|
| Error exposure | Accumulated | Independent per frame |
| Forward passes | 127 | 2 |
| Gradient flow | Through predictions | Through both passes |
| Context | Windowed | Full sequence |
| Speed | Slow | Fast |

**Conclusion**: Parallel scheduled sampling is a **practical approximation** that maintains the benefits while being much faster.

## Implementation Considerations

### Mask Granularity
The context shows:
```python
mask = tf.random.uniform(tf.shape(x)) < tf_ratio
```

This creates a mask at **per-element** granularity (every mel bin independently).

**Alternative**: Per-frame granularity:
```python
mask = tf.random.uniform([batch, seq_len, 1]) < tf_ratio
mask = tf.broadcast_to(mask, tf.shape(x))
```

**Recommendation**: Per-frame is more sensible (either use entire frame or don't).

### Memory Efficiency
- **Current**: Stores growing `ar_input` (1 → 128 frames)
- **Parallel**: Stores fixed-size tensors
- **Benefit**: More memory efficient, better for large batch sizes

### Graph Compatibility
- **Current**: Uses Python loop with `range(1, SEQ_LEN)` - works in graph mode because SEQ_LEN is static
- **Parallel**: Pure tensor operations - fully graph-compatible
- **Benefit**: Can remove `run_eagerly=True` for additional speedup

## Validation Strategy

To ensure correctness, we should verify:

1. **Shape preservation**: All tensors have expected shapes
2. **Loss equivalence**: Loss values in similar range to current implementation
3. **Training stability**: Model converges, no NaN/Inf
4. **Gradient flow**: Gradients are non-zero and reasonable magnitude
5. **tf_ratio behavior**: Mask respects tf_ratio distribution

## References

- Context provided by user describing parallel scheduled sampling
- Current implementation in `src/music_generation/train.py`
- Scheduled sampling paper: Bengio et al. (2015) "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"
