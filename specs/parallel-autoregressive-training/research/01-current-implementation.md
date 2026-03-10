# Research: Current Implementation Analysis

## Overview

Analyzed the existing training pipeline in `src/music_generation/train.py`, model architecture in `model.py`, and custom layers in `layers.py`.

## Current Training Approach

### The Problem: Inefficient Autoregressive Loop

The `MelGeneratorTraining.train_step()` method implements two training paths:

1. **Pure Teacher Forcing (tf_ratio ≥ 1.0):**
   - Single forward pass with full sequence
   - Loss computed as `preds[:, :-1, :] vs targets[:, 1:, :]`
   - **This is already efficient and correct**

2. **Autoregressive Training (tf_ratio < 1.0):**
   - **Inefficient O(T) loop** that grows input sequence frame-by-frame
   - For each timestep `t` from 1 to SEQ_LEN:
     - Takes last MAX_CONTEXT_FRAMES as context
     - Calls `self.base_model(context, training=True)` 
     - Extracts last frame prediction
     - Applies scheduled sampling (mix ground truth vs prediction based on tf_ratio)
     - Concatenates to growing input sequence
   - **This forces SEQ_LEN forward passes per batch**

### Code Evidence

```python
# From train.py lines ~180-210 (autoregressive_training function)
for t in range(1, SEQ_LEN):
    context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]
    pred = self.base_model(context, training=True)  # ← Repeated model call
    next_frame_pred = pred[:, -1:, :]
    
    use_teacher = tf.random.uniform([batch_size, 1, 1]) < self.tf_ratio
    next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame_pred)
    
    preds.append(next_frame_pred)
    ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
```

**Performance Impact:**
- SEQ_LEN = 256 (from config)
- Each batch requires 255 forward passes instead of 1
- **~255x computational overhead** during scheduled sampling phase

## Model Architecture: Already Causal

### Good News: Architecture Supports Parallel Training

The `MelGenerator` model is **already fully causal**:

1. **Causal Convolutions** (`CausalConv1D`):
   - Left-padding only: `padding = (kernel_size - 1) * dilation_rate`
   - Output at time `t` depends only on inputs `≤ t`

2. **Local Window Attention** (`LocalWindowAttention`):
   - Sliding windows with left-padding
   - Relative position bias
   - Memory efficient: O(B × H × T × window_size) vs O(B × H × T²)
   - **Already implements causal masking via windowing**

3. **Transformer Blocks**:
   - Pre-norm architecture (norm before attention/FFN)
   - Residual connections

### Architecture Flow

```
Input [B, T, 80] 
  ↓ Dense projection
[B, T, D_MODEL=512]
  ↓ CausalConvBlock (kernel=3)
[B, T, 512]
  ↓ CausalConvBlock (kernel=3)
[B, T, 512]
  ↓ TransformerBlock (window=128)
[B, T, 512]
  ↓ TransformerBlock (window=256)
[B, T, 512]
  ↓ TransformerBlock (window=512)
[B, T, 512]
  ↓ CausalConvBlock (kernel=3)
[B, T, 512]
  ↓ Dense projection
[B, T, 80]
```

**Key Insight:** The model's `call()` method already processes full sequences in parallel. The causal structure is baked into the layers themselves.

## Teacher Forcing Schedule

### Current Implementation

- **Exponential decay:** `ε(step) = max(ε_min, ε_initial × exp(-k × step))`
- **Parameters:**
  - `initial_tf_ratio = 1.0` (pure teacher forcing)
  - `min_tf_ratio = 0.05` (maintains 5% stability)
  - `decay_k = 1e-5` (slow decay)
  - `warmup_steps = 5000` (keeps ratio at 1.0 initially)

### Scheduled Sampling Logic

In the autoregressive loop:
```python
use_teacher = tf.random.uniform([batch_size, 1, 1]) < self.tf_ratio
next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame_pred)
```

**Per-frame stochastic mixing:**
- Each frame independently sampled
- Probability `tf_ratio` → use ground truth
- Probability `1 - tf_ratio` → use model prediction

## The Refactoring Challenge

### What Needs to Change

1. **Eliminate the timestep loop** in `autoregressive_training()`
2. **Implement parallel scheduled sampling** that:
   - Processes full sequence in one forward pass
   - Applies scheduled sampling to the input sequence before the forward pass
   - Maintains the same stochastic mixing behavior

### What Stays the Same

1. Model architecture (already causal)
2. Teacher forcing schedule (exponential decay)
3. Loss computation (`preds[:, :-1, :] vs targets[:, 1:, :]`)
4. The pure teacher forcing path (already optimal)

### Key Technical Question

**How to implement scheduled sampling without the loop?**

The current approach mixes ground truth and predictions **during** the forward pass. In parallel training, we need to:
- Pre-construct a mixed input sequence
- But predictions aren't available yet...

**Possible solutions:**
1. **Parallel scheduled sampling with input corruption** (add noise to ground truth instead of mixing with predictions)
2. **Two-pass approach** (forward pass to get predictions, then mix and re-forward)
3. **Eliminate scheduled sampling entirely** (use pure teacher forcing, rely on inference-time techniques)

This requires further research into modern scheduled sampling techniques for parallel training.

## Performance Expectations

### Current Performance
- Autoregressive path: O(T) forward passes per batch
- With SEQ_LEN=256: ~255 model calls per batch

### Target Performance
- Parallel path: O(1) forward pass per batch
- **Expected speedup: ~100-200x** (accounting for overhead)

### Memory Trade-offs
- Current: Small memory footprint (only stores growing sequence)
- Parallel: Full sequence in memory (batch_size × SEQ_LEN × mel_dim)
- With batch_size=16, SEQ_LEN=256, mel_dim=80: ~5MB per batch (negligible)

## References

- `src/music_generation/train.py` - Training loop implementation
- `src/music_generation/model.py` - MelGenerator architecture
- `src/music_generation/layers.py` - Causal layers (CausalConv1D, LocalWindowAttention)
- `src/music_generation/config.py` - Hyperparameters (SEQ_LEN=256, D_MODEL=512, etc.)
