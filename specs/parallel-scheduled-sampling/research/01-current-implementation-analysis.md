# Research: Current Implementation Analysis

## Overview

Analysis of the existing `MelGeneratorTraining.train_step()` method in `src/music_generation/train.py` to understand the autoregressive loop and identify optimization opportunities.

## Current Architecture

### Model: MelGenerator
- **Type**: Autoregressive mel spectrogram generator
- **Architecture**: Transformer-based with causal convolutions
- **Input/Output**: `[batch, seq_len, 80]` mel spectrograms
- **Key property**: Strictly causal - output at time t depends only on inputs ≤ t

**Layers:**
1. Input projection: Dense(80 → 128)
2. 2 causal conv blocks (kernel_size=3, residual)
3. 3 transformer blocks with local attention (windows: 128, 256, 512)
4. 1 causal conv head (kernel_size=3, residual)
5. Output projection: Dense(128 → 80)

### Training Configuration
- **SEQ_LEN**: 128 frames
- **BATCH_SIZE**: 16 (default)
- **D_MODEL**: 128
- **NUM_HEADS**: 4
- **MAX_CONTEXT_FRAMES**: Context window limit for efficiency

## Current train_step Implementation

The training uses **two separate code paths** based on teacher forcing ratio:

### Path 1: Pure Teacher Forcing (tf_ratio ≥ 1.0)
```python
def pure_teacher_forcing():
    preds = self.base_model(x, training=True)  # 1 forward pass
    loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
```
- **Forward passes**: 1
- **Behavior**: Model sees only ground truth inputs
- **Speed**: Fast (parallel)
- **Used**: Early training when tf_ratio = 1.0

### Path 2: Autoregressive Training (tf_ratio < 1.0)
```python
def autoregressive_training():
    ar_input = x[:, :1, :]  # Start with first frame
    preds = []
    
    for t in range(1, SEQ_LEN):  # Loop 127 times
        context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]
        pred = self.base_model(context, training=True)  # Forward pass #t
        next_frame_pred = pred[:, -1:, :]
        
        # Scheduled sampling
        use_teacher = tf.random.uniform([batch_size, 1, 1]) < self.tf_ratio
        next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame_pred)
        
        preds.append(next_frame_pred)
        ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
    
    pred_seq = tf.concat(preds, axis=1)
    loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
```
- **Forward passes**: 127 (one per timestep, excluding first frame)
- **Behavior**: Model sees mix of ground truth and its own predictions
- **Speed**: Very slow (sequential bottleneck)
- **Used**: After warmup when tf_ratio < 1.0

## Performance Bottleneck

### The Problem
The autoregressive loop is **O(seq_len)** in forward passes:
- Each timestep requires a separate forward pass
- Cannot be parallelized due to data dependency
- With SEQ_LEN=128: **127 forward passes per training step**

### Why It's Slow
1. **Sequential execution**: Must wait for prediction at time t before computing t+1
2. **Gradient tape overhead**: All 127 forward passes tracked in single tape
3. **Memory growth**: `ar_input` grows from length 1 to 128
4. **Context slicing**: Each iteration slices last MAX_CONTEXT_FRAMES

### Estimated Impact
- Pure teacher forcing: ~1 second per batch
- Autoregressive training: ~50-100 seconds per batch
- **Slowdown factor**: 50-100×

## Key Observations

### 1. Model Already Supports Parallel Forward Pass
The `MelGenerator.call()` method accepts full sequences:
```python
def call(self, mel, training=False):
    # Input: [batch, seq_len, N_MELS]
    # Output: [batch, seq_len, N_MELS]
```

The model is **already causal** through:
- Causal convolutions (padding on left only)
- Causal attention masks in transformer blocks

This means the model can process entire sequences in parallel while maintaining causality.

### 2. Scheduled Sampling Logic is Simple
The mixing logic is straightforward:
```python
use_teacher = tf.random.uniform([batch_size, 1, 1]) < self.tf_ratio
next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame_pred)
```

This can be vectorized across the entire sequence.

### 3. Loss Computation is Already Parallel
```python
loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
```

The loss doesn't require sequential computation.

### 4. Teacher Forcing Schedule is Working
The exponential decay schedule is implemented correctly:
- Starts at 1.0 (pure teacher forcing)
- Decays exponentially with warmup
- Floors at min_tf_ratio (0.05)

## Optimization Opportunity

The autoregressive loop is **unnecessary for training**. The model's causal architecture already ensures correct dependencies.

**Key insight**: We can:
1. Do one forward pass to get predictions for entire sequence
2. Mix predictions with ground truth based on tf_ratio (vectorized)
3. Do second forward pass with mixed inputs
4. Compute loss

This maintains scheduled sampling behavior while reducing from 127 to 2 forward passes.

## Dependencies and Constraints

### Must Preserve
- Teacher forcing ratio schedule (exponential decay)
- Scheduled sampling behavior (model sees own errors)
- Loss computation (MAE on next-frame prediction)
- Gradient computation and optimization
- Metrics tracking (loss, grad_norm, tf_ratio)

### Can Change
- Number of forward passes (127 → 2)
- Loop structure (sequential → parallel)
- Input construction (incremental concat → vectorized mixing)

### TensorFlow Constraints
- Currently uses `run_eagerly=True` in model.compile()
- Uses `tf.cond` to switch between training paths
- Uses `tf.debugging.assert_equal` for shape validation
- Uses `tf.print` for debugging

## Next Steps

The current implementation is correct but inefficient. The parallel scheduled sampling approach can maintain identical behavior while achieving 50-100× speedup.
