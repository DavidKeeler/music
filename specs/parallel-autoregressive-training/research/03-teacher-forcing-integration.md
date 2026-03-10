# Research: Teacher Forcing with Parallel Training

## Overview

Analysis of how teacher forcing and scheduled sampling integrate with parallel autoregressive training using causal masks.

## Teacher Forcing Fundamentals

### Pure Teacher Forcing (tf_ratio = 1.0)

**Definition:** During training, always use ground truth as input for predicting the next token.

**In parallel training:**
```python
# Input:  ground_truth[0:T-1]
# Output: predictions[0:T-1]
# Target: ground_truth[1:T]

preds = model(ground_truth, training=True)  # Single forward pass
loss = loss_fn(preds[:, :-1, :], ground_truth[:, 1:, :])
```

**Advantages:**
- Fastest training (single forward pass)
- Stable gradients
- Works perfectly with causal masking

**Disadvantage:**
- **Exposure bias:** Model never sees its own predictions during training, leading to error accumulation at inference time

### The Exposure Bias Problem

**Training:** Model always conditions on perfect ground truth
```
Input:  [GT_0, GT_1, GT_2, GT_3]
Predict: [P_1,  P_2,  P_3,  P_4]
```

**Inference:** Model conditions on its own (potentially wrong) predictions
```
Input:  [GT_0, P_1,  P_2,  P_3]  ← P_1 might be wrong!
Predict: [P_1,  P_2,  P_3,  P_4]
```

**Result:** Small errors compound over time, causing distribution shift and degraded long-sequence generation.

## Scheduled Sampling

### Concept

**Gradually transition** from teacher forcing to autoregressive generation during training by randomly mixing ground truth and model predictions.

**Teacher forcing ratio (ε):** Probability of using ground truth at each position
- ε = 1.0: Pure teacher forcing
- ε = 0.0: Pure autoregressive (always use predictions)
- 0 < ε < 1: Mixed

### Decay Schedules

**Current implementation uses exponential decay:**

```python
ε(step) = max(ε_min, ε_initial × exp(-k × step))
```

**Parameters:**
- `ε_initial = 1.0` (start with pure teacher forcing)
- `ε_min = 0.05` (maintain 5% stability floor)
- `k = 1e-5` (decay rate)
- `warmup_steps = 5000` (delay decay start)

**Typical schedule:**
- Steps 0-5000: ε = 1.0 (pure teacher forcing)
- Step 10000: ε ≈ 0.95
- Step 50000: ε ≈ 0.60
- Step 100000: ε ≈ 0.37
- Step 200000+: ε ≈ 0.05 (floor)

**Alternative schedules:**
1. **Linear decay:** `ε(step) = max(ε_min, ε_initial - k × step)`
2. **Inverse sigmoid:** `ε(step) = k / (k + exp(step / k))`
3. **Step decay:** Drop ε at fixed intervals

## Integrating Scheduled Sampling with Parallel Training

### Challenge

Traditional scheduled sampling requires sequential processing:
```python
for t in range(seq_len):
    if random() < ε:
        input[t] = ground_truth[t]  # Teacher forcing
    else:
        input[t] = model_prediction[t-1]  # Autoregressive
```

This breaks parallelization.

### Solution 1: Two-Pass Parallel Scheduled Sampling

**Algorithm from "Parallel Scheduled Sampling" (ICLR 2020):**

```python
def two_pass_scheduled_sampling(x, model, tf_ratio):
    """
    Two forward passes for parallel scheduled sampling.
    
    Pass 1: Get predictions for all positions
    Pass 2: Train on mixed input (ground truth + predictions)
    """
    # Pass 1: Forward with ground truth to get predictions
    preds_pass1 = model(x, training=False)  # [B, T, D]
    
    # Create sampling mask (per-position, per-batch)
    use_teacher = tf.random.uniform([B, T, 1]) < tf_ratio
    
    # Shift predictions: pred[t-1] is input for position t
    preds_shifted = tf.concat([x[:, :1, :], preds_pass1[:, :-1, :]], axis=1)
    
    # Mix ground truth and predictions
    mixed_input = tf.where(use_teacher, x, preds_shifted)
    
    # Pass 2: Forward with mixed input (with gradients)
    with tf.GradientTape() as tape:
        preds_pass2 = model(mixed_input, training=True)
        loss = loss_fn(preds_pass2[:, :-1, :], x[:, 1:, :])
    
    # Backprop through pass 2 only
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss
```

**Complexity:** O(2) forward passes vs O(T) in sequential scheduled sampling

**Key insight:** First pass is inference-only (no gradients), second pass trains on mixed input.

### Solution 2: Single-Pass with Input Corruption

**Simpler approximation** that avoids two passes:

```python
def input_corruption_scheduled_sampling(x, model, corruption_ratio):
    """
    Single-pass approximation using noise instead of predictions.
    
    Corruption ratio = 1 - tf_ratio
    """
    # Add Gaussian noise to simulate prediction errors
    noise_scale = 0.1  # Tune based on data scale
    noise = tf.random.normal(tf.shape(x)) * noise_scale
    
    # Randomly corrupt positions
    use_corruption = tf.random.uniform(tf.shape(x)) < corruption_ratio
    corrupted_input = tf.where(use_corruption, x + noise, x)
    
    # Single forward pass
    with tf.GradientTape() as tape:
        preds = model(corrupted_input, training=True)
        loss = loss_fn(preds[:, :-1, :], x[:, 1:, :])
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss
```

**Trade-offs:**
- ✅ Faster: Single forward pass
- ✅ Simpler: No need to manage two passes
- ❌ Less accurate: Noise doesn't perfectly simulate model predictions
- ❌ Requires tuning: Noise scale must match typical prediction errors

### Solution 3: Hybrid Approach

**Combine both methods** based on training phase:

```python
def hybrid_scheduled_sampling(x, model, tf_ratio, step):
    """
    Use different strategies based on training progress.
    """
    if tf_ratio >= 0.95:
        # Early training: Pure teacher forcing (fastest)
        return pure_teacher_forcing(x, model)
    
    elif tf_ratio >= 0.5:
        # Mid training: Input corruption (fast approximation)
        corruption_ratio = 1 - tf_ratio
        return input_corruption_scheduled_sampling(x, model, corruption_ratio)
    
    else:
        # Late training: Two-pass PSS (accurate)
        return two_pass_scheduled_sampling(x, model, tf_ratio)
```

**Rationale:**
- Early: Model needs stable training → pure teacher forcing
- Mid: Model is learning → fast approximation sufficient
- Late: Model needs realistic errors → accurate two-pass method

## Current Implementation Analysis

### What Works

The current code already implements **pure teacher forcing efficiently**:

```python
# From train.py - pure_teacher_forcing() function
def pure_teacher_forcing():
    with tf.GradientTape() as tape:
        preds = self.base_model(x, training=True)  # ← Single pass
        loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
    
    grads = tape.gradient(loss, self.base_model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
    return {"loss": loss, ...}
```

**This is already optimal** - no changes needed for tf_ratio = 1.0.

### What Needs Refactoring

The **autoregressive_training()** function uses O(T) sequential passes:

```python
# Current inefficient implementation
for t in range(1, SEQ_LEN):
    context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]
    pred = self.base_model(context, training=True)  # ← Repeated call
    next_frame_pred = pred[:, -1:, :]
    
    use_teacher = tf.random.uniform([batch_size, 1, 1]) < self.tf_ratio
    next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame_pred)
    
    preds.append(next_frame_pred)
    ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
```

**Should be replaced with** two-pass parallel scheduled sampling.

## Recommended Approach for Refactoring

### Phase 1: Eliminate the Loop (Immediate Win)

**Replace autoregressive loop with two-pass PSS:**

```python
def parallel_scheduled_sampling():
    # Pass 1: Get predictions
    preds_pass1 = self.base_model(x, training=False)
    
    # Sample and mix
    use_teacher = tf.random.uniform([batch_size, seq_len, 1]) < self.tf_ratio
    preds_shifted = tf.concat([x[:, :1, :], preds_pass1[:, :-1, :]], axis=1)
    mixed_input = tf.where(use_teacher, x, preds_shifted)
    
    # Pass 2: Train
    with tf.GradientTape() as tape:
        preds_pass2 = self.base_model(mixed_input, training=True)
        loss = tf.reduce_mean(tf.abs(preds_pass2[:, :-1, :] - y[:, 1:, :]))
    
    grads = tape.gradient(loss, self.base_model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
    
    return {"loss": loss, ...}
```

**Expected speedup:** ~100x (from 255 passes to 2 passes)

### Phase 2: Optimize Further (Optional)

**Add hybrid strategy:**
- tf_ratio ≥ 0.99: Pure teacher forcing (1 pass)
- 0.5 ≤ tf_ratio < 0.99: Input corruption (1 pass)
- tf_ratio < 0.5: Two-pass PSS (2 passes)

**Expected speedup:** Additional 2x during mid-training phase

## Continuous vs Discrete Tokens

### Important Consideration

Most scheduled sampling literature focuses on **discrete tokens** (text, where you sample from a distribution).

**Mel spectrograms are continuous** - each frame is a vector of 80 continuous values.

### Implications

**For discrete tokens:**
```python
# Sample from predicted distribution
next_token = tf.random.categorical(logits, num_samples=1)
```

**For continuous mel frames:**
```python
# Use predicted values directly (no sampling)
next_frame = predictions[:, -1, :]

# Optional: Add temperature-scaled noise
if temperature > 0:
    noise = tf.random.normal(tf.shape(next_frame)) * temperature * 0.1
    next_frame = next_frame + noise
```

**Current implementation already handles this correctly** - it uses predictions directly without sampling.

## Summary

### Key Findings

1. **Pure teacher forcing** (tf_ratio = 1.0) is already optimal in current code
2. **Scheduled sampling** (tf_ratio < 1.0) currently uses inefficient O(T) loop
3. **Two-pass PSS** can reduce to O(2) while maintaining scheduled sampling semantics
4. **Input corruption** offers O(1) approximation for mid-training
5. Model architecture already supports parallel training (causal layers)

### Recommended Implementation

**Replace the autoregressive_training() function with:**
- Two-pass parallel scheduled sampling
- Keep pure teacher forcing path as-is
- Use tf.cond to switch based on tf_ratio threshold

**Expected performance:**
- Pure TF (tf_ratio=1.0): 1 forward pass (no change)
- Scheduled sampling (tf_ratio<1.0): 2 forward passes (vs 255 currently)
- **Overall speedup: ~50-100x** during scheduled sampling phase

## References

- "Parallel Scheduled Sampling" (Duckworth et al., ICLR 2020) - Two-pass algorithm
- "Scheduled Sampling for Sequence Prediction" (Bengio et al., 2015) - Original scheduled sampling
- "Professor Forcing" (Lamb et al., 2016) - Alternative to scheduled sampling
- Current implementation: `src/music_generation/train.py`
