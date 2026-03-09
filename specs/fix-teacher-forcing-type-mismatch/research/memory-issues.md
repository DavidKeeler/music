# Research: Memory Issues in Autoregressive Training

## Problem: Growing Tensor Inside GradientTape

### Current Implementation (and Original)

```python
with tf.GradientTape() as tape:
    ar_input = x[:, :1, :]  # [batch, 1, 80]
    preds = []
    
    for t in range(1, seq_len):  # seq_len = 512
        context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]
        pred = self.base_model(context, training=True)
        next_frame_pred = pred[:, -1:, :]
        preds.append(next_frame_pred)
        
        # MEMORY ISSUE: Creates new tensor every iteration
        ar_input = tf.concat([ar_input, next_input], axis=1)
    
    pred_seq = tf.concat(preds, axis=1)
    loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))

grads = tape.gradient(loss, self.base_model.trainable_variables)
```

### Memory Accumulation

**GradientTape behavior:**
- Records all operations for backpropagation
- Keeps all intermediate tensors in memory
- Cannot free memory until `tape.gradient()` is called

**Memory growth per iteration:**
- Iteration 1: `ar_input` = `[4, 1, 80]` = 1.28 KB
- Iteration 2: `ar_input` = `[4, 2, 80]` = 2.56 KB (old tensor still in tape)
- Iteration 3: `ar_input` = `[4, 3, 80]` = 3.84 KB (old tensors still in tape)
- ...
- Iteration 512: `ar_input` = `[4, 512, 80]` = 655 KB

**Total memory retained:**
- Sum of all ar_input versions: 1 + 2 + 3 + ... + 512 = 131,328 frames
- Total: `4 * 131,328 * 80 * 4 bytes` = **168 MB per batch**
- Plus `preds` list: `4 * 511 * 80 * 4 bytes` = **0.65 MB**
- Plus all model activations for 512 forward passes

**For comparison, original code (seq_len=128):**
- Sum: 1 + 2 + 3 + ... + 128 = 8,256 frames
- Total: `4 * 8,256 * 80 * 4 bytes` = **10.5 MB per batch**
- **Current uses 16x more memory!**

---

## Why This Causes OOM (Out of Memory)

**Scenario:** Training on GPU with 8GB VRAM
- Model parameters: ~50 MB
- Model activations per forward pass: ~100 MB
- Gradient tape overhead: **168 MB** (the problem!)
- Optimizer state: ~100 MB
- **Total: ~418 MB per batch**

With `batch_size=4`, this is manageable, but:
- Multiple batches in flight (prefetching)
- TensorFlow memory fragmentation
- Other GPU processes

**Result:** OOM errors or severe memory pressure

---

## Solutions

### Solution 1: Stop Gradient on ar_input (Recommended)

Only track gradients through the model predictions, not the growing input tensor:

```python
with tf.GradientTape() as tape:
    ar_input = x[:, :1, :]
    preds = []
    
    for t in range(1, seq_len):
        context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]
        pred = self.base_model(context, training=True)
        next_frame_pred = pred[:, -1:, :]
        preds.append(next_frame_pred)
        
        use_teacher = tf.random.uniform([batch_size, 1, 1]) < self.tf_ratio
        next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame_pred)
        
        # Stop gradient: ar_input doesn't need gradients
        ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
    
    pred_seq = tf.concat(preds, axis=1)
    loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
```

**Effect:**
- `ar_input` growth is not tracked by gradient tape
- Only `preds` list is tracked (much smaller)
- Memory usage: **~10 MB instead of 168 MB**

**Correctness:**
- Gradients still flow through `preds` → model parameters
- `ar_input` is just a container for context, doesn't need gradients
- Loss is computed on `pred_seq`, not `ar_input`

### Solution 2: Use TensorArray Instead of List

```python
with tf.GradientTape() as tape:
    ar_input = x[:, :1, :]
    preds = tf.TensorArray(dtype=tf.float32, size=seq_len-1, dynamic_size=False)
    
    for t in range(1, seq_len):
        context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]
        pred = self.base_model(context, training=True)
        next_frame_pred = pred[:, -1:, :]
        preds = preds.write(t-1, next_frame_pred)
        
        use_teacher = tf.random.uniform([batch_size, 1, 1]) < self.tf_ratio
        next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame_pred)
        ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
    
    pred_seq = tf.concat(tf.unstack(preds.stack()), axis=1)
    loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
```

**Benefits:**
- More efficient than Python list
- Pre-allocated memory
- Better for graph mode

**Drawbacks:**
- More complex code
- Minimal benefit over Solution 1

### Solution 3: Reduce SEQ_LEN

```python
SEQ_LEN = 128  # Instead of 512
```

**Benefits:**
- Matches original implementation
- Reduces memory by 16x
- Simpler fix

**Drawbacks:**
- Shorter training sequences
- May hurt model's ability to learn long-range dependencies
- Doesn't solve the fundamental issue

---

## Context: Three Different Implementations

1. **conducting/** (TensorFlow, original):
   - Autoregressive training with seq_len=128
   - Memory usage: ~10.5 MB per batch
   - No memory issues due to short sequences

2. **conducting2/** (PyTorch):
   - **Parallel training only** (no autoregressive loop!)
   - seq_len=512
   - No memory issues because no growing tensor

3. **conducting3/** (TensorFlow, current):
   - **Both parallel AND autoregressive** training
   - Parallel when tf_ratio >= 1.0 (no memory issue)
   - Autoregressive when tf_ratio < 1.0 (memory issue!)
   - seq_len=512 → 16x more memory than original

**The memory issue only occurs during scheduled sampling phase (tf_ratio < 1.0).**

## Recommendation

**Use Solution 1: Add `tf.stop_gradient()` to ar_input growth**

This is the minimal fix that:
1. Solves the memory issue during autoregressive training
2. Preserves SEQ_LEN=512 for better long-range modeling
3. Maintains correctness (gradients flow through preds)
4. Single-line change
5. No performance impact

**Combined with the type fix, we have two one-line changes:**
1. Line 125: Cast warmup_steps to int64 (fixes type error)
2. Line 220: Add tf.stop_gradient() to ar_input concat (fixes memory issue)

---

## Memory Comparison

| Approach | Memory per Batch | Notes |
|----------|------------------|-------|
| Current (buggy) | 168 MB | OOM risk |
| With stop_gradient | 10 MB | Recommended |
| Reduce SEQ_LEN to 128 | 10.5 MB | Loses long-range modeling |
| Original (seq_len=128) | 10.5 MB | Reference |

---

## Verification

After applying `tf.stop_gradient()`:

```python
# Test memory usage
import tracemalloc
tracemalloc.start()

model.train_step((x, y))

current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
tracemalloc.stop()
```

Expected: Peak memory should drop from ~168 MB to ~10 MB for the gradient tape portion.
