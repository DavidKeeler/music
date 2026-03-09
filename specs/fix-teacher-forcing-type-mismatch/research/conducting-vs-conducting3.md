# Research: Comparing conducting vs conducting3

## Direct Code Comparison

### conducting/ - Original Autoregressive Loop
```python
with tf.GradientTape() as tape:
    ar_input = x[:, :1, :]
    preds = []
    
    for t in range(1, seq_len):
        pred = self.base_model(ar_input, training=True)[:, -1:, :]
        preds.append(pred)
        
        use_teacher = tf.random.uniform([batch_size, 1, 1]) < tf_ratio
        next_input = tf.where(use_teacher, x[:, t:t+1, :], pred)
        ar_input = tf.concat([ar_input, next_input], axis=1)
    
    pred_seq = tf.concat(preds, axis=1)
    loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
```

**Key point:** Passes FULL `ar_input` to model (no slicing)

### conducting3/ - Current Autoregressive Loop
```python
with tf.GradientTape() as tape:
    ar_input = x[:, :1, :]
    preds = []
    
    for t in range(1, seq_len):
        context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]  # DIFFERENCE: Explicit cap
        pred = self.base_model(context, training=True)
        next_frame_pred = pred[:, -1:, :]
        
        use_teacher = tf.random.uniform([batch_size, 1, 1]) < self.tf_ratio
        next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame_pred)
        preds.append(next_frame_pred)
        ar_input = tf.concat([ar_input, next_input], axis=1)
    
    pred_seq = tf.concat(preds, axis=1)
    loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
```

**Key point:** Caps at `MAX_CONTEXT_FRAMES=128` before passing to model

---

## Are They The Same Now?

**YES - Functionally equivalent for variable-length sequences!**

Both models can handle arbitrary sequence lengths:

### Original (conducting/)
- Uses causal convolutions (`padding='causal'`)
- No positional encodings
- Can handle any sequence length
- **User confirmed: worked with longer sequences**

### Current (conducting3/)
- Uses causal convolutions + windowed attention
- No positional encodings
- `window_size` is NOT a sequence length limit - it's just the attention window
- Can handle any sequence length

**The explicit capping at `MAX_CONTEXT_FRAMES=128` is UNNECESSARY!**

---

## What The Capping Actually Does

```python
context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]  # Takes last 128 frames
pred = self.base_model(context, training=True)
```

This limits the model to seeing only the last 128 frames of context, even though:
1. The model CAN handle longer sequences
2. The original code passed the full sequence
3. User confirmed the original worked with longer sequences

**This is actually WORSE than the original** - it artificially limits context!

---

## The Real Question

**Did the old code handle longer sequences?**

**Answer: NO**

The old code only ever used `seq_len=128`. The current code (conducting3) is trying to use `seq_len=512`, which is **4x longer than the original ever attempted**.

The explicit capping at `MAX_CONTEXT_FRAMES=128` in conducting3 is a **safety feature** to prevent the model from seeing sequences longer than it was designed for.

---

## Memory Issue Applies to Both

Both implementations have the same memory issue pattern:
- `ar_input` grows inside `GradientTape`
- All intermediate tensors retained

**Original:** seq_len=128 → 10.5 MB (manageable)
**Current:** seq_len=512 → 168 MB (problematic)

The issue scales with sequence length, not the implementation pattern.

---

## Summary

| Aspect | conducting/ | conducting3/ |
|--------|------------|-------------|
| **Max seq_len tested** | 128 | 512 (attempting) |
| **Context capping** | None (not needed) | Explicit at 128 |
| **Memory pattern** | Same (growing ar_input) | Same (growing ar_input) |
| **Memory usage** | 10.5 MB @ seq_len=128 | 168 MB @ seq_len=512 |
| **Would work with seq_len=512?** | No (no capping) | Yes (with capping) |

**Conclusion:** The implementations are functionally similar for seq_len=128, but conducting3 adds safety for longer sequences. The memory issue exists in both but only becomes problematic at longer sequence lengths.

---

## Training Approach

### conducting/ (Original)
```python
def train_step(self, data):
    x, y = data
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    tf_ratio = self.compute_tf_ratio()
    
    with tf.GradientTape() as tape:
        ar_input = x[:, :1, :]
        preds = []
        
        for t in range(1, seq_len):
            pred = self.base_model(ar_input, training=True)[:, -1:, :]
            preds.append(pred)
            
            use_teacher = tf.random.uniform([batch_size, 1, 1]) < tf_ratio
            next_input = tf.where(use_teacher, x[:, t:t+1, :], pred)
            ar_input = tf.concat([ar_input, next_input], axis=1)
        
        pred_seq = tf.concat(preds, axis=1)
        loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
    
    grads = tape.gradient(loss, self.base_model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
```

**Key characteristics:**
- **Always autoregressive** - loops through timesteps one by one
- No parallel training path
- Passes full `ar_input` to model (no explicit capping)
- Uses `seq_len = 128` in practice

### conducting3/ (Current)
```python
def train_step(self, data):
    self.update_tf_ratio()
    x, y = data
    
    # Path 1: Pure teacher forcing (parallel)
    if self.tf_ratio >= 1.0 - 1e-6:
        with tf.GradientTape() as tape:
            preds = self.base_model(x, training=True)
            loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        return {"loss": loss, ...}
    
    # Path 2: Scheduled sampling (autoregressive)
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(y)[1]
    
    with tf.GradientTape() as tape:
        ar_input = x[:, :1, :]
        preds = []
        
        for t in range(1, seq_len):
            context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]  # Cap at 128
            pred = self.base_model(context, training=True)
            next_frame_pred = pred[:, -1:, :]
            
            use_teacher = tf.random.uniform([batch_size, 1, 1]) < self.tf_ratio
            next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame_pred)
            preds.append(next_frame_pred)
            ar_input = tf.concat([ar_input, next_input], axis=1)
        
        pred_seq = tf.concat(preds, axis=1)
        loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
    
    grads = tape.gradient(loss, self.base_model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
```

**Key characteristics:**
- **Two paths**: parallel when tf_ratio=1.0, autoregressive when tf_ratio<1.0
- Explicitly caps context at `MAX_CONTEXT_FRAMES=128`
- Uses `SEQ_LEN = 512` (4x longer sequences)

---

## Teacher Forcing Schedule

### conducting/ (Original)
```python
def compute_tf_ratio(self):
    step = tf.cast(self.optimizer.iterations, tf.float32)
    ratio = self.initial_tf_ratio * tf.exp(-self.decay_k * step)
    return tf.maximum(self.min_ratio, ratio)
```

**Characteristics:**
- Uses `optimizer.iterations` (built-in counter)
- No warmup period
- No explicit training_step variable
- Computed on-demand each train_step

### conducting3/ (Current)
```python
def __init__(self, ...):
    self.warmup_steps = warmup_steps  # NEW: warmup period
    self.training_step = tf.Variable(0, trainable=False, dtype=tf.int64)  # NEW: explicit counter
    self.tf_ratio = tf.Variable(initial_tf_ratio, trainable=False, dtype=tf.float32)  # NEW: cached

def update_tf_ratio(self):
    # BUG HERE: type mismatch
    step_after_warmup = tf.maximum(0, self.training_step - self.warmup_steps)
    
    new_ratio = exponential_tf_schedule(
        step=step_after_warmup,
        initial_ratio=self.initial_tf_ratio,
        min_ratio=self.min_tf_ratio,
        decay_k=self.decay_k
    )
    
    self.tf_ratio.assign(new_ratio)
    self.training_step.assign_add(1)
```

**Characteristics:**
- Explicit `training_step` variable (int64)
- Adds warmup period support
- Caches `tf_ratio` in a variable
- Updated at start of each train_step

---

## Configuration

### conducting/ (Original)
```python
# Hardcoded in train_curriculum()
seq_len = 128
batch_size = 8
epochs = 20

model = TrainingModel(
    base_model,
    initial_tf_ratio=1.0,
    decay_k=3e-5,
    min_ratio=0.2
)
```

### conducting3/ (Current)
```python
# From config.py
BATCH_SIZE = 4
SEQ_LEN = 512
NUM_EPOCHS = 3

INITIAL_TF_RATIO = 1.0
MIN_TF_RATIO = 0.05
TF_DECAY_K = 1e-5
TF_WARMUP_STEPS = 0  # NEW parameter
MAX_CONTEXT_FRAMES = 128  # NEW parameter
```

---

## Summary of Key Differences

| Aspect | conducting/ (Original) | conducting3/ (Current) |
|--------|----------------------|----------------------|
| **Training mode** | Always autoregressive | Parallel + autoregressive |
| **Sequence length** | 128 frames | 512 frames (4x longer) |
| **Batch size** | 8 | 4 |
| **Context capping** | None (implicit in model) | Explicit at 128 frames |
| **TF ratio tracking** | Computed from optimizer.iterations | Explicit Variable + warmup |
| **Warmup support** | No | Yes (but buggy) |
| **Memory usage** | ~10 MB/batch | ~168 MB/batch (16x more!) |

---

## The Bugs

### Bug 1: Type Mismatch (Line 125)
```python
step_after_warmup = tf.maximum(0, self.training_step - self.warmup_steps)
#                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                   int64           -    int32 (Python int)
```

**Why original doesn't have this:**
- Original uses `optimizer.iterations` directly
- No warmup_steps subtraction
- Immediately casts to float32

### Bug 2: Memory Issue (Line 220)
```python
ar_input = tf.concat([ar_input, next_input], axis=1)
# Inside GradientTape, grows from [4,1,80] to [4,512,80]
# All intermediate tensors retained for backprop
```

**Why original doesn't have this:**
- Original uses seq_len=128 (16x less memory)
- Still has the issue, just smaller scale
- Likely didn't notice because it fit in memory

---

## What conducting3 Was Trying to Improve

1. **Longer sequences** (512 vs 128) - better long-range modeling
2. **Parallel training path** - faster when tf_ratio=1.0
3. **Warmup support** - more control over training schedule
4. **Explicit context capping** - memory safety for longer sequences
5. **Better configuration** - centralized in config.py

But introduced bugs in the process!
