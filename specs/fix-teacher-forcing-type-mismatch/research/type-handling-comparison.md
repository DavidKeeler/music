# Research: Type Handling in Original vs Current Implementation

## Original Implementation Analysis

**File:** `/Users/davidkeeler/code/conducting/src/main/python/complex_music_model/train_audio.py`

### Type Handling in compute_tf_ratio()

```python
def compute_tf_ratio(self):
    step = tf.cast(self.optimizer.iterations, tf.float32)
    ratio = self.initial_tf_ratio * tf.exp(-self.decay_k * step)
    return tf.maximum(self.min_ratio, ratio)
```

**Key observations:**
- Uses `self.optimizer.iterations` (built-in counter, type: `int64`)
- Immediately casts to `tf.float32` for computation
- No warmup_steps parameter - warmup is handled separately in learning rate schedule
- No explicit training_step variable
- `tf.maximum()` compares two `float32` values (no type mismatch possible)

### Autoregressive Loop (Growing Context)

```python
def train_step(self, data):
    x, y = data
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    
    with tf.GradientTape() as tape:
        ar_input = x[:, :1, :]  # Start with 1 frame
        preds = []
        
        for t in range(1, seq_len):
            pred = self.base_model(ar_input, training=True)[:, -1:, :]
            preds.append(pred)
            
            use_teacher = tf.random.uniform([batch_size, 1, 1]) < tf_ratio
            next_input = tf.where(use_teacher, x[:, t:t+1, :], pred)
            ar_input = tf.concat([ar_input, next_input], axis=1)  # Grows unbounded
        
        pred_seq = tf.concat(preds, axis=1)
        loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
```

**Key characteristics:**
- `ar_input` starts at shape `[batch, 1, mel_dim]`
- Grows by 1 frame each iteration: `[batch, 2, mel_dim]`, `[batch, 3, mel_dim]`, etc.
- No explicit capping - passes full growing sequence to model
- Model internally handles context (likely has positional encoding limits)

**Performance implication:**
- Iteration 1: Process 1 frame
- Iteration 2: Process 2 frames
- Iteration 128: Process 128 frames
- Iteration 512: Process 512 frames
- Average: ~256 frames per iteration for 512-step sequence

---

## Current Implementation Analysis

**File:** `/Users/davidkeeler/code/conducting3/src/music_generation/train.py`

### Type Handling in update_tf_ratio()

```python
def __init__(self, base_model, initial_tf_ratio=INITIAL_TF_RATIO, 
             min_tf_ratio=MIN_TF_RATIO, decay_k=TF_DECAY_K, warmup_steps=TF_WARMUP_STEPS):
    super().__init__()
    # ...
    self.warmup_steps = warmup_steps  # Python int from config (TF_WARMUP_STEPS = 0)
    self.training_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="training_step")

def update_tf_ratio(self):
    # Line 125 - THE BUG
    step_after_warmup = tf.maximum(0, self.training_step - self.warmup_steps)
    # self.training_step is int64, self.warmup_steps is Python int (becomes int32 in TF)
```

**The type mismatch:**
- `self.training_step`: `tf.Variable` with explicit `dtype=tf.int64`
- `self.warmup_steps`: Python `int` → TensorFlow converts to `int32` by default
- `tf.maximum(0, int64 - int32)` → Type error!

### Autoregressive Loop (Growing Context with Cap)

```python
def train_step(self, data):
    # ...
    with tf.GradientTape() as tape:
        ar_input = x[:, :1, :]  # Start with 1 frame
        preds = []
        
        for t in range(1, seq_len):
            # Cap at MAX_CONTEXT_FRAMES before passing to model
            context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]
            pred = self.base_model(context, training=True)
            next_frame_pred = pred[:, -1:, :]
            
            use_teacher = tf.random.uniform([batch_size, 1, 1]) < self.tf_ratio
            next_input = tf.where(use_teacher, x[:, t:t+1, :], next_frame_pred)
            
            preds.append(next_frame_pred)
            ar_input = tf.concat([ar_input, next_input], axis=1)  # Grows unbounded
```

**Key characteristics:**
- `ar_input` grows unbounded (same as original)
- Caps at `MAX_CONTEXT_FRAMES=128` only when passing to model
- Slicing `[:, -128:, :]` is cheap (no copy, just view)

**Performance implication:**
- Iteration 1: Process 1 frame
- Iteration 2: Process 2 frames
- Iteration 128: Process 128 frames
- Iteration 129-512: Process 128 frames (capped)
- Average: ~100 frames per iteration for 512-step sequence

**Comparison to broken fixed-window approach:**
- Fixed window would process 128 frames every iteration
- Average: 128 frames per iteration
- Current approach is ~22% faster on average

---

## Root Cause Summary

The type mismatch occurs because:
1. `self.training_step` is explicitly created as `int64`
2. `self.warmup_steps` is a Python `int` (defaults to `int32` in TensorFlow)
3. TensorFlow's `tf.maximum()` requires matching types

The original code avoided this by:
- Using `optimizer.iterations` (always `int64`)
- Immediately casting to `float32` for all computations
- No warmup_steps subtraction

---

## Fix Options

### Option 1: Cast warmup_steps to int64 (Recommended)
```python
def update_tf_ratio(self):
    step_after_warmup = tf.maximum(0, self.training_step - tf.cast(self.warmup_steps, tf.int64))
```

**Pros:**
- Minimal change (1 line)
- Preserves int64 precision for training_step
- Explicit and clear

**Cons:**
- None

### Option 2: Cast training_step to int32
```python
self.training_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="training_step")
```

**Pros:**
- No change to update_tf_ratio()

**Cons:**
- int32 max value is ~2.1B - could overflow for very long training runs
- Less future-proof

### Option 3: Cast both to float32 (like original)
```python
def update_tf_ratio(self):
    step_float = tf.cast(self.training_step, tf.float32)
    warmup_float = tf.cast(self.warmup_steps, tf.float32)
    step_after_warmup = tf.maximum(0.0, step_float - warmup_float)
```

**Pros:**
- Matches original pattern
- No precision issues for exponential computation

**Cons:**
- More verbose
- Unnecessary since exponential_tf_schedule already casts to float32

---

## Recommendation

**Use Option 1:** Cast `self.warmup_steps` to `int64` in the subtraction.

This is the minimal fix that:
- Resolves the type mismatch
- Preserves int64 precision
- Requires only 1 character change
- Follows TensorFlow best practices
