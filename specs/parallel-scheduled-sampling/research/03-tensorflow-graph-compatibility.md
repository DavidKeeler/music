# Research: TensorFlow Graph Compatibility

## Overview

Analysis of TensorFlow graph mode compatibility for parallel scheduled sampling, including considerations for `tf.function` compilation and performance optimization.

## Current Graph Mode Status

### Training Configuration
```python
model.compile(optimizer=optimizer, run_eagerly=True)
```

**Current state**: Eager execution enabled (`run_eagerly=True`)

**Why**: Likely due to:
1. Debugging convenience (tf.print, assertions work)
2. Python loop in autoregressive training
3. Dynamic control flow with tf.cond

### Impact of Eager Mode
- **Pros**: Easy debugging, flexible control flow
- **Cons**: ~2-5× slower than graph mode
- **Memory**: Higher overhead per operation

## Graph Mode Requirements

For `tf.function` compilation (graph mode), operations must be:

1. **Tensor operations**: No Python control flow
2. **Static shapes**: Or explicit shape handling
3. **Serializable**: No Python closures with external state

## Parallel Scheduled Sampling Graph Compatibility

### Algorithm Review
```python
def train_step(self, data):
    x, y = data
    tf_ratio = self.compute_tf_ratio()
    
    with tf.GradientTape() as tape:
        # Step 1: First forward pass
        preds = self.base_model(x, training=True)
        
        # Step 2: Create mask
        mask = tf.random.uniform(tf.shape(x)) < tf_ratio
        
        # Step 3: Mix
        mixed = tf.where(mask, x, preds)
        
        # Step 4: Shift
        inputs = mixed[:, :-1]
        targets = y[:, 1:]
        
        # Step 5: Second forward pass
        preds2 = self.base_model(inputs, training=True)
        
        # Step 6: Loss
        loss = tf.reduce_mean(tf.abs(preds2 - targets))
    
    grads = tape.gradient(loss, self.base_model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
    
    return {"loss": loss, "tf_ratio": tf_ratio}
```

### Compatibility Analysis

✅ **Fully graph-compatible**

All operations are pure tensor operations:
- `tf.random.uniform`: Graph-compatible random op
- `tf.where`: Element-wise conditional
- Slicing (`[:, :-1]`): Static slicing
- `tf.reduce_mean`, `tf.abs`: Standard ops
- `tape.gradient`: Core TensorFlow API

**No Python control flow** - everything is TensorFlow ops.

## Comparison to Current Implementation

### Current Autoregressive Path
```python
def autoregressive_training():
    ar_input = x[:, :1, :]
    preds = []
    
    for t in range(1, SEQ_LEN):  # Python loop
        context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]
        pred = self.base_model(context, training=True)
        # ... more operations
        ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
    
    pred_seq = tf.concat(preds, axis=1)
    loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
```

**Graph compatibility**: ✅ Actually compatible!

The Python `for` loop with `range(SEQ_LEN)` is **unrolled at trace time** because:
- `SEQ_LEN` is a Python constant (not a tensor)
- Loop bounds are static
- TensorFlow converts it to a sequence of ops in the graph

**However**: This creates a **very large graph** (127 model calls), which:
- Increases compilation time
- Increases memory usage
- Slows down execution

### Current tf.cond Usage
```python
return tf.cond(
    self.tf_ratio >= 1.0 - 1e-6,
    pure_teacher_forcing,
    autoregressive_training
)
```

**Graph compatibility**: ✅ Compatible

`tf.cond` is designed for graph mode - it creates conditional branches in the graph.

**Issue**: Both branches are compiled into the graph, even though only one executes.

## Optimization Opportunities

### 1. Remove run_eagerly=True

With parallel scheduled sampling, we can safely use graph mode:

```python
model.compile(optimizer=optimizer, run_eagerly=False)
```

**Expected speedup**: 2-5× on top of the 50-100× from parallel sampling

**Total speedup**: 100-500× compared to current eager autoregressive

### 2. Remove tf.cond Branching

With parallel scheduled sampling, we don't need separate paths:

```python
def train_step(self, data):
    # Single unified path works for all tf_ratio values
    # No branching needed
```

**Benefits**:
- Simpler code
- Smaller graph
- Faster compilation
- No branch prediction overhead

### 3. Remove Debugging Ops

For production training, remove:
```python
tf.debugging.assert_equal(...)  # Adds overhead
tf.print(...)                    # Forces eager-like execution
```

**Alternative**: Use TensorBoard callbacks for monitoring

### 4. Use XLA Compilation

```python
@tf.function(jit_compile=True)
def train_step(self, data):
    # ... parallel scheduled sampling
```

**Benefits**:
- Fuses operations
- Optimizes memory layout
- Further 1.5-3× speedup

**Compatibility**: Parallel scheduled sampling should be XLA-compatible (all standard ops)

## Shape Handling

### Dynamic Shapes
```python
batch_size = tf.shape(x)[0]  # Dynamic
seq_len = tf.shape(x)[1]      # Could be dynamic
mel_dim = tf.shape(x)[2]      # Static (80)
```

**Current implementation**: Uses `tf.shape()` for dynamic batch size

**Parallel scheduled sampling**: Also uses `tf.shape()` - compatible

### Static Shape Assertions

For graph mode, we can add static shape hints:

```python
x.set_shape([None, SEQ_LEN, N_MELS])
y.set_shape([None, SEQ_LEN, N_MELS])
```

This helps TensorFlow optimize the graph.

## Gradient Tape Considerations

### Current Implementation
```python
with tf.GradientTape() as tape:
    for t in range(1, SEQ_LEN):
        pred = self.base_model(context, training=True)
        # ... 127 forward passes tracked
```

**Issue**: Tape tracks 127 forward passes - large memory footprint

### Parallel Scheduled Sampling
```python
with tf.GradientTape() as tape:
    preds = self.base_model(x, training=True)      # Pass 1
    mixed = tf.where(mask, x, preds)
    preds2 = self.base_model(inputs, training=True) # Pass 2
    loss = tf.reduce_mean(tf.abs(preds2 - targets))
```

**Benefit**: Tape tracks only 2 forward passes - much smaller memory footprint

### Persistent Tape?

Not needed - we only compute gradients once per step.

## Random Operations in Graph Mode

### tf.random.uniform Behavior

```python
mask = tf.random.uniform(tf.shape(x)) < tf_ratio
```

**Graph mode**: Creates a random op in the graph that generates new values each execution

**Seed handling**: Can set seed for reproducibility:
```python
tf.random.set_seed(42)
```

**Performance**: Random ops are fast and graph-compatible

## Metrics and Callbacks

### Current Metrics
```python
self.tf_ratio_metric = tf.keras.metrics.Mean(name="tf_ratio")
```

**Graph compatibility**: ✅ Keras metrics are graph-compatible

### TensorBoard Logging

Replace `tf.print` with TensorBoard:
```python
tf.summary.scalar('loss', loss, step=self.training_step)
tf.summary.scalar('tf_ratio', self.tf_ratio, step=self.training_step)
```

**Graph compatibility**: ✅ Summary ops work in graph mode

## Migration Path

### Phase 1: Implement Parallel Scheduled Sampling (Eager Mode)
```python
model.compile(optimizer=optimizer, run_eagerly=True)
```
- Keep eager mode for debugging
- Verify correctness
- Measure speedup (50-100×)

### Phase 2: Enable Graph Mode
```python
model.compile(optimizer=optimizer, run_eagerly=False)
```
- Remove tf.print statements
- Keep tf.debugging.assert_equal (they're no-ops in graph mode by default)
- Measure additional speedup (2-5×)

### Phase 3: Enable XLA (Optional)
```python
@tf.function(jit_compile=True)
def train_step(self, data):
    # ...
```
- Test compatibility
- Measure additional speedup (1.5-3×)

## Potential Issues and Solutions

### Issue 1: Shape Inference Failures

**Symptom**: "Cannot infer shape" errors in graph mode

**Solution**: Add explicit shape hints:
```python
x.set_shape([None, SEQ_LEN, N_MELS])
```

### Issue 2: Variable Creation in train_step

**Symptom**: "Variables cannot be created in tf.function" error

**Solution**: Create all variables in `__init__`, not in `train_step`

**Current code**: ✅ Already correct (variables created in `__init__`)

### Issue 3: Python State Access

**Symptom**: Retracing on every call

**Solution**: Use `tf.Variable` for mutable state (already done):
```python
self.tf_ratio = tf.Variable(initial_tf_ratio, trainable=False)
```

## Conclusion

Parallel scheduled sampling is **fully graph-compatible** and enables:

1. ✅ Removal of `run_eagerly=True` (2-5× speedup)
2. ✅ Removal of `tf.cond` branching (simpler code)
3. ✅ Smaller gradient tape (less memory)
4. ✅ XLA compilation (optional 1.5-3× speedup)

**Total potential speedup**: 100-500× compared to current eager autoregressive implementation.

**Recommendation**: Implement in eager mode first for debugging, then enable graph mode for production training.
