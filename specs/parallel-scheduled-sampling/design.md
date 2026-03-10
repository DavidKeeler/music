# Design: Parallel Scheduled Sampling for Mel Generator Training

## Overview

Optimize mel generator training by replacing the slow autoregressive loop (127 forward passes) with parallel scheduled sampling (2 forward passes), achieving 50-100× speedup while maintaining scheduled sampling behavior.

## Detailed Requirements

### Functional Requirements

1. **Maintain scheduled sampling behavior**: Model must see its own prediction errors during training
2. **Preserve teacher forcing schedule**: Exponential decay from 1.0 to 0.05 with warmup
3. **Match loss computation**: Continue using MAE on next-frame prediction
4. **Support all tf_ratio values**: From 1.0 (pure teacher forcing) to 0.05 (minimal)
5. **Maintain gradient flow**: Ensure proper backpropagation through both forward passes
6. **Preserve metrics**: Continue tracking loss, grad_norm, and tf_ratio

### Performance Requirements

1. **Primary goal**: Reduce forward passes from 127 to 2 (50-100× speedup)
2. **Secondary goal**: Enable graph mode for additional 2-5× speedup
3. **Memory efficiency**: Reduce gradient tape size (no 127-step accumulation)
4. **Correctness**: Loss values within 10% of baseline implementation

### Compatibility Requirements

1. **TensorFlow version**: 2.13+ (current requirement)
2. **Keras API**: Use standard Keras Model.train_step override
3. **Existing config**: Preserve all hyperparameters (SEQ_LEN, BATCH_SIZE, etc.)
4. **Checkpoint format**: Maintain compatibility with existing checkpoints
5. **Dataset format**: No changes to data pipeline

## Architecture Overview

### High-Level Flow

```
Input: (x, y) batch from dataset
  ↓
Update teacher forcing ratio
  ↓
Forward Pass 1: Predict from ground truth
  ↓
Create scheduled sampling mask
  ↓
Mix ground truth + predictions
  ↓
Shift for next-frame prediction
  ↓
Forward Pass 2: Predict from mixed input
  ↓
Compute loss (MAE)
  ↓
Backpropagate and update weights
  ↓
Return metrics
```

### Algorithm Diagram

```mermaid
graph TD
    A[Input: x, y] --> B[Forward Pass 1: preds = model(x)]
    B --> C[Create mask: random < tf_ratio]
    C --> D[Mix: where(mask, x, preds)]
    D --> E[Shift: inputs = mixed[:, :-1], targets = y[:, 1:]]
    E --> F[Forward Pass 2: preds2 = model(inputs)]
    F --> G[Loss: MAE(preds2, targets)]
    G --> H[Compute gradients]
    H --> I[Update weights]
    I --> J[Return metrics]
```

## Components and Interfaces

### Component 1: MelGeneratorTraining.train_step()

**Current signature**:
```python
def train_step(self, data) -> dict
```

**Inputs**:
- `data`: Tuple of (x, y) tensors
  - `x`: Input mel spectrograms `[batch, seq_len, 80]`
  - `y`: Target mel spectrograms `[batch, seq_len, 80]` (same as x for autoregressive)

**Outputs**:
- `dict`: Metrics dictionary
  - `"loss"`: Training loss (float)
  - `"grad_norm"`: Gradient norm (float)
  - `"tf_ratio"`: Current teacher forcing ratio (float)

**Changes**:
- Remove `tf.cond` branching (pure_teacher_forcing vs autoregressive_training)
- Replace autoregressive loop with parallel scheduled sampling
- Simplify to single unified path

### Component 2: Scheduled Sampling Mask

**New helper method**:
```python
def create_scheduled_sampling_mask(self, shape, tf_ratio) -> tf.Tensor
```

**Purpose**: Create boolean mask for mixing ground truth and predictions

**Inputs**:
- `shape`: Tensor shape `[batch, seq_len, mel_dim]`
- `tf_ratio`: Teacher forcing ratio (float)

**Output**:
- Boolean tensor `[batch, seq_len, mel_dim]`
  - `True`: Use ground truth
  - `False`: Use prediction

**Implementation**:
```python
# Per-frame granularity (not per-element)
mask = tf.random.uniform([shape[0], shape[1], 1]) < tf_ratio
mask = tf.broadcast_to(mask, shape)
return mask
```

**Rationale**: Per-frame granularity is more sensible than per-element (use entire frame or don't).

### Component 3: Teacher Forcing Ratio Optimization

**Edge case optimization**: When `tf_ratio >= 1.0 - 1e-6`, skip first forward pass.

**Rationale**: 
- At tf_ratio=1.0, mask is all True → mixed = x
- First forward pass is wasted computation
- Optimization saves ~50% time during early training

**Implementation**:
```python
if self.tf_ratio >= 1.0 - 1e-6:
    # Pure teacher forcing: skip first pass
    inputs = x[:, :-1]
    targets = y[:, 1:]
    preds = self.base_model(inputs, training=True)
    loss = tf.reduce_mean(tf.abs(preds - targets))
else:
    # Parallel scheduled sampling: 2 passes
    # ... (full algorithm)
```

## Data Models

### Input Data
```python
x: tf.Tensor  # [batch_size, seq_len, n_mels]
y: tf.Tensor  # [batch_size, seq_len, n_mels]
```

**Constraints**:
- `batch_size`: Dynamic (typically 16)
- `seq_len`: Static (128)
- `n_mels`: Static (80)
- `x` and `y` have identical shapes (autoregressive training)

### Intermediate Tensors

```python
# Forward pass 1
preds: tf.Tensor  # [batch_size, seq_len, n_mels]

# Scheduled sampling
mask: tf.Tensor   # [batch_size, seq_len, n_mels], dtype=bool
mixed: tf.Tensor  # [batch_size, seq_len, n_mels]

# Shifted for next-frame prediction
inputs: tf.Tensor   # [batch_size, seq_len-1, n_mels]
targets: tf.Tensor  # [batch_size, seq_len-1, n_mels]

# Forward pass 2
preds2: tf.Tensor  # [batch_size, seq_len-1, n_mels]

# Loss
loss: tf.Tensor    # scalar
```

### Metrics Output
```python
{
    "loss": tf.Tensor,      # scalar, float32
    "grad_norm": tf.Tensor, # scalar, float32
    "tf_ratio": tf.Tensor   # scalar, float32
}
```

## Error Handling

### Shape Validation

**Assertions to keep**:
```python
tf.debugging.assert_equal(tf.shape(x), tf.shape(y), 
                         message="Input and target shapes must match")
```

**Assertions to add**:
```python
tf.debugging.assert_equal(tf.shape(preds), tf.shape(x),
                         message="Predictions must match input shape")
tf.debugging.assert_equal(tf.shape(preds2)[1], tf.shape(x)[1] - 1,
                         message="Shifted predictions must be seq_len-1")
```

### NaN/Inf Detection

**Current approach**: `tf.print` statements

**Recommendation for production**:
- Remove `tf.print` (breaks graph mode)
- Use TensorBoard scalar logging
- Add `tf.debugging.check_numerics` in debug mode

```python
if DEBUG:
    loss = tf.debugging.check_numerics(loss, "Loss contains NaN/Inf")
```

### Gradient Issues

**Current handling**: Compute gradient norm, log to metrics

**Keep**: This is good practice for monitoring training stability

**Add**: Optional gradient clipping (already in optimizer with `clipnorm=0.5`)

## Acceptance Criteria

### Given: Model with parallel scheduled sampling
### When: Training for 100 steps with tf_ratio=0.5
### Then: 
- Training completes without errors
- Steps per second is 50-100× faster than baseline
- Loss values are within 10% of baseline
- Gradients are non-zero and finite
- Metrics are logged correctly

---

### Given: Model with tf_ratio=1.0 (pure teacher forcing)
### When: Training for 100 steps
### Then:
- Only 1 forward pass per step (optimization applied)
- Training speed matches or exceeds baseline
- Loss matches baseline exactly

---

### Given: Model with tf_ratio=0.05 (minimal teacher forcing)
### When: Training for 100 steps
### Then:
- 2 forward passes per step
- ~95% of frames use model predictions
- Training remains stable (no divergence)

---

### Given: Parallel scheduled sampling implementation
### When: Enabling graph mode (run_eagerly=False)
### Then:
- Training completes without errors
- Additional 2-5× speedup over eager mode
- Loss convergence is identical

---

### Given: Both baseline and optimized models
### When: Training for 10 epochs with same initialization
### Then:
- Final loss values differ by <10%
- Convergence curves have similar shape
- Generated audio quality is comparable

## Testing Strategy

### Unit Tests

**Test 1: Shape consistency**
```python
def test_train_step_shapes():
    model = MelGeneratorTraining(base_model)
    x = tf.random.normal([16, 128, 80])
    y = tf.random.normal([16, 128, 80])
    metrics = model.train_step((x, y))
    assert 'loss' in metrics
    assert metrics['loss'].shape == []
```

**Test 2: Mask distribution**
```python
def test_mask_distribution():
    model = MelGeneratorTraining(base_model)
    model.tf_ratio.assign(0.5)
    shape = [16, 128, 80]
    mask = model.create_scheduled_sampling_mask(shape, model.tf_ratio)
    ratio = tf.reduce_mean(tf.cast(mask, tf.float32))
    assert abs(ratio - 0.5) < 0.05  # Within 5%
```

**Test 3: Pure teacher forcing optimization**
```python
def test_pure_teacher_forcing():
    model = MelGeneratorTraining(base_model)
    model.tf_ratio.assign(1.0)
    # Should use optimized path (1 forward pass)
    # Verify by checking execution time or profiling
```

### Integration Tests

**Test 4: Training convergence**
```python
def test_training_convergence():
    model = MelGeneratorTraining(base_model)
    dataset = create_dataset(...)
    history = model.fit(dataset, epochs=5, steps_per_epoch=100)
    assert history.history['loss'][-1] < history.history['loss'][0]
```

**Test 5: Checkpoint compatibility**
```python
def test_checkpoint_compatibility():
    model = MelGeneratorTraining(base_model)
    model.fit(dataset, epochs=1)
    model.save_weights('checkpoint.h5')
    
    new_model = MelGeneratorTraining(base_model)
    new_model.load_weights('checkpoint.h5')
    # Verify weights loaded correctly
```

### Performance Tests

**Test 6: Speedup measurement**
```python
def test_speedup():
    baseline_time = benchmark_train_step(baseline_model, dataset)
    optimized_time = benchmark_train_step(optimized_model, dataset)
    speedup = baseline_time / optimized_time
    assert speedup > 40  # At least 40× speedup
```

**Test 7: Memory usage**
```python
def test_memory_usage():
    baseline_memory = measure_peak_memory(baseline_model, dataset)
    optimized_memory = measure_peak_memory(optimized_model, dataset)
    assert optimized_memory < baseline_memory
```

### Validation Tests

**Test 8: Loss equivalence**
```python
def test_loss_equivalence():
    tf.random.set_seed(42)
    baseline_loss = train_n_steps(baseline_model, dataset, 10)
    
    tf.random.set_seed(42)
    optimized_loss = train_n_steps(optimized_model, dataset, 10)
    
    relative_diff = abs(baseline_loss - optimized_loss) / baseline_loss
    assert relative_diff < 0.1  # Within 10%
```

## Appendices

### Appendix A: Technology Choices

**TensorFlow/Keras**: Already in use, no changes needed

**Graph mode**: Optional optimization after validating eager mode

**XLA compilation**: Optional further optimization (not in initial implementation)

### Appendix B: Research Findings Summary

From research documents in `research/`:

1. **Current bottleneck**: 127 forward passes in autoregressive loop
2. **Parallel algorithm**: Reduces to 2 forward passes while maintaining scheduled sampling
3. **Graph compatibility**: Fully compatible, enables additional speedups
4. **Benchmarking strategy**: Comprehensive validation and measurement approach

### Appendix C: Alternative Approaches Considered

**Alternative 1: Causal masking with single forward pass**

Mentioned in original context as "one more improvement":
> "For transformers or conv models you should instead use a causal mask and train on the whole sequence at once."

**Analysis**: 
- Would reduce to 1 forward pass
- Requires architectural changes to model
- More complex to implement
- Deferred to future optimization

**Decision**: Implement 2-pass parallel scheduled sampling first (simpler, proven approach)

---

**Alternative 2: Keep autoregressive loop, optimize with tf.while_loop**

**Analysis**:
- Still O(seq_len) complexity
- Marginal speedup (maybe 2-3×)
- Doesn't address fundamental bottleneck

**Decision**: Rejected in favor of parallel approach

---

**Alternative 3: Hybrid approach (autoregressive for low tf_ratio, parallel for high)**

**Analysis**:
- Adds complexity
- Parallel approach works well for all tf_ratio values
- No clear benefit

**Decision**: Rejected in favor of unified parallel approach

### Appendix D: Migration Strategy

**Phase 1: Implement in eager mode**
- Keep `run_eagerly=True`
- Validate correctness
- Measure speedup
- Duration: 1-2 days

**Phase 2: Enable graph mode**
- Set `run_eagerly=False`
- Remove `tf.print` statements
- Validate correctness
- Measure additional speedup
- Duration: 1 day

**Phase 3: Optional XLA**
- Add `@tf.function(jit_compile=True)`
- Test compatibility
- Measure additional speedup
- Duration: 1 day

**Phase 4: Production deployment**
- Update documentation
- Update training scripts
- Retrain models
- Duration: 1-2 days

### Appendix E: Backward Compatibility

**Checkpoint format**: No changes to model architecture → checkpoints remain compatible

**Config parameters**: All existing hyperparameters preserved

**Dataset format**: No changes required

**Inference**: No changes (inference already uses autoregressive generation)

**Migration path**: Drop-in replacement for existing training code
