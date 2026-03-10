# Refactor Training Pipeline to Parallel Autoregressive Training

## Objective

Refactor `src/music_generation/train.py` to eliminate the O(T) autoregressive loop and replace it with parallel training using two-pass scheduled sampling. Achieve ~100x speedup while preserving exact autoregressive semantics and checkpoint compatibility.

## Context

- **Spec directory:** `specs/parallel-autoregressive-training/`
- **Design:** See `design.md` for complete architecture and requirements
- **Plan:** See `plan.md` for 7-step implementation guide
- **Research:** See `research/` for technical background

## Key Requirements

### Functional

1. **Replace O(T) loop** in `autoregressive_training()` with 2-pass parallel scheduled sampling
2. **Preserve pure teacher forcing** path (tf_ratio ≥ 0.99) - already optimal, just extract to method
3. **Maintain exact behavior:** Same loss computation, same teacher forcing schedule, same metrics
4. **Checkpoint compatibility:** No model architecture changes, existing checkpoints must work

### Performance Targets

- Forward passes per batch: 255 → 2 (scheduled sampling path)
- Training speed: 50-100x faster
- Memory usage: ~200x reduction
- Support batch_size=64 on 16GB RAM (vs 4-8 currently)

## Implementation Summary

### Changes to `src/music_generation/train.py`

**1. Extract pure teacher forcing method:**
```python
def _pure_teacher_forcing(self, x, y):
    """Single forward pass with ground truth."""
    with tf.GradientTape() as tape:
        preds = self.base_model(x, training=True)
        loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
    
    grads = tape.gradient(loss, self.base_model.trainable_variables)
    grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in grads if g is not None]))
    self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
    self.tf_ratio_metric.update_state(self.tf_ratio)
    
    return {"loss": loss, "grad_norm": grad_norm, "tf_ratio": self.tf_ratio_metric.result()}
```

**2. Add parallel scheduled sampling method:**
```python
def _parallel_scheduled_sampling(self, x, y):
    """Two-pass parallel scheduled sampling."""
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    
    # Pass 1: Get predictions (no gradients)
    preds_pass1 = self.base_model(x, training=False)
    
    # Sample and mix
    use_teacher = tf.random.uniform([batch_size, seq_len, 1]) < self.tf_ratio
    preds_shifted = tf.concat([x[:, :1, :], preds_pass1[:, :-1, :]], axis=1)
    mixed_input = tf.where(use_teacher, x, preds_shifted)
    
    # Pass 2: Train with mixed input
    with tf.GradientTape() as tape:
        preds_pass2 = self.base_model(mixed_input, training=True)
        loss = tf.reduce_mean(tf.abs(preds_pass2[:, :-1, :] - y[:, 1:, :]))
    
    grads = tape.gradient(loss, self.base_model.trainable_variables)
    grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in grads if g is not None]))
    self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
    self.tf_ratio_metric.update_state(self.tf_ratio)
    
    return {"loss": loss, "grad_norm": grad_norm, "tf_ratio": self.tf_ratio_metric.result()}
```

**3. Simplify train_step:**
```python
def train_step(self, data):
    """Training step with parallel processing."""
    self.update_tf_ratio()
    
    x, y = data
    tf.debugging.assert_equal(tf.shape(x), tf.shape(y))
    
    # Choose training path
    if self.tf_ratio >= 0.99:
        return self._pure_teacher_forcing(x, y)
    else:
        return self._parallel_scheduled_sampling(x, y)
```

**4. Remove old code:**
- Delete nested `pure_teacher_forcing()` function (~50 lines)
- Delete nested `autoregressive_training()` function (~50 lines)
- Remove `tf.cond` dispatch logic

## Acceptance Criteria

### Must Pass

**AC1: Pure teacher forcing (tf_ratio=1.0)**
- Exactly 1 forward pass per batch
- Loss computed as `mean(|preds[:, :-1, :] - targets[:, 1:, :]|)`
- Training completes successfully

**AC2: Parallel scheduled sampling (tf_ratio=0.5)**
- Exactly 2 forward passes per batch
- Sampling mask has ~50% True values (±5%)
- Loss is numerically similar to baseline

**AC3: Performance**
- Training 100 steps completes in < 20 seconds (vs ~1000 seconds baseline)
- Speedup >= 50x

**AC4: Compatibility**
- Existing checkpoints load without errors
- tf_ratio and training_step variables preserved
- Model architecture unchanged

**AC5: End-to-end training**
- Full training loop works with `model.fit()`
- Loss decreases over epochs
- Callbacks work correctly

## Testing Requirements

### Unit Tests (create `tests/test_parallel_training.py`)

1. **test_pure_teacher_forcing_shapes** - Verify output dict structure
2. **test_parallel_sampling_mask_distribution** - Verify sampling statistics
3. **test_train_step_dispatch** - Verify correct method selection based on tf_ratio

### Integration Tests

4. **test_end_to_end_training** - Full training loop with dataset
5. **test_checkpoint_save_load** - Checkpoint compatibility

### Benchmark

6. **scripts/benchmark_training.py** - Measure actual speedup

Run all tests with: `pytest tests/test_parallel_training.py -v`

## Validation Steps

1. **Code review:** Verify implementation matches design
2. **Unit tests:** All tests pass
3. **Benchmark:** Measure speedup (target: 50-100x)
4. **Integration:** Run 10 epochs on small dataset, verify loss decreases
5. **Checkpoint test:** Save/load checkpoint, verify compatibility

## Notes

- **No model changes:** `model.py`, `layers.py`, `config.py` remain unchanged
- **Keep existing logic:** `update_tf_ratio()`, `exponential_tf_schedule()`, optimizer setup unchanged
- **Preserve behavior:** Same loss function, same metrics, same schedule
- **Simplify code:** Net reduction in lines (~100 deleted, ~55 added)

## Success Criteria

- ✅ All tests pass
- ✅ 50x+ speedup achieved
- ✅ Checkpoints compatible
- ✅ Code is simpler and more maintainable
- ✅ Training quality maintained or improved

## Reference

- Full design: `specs/parallel-autoregressive-training/design.md`
- Implementation plan: `specs/parallel-autoregressive-training/plan.md`
- Research: `specs/parallel-autoregressive-training/research/`
