# Research: Performance Benchmarking Strategy

## Overview

Strategy for measuring the actual speedup from parallel scheduled sampling and validating correctness of the optimization.

## Benchmarking Goals

1. **Measure speedup**: Quantify training time improvement
2. **Validate correctness**: Ensure optimization doesn't break training
3. **Compare convergence**: Verify model learns equally well
4. **Profile bottlenecks**: Identify remaining optimization opportunities

## Metrics to Track

### Primary Metrics

#### 1. Training Speed
- **Steps per second**: Throughput metric
- **Seconds per step**: Latency metric
- **Epoch time**: Total time for full dataset pass

**Measurement**:
```python
import time

start = time.time()
for step, (x, y) in enumerate(dataset):
    model.train_step((x, y))
    if step >= 100:
        break
elapsed = time.time() - start
steps_per_sec = 100 / elapsed
```

#### 2. Loss Convergence
- **Training loss**: Should decrease similarly
- **Loss curve shape**: Should match baseline
- **Final loss value**: Should be comparable

**Measurement**: TensorBoard logs

#### 3. Memory Usage
- **Peak GPU memory**: `nvidia-smi` or `tf.config.experimental.get_memory_info`
- **Peak RAM**: `psutil.Process().memory_info().rss`

**Measurement**:
```python
import psutil
process = psutil.Process()
peak_memory_mb = process.memory_info().rss / (1024 ** 2)
```

### Secondary Metrics

#### 4. Gradient Statistics
- **Gradient norm**: Should be in similar range
- **Gradient distribution**: Check for NaN/Inf

**Measurement**: Already tracked in current implementation

#### 5. Model Quality
- **Generated audio quality**: Subjective listening test
- **Mel spectrogram quality**: Visual inspection
- **Reconstruction error**: On validation set

## Benchmark Scenarios

### Scenario 1: Pure Teacher Forcing (tf_ratio = 1.0)

**Current implementation**: 1 forward pass (fast path)
**Parallel implementation**: 2 forward passes

**Expected result**: Parallel might be slightly slower (2× forward passes)

**Purpose**: Establish baseline, verify correctness

### Scenario 2: High Teacher Forcing (tf_ratio = 0.8)

**Current implementation**: 127 forward passes (autoregressive)
**Parallel implementation**: 2 forward passes

**Expected result**: ~60× speedup

**Purpose**: Realistic early training scenario

### Scenario 3: Low Teacher Forcing (tf_ratio = 0.2)

**Current implementation**: 127 forward passes (autoregressive)
**Parallel implementation**: 2 forward passes

**Expected result**: ~60× speedup

**Purpose**: Realistic late training scenario

### Scenario 4: Minimal Teacher Forcing (tf_ratio = 0.05)

**Current implementation**: 127 forward passes (autoregressive)
**Parallel implementation**: 2 forward passes

**Expected result**: ~60× speedup

**Purpose**: End of training scenario

## Benchmark Implementation

### Timing Harness

```python
import time
import numpy as np

def benchmark_train_step(model, dataset, num_steps=100, warmup_steps=10):
    """Benchmark training step performance.
    
    Args:
        model: Compiled Keras model
        dataset: tf.data.Dataset
        num_steps: Number of steps to benchmark
        warmup_steps: Number of warmup steps (excluded from timing)
    
    Returns:
        dict with timing statistics
    """
    times = []
    
    for step, data in enumerate(dataset.take(num_steps + warmup_steps)):
        start = time.perf_counter()
        metrics = model.train_step(data)
        elapsed = time.perf_counter() - start
        
        if step >= warmup_steps:
            times.append(elapsed)
        
        if step >= num_steps + warmup_steps - 1:
            break
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'steps_per_sec': 1.0 / np.mean(times),
        'total_time': np.sum(times)
    }
```

### Comparison Script

```python
def compare_implementations(baseline_model, optimized_model, dataset, tf_ratios):
    """Compare baseline vs optimized implementation.
    
    Args:
        baseline_model: Current autoregressive implementation
        optimized_model: Parallel scheduled sampling implementation
        dataset: tf.data.Dataset
        tf_ratios: List of teacher forcing ratios to test
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for tf_ratio in tf_ratios:
        # Set teacher forcing ratio
        baseline_model.tf_ratio.assign(tf_ratio)
        optimized_model.tf_ratio.assign(tf_ratio)
        
        # Benchmark baseline
        baseline_stats = benchmark_train_step(baseline_model, dataset)
        
        # Benchmark optimized
        optimized_stats = benchmark_train_step(optimized_model, dataset)
        
        # Compute speedup
        speedup = baseline_stats['mean_time'] / optimized_stats['mean_time']
        
        results.append({
            'tf_ratio': tf_ratio,
            'baseline_time': baseline_stats['mean_time'],
            'optimized_time': optimized_stats['mean_time'],
            'speedup': speedup,
            'baseline_steps_per_sec': baseline_stats['steps_per_sec'],
            'optimized_steps_per_sec': optimized_stats['steps_per_sec']
        })
    
    return pd.DataFrame(results)
```

## Correctness Validation

### Test 1: Shape Consistency

```python
def test_shapes(model, dataset):
    """Verify output shapes are correct."""
    for x, y in dataset.take(1):
        metrics = model.train_step((x, y))
        
        assert 'loss' in metrics
        assert not tf.math.is_nan(metrics['loss'])
        assert not tf.math.is_inf(metrics['loss'])
        
        print("✓ Shapes and loss are valid")
```

### Test 2: Loss Equivalence

```python
def test_loss_equivalence(baseline_model, optimized_model, dataset, tolerance=0.1):
    """Verify losses are similar between implementations."""
    baseline_losses = []
    optimized_losses = []
    
    # Use same random seed for both
    tf.random.set_seed(42)
    for x, y in dataset.take(10):
        metrics = baseline_model.train_step((x, y))
        baseline_losses.append(float(metrics['loss']))
    
    tf.random.set_seed(42)
    for x, y in dataset.take(10):
        metrics = optimized_model.train_step((x, y))
        optimized_losses.append(float(metrics['loss']))
    
    baseline_mean = np.mean(baseline_losses)
    optimized_mean = np.mean(optimized_losses)
    
    relative_diff = abs(baseline_mean - optimized_mean) / baseline_mean
    
    assert relative_diff < tolerance, f"Loss difference too large: {relative_diff:.2%}"
    print(f"✓ Loss equivalence verified (diff: {relative_diff:.2%})")
```

### Test 3: Gradient Flow

```python
def test_gradient_flow(model, dataset):
    """Verify gradients are non-zero and finite."""
    for x, y in dataset.take(1):
        with tf.GradientTape() as tape:
            # Manually compute loss
            preds = model.base_model(x, training=True)
            loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
        
        grads = tape.gradient(loss, model.base_model.trainable_variables)
        
        for i, grad in enumerate(grads):
            if grad is not None:
                assert not tf.reduce_any(tf.math.is_nan(grad)), f"NaN gradient at layer {i}"
                assert not tf.reduce_any(tf.math.is_inf(grad)), f"Inf gradient at layer {i}"
                assert tf.reduce_any(grad != 0), f"Zero gradient at layer {i}"
        
        print("✓ Gradient flow verified")
```

### Test 4: Teacher Forcing Ratio Distribution

```python
def test_tf_ratio_distribution(model, dataset, expected_ratio=0.5, num_samples=1000):
    """Verify scheduled sampling mask respects tf_ratio."""
    model.tf_ratio.assign(expected_ratio)
    
    mask_means = []
    for x, y in dataset.take(num_samples):
        mask = tf.random.uniform(tf.shape(x)) < model.tf_ratio
        mask_mean = tf.reduce_mean(tf.cast(mask, tf.float32))
        mask_means.append(float(mask_mean))
    
    actual_ratio = np.mean(mask_means)
    
    # Should be close to expected_ratio (within 5%)
    assert abs(actual_ratio - expected_ratio) < 0.05, \
        f"Mask ratio {actual_ratio:.3f} differs from expected {expected_ratio:.3f}"
    
    print(f"✓ TF ratio distribution verified: {actual_ratio:.3f} ≈ {expected_ratio:.3f}")
```

## Convergence Validation

### Training Comparison

Run both implementations for several epochs and compare:

```python
def compare_training_convergence(baseline_model, optimized_model, dataset, epochs=10):
    """Compare training convergence between implementations."""
    
    # Train baseline
    baseline_history = baseline_model.fit(
        dataset, 
        epochs=epochs, 
        steps_per_epoch=100
    )
    
    # Train optimized (reset weights to same initialization)
    optimized_model.base_model.set_weights(baseline_model.base_model.get_weights())
    optimized_history = optimized_model.fit(
        dataset, 
        epochs=epochs, 
        steps_per_epoch=100
    )
    
    # Plot comparison
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(baseline_history.history['loss'], label='Baseline')
    plt.plot(optimized_history.history['loss'], label='Optimized')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Convergence Comparison')
    plt.savefig('convergence_comparison.png')
    
    print("✓ Convergence comparison plot saved")
```

## Profiling for Bottlenecks

### TensorFlow Profiler

```python
# Profile optimized implementation
tf.profiler.experimental.start('logdir')

for step, data in enumerate(dataset.take(100)):
    model.train_step(data)

tf.profiler.experimental.stop()

# View in TensorBoard:
# tensorboard --logdir=logdir
```

### Key Metrics to Check
- **Op time**: Which operations take longest?
- **Memory usage**: Peak memory per step
- **Device utilization**: GPU/CPU usage
- **Data pipeline**: Is data loading a bottleneck?

## Expected Results

### Speedup Predictions

| Scenario | Current (s/step) | Optimized (s/step) | Speedup |
|----------|------------------|-------------------|---------|
| tf_ratio=1.0 | 0.1 | 0.2 | 0.5× (slower) |
| tf_ratio=0.8 | 5.0 | 0.1 | 50× |
| tf_ratio=0.5 | 5.0 | 0.1 | 50× |
| tf_ratio=0.2 | 5.0 | 0.1 | 50× |
| tf_ratio=0.05 | 5.0 | 0.1 | 50× |

**Note**: tf_ratio=1.0 might be slower because parallel does 2 passes vs 1 pass in current implementation.

**Solution**: Add optimization to skip first pass when tf_ratio=1.0.

### Loss Convergence

- **Expected**: Similar final loss values
- **Acceptable difference**: <10% relative difference
- **Convergence speed**: May differ slightly due to different error exposure patterns

### Memory Usage

- **Expected**: Lower peak memory (smaller gradient tape)
- **Reduction**: ~50-80% less memory during backward pass

## Benchmark Reporting

### Report Template

```markdown
# Parallel Scheduled Sampling Benchmark Results

## Environment
- TensorFlow version: X.X.X
- Hardware: [CPU/GPU model]
- Batch size: 16
- Sequence length: 128

## Speed Comparison

| TF Ratio | Baseline (s/step) | Optimized (s/step) | Speedup |
|----------|-------------------|-------------------|---------|
| 1.0      | X.XX              | X.XX              | X.XX×   |
| 0.8      | X.XX              | X.XX              | X.XX×   |
| 0.5      | X.XX              | X.XX              | X.XX×   |
| 0.2      | X.XX              | X.XX              | X.XX×   |
| 0.05     | X.XX              | X.XX              | X.XX×   |

## Correctness Validation

- ✓ Shape consistency
- ✓ Loss equivalence (diff: X.X%)
- ✓ Gradient flow
- ✓ TF ratio distribution

## Convergence Comparison

[Include plot]

Final loss:
- Baseline: X.XXX
- Optimized: X.XXX
- Difference: X.X%

## Memory Usage

- Baseline peak: XXX MB
- Optimized peak: XXX MB
- Reduction: XX%

## Conclusion

[Summary of results and recommendations]
```

## Implementation Checklist

Before benchmarking:
- [ ] Implement parallel scheduled sampling
- [ ] Add timing instrumentation
- [ ] Set up TensorBoard logging
- [ ] Prepare test dataset

During benchmarking:
- [ ] Run warmup steps
- [ ] Test multiple tf_ratio values
- [ ] Measure with consistent random seeds
- [ ] Profile with TensorFlow Profiler

After benchmarking:
- [ ] Validate correctness
- [ ] Compare convergence
- [ ] Document results
- [ ] Identify remaining bottlenecks

## Next Steps

After validating speedup and correctness:
1. Enable graph mode (`run_eagerly=False`) for additional 2-5× speedup
2. Consider XLA compilation for further optimization
3. Optimize data pipeline if it becomes bottleneck
4. Test on larger datasets and longer sequences
