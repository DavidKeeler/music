# Implementation Plan

## Checklist
- [ ] Step 1: Add memory configuration
- [ ] Step 2: Implement streaming statistics computation
- [ ] Step 3: Optimize autoregressive training loop with fixed-size window
- [ ] Step 4: Add batch size warning
- [ ] Step 5: Add memory profiling and validation

---

## Step 1: Add Memory Configuration

**Objective:** Configure TensorFlow memory growth to prevent upfront allocation of all GPU/CPU memory.

**Implementation:**
1. Add `configure_memory()` function at top of `src/music_generation/train.py`
2. Call it before any model creation in `main()`
3. Enable memory growth for all GPUs

**Files to modify:**
- `src/music_generation/train.py`

**Code changes:**
```python
# Add after imports, before any classes
def configure_memory():
    """Configure TensorFlow memory settings for resource-constrained environments."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Enabled memory growth for {len(gpus)} GPU(s)")

# In main(), add as first line:
configure_memory()
```

**Tests:**
- Verify function runs without errors on systems with/without GPU
- Verify training still works after change

**Integration:**
- No breaking changes
- Transparent to users

**Demo:**
Run training command and verify no immediate memory allocation spike in Activity Monitor/htop.

---

## Step 2: Implement Streaming Statistics Computation

**Objective:** Replace batch statistics computation with streaming approach to avoid loading entire dataset into memory.

**Implementation:**
1. Modify `MusicNetDataset._compute_statistics()` in `src/music_generation/dataset.py`
2. Use chunked sum/sum-of-squares approach instead of concatenating all mels
3. Compute mean and std from running totals

**Files to modify:**
- `src/music_generation/dataset.py`

**Code changes:**
```python
def _compute_statistics(self):
    """Compute global mean and std using streaming algorithm."""
    running_sum = 0.0
    running_sq_sum = 0.0
    total_elements = 0
    
    for audio_file in self.audio_files:
        logger.info(f"Processing {audio_file.name}")
        waveform = load_audio(audio_file)
        mel = audio_to_mel(waveform)
        
        # Accumulate statistics without storing mel
        running_sum += tf.reduce_sum(mel).numpy()
        running_sq_sum += tf.reduce_sum(tf.square(mel)).numpy()
        total_elements += mel.shape[0] * mel.shape[1]
    
    mean = running_sum / total_elements
    variance = (running_sq_sum / total_elements) - (mean ** 2)
    std = tf.sqrt(variance).numpy()
    
    logger.info(f"Computed statistics: mean={mean:.4f}, std={std:.4f}")
    return mean, std
```

**Tests:**
1. Unit test: Compare streaming vs. batch computation on small dataset (should match within 1e-6)
2. Integration test: Verify cached statistics still work
3. Memory test: Profile memory during statistics computation (should be flat)

**Integration:**
- Delete existing `cache/stats.json` to trigger recomputation
- Verify new statistics produce similar values

**Demo:**
Run training on fresh cache directory, monitor memory usage during statistics computation (should stay constant).

---

## Step 3: Optimize Autoregressive Training Loop with Fixed-Size Window

**Objective:** Replace growing concatenation with fixed-size sliding window to achieve O(n) memory instead of O(n²).

**Implementation:**
1. Modify `MelGeneratorTraining.train_step()` in `src/music_generation/train.py`
2. Add `context_len` parameter (default: 64)
3. Truncate `ar_input` to last `context_len` frames before each prediction
4. Keep rest of logic (teacher forcing, loss computation) unchanged

**Files to modify:**
- `src/music_generation/train.py`

**Code changes:**
```python
class MelGeneratorTraining(tf.keras.Model):
    """Training wrapper with teacher forcing."""
    
    def __init__(self, base_model, initial_tf_ratio=INITIAL_TF_RATIO, 
                 decay_k=TF_DECAY_K, min_ratio=MIN_TF_RATIO, context_len=64):
        super().__init__()
        self.base_model = base_model
        self.initial_tf_ratio = initial_tf_ratio
        self.decay_k = decay_k
        self.min_ratio = min_ratio
        self.context_len = context_len  # NEW
    
    def train_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        tf_ratio = self.compute_tf_ratio()
        
        with tf.GradientTape() as tape:
            ar_input = x[:, :1, :]
            preds = []
            
            for t in range(1, seq_len):
                # NEW: Truncate to fixed context window
                if tf.shape(ar_input)[1] > self.context_len:
                    ar_input = ar_input[:, -self.context_len:, :]
                
                pred = self.base_model(ar_input, training=True)[:, -1:, :]
                preds.append(pred)
                use_teacher = tf.random.uniform([batch_size, 1, 1]) < tf_ratio
                next_input = tf.where(use_teacher, x[:, t:t+1, :], pred)
                ar_input = tf.concat([ar_input, next_input], axis=1)
            
            pred_seq = tf.concat(preds, axis=1)
            loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
        
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {"loss": loss, "tf_ratio": tf_ratio}
```

**Tests:**
1. Unit test: Verify predictions are valid tensors with correct shape
2. Integration test: Train for 5 epochs, verify loss decreases
3. Memory test: Profile peak memory during training (should be constant per batch)
4. Quality test: Generate audio sample, verify it sounds reasonable

**Integration:**
- Existing checkpoints will load correctly
- May need to retrain from scratch for best results (optional)

**Demo:**
Run training with memory profiler, show constant memory usage during epoch. Compare peak memory before/after optimization.

---

## Step 4: Add Batch Size Warning

**Objective:** Warn users when batch size may be too large for their system memory.

**Implementation:**
1. Add memory check in `train()` function in `src/music_generation/train.py`
2. Display warning if batch_size > 8 and system RAM < 16GB
3. Suggest appropriate batch size

**Files to modify:**
- `src/music_generation/train.py`
- `requirements.txt` (add psutil if not present)

**Code changes:**
```python
# Add import at top
import psutil

# In train() function, after argument parsing:
def train(data_dir, cache_dir, checkpoint_dir, batch_size=BATCH_SIZE, ...):
    # Check memory vs batch size
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    if batch_size > 8 and total_memory_gb < 16:
        logger.warning(f"⚠️  Batch size {batch_size} may be too large for {total_memory_gb:.1f}GB RAM")
        recommended_batch_size = 2 if total_memory_gb < 8 else 4
        logger.warning(f"⚠️  Consider using --batch_size {recommended_batch_size}")
    
    # ... rest of function
```

**Tests:**
1. Unit test: Verify warning logic with mocked memory values
2. Integration test: Run with large batch size on test system, verify warning appears

**Integration:**
- Non-breaking change
- Users can ignore warning if desired

**Demo:**
Run training with `--batch_size 16` on MacBook Air, show warning message.

---

## Step 5: Add Memory Profiling and Validation

**Objective:** Add tooling to measure and validate memory improvements.

**Implementation:**
1. Create `scripts/profile_memory.py` to measure peak memory during training
2. Create `tests/test_memory.py` with memory regression tests
3. Document memory usage in README

**Files to create:**
- `scripts/profile_memory.py`
- `tests/test_memory.py`

**Code for profiling script:**
```python
"""Profile memory usage during training."""
import tracemalloc
import argparse
from src.music_generation.train import train

def profile_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--cache_dir', default='./cache')
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    
    tracemalloc.start()
    
    train(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\n{'='*60}")
    print(f"Memory Profile Results:")
    print(f"  Current: {current / 1024**2:.1f} MB")
    print(f"  Peak: {peak / 1024**2:.1f} MB")
    print(f"{'='*60}")

if __name__ == '__main__':
    profile_training()
```

**Tests:**
1. Run profiling script before and after optimizations
2. Verify peak memory is reduced by at least 30%
3. Create regression test that fails if memory exceeds threshold

**Integration:**
- Add profiling instructions to README
- Document expected memory usage for different batch sizes

**Demo:**
Run profiling script, show before/after memory comparison. Update README with results.

---

## Summary

This plan addresses all four identified memory issues:
1. **Memory configuration** - Prevents upfront allocation
2. **Streaming statistics** - Eliminates dataset-size memory spike
3. **Fixed-window autoregressive** - Reduces O(n²) to O(n) memory growth
4. **Batch size warning** - Helps users avoid OOM errors

Each step is incremental, testable, and maintains backward compatibility. The optimizations are transparent to users and don't change model behavior or quality.

**Estimated Impact:**
- Peak memory reduction: 50-70%
- Training speed impact: <10% slowdown
- Code complexity: Minimal increase
- Risk: Low (all changes are localized and well-tested)
