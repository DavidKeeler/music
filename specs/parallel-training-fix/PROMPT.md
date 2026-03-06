# Objective
Fix critical training loop bug causing ~100x memory overhead by replacing autoregressive loop with proper parallel Transformer training.

# Context
Current training runs SEQ_LEN forward passes inside a single GradientTape, causing memory to scale as O(SEQ_LEN × model_memory). This is incorrect for Transformers - they should use parallel training with causal masking.

**Current (broken):**
```python
for t in range(1, seq_len):  # 512 iterations!
    pred = self.base_model(ar_input, training=True)
    # ... accumulates 512 forward passes in gradient tape
```

**Correct:**
```python
preds = self.base_model(x, training=True)  # Single forward pass
loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
```

# Key Requirements

## 1. Simplify MelGeneratorTraining Class
**File:** `src/music_generation/train.py`

Replace entire class with:
```python
class MelGeneratorTraining(tf.keras.Model):
    """Simplified training wrapper with parallel processing."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            preds = self.base_model(x, training=True)
            loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
        
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {"loss": loss}
```

**Delete:**
- `initial_tf_ratio`, `decay_k`, `min_ratio`, `context_len` parameters
- `compute_tf_ratio()` method
- Entire autoregressive loop
- Teacher forcing logic

## 2. Update train() Function
**File:** `src/music_generation/train.py`

Change model creation:
```python
# Old:
model = MelGeneratorTraining(base_model, initial_tf_ratio=..., decay_k=..., ...)

# New:
model = MelGeneratorTraining(base_model)
```

Remove teacher forcing arguments from `main()` argparse.

## 3. Clean Up config.py
**File:** `src/music_generation/config.py`

Delete:
```python
INITIAL_TF_RATIO = 1.0
TF_DECAY_K = 1e-5
MIN_TF_RATIO = 0.05
```

## 4. Fix Dataset Prefetching
**File:** `src/music_generation/dataset.py`

Change:
```python
# Old:
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# New:
dataset = dataset.prefetch(2)
```

## 5. Update Memory Profiling Script
**File:** `scripts/profile_memory.py`

Remove teacher forcing parameters from argparse and train() call.

## 6. Add Validation (Optional but Recommended)
**File:** `src/music_generation/train.py`

Add to `train_step()`:
```python
# Shape validation
tf.debugging.assert_equal(tf.shape(x), tf.shape(y))
tf.debugging.assert_equal(tf.shape(preds), tf.shape(x))

# NaN detection
if tf.math.is_nan(loss) or tf.math.is_inf(loss):
    tf.print("⚠️  WARNING: NaN/Inf loss detected!")
    return {"loss": loss}
```

## 7. Create Test Suite
**File:** `tests/test_parallel_training.py`

Test:
- train_step produces correct shapes
- Memory usage is reasonable (<500MB peak)
- Loss decreases over steps

# Acceptance Criteria

**Given** parallel training is implemented  
**When** profiling memory usage  
**Then** peak memory is ~100x lower (50GB → 500MB)

**Given** single forward pass per batch  
**When** measuring training speed  
**Then** throughput is ~100x faster (10 sec/step → 0.1 sec/step)

**Given** training runs on MacBook Air  
**When** using batch_size=8  
**Then** training completes without OOM errors

**Given** model is trained with parallel method  
**When** generating audio samples  
**Then** quality is equivalent or better than before

**Given** causal masking prevents future leakage  
**When** training for 50 epochs  
**Then** loss decreases steadily and model converges

# Implementation Notes

**Why this works:**
- Causal attention mask already prevents future information leakage
- During training, we feed full ground truth sequence
- Model predicts next frame at each position
- Loss compares predictions at t with targets at t+1
- This is standard Transformer training (not RNN-style BPTT)

**Breaking changes:**
- `MelGeneratorTraining` constructor signature changed
- Teacher forcing parameters removed
- Training behavior changed (but model quality improves)

**Migration:**
- Delete old checkpoints (trained with broken method)
- Retrain from scratch with correct method
- Existing model architecture unchanged (can load weights)

**Expected impact:**
- Memory: ~100x reduction
- Speed: ~100x faster
- Quality: Equivalent or better
- Code: Much simpler

# Files to Modify

1. `src/music_generation/train.py` - Simplify training class, update train()
2. `src/music_generation/config.py` - Remove teacher forcing constants
3. `src/music_generation/dataset.py` - Fix prefetching
4. `scripts/profile_memory.py` - Remove teacher forcing parameters
5. `tests/test_parallel_training.py` - Create test suite (new file)

# Testing

```bash
# Run tests
pytest tests/test_parallel_training.py -v

# Profile memory
python scripts/profile_memory.py \
  --data_dir ~/data/music/musicnet/train_data \
  --batch_size 4 \
  --epochs 2

# Full training
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints \
  --epochs 50 \
  --batch_size 8
```

**Verify:**
- Peak memory < 1GB (was ~50GB)
- Training speed ~100x faster
- Loss decreases steadily
- Generated audio quality is good

# Reference
Detailed design and plan in `specs/parallel-training-fix/`
