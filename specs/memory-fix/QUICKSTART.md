# Quick Reference: Memory Fix Implementation

## Problem
Training script OOMs on MacBook Air due to:
1. O(n²) memory growth in autoregressive loop
2. Loading entire dataset for statistics
3. No memory limits configured
4. Batch size too large (16)

## Solution Summary
1. Enable TensorFlow memory growth
2. Stream statistics computation
3. Fixed-size sliding window (64 frames)
4. Batch size warnings
5. Memory profiling tools

## To Implement with Ralph

```bash
cd /Users/davidkeeler/code/conducting3
ralph run --config presets/spec-driven.yml
# When prompted, specify: specs/memory-fix/PROMPT.md
```

## To Implement Manually

Follow 5 steps in `specs/memory-fix/plan.md`:

### Step 1: Memory Config (5 min)
Add to `src/music_generation/train.py`:
```python
def configure_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
```

### Step 2: Streaming Stats (15 min)
Replace `_compute_statistics()` in `src/music_generation/dataset.py` with chunked approach.

### Step 3: Fixed Window (20 min)
Add `context_len=64` to `MelGeneratorTraining` and truncate `ar_input` in loop.

### Step 4: Warning (10 min)
Add batch size check using `psutil` in `train()`.

### Step 5: Profiling (15 min)
Create `scripts/profile_memory.py` with `tracemalloc`.

## To Test After Implementation

```bash
# Profile memory
python scripts/profile_memory.py \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --batch_size 4 \
  --epochs 2

# Full training
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints \
  --epochs 100 \
  --batch_size 4
```

## Expected Results
- Peak memory: 50-70% reduction
- Training speed: <10% slower
- Model quality: Unchanged
- Batch size: 4 instead of 16 for MacBook Air

## Files Changed
- `src/music_generation/train.py`
- `src/music_generation/dataset.py`
- `scripts/profile_memory.py` (new)
- `requirements.txt` (add psutil)
