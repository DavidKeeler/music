# Parallel Training Fix Summary

## Project Overview

Fix critical training loop bug causing ~100x memory overhead. Current implementation runs 512 forward passes per batch inside a single GradientTape. Correct implementation uses single forward pass with causal masking (standard Transformer training).

## Problem

**Current (broken):**
```python
for t in range(1, seq_len):  # 512 iterations
    pred = self.base_model(ar_input, training=True)
    # Accumulates 512 forward passes in gradient tape
```

**Memory:** O(SEQ_LEN × model) = ~50GB per batch  
**Speed:** ~10 seconds per step

## Solution

**Correct:**
```python
preds = self.base_model(x, training=True)  # Single forward pass
loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
```

**Memory:** O(model) = ~500MB per batch  
**Speed:** ~0.1 seconds per step

## Changes Required

### 1. Simplify MelGeneratorTraining
- Remove autoregressive loop
- Remove teacher forcing (initial_tf_ratio, decay_k, min_ratio, context_len)
- Single forward pass with causal masking
- Simple loss computation

### 2. Update train() Function
- Remove teacher forcing parameters
- Simplify model creation

### 3. Clean Up config.py
- Delete INITIAL_TF_RATIO, TF_DECAY_K, MIN_TF_RATIO

### 4. Fix Dataset Prefetching
- Change prefetch(AUTOTUNE) → prefetch(2)

### 5. Update Profiling Script
- Remove teacher forcing parameters

### 6. Add Tests
- Shape validation
- Memory profiling
- Loss convergence

## Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Memory | ~50GB | ~500MB | ~100x |
| Speed | ~10 sec/step | ~0.1 sec/step | ~100x |
| Quality | Baseline | Same or better | - |
| Code complexity | High | Low | Much simpler |

## To Run with Ralph

```bash
ralph run -P specs/parallel-training-fix/PROMPT.md
```

## To Test After Implementation

```bash
# Profile memory
python scripts/profile_memory.py \
  --data_dir ~/data/music/musicnet/train_data \
  --batch_size 4 \
  --epochs 2

# Full training (can use larger batch now!)
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints \
  --epochs 50 \
  --batch_size 8
```

**Verify:**
- Peak memory < 1GB (was ~50GB)
- Training completes in minutes (was hours)
- Loss decreases steadily
- Generated audio quality is good

## Why This Works

**Causal Attention Mask:**
- Already prevents future information leakage
- Output at position t only depends on inputs ≤ t
- No explicit loop needed during training

**Training:**
- Feed full ground truth sequence
- Model predicts next frame at each position (in parallel)
- Loss compares predictions at t with targets at t+1
- Standard Transformer training method

**Inference:**
- Still autoregressive (generate one frame at a time)
- Use model.generate() method
- Unchanged from before

## Files Modified

1. `src/music_generation/train.py` - Simplify training class
2. `src/music_generation/config.py` - Remove teacher forcing
3. `src/music_generation/dataset.py` - Fix prefetching
4. `scripts/profile_memory.py` - Remove parameters
5. `tests/test_parallel_training.py` - New test suite

## Migration Notes

**Breaking changes:**
- Teacher forcing parameters removed
- Training behavior changed

**Recommended:**
- Delete old checkpoints
- Retrain from scratch
- Model architecture unchanged (can load weights if needed)

## References

- Problem analysis: `specs/parallel-training-fix/rough-idea.md`
- Design document: `specs/parallel-training-fix/design.md`
- Implementation plan: `specs/parallel-training-fix/plan.md`
