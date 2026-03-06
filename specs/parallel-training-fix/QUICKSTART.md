# Quick Reference: Parallel Training Fix

## Problem
Training loop runs 512 forward passes per batch → ~50GB memory, ~10 sec/step

## Solution
Single forward pass with causal masking → ~500MB memory, ~0.1 sec/step

## Key Change

**Before (broken):**
```python
for t in range(1, seq_len):
    pred = self.base_model(ar_input, training=True)
    # ... 512 iterations
```

**After (correct):**
```python
preds = self.base_model(x, training=True)  # Single pass
loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
```

## To Implement

```bash
ralph run -P specs/parallel-training-fix/PROMPT.md
```

## To Test

```bash
# Profile memory (should be ~100x lower)
python scripts/profile_memory.py \
  --data_dir ~/data/music/musicnet/train_data \
  --batch_size 4 \
  --epochs 2

# Full training (can use batch_size=8 now!)
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --epochs 50 \
  --batch_size 8
```

## Expected Results
- Memory: 50GB → 500MB (~100x reduction)
- Speed: 10 sec/step → 0.1 sec/step (~100x faster)
- Quality: Same or better
- Code: Much simpler

## Files Changed
- `src/music_generation/train.py` (simplify training class)
- `src/music_generation/config.py` (remove teacher forcing)
- `src/music_generation/dataset.py` (fix prefetch)
- `scripts/profile_memory.py` (remove parameters)
- `tests/test_parallel_training.py` (new tests)

## Why It Works
Causal attention mask already prevents future leakage. No loop needed during training. This is standard Transformer training.
