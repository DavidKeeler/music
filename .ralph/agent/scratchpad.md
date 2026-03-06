# Scratchpad - Parallel Training Fix

## Objective
Fix critical training loop bug causing ~100x memory overhead by replacing autoregressive loop with proper parallel Transformer training.

## Understanding the Problem

**Current (broken) approach:**
- Training runs SEQ_LEN (512) forward passes inside a single GradientTape
- Memory scales as O(SEQ_LEN × model_memory) ≈ 50GB
- Each iteration accumulates gradients for 512 forward passes
- This is RNN-style BPTT, not proper Transformer training

**Correct approach:**
- Single forward pass per batch with causal masking
- Model predicts next frame at each position in parallel
- Loss compares predictions at t with targets at t+1
- Memory scales as O(model_memory) ≈ 500MB
- This is standard Transformer training

## Key Insight
The Transformer already has causal attention masking built-in. We don't need to manually iterate through the sequence - we can feed the entire sequence at once and let the model handle causality through its attention mechanism.

## Implementation Plan

1. **Simplify MelGeneratorTraining class** - Remove autoregressive loop, teacher forcing logic
2. **Update train() function** - Remove teacher forcing parameters
3. **Clean up config.py** - Remove teacher forcing constants
4. **Fix dataset prefetching** - Change from AUTOTUNE to fixed value
5. **Update memory profiling script** - Remove teacher forcing parameters
6. **Create test suite** - Verify parallel training works correctly

## Current Status
- Previous work focused on memory optimizations (streaming stats, fixed windows, etc.)
- Old task "Verify training compatibility" exists but is for the OLD approach
- Need to create new tasks for the parallel training fix


## Iteration 1 - Completed

**Task:** Simplify MelGeneratorTraining class to use parallel training

**Changes made:**
1. Replaced autoregressive loop with single forward pass
2. Removed teacher forcing parameters (initial_tf_ratio, decay_k, min_ratio, context_len)
3. Removed compute_tf_ratio() method
4. Added call() method for forward pass
5. Simplified train_step() to use parallel processing
6. Added shape validation with tf.debugging.assert_equal
7. Added NaN/Inf detection
8. Updated tests to match new interface

**Key insight:**
The Transformer already has causal attention masking. We don't need to manually iterate through the sequence - we feed the entire sequence at once and let the model handle causality through its attention mechanism.

**Loss computation:**
```python
loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
```
This compares predictions at position t with targets at position t+1, which is the standard next-frame prediction task.

**Commit:** 6de74e0 - "refactor: replace autoregressive loop with parallel Transformer training"

**Next:** Update train() function and remove teacher forcing parameters from argparse


## Iteration 2 - Completed

**Task:** Update train() function and remove teacher forcing parameters (task-1772814499-2c6a)

**Changes made:**
1. Updated scripts/profile_memory.py:
   - Removed teacher forcing parameters from argparse (initial_tf_ratio, tf_decay_k, min_tf_ratio)
   - Removed teacher forcing params from train() call
   - Removed unused config import
2. Verified train.py train() function was already clean (no teacher forcing params)

**Key insight:**
The train() function in train.py was already updated in Iteration 1 when MelGeneratorTraining was simplified. This task primarily involved cleaning up the memory profiling script to match.

**Commit:** 1dbb3c7 - "refactor: remove teacher forcing parameters from memory profiling script"

**Next:** Fix dataset prefetching to use fixed value (task-1772814505-4127)


## Iteration 3 - Completed

**Task:** Clean up config.py - remove teacher forcing constants (task-1772814502-7f9d)

**Changes made:**
1. Removed three teacher forcing constants from config.py:
   - INITIAL_TF_RATIO = 1.0
   - TF_DECAY_K = 1e-5
   - MIN_TF_RATIO = 0.05
2. Verified no other files reference these constants (grep found 0 matches)
3. Verified config.py imports successfully

**Key insight:**
These constants were only used by the old autoregressive training loop with teacher forcing. Since we've replaced that with parallel training, they're no longer needed.

**Commit:** 1c976db - "refactor: remove teacher forcing constants from config"

**Next:** Fix dataset prefetching to use fixed value (task-1772814505-4127)


## Iteration 4 - Completed

**Task:** Fix dataset prefetching to use fixed value (task-1772814505-4127)

**Changes made:**
1. Updated src/music_generation/dataset.py:
   - Changed `ds.prefetch(tf.data.AUTOTUNE)` to `ds.prefetch(2)`
2. Updated src/music_generation/vocoder.py:
   - Changed `dataset.prefetch(tf.data.AUTOTUNE)` to `dataset.prefetch(2)`

**Key insight:**
Using a fixed prefetch value of 2 instead of AUTOTUNE provides more predictable memory behavior during parallel training. AUTOTUNE can dynamically adjust prefetch buffer size, which can lead to unpredictable memory usage spikes.

**Commit:** c7e7e3e - "fix: replace AUTOTUNE with fixed prefetch value of 2"

**Next:** Update memory profiling script (task-1772814508-0356) - Note: This was already completed in Iteration 2, may need to verify or close


## Iteration 5 - Completed

**Task:** Create test suite for parallel training (task-1772814511-6c5e)

**Changes made:**
1. Created tests/test_parallel_training.py with 3 tests:
   - test_train_step_shapes: Verifies train_step produces correct shapes
   - test_single_forward_pass: Confirms single forward pass (not autoregressive loop)
   - test_loss_decreases: Validates training convergence over 10 steps
2. Used SimpleMelGenerator mock model to avoid complex dependencies
3. All tests pass successfully

**Key insight:**
The test suite confirms the parallel training implementation works correctly:
- Shapes are validated properly
- Single forward pass replaces autoregressive loop
- Loss decreases over training steps, showing convergence

**Commit:** 198ec75 - "test: add parallel training test suite"

**Status:** All tasks complete! The parallel training fix is fully implemented and tested.

## Summary

The objective has been completed successfully. All 6 requirements have been implemented:

1. ✅ Simplified MelGeneratorTraining class (Iteration 1)
2. ✅ Updated train() function (Iteration 2)
3. ✅ Cleaned up config.py (Iteration 3)
4. ✅ Fixed dataset prefetching (Iteration 4)
5. ✅ Updated memory profiling script (Iteration 2)
6. ✅ Created test suite (Iteration 5)

**Expected impact:**
- Memory: ~100x reduction (50GB → 500MB)
- Speed: ~100x faster (10 sec/step → 0.1 sec/step)
- Code: Much simpler (removed autoregressive loop and teacher forcing)
- Quality: Equivalent or better (proper Transformer training)

The training loop now uses proper parallel Transformer training with causal masking instead of the broken autoregressive approach.
