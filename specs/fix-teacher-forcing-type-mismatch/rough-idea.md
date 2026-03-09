# Rough Idea: Fix Teacher Forcing Type Mismatch

## Problem Statement

The teacher forcing implementation in `src/music_generation/train.py` has a type mismatch error that prevents training from running:

**Error:** `TypeError: Input 'y' of 'Maximum' Op has type int64 that does not match type int32 of argument 'x'`

**Location:** Line 125 in `update_tf_ratio()` method

**Root Cause:** `self.training_step` is `int64` but `self.warmup_steps` is `int32` (or vice versa), causing a type mismatch in `tf.maximum()`.

## Performance Context

A recent refactor updated the autoregressive training loop to match the original implementation's "growing context" approach:

**Original approach (conducting/):**
- Starts with 1 frame
- Grows context naturally (1, 2, 3... frames)
- Uses Python list + concat
- Only caps at MAX_CONTEXT_FRAMES when passing to model
- Faster because early iterations process small sequences

**Previous broken approach:**
- Used fixed sliding window (always processing MAX_CONTEXT_FRAMES=128 frames)
- Much slower for 512 timestep sequences

**Current state:**
- Code updated to match original growing context approach
- Type mismatch bug prevents execution

## Goal

Fix the type mismatch error with minimal changes while preserving the growing context implementation that matches the original code's performance characteristics.

## Files Involved

- `src/music_generation/train.py` - Line 125 in `update_tf_ratio()`
- `src/music_generation/config.py` - TF_WARMUP_STEPS definition

## Comparison Needed

Compare the current implementation against the original in `/Users/davidkeeler/code/conducting/src/main/python/complex_music_model/train_audio.py` to ensure:
1. Type consistency is correct
2. Growing context approach is properly implemented
3. No performance regressions
