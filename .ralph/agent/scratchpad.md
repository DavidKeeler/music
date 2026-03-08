# Scratchpad: Debug Training Pipeline

## Understanding

The objective is to fix runtime issues in the TensorFlow music generation training pipeline to ensure it runs without crashes for at least one complete epoch.

## Current State Analysis

Reviewed `src/music_generation/train.py` and found:

**Step 1 (Graph execution error) - ALREADY FIXED ✓**
- Lines 87-91 already use `tf.cond` with `tf.math.logical_or` for NaN/Inf detection
- This matches the required fix from the plan

**Remaining Steps:**
- Step 2: Add shape validation for dataset batches (before training)
- Step 3: Verify model builds correctly on first batch
- Step 4: Test training for one complete epoch
- Step 5: Validate checkpoint saving

## Plan

Since Step 1 is already complete, I'll focus on Steps 2-5:

1. **Add dataset shape validation** - Insert validation code after dataset creation to verify batch shapes
2. **Add model build verification** - Trigger model build with sample batch and print summary
3. **Add checkpoint validation** - Verify checkpoint files are created after training
4. **Run end-to-end test** - Execute training for 1 epoch to verify everything works

These changes will be minimal additions to the `train()` function to add diagnostic output and validation checks.

## Completed Work

### Task 1: Dataset Shape Validation ✓
- Added validation code after dataset creation in `train()` function
- Validates batch_size, mel channels (80), and input/target shape match
- Prints shapes for debugging
- Committed: f7c2dce

### Task 2: Model Build Verification ✓
- Added model build verification after model compilation
- Tests forward pass with sample batch from dataset
- Asserts output shape matches input shape
- Provides early failure detection before training starts
- Committed: ece16d7

### Task 3: Checkpoint Validation ✓
- Added validation after model.fit() to verify checkpoint exists
- Displays checkpoint size for diagnostics
- Raises FileNotFoundError if checkpoint missing
- Provides early detection of checkpoint save failures
- Committed: bce075f

## Next Steps

The blocked task "Run end-to-end training test for 1 epoch" should now be unblocked and ready to execute. This will verify all the validation steps work together in a complete training run.
