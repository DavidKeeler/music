# Scratchpad: Parallel Autoregressive Training Refactor

## Current State Analysis

The current `train.py` has:
1. ✅ Teacher forcing schedule tracking (tf_ratio, training_step)
2. ✅ Pure teacher forcing path (single forward pass)
3. ❌ O(T) autoregressive loop (255 iterations per batch)
4. ✅ tf.cond dispatch between paths

## Objective

Replace the O(T) autoregressive loop with 2-pass parallel scheduled sampling:
- Pass 1: Get predictions without gradients
- Pass 2: Mix predictions with ground truth, train with gradients
- Result: 255 forward passes → 2 forward passes (127x reduction)

## Implementation Plan

The objective provides exact implementation code. I need to:

1. **Extract pure teacher forcing to method** - Already exists as nested function, just move to class method
2. **Add parallel scheduled sampling method** - New 2-pass implementation
3. **Simplify train_step** - Replace tf.cond with simple if/else (Python control flow is fine)
4. **Remove old code** - Delete nested functions and tf.cond

## Key Insight

The current code uses `tf.cond` which requires both branches to be graph-compatible. This was causing the OperatorNotAllowedInGraphError. The new approach uses simple Python `if` statement because:
- We're checking a Python-accessible value (self.tf_ratio)
- Only ONE branch executes per call
- No need for symbolic control flow

## Tasks to Create

Based on the existing task list, I see tasks are already created but blocked. Let me check what's blocking them and if I need to create new tasks for this specific refactor.

Looking at the objective, this is a COMPLETE REFACTOR that supersedes the incremental approach. I should:
1. Close the old blocked tasks (they're for incremental approach)
2. Create new tasks for the parallel refactor approach
3. Implement the refactor

## Next Steps

1. Review existing tasks and close obsolete ones
2. Create tasks for parallel refactor
3. Implement first task
