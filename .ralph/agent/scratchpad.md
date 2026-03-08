# Scratchpad: Teacher Forcing Schedule Implementation

## Iteration 1 - Initial Analysis

### Understanding the Objective
Implement exponential decay scheduled sampling for the mel spectrogram generator to reduce exposure bias. The current training uses pure teacher forcing (always feeding ground truth), but during inference the model uses its own predictions, causing train-test mismatch.

### Current State Analysis
From examining the codebase:

1. **train.py** - Has `MelGeneratorTraining` wrapper with `train_step()` that:
   - Uses parallel training (single forward pass, not autoregressive loop)
   - Compares predictions at t with targets at t+1
   - Loss: `preds[:, :-1, :] - y[:, 1:, :]`
   - This is NOT autoregressive teacher forcing - it's parallel training

2. **config.py** - Has basic training params but NO teacher forcing schedule params yet

3. **Memory patterns from memories**:
   - Parallel training already implemented (mem-1772814618-c42c)
   - Single forward pass with causal masking, not autoregressive loop
   - Memory scales O(model) not O(seq_len × model)

### Critical Realization
The current training is ALREADY parallel (not autoregressive). The design doc assumes autoregressive training with teacher forcing, but the actual implementation uses parallel training with causal attention.

**This changes the implementation approach:**
- Can't do "scheduled sampling" in the traditional sense (mixing ground truth and predictions per timestep)
- Need to implement scheduled sampling at the SEQUENCE level, not timestep level
- OR need to switch to autoregressive training loop (but this contradicts the parallel training optimization)

### Decision Point
Two options:
1. **Keep parallel training** - Implement sequence-level scheduled sampling (use model predictions for some sequences, ground truth for others)
2. **Switch to autoregressive** - Implement timestep-level scheduled sampling as designed (but loses parallel training benefits)

The design doc explicitly requires timestep-level scheduled sampling with autoregressive generation. This means we need to implement the autoregressive training loop with scheduled sampling, even though it's slower than parallel training.

### Plan
Following the 8-step plan from specs:

1. **Add schedule function** - Pure function for exponential decay
2. **Update config** - Add 4 new parameters
3. **Add ratio tracking** - Modify trainer to track and update ratio
4. **Implement autoregressive training step** - Core implementation with memory optimizations
5. **Update training loop** - Integrate scheduled sampling
6. **Add validation** - Pure autoregressive evaluation
7. **Update checkpoints** - Save/restore ratio state
8. **Add tests** - Comprehensive coverage

### Next Action
Start with Step 1: Add the schedule function. This is a pure function with no dependencies, easy to test.


---

## Iteration 1 - Completed

### Task: task-1772945574-e628 - Step 1: Add exponential_tf_schedule function

**What I did:**
- Added `exponential_tf_schedule()` function to `src/music_generation/train.py`
- Implemented formula: `ε(step) = max(ε_min, ε_initial * exp(-k * step))`
- Created comprehensive unit tests in `tests/test_scheduled_sampling.py`
- All 6 tests pass

**Implementation details:**
- Pure function with no side effects
- Accepts int or tf.Tensor for step parameter
- Returns float32 tensor
- Default parameters: initial_ratio=1.0, min_ratio=0.05, decay_k=1e-5

**Tests:**
1. ✓ Initial step returns initial_ratio
2. ✓ Respects min_ratio floor
3. ✓ Exponential decay formula correct
4. ✓ Different decay rates work as expected
5. ✓ Accepts tensor input
6. ✓ Batch computation works

**Commit:** a64cbbf - feat: add exponential_tf_schedule function for teacher forcing decay

**Next task:** task-1772945577-8607 - Step 2: Update config with TF schedule params
