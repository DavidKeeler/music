# Scratchpad: Mel Frame Reduction Factor Implementation

## Objective
Implement a reduction factor (R=4) for the mel spectrogram generator to predict multiple consecutive frames per autoregressive step, reducing sequence length 4x and improving training efficiency.

## Understanding

The goal is to modify the music generation system to predict 4 mel frames at once instead of 1 frame per step. This will:
- Reduce effective sequence length from 512 to 128 (4x speedup)
- Reduce attention complexity from O(512²) to O(128²) = 16x improvement
- Maintain causal structure and model quality

Key insight: External APIs (inference) still work with individual frames [T, 80], but internally the model operates on grouped frames [T/R, R*80].

## Implementation Plan

Following the 7-step incremental plan from specs/mel-frame-reduction-factor/plan.md:

1. **Config**: Add REDUCTION_FACTOR=4, GROUPED_MEL_DIM=320, EFFECTIVE_SEQ_LEN=128
2. **Dataset**: Truncate to R-divisible length, reshape [T,80] → [T/R, R*80]
3. **Model I/O**: Update input/output projections to handle GROUPED_MEL_DIM
4. **Generation**: Handle grouped frame prediction and reshaping
5. **Training**: Compute per-frame loss and average across R frames
6. **Tests**: Unit tests for all shape transformations
7. **E2E Verification**: Run training and inference to validate

## Current Status
✓ Step 1 complete: Config changes committed (4009dee)
✓ Step 2 complete: Dataset preprocessing committed (690828b)

**Step 2 Summary:**
- Modified dataset.py to truncate and reshape mel sequences
- Truncation: mel_length → (mel_length // R) * R to ensure divisibility
- Reshape: [T, 80] → [T/R, 320] for grouped frame processing
- Updated all sequence length checks to use EFFECTIVE_SEQ_LEN (128)
- Updated stride calculation for grouped frames
- Created 11 unit tests covering all transformations
- Verified end-to-end with real MusicNet data: [2, 128, 320] ✓

**Key Implementation Details:**
- Truncation happens before reshaping to avoid dimension errors
- Normalization applied before reshaping (preserves per-frame statistics)
- Input/target pairs use 1-step offset on grouped sequences
- Dataset API unchanged: still returns (input, target) batches

## Next Steps
1. Step 3: Modify model input/output projections to handle GROUPED_MEL_DIM


## Step 3 Complete: Model I/O Projections Updated (beb0bcc)

Successfully modified MelGenerator to handle grouped frames with R=4 reduction factor.

**Changes Made:**
1. Updated input projection: `Dense(N_MELS)` → `Dense(GROUPED_MEL_DIM)` (80 → 320)
2. Updated output projection: `Dense(N_MELS)` → `Dense(GROUPED_MEL_DIM)` (80 → 320)
3. Modified `generate()` method for grouped frame prediction:
   - Truncates seed to R-divisible length (handles edge cases)
   - Reshapes seed: `[T, 80]` → `[T/R, 320]` for internal processing
   - Generates in steps of R frames: `num_steps = (num_frames + R - 1) // R`
   - Reshapes output back: `[num_steps, 320]` → `[num_steps*R, 80]`
   - Truncates to exact `num_frames` requested
4. Updated all model tests to use `GROUPED_MEL_DIM` and `EFFECTIVE_SEQ_LEN`

**Test Results:**
- ✓ All model tests pass (4/4)
- ✓ All reduction factor tests pass (11/11)
- ✓ Model correctly processes grouped frames `[B, T/R, 320]`
- ✓ Generate method maintains API compatibility with individual frames

**Key Design Decisions:**
- External API unchanged: `generate()` still accepts/returns `[T, 80]` individual frames
- Internal operations use grouped frames `[T/R, 320]` for efficiency
- Seed truncation handles non-divisible lengths gracefully
- Context window now uses `EFFECTIVE_SEQ_LEN` (128) instead of `SEQ_LEN` (512)

**Next Steps:**
Step 4 will update the model generation method (already done in this step), so we can move to Step 5: modify training loss computation to handle grouped frames.


## Step 4 Complete: Generation Method Already Updated

Step 4 (update model generation method) was completed as part of Step 3 since the I/O projections and generation logic are tightly coupled. The `generate()` method now:
- Handles grouped frame prediction (R frames per step)
- Truncates seed to R-divisible length
- Reshapes seed and output between individual and grouped frames
- Maintains API compatibility with external callers

No additional work needed for Step 4.


## Step 5: Training Loss Computation

**Task:** Modify `train_step()` to compute per-frame loss and average across R frames.

**Current State:**
- Loss computed on grouped frames: `[B, T/R, 320]`
- Uses MAE: `tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))`
- Applied in both `_pure_teacher_forcing()` and `_parallel_scheduled_sampling()`

**Required Changes:**
1. Reshape predictions: `[B, T/R, 320] → [B, T/R, 4, 80]`
2. Reshape targets: `[B, T/R, 320] → [B, T/R, 4, 80]`
3. Compute per-frame loss: MSE per frame → `[B, T/R, 4]`
4. Average across all dimensions: scalar loss

**Implementation Plan:**
- Update both training methods with identical loss computation
- Add inline comments explaining reshaping logic
- Maintain existing teacher forcing schedule logic
- Use MSE instead of MAE for per-frame loss (as specified in plan)
