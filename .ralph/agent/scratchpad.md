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


## Step 5 Complete: Training Loss Computation (22fc63d)

Successfully modified training loss computation to handle grouped frames with per-frame loss averaging.

**Changes Made:**
1. Updated `_pure_teacher_forcing()`:
   - Reshape predictions: `[B, T/R-1, R*80] → [B, T/R-1, R, 80]`
   - Reshape targets: `[B, T/R-1, R*80] → [B, T/R-1, R, 80]`
   - Compute MSE per frame: `tf.square(preds_frames - target_frames)`
   - Average across all dimensions: `tf.reduce_mean()`

2. Updated `_parallel_scheduled_sampling()`:
   - Same reshaping and loss computation logic
   - Applied to Pass 2 (training pass with mixed input)

3. Updated test infrastructure:
   - `SimpleMelGenerator` now uses `GROUPED_MEL_DIM` (320) instead of 80
   - All test data updated to use `GROUPED_MEL_DIM`
   - Added `GROUPED_MEL_DIM` to test imports

**Test Results:**
- ✓ All training tests pass (8/8)
- ✓ All reduction factor tests pass (11/11)
- ✓ All model tests pass (4/4)
- ✓ Loss computation correctly handles grouped frames
- ✓ Gradients flow correctly through reshaped tensors

**Key Implementation Details:**
- Loss changed from MAE to MSE (as specified in plan)
- Reshaping uses dynamic batch_size and seq_len from tf.shape()
- Both training methods use identical loss computation
- Teacher forcing schedule logic unchanged
- Inline comments explain reshaping rationale

**Next Steps:**
Step 6: Add unit tests for reduction factor (already exists - tests/test_reduction_factor.py with 11 tests)
Step 7: Verify end-to-end training and inference


## Step 6 Complete: Comprehensive Unit Tests (3cd8dc6)

Successfully added comprehensive unit tests for all reduction factor components.

**Changes Made:**
1. Added TestModelProjections (2 tests):
   - test_model_input_shape: Verify model accepts grouped frames [B, T/R, R*80]
   - test_model_output_shape: Verify model outputs grouped frames [B, T/R, R*80]

2. Added TestLossComputation (2 tests):
   - test_loss_reshaping: Verify loss reshapes to [B, T/R-1, R, 80] format
   - test_loss_scalar_output: Verify loss produces scalar value

3. Added TestGenerationWithReduction (4 tests):
   - test_seed_truncation: Verify seed truncated to R-divisible length
   - test_seed_reshaping: Verify seed reshaped to grouped format
   - test_generation_steps: Verify correct number of steps (ceil(frames/R))
   - test_output_reshaping: Verify output reshaped back to individual frames

**Test Results:**
- ✓ All 19 tests pass (11 existing + 8 new)
- ✓ All core tests pass: test_reduction_factor.py, test_model.py, test_train.py
- ✓ Tests cover all components: config, dataset, model, training, generation
- ✓ All shape transformations validated for R=4

**Test Coverage:**
- Config values: REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN
- Dataset: truncation, reshaping, sequence pairs
- Model: input/output projections, forward pass
- Training: loss computation, per-frame averaging
- Generation: seed preprocessing, step calculation, output reshaping

**Next Steps:**
Step 7: Verify end-to-end training and inference
