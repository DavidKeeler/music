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
- Added REDUCTION_FACTOR=4, GROUPED_MEL_DIM=320, EFFECTIVE_SEQ_LEN=128
- All values verified correct
- Ready for Step 2: Dataset preprocessing

## Next Steps
1. Pick up Step 2: Update dataset preprocessing for grouped frames


## Step 2 Implementation: Dataset Preprocessing

Starting implementation of grouped frame preprocessing in dataset.py.

Key changes needed:
1. Import REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN from config
2. In _generator(), after loading mel:
   - Truncate to R-divisible length
   - Reshape [T, 80] → [T/R, 320]
3. Update sequence length check from SEQ_LEN to EFFECTIVE_SEQ_LEN
4. Update stride calculation to use EFFECTIVE_SEQ_LEN
5. Update output signature to use EFFECTIVE_SEQ_LEN and GROUPED_MEL_DIM

This maintains the same API (returns input/target pairs) but internally operates on grouped frames.
