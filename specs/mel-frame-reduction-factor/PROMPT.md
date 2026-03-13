# PROMPT: Implement Mel Frame Reduction Factor

## Objective

Implement a reduction factor (R=4) for the mel spectrogram generator to predict multiple consecutive frames per autoregressive step, reducing sequence length 4x and improving training efficiency.

## Key Requirements

1. Add `REDUCTION_FACTOR = 4`, `GROUPED_MEL_DIM = N_MELS * REDUCTION_FACTOR`, and `EFFECTIVE_SEQ_LEN = SEQ_LEN // REDUCTION_FACTOR` to `config.py`
2. Update dataset pipeline to truncate mel sequences to R-divisible length and reshape to `[T/R, R*80]`
3. Modify `MelGenerator` input projection to accept `GROUPED_MEL_DIM` and output projection to produce `GROUPED_MEL_DIM`
4. Update `generate()` method to handle grouped frame prediction and reshape output to individual frames
5. Modify `train_step()` to compute loss per frame and average across R frames
6. Add unit tests for shape transformations and layer dimensions
7. Verify end-to-end training and inference work correctly

## Acceptance Criteria

**Given** a configured reduction factor R=4  
**When** training the model on MusicNet dataset  
**Then** the effective sequence length should be 128 (512/4)  
**And** each training step should process 4 frames per position

**Given** a trained model with R=4  
**When** generating 500 frames autoregressively  
**Then** the model should produce 125 generation steps (500/4)  
**And** each step should predict 4 consecutive frames

**Given** a mel sequence of length 515  
**When** preprocessing for R=4  
**Then** the sequence should be truncated to 512 frames  
**And** reshaped to [128, 320] for model input

**Given** predictions and targets of shape [B, T/R, R*80]  
**When** computing loss  
**Then** loss should be computed independently for each of R frames  
**And** averaged across all frames and batch

**Given** a seed mel of 100 frames  
**When** generating with R=4  
**Then** seed should be truncated to 96 frames (divisible by 4)  
**And** reshaped to [24, 320] for model context

## Reference

Full design and implementation plan: `specs/mel-frame-reduction-factor/`

- `design.md` - Complete architecture and component specifications
- `plan.md` - 7-step incremental implementation plan
- `requirements.md` - Detailed requirements from Q&A clarification

## Implementation Notes

- Follow the 7-step plan in `plan.md` for incremental, testable progress
- Maintain causal structure throughout the model
- External APIs (inference) should accept/return individual frames `[T, 80]`
- Internal model operations use grouped frames `[T/R, R*80]`
- No backward compatibility needed; requires retraining from scratch
- Add inline code comments explaining reshaping logic
