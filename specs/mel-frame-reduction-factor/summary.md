# Mel Frame Reduction Factor - Summary

## Project Overview

This project implements a **reduction factor** (R) for the mel spectrogram generator, enabling the model to predict multiple consecutive mel frames per autoregressive step. This reduces sequence length by 4x, accelerates training, improves convergence stability, and decreases error accumulation during generation.

## Artifacts

All project files are located in `specs/mel-frame-reduction-factor/`:

1. **rough-idea.md** - Initial concept describing the reduction factor technique
2. **requirements.md** - Q&A record from requirements clarification (12 questions)
3. **design.md** - Detailed design document with architecture, components, and acceptance criteria
4. **plan.md** - Implementation plan with 7 incremental steps
5. **summary.md** - This file

## Key Design Decisions

- **Reduction Factor**: R=4 (configurable in config.py)
- **Dataset**: Truncate sequences to be divisible by R, reshape to grouped frames
- **Model**: Update both input and output projections to handle R×N_MELS dimensions
- **Training**: Compute loss per frame, average across R frames
- **Inference**: Use all R predicted frames as context for next prediction
- **Testing**: Unit tests only (shape transformations and layer dimensions)
- **Compatibility**: No backward compatibility; requires retraining from scratch

## Implementation Approach

The plan follows a 7-step incremental approach:

1. **Config** - Add REDUCTION_FACTOR and derived constants
2. **Dataset** - Truncate and reshape mel sequences
3. **Model Architecture** - Update input/output projections
4. **Generation** - Handle grouped frame prediction
5. **Training** - Modify loss computation
6. **Tests** - Add unit tests for all components
7. **Verification** - End-to-end training and inference

Each step produces working, testable code that integrates with previous steps.

## Expected Benefits

- **4x shorter sequences**: 512 frames → 128 grouped frames
- **16x less attention compute**: O(T²) → O((T/4)²)
- **Faster convergence**: Larger temporal chunks provide stronger training signals
- **More stable generation**: Fewer autoregressive steps reduce error accumulation

## Next Steps

1. Review the design and plan documents
2. Create a PROMPT.md for Ralph to implement autonomously (optional)
3. Run `ralph run --config presets/pdd-to-code-assist.yml` to begin implementation
4. Or implement manually following the 7-step plan

## References

- Design Document: `specs/mel-frame-reduction-factor/design.md`
- Implementation Plan: `specs/mel-frame-reduction-factor/plan.md`
- Requirements: `specs/mel-frame-reduction-factor/requirements.md`
