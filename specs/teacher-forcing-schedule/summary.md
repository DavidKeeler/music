# Teacher Forcing Schedule - Project Summary

## Overview

Implementation of exponential decay scheduled sampling for the mel spectrogram generator to reduce exposure bias and improve autoregressive inference quality.

## Problem

The current training uses pure teacher forcing (always feeding ground truth previous frames). This creates a train-test mismatch: during inference, the model must use its own predictions, leading to error accumulation and poor generation quality.

## Solution

Gradually transition from teacher forcing to using model predictions during training via exponential decay schedule: `ε(step) = max(ε_min, ε_initial * exp(-k * step))`

## Artifacts

All artifacts are located in `specs/teacher-forcing-schedule/`:

### Planning Documents
- **`rough-idea.md`** - Initial problem statement
- **`requirements.md`** - Requirements clarification Q&A (minimal, research-driven)
- **`design.md`** - Comprehensive design document with:
  - Architecture and components
  - Memory-efficient implementation patterns
  - Acceptance criteria (Given-When-Then)
  - Testing strategy
  - Performance analysis
  - Hyperparameter tuning guide

### Research
- **`research/scheduled-sampling-overview.md`** - Background on scheduled sampling, schedule types, research findings
- **`research/tensorflow-implementation.md`** - TensorFlow/Keras implementation patterns and best practices
- **`research/summary.md`** - Key findings and recommendations

### Implementation
- **`plan.md`** - 8-step incremental implementation plan with:
  - Checklist for tracking progress
  - Detailed steps with objectives, tests, integration notes
  - Demo commands for each step
  - Estimated effort: 12-18 hours

## Key Design Decisions

1. **Exponential decay schedule** - Most practical, widely used, simple to tune
2. **Memory-efficient implementation** - TensorArray + sliding window, avoid growing concatenation
3. **Custom training loop** - No TensorFlow Addons dependency, better control
4. **Per-step updates** - Smooth, continuous decay
5. **Minimum ratio 0.05** - Maintains training stability

## Configuration Parameters

```python
initial_tf_ratio: float = 1.0      # Start with pure teacher forcing
min_tf_ratio: float = 0.05         # Minimum ratio for stability
tf_decay_k: float = 1e-5           # Decay rate (tune based on dataset)
tf_warmup_steps: int = 0           # Steps before decay starts
```

## Implementation Steps

1. Add schedule function
2. Update configuration
3. Add teacher forcing ratio tracking to trainer
4. Implement memory-efficient training step
5. Update training loop
6. Add validation with pure autoregressive
7. Update checkpoint save/load
8. Add comprehensive tests

## Expected Benefits

- Reduced exposure bias
- Better autoregressive inference quality
- Less error accumulation during generation
- More coherent long-term sequences

## Next Steps

### Option 1: Manual Implementation
Follow the implementation plan in `plan.md` step by step.

### Option 2: Autonomous Implementation with Ralph
Create a PROMPT.md for Ralph to implement autonomously:

```bash
# Full pipeline with code assist
ralph run --config presets/pdd-to-code-assist.yml

# Simpler spec-driven flow
ralph run --config presets/spec-driven.yml
```

### Option 3: Hybrid Approach
Implement critical steps manually (Steps 1-4), use Ralph for integration and testing (Steps 5-8).

## References

- Bengio et al. (2015): Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
- Current codebase: `src/music_generation/train.py`
- Related specs: `specs/tensorflow-music-generation/`
