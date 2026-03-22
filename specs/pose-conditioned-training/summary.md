# Pose-Conditioned Training — Summary

## Artifacts

| File | Description |
|------|-------------|
| `specs/pose-conditioned-training/rough-idea.md` | Original idea + knowledge base summary |
| `specs/pose-conditioned-training/requirements.md` | 11 Q&A pairs covering all design decisions |
| `specs/pose-conditioned-training/design.md` | Full design doc: architecture, components, data models, acceptance criteria |
| `specs/pose-conditioned-training/plan.md` | 10-step implementation plan with test requirements per step |

## Overview

Integrate pose (body point) conditioning into MelGenerator via cross-attention in transformer2/transformer3, gated by a cosine-scheduled alpha parameter. Training runs in phases via separate `train_pose.py` invocations. Existing audio-only training and tests are unaffected.

**Key decisions:**
- Pre-extracted pose data `[T, 85]`, resampled to mel frame rate, same dir as audio
- TransformerBlock modified with `enable_cross_attn` flag (backward compatible)
- `pose=None` default on all MelGenerator methods
- Pose encoder supports frozen and trainable modes
- Per-sample conditioning dropout (skip cross-attention)
- Auto-detect checkpoint type when loading Phase 1 → Phase 2
- Alignment losses implemented but not active

**Files to modify:** `encoder.py`, `layers.py`, `model.py`, `dataset.py`, `config.py`, `losses.py`
**Files to create:** `train_pose.py`

## Suggested Next Steps

Implement using the plan at `specs/pose-conditioned-training/plan.md`. Steps 1-3 (encoder, layers, model) can be built and tested before the training pipeline (steps 4-7).
