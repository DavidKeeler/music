# Summary — Learned Continuous Mel Tokens

## Artifacts

| File | Description |
|------|-------------|
| `specs/learned-mel-tokens/rough-idea.md` | Original specification |
| `specs/learned-mel-tokens/requirements.md` | 9 Q&A clarifications |
| `specs/learned-mel-tokens/design.md` | Full design with architecture, components, data models, acceptance criteria |
| `specs/learned-mel-tokens/plan.md` | 7-step incremental implementation plan |
| `specs/learned-mel-tokens/summary.md` | This file |

## Overview

Refactor the TensorFlow mel spectrogram generator to replace fixed reduction-factor frame grouping (`GROUPED_MEL_DIM`, `REDUCTION_FACTOR`) with a learned tokenizer/detokenizer architecture. A causal strided Conv1D tokenizer compresses `[B, T, N_MELS]` mel spectrograms into `[B, T/C, D_MODEL]` latent tokens. The transformer and conv stack operate in token space. A causal upsample+Conv1D detokenizer reconstructs full-resolution mel output.

**Key decisions:**
- Configurable compression ratio (default 4x)
- Strictly causal tokenizer and detokenizer
- `tf.repeat` + causal Conv1D for upsampling (no Conv1DTranspose)
- Joint end-to-end training (no pre-training phase)
- Token-space scheduled sampling
- Dataset yields raw `[SEQ_LEN, N_MELS]` — no grouped frames
- Clean removal of all grouped-mel constants

**Files modified:** `config.py`, `model.py`, `train.py`, `dataset.py`
**Files unchanged:** `layers.py`, `inference.py`

## Next Steps

Implementation via the 7-step plan in `plan.md`. Each step builds on the previous and ends with a demoable state.
