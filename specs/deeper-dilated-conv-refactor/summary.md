# Summary: Deeper Dilated Causal Conv Refactor

## Artifacts

| File | Description |
|---|---|
| `specs/deeper-dilated-conv-refactor/rough-idea.md` | Original idea and goals |
| `specs/deeper-dilated-conv-refactor/requirements.md` | Q&A clarifications |
| `specs/deeper-dilated-conv-refactor/design.md` | Full design document |
| `specs/deeper-dilated-conv-refactor/plan.md` | Implementation plan (3 steps) |

## Overview

Refactor MelGenerator's conv stack from 3 hardcoded layers to a dynamic 5-layer dilated causal architecture driven by `CONV_DILATION_RATES = [1, 2, 4, 8, 16]`. Increases receptive field from ~160 ms to ~672 ms. All conv layers placed before the transformer stack. D_MODEL stays at 128. ~99K additional params (~7.6% increase).

## Files to Modify

- `src/music_generation/config.py` — update `CONV_DILATION_RATES`
- `src/music_generation/model.py` — refactor `MelGenerator.__init__` and `call()`

## Next Steps

- Implement using the plan: `specs/deeper-dilated-conv-refactor/plan.md`
- Or generate a PROMPT.md for Ralph to implement autonomously
