# Summary: Token-Space Autoregressive Refactor

## Artifacts

| File | Description |
|------|-------------|
| `specs/token-space-refactor/rough-idea.md` | Original refactor spec |
| `specs/token-space-refactor/requirements.md` | 9 Q&A clarifications |
| `specs/token-space-refactor/research/tokenizer-ceil-behavior.md` | Tokenizer output length analysis for non-divisible inputs |
| `specs/token-space-refactor/research/causal-upsampling.md` | UpSampling1D + CausalConv1D causality verification |
| `specs/token-space-refactor/research/test-mock-inventory.md` | Catalog of test files requiring updates |
| `specs/token-space-refactor/design.md` | Full design doc with architecture, interfaces, acceptance criteria |
| `specs/token-space-refactor/plan.md` | 9-step incremental implementation plan |

## Overview

This refactor fixes critical autoregressive correctness issues in the mel generator by eliminating mel↔token roundtrips in both generation and training. Key changes:

- New `forward_tokens()` method with `Dense(D_MODEL)` projection head as the core token-space primitive
- Generation loop uses `forward_tokens()` — tokenize once, autoregress in token space, detokenize once
- Scheduled sampling uses `forward_tokens()` for predicted tokens instead of re-tokenizing model output
- Loss simplified to `MSE(preds, y)` with no temporal shifting; dataset yields `input == target`
- Tokenizer/detokenizer handle arbitrary input lengths via pad-on-encode / trim-on-decode
- Detokenizer upgraded from `tf.repeat` to `UpSampling1D` + `CausalConv1D`

## Files Modified

| File | Scope |
|------|-------|
| `src/music_generation/model.py` | MelTokenizer, MelDetokenizer, MelGenerator |
| `src/music_generation/train.py` | MelGeneratorTraining (loss, scheduled sampling) |
| `src/music_generation/dataset.py` | Remove truncation and target shift |
| `tests/test_train.py` | Mock updates, loss assertions, new test |
| `tests/test_parallel_training.py` | Mock updates, docstring |
| `tests/test_scheduled_sampling.py` | Mock updates, re-tokenization logic |
| `tests/test_model.py` | New test cases |
| `tests/test_dataset.py` | New test case |

## Next Steps

To implement this refactor, use the 9-step plan in `specs/token-space-refactor/plan.md`.
