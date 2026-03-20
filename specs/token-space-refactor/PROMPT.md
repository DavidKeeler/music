# Token-Space Autoregressive Refactor

## Objective

Fix autoregressive correctness in the mel generator by eliminating mel↔token roundtrips in generation and training. Add `forward_tokens()` as the core token-space primitive. Simplify loss and dataset.

## Key Requirements

- Add `forward_tokens()` method with `Dense(D_MODEL)` projection head returning `[B, T_tok, D_MODEL]`
- Generation loop: tokenize seed once → autoregress via `forward_tokens()` → detokenize once at end
- Scheduled sampling: use `forward_tokens(gt_tokens)` for predicted tokens (no re-tokenization)
- Scheduled sampling pass 2: latent encoder receives original mel `x`, not detokenized mixed tokens
- Keep right-shift of pred tokens before mixing with ground truth
- Loss: `MSE(preds, y)` — no temporal shifting
- Dataset: `input == target` (same mel slice, no frame shift)
- Remove dataset truncation to `TOKEN_COMPRESSION_RATIO` multiples
- Tokenizer: pad input to next multiple of C
- Detokenizer: `UpSampling1D(size=2)` + `CausalConv1D` replacing `tf.repeat`; trim output via `target_length` param
- `forward_from_tokens()` delegates to `forward_tokens()` + detokenizer (no trimming)
- `call()` captures original T, passes `target_length` to detokenizer
- `inference.py` unchanged
- Update test mocks in `test_train.py`, `test_parallel_training.py`, `test_scheduled_sampling.py`
- Add new tests for `forward_tokens()` shape, arbitrary input lengths, no-retokenization in generation/training, input==target dataset

## Acceptance Criteria

- Given a MelGenerator, when `generate()` is called, then `tokenizer()` is called once (seed) and `detokenizer()` once (end). No intermediate mel→token conversions.
- Given `tf_ratio < 0.99`, when `_parallel_scheduled_sampling()` runs, then predicted tokens come from `forward_tokens()`, not `tokenizer(model_output)`. `detokenizer()` is not called in training logic.
- Given any training step, when loss is computed, then it is `MSE(preds, y)` with no `[:,:-1,:]` or `[:,1:,:]` slicing.
- Given the dataset generator, when it yields `(input, target)`, then they are the same tensor.
- Given mel with length T not divisible by 4, when passed through `model(mel)`, then output shape is `[B, T, N_MELS]`.
- Given tokens `[B, T_tok, D_MODEL]`, when `forward_tokens()` is called, then output is `[B, T_tok, D_MODEL]`.
- Given the MelDetokenizer, when inspected, then it contains `UpSampling1D` layers.
- Given updated mocks and new tests, when `pytest tests/` runs, then all tests pass.

## Reference

See `specs/token-space-refactor/` for design doc, research, and 9-step implementation plan.
