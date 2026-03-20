# Implementation Plan: Token-Space Autoregressive Refactor

## Checklist

- [ ] Step 1: Add `forward_tokens()` and projection head to MelGenerator
- [ ] Step 2: Update MelDetokenizer with learned upsampling and trimming
- [ ] Step 3: Update MelTokenizer with input padding
- [ ] Step 4: Wire padding/trimming through `call()` and `forward_from_tokens()`
- [ ] Step 5: Rewrite `generate()` to use token-space autoregression
- [ ] Step 6: Update training — loss, scheduled sampling, latent encoder
- [ ] Step 7: Simplify dataset — remove truncation and target shift
- [ ] Step 8: Update test mocks and assertions
- [ ] Step 9: Add new test cases

---

## Step 1: Add `forward_tokens()` and projection head to MelGenerator

**Objective:** Introduce the core token-space forward primitive.

**Implementation guidance:**
- Add `self.projection_head = tf.keras.layers.Dense(D_MODEL)` in `MelGenerator.__init__()`
- Add `forward_tokens()` method that runs z-conditioning → conv stack → transformer stack → `projection_head`, returning `[B, T_tok, D_MODEL]`
- This is the shared transformer pipeline extracted from `forward_from_tokens()`, minus the detokenizer call

**Test requirements:**
- Run existing `tests/test_model.py::test_forward_from_tokens` — should still pass (API unchanged)
- Manually verify `forward_tokens()` returns `[B, T_tok, D_MODEL]`

**Integration notes:**
- `forward_from_tokens()` is NOT yet refactored to use `forward_tokens()` — that happens in Step 4
- No other files change yet

**Demo:** `model.forward_tokens(tokens).shape == [B, T_tok, D_MODEL]`

---

## Step 2: Update MelDetokenizer with learned upsampling and trimming

**Objective:** Replace `tf.repeat` with `UpSampling1D` + `CausalConv1D` and add `target_length` trimming.

**Implementation guidance:**
- In `MelDetokenizer.__init__()`, change block tuples from `(conv, norm)` to `(up, conv, norm)` where `up = tf.keras.layers.UpSampling1D(size=2)`
- In `call()`, replace `tf.repeat(x, repeats=2, axis=1)` with `up(x)`
- Add `target_length=None` parameter to `call()`. If provided, slice output to `[:, :target_length, :]`

**Test requirements:**
- Run `tests/test_model.py::test_forward_shape` — output shape should be unchanged for divisible-length inputs
- Verify `UpSampling1D` layers appear in model summary

**Integration notes:**
- Existing callers pass no `target_length`, so behavior is identical for now
- `forward_from_tokens()` still calls detokenizer without trimming — correct, since it expects padded tokens

**Demo:** `model.detokenizer(tokens).shape == [B, T_tok * 4, N_MELS]` (unchanged)

---

## Step 3: Update MelTokenizer with input padding

**Objective:** Pad input to next multiple of `TOKEN_COMPRESSION_RATIO` so arbitrary lengths work.

**Implementation guidance:**
- In `MelTokenizer.call()`, compute `pad_amount = (-tf.shape(x)[1]) % TOKEN_COMPRESSION_RATIO`
- Right-pad with zeros: `x = tf.pad(x, [[0,0], [0, pad_amount], [0,0]])`
- Do this before the conv layers
- Update docstring to remove "T is divisible by TOKEN_COMPRESSION_RATIO" constraint

**Test requirements:**
- Verify `MelTokenizer()(mel_13_frames).shape[1] == ceil(13/4) == 4`
- Verify `MelTokenizer()(mel_12_frames).shape[1] == 3` (no change for divisible)

**Integration notes:**
- Tokenizer is now safe for arbitrary lengths
- Callers don't need to pre-truncate

**Demo:** Non-divisible input length tokenizes without error.

---

## Step 4: Wire padding/trimming through `call()` and `forward_from_tokens()`

**Objective:** Make the full forward pass handle arbitrary lengths transparently. Refactor `forward_from_tokens()` to use `forward_tokens()`.

**Implementation guidance:**
- `MelGenerator.call()`: capture `original_T = tf.shape(mel)[1]`, tokenize, call `forward_tokens()`, detokenize with `target_length=original_T`
- `MelGenerator.forward_from_tokens()`: call `forward_tokens()` then `self.detokenizer(token_preds)` (no trimming — caller provides padded tokens, expects `T_tok * C` output)

**Test requirements:**
- `test_forward_shape`: `model(mel_512).shape == [B, 512, N_MELS]` (unchanged)
- `test_forward_from_tokens`: `model.forward_from_tokens(tokens_128).shape == [B, 512, N_MELS]` (unchanged)
- New: `model(mel_510).shape == [B, 510, N_MELS]` (arbitrary length)

**Integration notes:**
- This is the step where `forward_from_tokens()` changes internally (delegates to `forward_tokens()` + detokenizer)
- All existing callers of `call()` and `forward_from_tokens()` get correct behavior

**Demo:** `model(tf.zeros([2, 510, 100])).shape == [2, 510, 100]`

---

## Step 5: Rewrite `generate()` to use token-space autoregression

**Objective:** Eliminate the mel→token roundtrip in the generation loop.

**Implementation guidance:**
- Tokenize seed once
- Loop: `forward_tokens(context)` → take last token `[:, -1:, :]` → add noise → append to context
- After loop: detokenize all generated tokens once with `self.detokenizer(all_tokens)`
- Remove: `self.forward_from_tokens()` call and `self.tokenizer(output_mel)` re-tokenization from the loop

**Test requirements:**
- `test_generate_shape`: output shape `[num_frames, N_MELS]` (unchanged API)
- `test_causality_generate`: determinism with `temperature=0.0` still holds

**Integration notes:**
- `inference.py` is unchanged — it calls `generate()` which still returns `[num_frames, N_MELS]`
- This is the highest-impact correctness fix

**Demo:** Generate 100 frames from a seed — no intermediate tokenizer calls.

---

## Step 6: Update training — loss, scheduled sampling, latent encoder

**Objective:** Fix scheduled sampling to stay in token space, simplify loss.

**Implementation guidance:**

`_pure_teacher_forcing(x, y)`:
- Change loss from `tf.reduce_mean(tf.square(preds[:, :-1, :] - y[:, 1:, :]))` to `tf.reduce_mean(tf.square(preds - y))`

`_parallel_scheduled_sampling(x, y)`:
- Pass 1: replace `pred_tokens = self.base_model.tokenizer(preds_pass1)` with `pred_tokens = self.base_model.forward_tokens(gt_tokens, z=z_pass1, training=False)`
- Pass 2: remove `x_mixed = self.base_model.detokenizer(mixed_tokens)`. Change latent encoder input from `x_mixed` to `x`.
- Change loss from shifted to `tf.reduce_mean(tf.square(preds - y))`

**Test requirements:**
- `test_training_step_no_errors` passes
- `test_loss_decreases` passes
- Gradient flow tests pass

**Integration notes:**
- This depends on Step 1 (`forward_tokens()` exists) and Step 7 (dataset yields `input == target`)
- If running tests before Step 7, temporarily use `x` as both input and target

**Demo:** Training step completes without error, loss decreases over steps.

---

## Step 7: Simplify dataset — remove truncation and target shift

**Objective:** Dataset yields `(mel_slice, mel_slice)` with no truncation.

**Implementation guidance:**

In `MusicNetDataset._generator()`:
- Remove `truncated_length = mel_length - (mel_length % TOKEN_COMPRESSION_RATIO)` and `mel = mel[:truncated_length]`
- Change yield from `(mel[start:start+SEQ_LEN], mel[start+1:start+SEQ_LEN+1])` to `(mel[start:start+SEQ_LEN], mel[start:start+SEQ_LEN])`

**Test requirements:**
- `test_create_dataset_returns_tf_dataset` passes (shape-only check)
- New: verify `input_mel == target_mel` for yielded pairs

**Integration notes:**
- This must be coordinated with Step 6 (loss change) — both assume `input == target`
- `SEQ_LEN` is 512 which is divisible by 4, so the padding from Step 3 won't activate for dataset sequences. But files shorter than `SEQ_LEN` are already skipped.

**Demo:** Dataset yields pairs where input and target are identical tensors.

---

## Step 8: Update test mocks and assertions

**Objective:** Fix broken tests due to interface changes.

**Implementation guidance:**

In `tests/test_train.py`, `tests/test_parallel_training.py`, `tests/test_scheduled_sampling.py`:
- Add `self.projection_head = tf.keras.layers.Dense(D_MODEL)` to `SimpleMelGenerator.__init__()`
- Add `forward_tokens()` method: `return self.projection_head(self.tok_dense(tokens))`
- Update loss assertions: remove `y[:, 1:, :]` shift (use `y` directly)
- Update `test_parallel_training.py` docstring referencing old shift pattern

In `tests/test_scheduled_sampling.py`:
- Update any assertions that expect re-tokenization behavior

**Test requirements:**
- `pytest tests/test_train.py tests/test_parallel_training.py tests/test_scheduled_sampling.py` — all pass

**Integration notes:**
- Mock `forward_tokens()` should mirror the real one: transformer pipeline output projected to D_MODEL, without detokenizer

**Demo:** `pytest tests/` — all existing tests pass.

---

## Step 9: Add new test cases

**Objective:** Verify the new behavior with dedicated tests.

**Implementation guidance:**

In `tests/test_model.py`:
- `test_forward_tokens_shape`: assert `forward_tokens(tokens).shape == [B, T_tok, D_MODEL]`
- `test_arbitrary_input_length`: assert `model(mel_510).shape == [B, 510, N_MELS]`
- `test_detokenizer_uses_upsampling`: assert `UpSampling1D` layers exist in detokenizer
- `test_generation_no_retokenization`: wrap tokenizer in a call-counting wrapper, assert called once during `generate()`

In `tests/test_train.py`:
- `test_scheduled_sampling_no_retokenization`: verify `forward_tokens()` is used for pred tokens, not `tokenizer(model_output)`

In `tests/test_dataset.py`:
- `test_input_equals_target`: verify yielded pairs are identical

**Test requirements:**
- All new tests pass
- `pytest tests/` — full suite green

**Integration notes:**
- These tests codify the acceptance criteria from the design doc

**Demo:** `pytest tests/ -v` — all tests pass including new ones.
