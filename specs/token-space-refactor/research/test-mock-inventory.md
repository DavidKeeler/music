# Test Mock Inventory for Token-Space Refactor

## Files with Mock Models

### 1. `test_train.py`

**Mock classes:**

| Class | Attributes/Methods |
|---|---|
| `SimpleMelTokenizer` | `proj` (Dense→D_MODEL), `call()` downsamples 4x via `x[:, ::4, :]` |
| `SimpleMelDetokenizer` | `proj` (Dense→N_MELS), `call()` upsamples 4x via `tf.repeat(x, 4)` |
| `SimpleMelGenerator` | `tokenizer`, `detokenizer`, `tok_dense`, `passthrough`, `call()`, `forward_from_tokens()` |

**`forward_from_tokens()` impl:** `self.detokenizer(self.tok_dense(tokens))`

**Changes needed for refactor:**
- Add `forward_tokens()` method returning `[B, T_tok, D_MODEL]` — e.g. `self.tok_dense(tokens)` (stop before detokenizer)
- Add `projection_head` (Dense→D_MODEL) attribute, apply in `forward_tokens()`
- Loss lines at L138, L171, L184, L212 all use `pred_seq - y[:, 1:, :]` — remove the `[:, 1:, :]` shift (loss becomes `pred_seq - y`)

---

### 2. `test_parallel_training.py`

**Mock classes:**

| Class | Attributes/Methods |
|---|---|
| `_MockTokenizer` | `proj` (Dense→D_MODEL), `call()` downsamples 4x via `x[:, ::4, :]` |
| `_MockDetokenizer` | `proj` (Dense→N_MELS), `call()` upsamples 4x via `tf.repeat(x, 4)` |
| `SimpleMelGenerator` | `tokenizer`, `detokenizer`, `tok_dense`, `call()`, `forward_from_tokens()` |

**`call()` impl:** `tokenizer(inputs)` → `forward_from_tokens(tokens)`
**`forward_from_tokens()` impl:** `self.detokenizer(self.tok_dense(tokens))`

**Changes needed for refactor:**
- Add `forward_tokens()` method returning `[B, T_tok, D_MODEL]`
- Add `projection_head` attribute
- L78 docstring references `preds[:, :-1, :] - targets[:, 1:, :]` — update docstring to reflect unshifted loss

---

### 3. `test_scheduled_sampling.py`

**Mock classes:**

| Class | Attributes/Methods |
|---|---|
| `_MockTokenizer` | `proj` (Dense→256), `call()` downsamples 4x via `tf.linspace` gather |
| `_MockDetokenizer` | `proj` (Dense→80), `call()` upsamples 4x via `tf.repeat(x, 4)` |
| `SimpleMelGenerator` | `tokenizer`, `detokenizer`, `dense`, `out`, `tok_dense`, `call()`, `forward_from_tokens()` |

**`call()` impl:** `self.out(self.dense(x))` (does NOT go through tokenizer/detokenizer)
**`forward_from_tokens()` impl:** `self.detokenizer(self.tok_dense(tokens))`

**Changes needed for refactor:**
- Add `forward_tokens()` method returning `[B, T_tok, D_MODEL]`
- Add `projection_head` attribute
- Scheduled sampling tests that re-tokenize predictions via `self.base_model.tokenizer(preds_pass1)` should instead use `forward_tokens()` output directly (no re-tokenization round-trip)

---

## Files That Test `generate()` on Real `MelGenerator`

### 4. `test_model.py`

- `test_forward_from_tokens()` (L26-30): Calls `model.forward_from_tokens(tokens)` on real `MelGenerator`, asserts shape `(2, SEQ_LEN, N_MELS)`
- `test_generate_shape()` (L33-38): Calls `model.generate(seed, num_frames=100)` on real `MelGenerator`

**Changes needed:**
- Add `test_forward_tokens()` test for the new method, asserting shape `[B, T_tok, D_MODEL]`
- `test_generate_shape()` will need updating if `generate()` internals change (currently it re-tokenizes output; refactor replaces that with `forward_tokens()`)
- Existing `test_forward_from_tokens()` can remain as-is (method still exists) or be updated

### 5. `test_causality_and_windows.py`

- `test_causality_generate()` (L36-38): Calls `model.generate(seed, num_frames=50, temperature=0.0, z=z)` twice for determinism check

**Changes needed:**
- No mock changes needed (uses real `MelGenerator`)
- Test should still pass after refactor since `generate()` API is unchanged; only internals change

### 6. `test_inference.py`

- `test_music_generation_model_generate()` (L11-27): Calls `model.generate(seed_mel, num_frames=100)` on `MusicGenerationModel` wrapper (which delegates to `MelGenerator.generate()`)
- `test_music_generation_model_generate_with_normalizer()` (L107-122): Same pattern with normalizer

**Mock classes:** `SimpleVocoder` only (no mel generator mock — uses real `MelGenerator`)

**Changes needed:**
- No mock changes needed
- Tests should pass as-is since `MusicGenerationModel.generate()` API is unchanged

### 7. `test_dilated_convolutions.py`

- `test_generation_works()` (L91): Calls `model.generate(seed, num_frames=10)` on real `MelGenerator`

**Changes needed:**
- No mock changes needed; should pass as-is

---

## Files That Test Dataset Shifted-Target Behavior

### 8. `test_dataset.py`

- `test_create_dataset_returns_tf_dataset()` (L48-57): Calls `create_dataset()`, checks shapes but does NOT assert `input == target` or `target == input[1:]`

**Changes needed:**
- No test changes strictly required — the test only checks shapes and types
- However, the underlying `dataset.py` generator currently yields `(mel[start:start+SEQ_LEN], mel[start+1:start+SEQ_LEN+1])` (shifted by 1 frame). The refactor changes this to `input == target` (same slice). The test will still pass since shapes remain `(SEQ_LEN, N_MELS)` for both.

---

## Files That Test Loss Shift (`preds[:,:-1,:] vs y[:,1:,:]`)

### `test_train.py`
- L138: `loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))`
- L171: `loss_with_stop = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))`
- L184: `loss_without_stop = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))`
- L212: `loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))`

### `test_parallel_training.py`
- L78 docstring: `mean(|preds[:, :-1, :] - targets[:, 1:, :]|)`

### Production code (`train.py`)
- `_pure_teacher_forcing()`: `preds[:, :-1, :] - y[:, 1:, :]`
- `_parallel_scheduled_sampling()`: `preds[:, :-1, :] - y[:, 1:, :]`

**All of these shift references must be removed** when the dataset changes to `input == target`.

---

## Summary of Required Changes

| File | Mock `forward_tokens()` | Mock `projection_head` | Remove loss shift | Other |
|---|---|---|---|---|
| `test_train.py` | ✅ Add | ✅ Add | ✅ 4 lines | — |
| `test_parallel_training.py` | ✅ Add | ✅ Add | ✅ docstring | — |
| `test_scheduled_sampling.py` | ✅ Add | ✅ Add | — | Update re-tokenization logic |
| `test_model.py` | — (real model) | — | — | Add `test_forward_tokens()` |
| `test_causality_and_windows.py` | — | — | — | Should pass as-is |
| `test_inference.py` | — | — | — | Should pass as-is |
| `test_dilated_convolutions.py` | — | — | — | Should pass as-is |
| `test_dataset.py` | — | — | — | Should pass as-is (shape-only checks) |
| `test_training_smoke.py` | — (real model) | — | — | Should pass as-is (delegates to `MelGeneratorTraining`) |
