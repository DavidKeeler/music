# Requirements — Token-Space Refactor

## Q&A Record

### Q1: forward_tokens() — should it return the raw transformer output, or a projected version?

Currently `forward_from_tokens()` runs: z-conditioning → conv stack → transformer stack → detokenizer. The new `forward_tokens()` would stop before the detokenizer, returning `[B, T_tok, D_MODEL]`.

But the detokenizer's final layer is `Dense(N_MELS)` — meaning the transformer output lives in D_MODEL space (256-dim), not mel space (100-dim). During generation, the next-token prediction you append to context would be a 256-dim vector, which is correct since context is also in D_MODEL space.

However: should `forward_tokens()` return the transformer output directly, or should there be a separate "token head" projection (e.g. a Dense(D_MODEL)) to map from "transformer hidden state" to "predicted next token embedding"? Or is the identity (raw transformer output = next token) the right choice here?

**A1:** Use a projection head. A `Dense(D_MODEL)` layer will map transformer hidden states back into token embedding space before returning from `forward_tokens()`.

---

### Q2: Loss simplification — what should the target be?

Currently the dataset yields `(input_mel, target_mel)` where target is input shifted by 1 *frame*:

```python
input_mel = mel[start:start + SEQ_LEN]       # frames [0..511]
target_mel = mel[start + 1:start + SEQ_LEN + 1]  # frames [1..512]
```

And then training applies an additional shift: `preds[:, :-1, :] vs y[:, 1:, :]`, which effectively means the model predicts frame `t+2` from frame `t`. That's a double-shift bug.

Your spec says to remove the loss shifting and just do `MSE(preds, y)`. With the dataset already providing a 1-frame-shifted target, this means the model learns to predict frame `t+1` from frame `t` — standard next-frame prediction.

Should we keep the dataset's 1-frame shift as-is and just remove the loss-level shift? Or do you want to change the dataset too (e.g. have `input == target` and let the model's internal causality handle everything)?

**A2:** Change the dataset. `input == target` — the dataset will yield the same mel slice for both input and target. The model's internal causality handles next-frame prediction. Loss becomes simply `MSE(preds, y)` with no shifting anywhere.

---

### Q3: Scheduled sampling pass 1 — how to get predicted tokens without re-tokenizing?

Currently pass 1 does:
```python
preds_pass1 = self.base_model(x, z=z_pass1, training=False)  # mel output
pred_tokens = self.base_model.tokenizer(preds_pass1, training=False)  # re-tokenize
```

Your spec says to replace this with:
```python
pred_tokens = self.base_model.forward_tokens(gt_tokens, z=z_pass1, training=False)
```

This means pass 1 runs the transformer on ground-truth tokens and takes the *token-space output* as the "predicted tokens" for mixing. That's clean — no re-tokenization.

But note: `gt_tokens` are the tokenized ground truth, and `forward_tokens()` output at position `t` predicts position `t+1` (due to causality). So when mixing, `pred_tokens` is already naturally shifted by one position relative to `gt_tokens`. Does the current shifted-concat logic still apply?

```python
pred_tokens_shifted = tf.concat([gt_tokens[:, :1, :], pred_tokens[:, :-1, :]], axis=1)
```

Or should we mix `pred_tokens` directly against `gt_tokens` without the shift, since the causal model output already represents the "next token" prediction?

**A3:** Keep the shift. `pred_tokens` from `forward_tokens()` at position `t` predicts `t+1`, so the right-shift aligns predictions with their target positions before mixing with ground truth.

---

### Q4: Latent encoder input during scheduled sampling pass 2

Currently pass 2 does:
```python
x_mixed = self.base_model.detokenizer(mixed_tokens, training=True)
z, z_mean, z_logvar = self.latent_encoder(x_mixed, training=True)
```

The latent encoder takes mel-space input. With the refactor removing the detokenizer call, what should the latent encoder receive?

Options:
- (a) Run latent encoder on the original mel input `x` (ground truth) — simple, no roundtrip
- (b) Add a token-space latent encoder that takes `mixed_tokens` directly
- (c) Keep detokenizing mixed_tokens just for the latent encoder input (partial roundtrip)

**A4:** Option (a) — run latent encoder on the original mel input `x` (ground truth). Simple, no roundtrip.

---

### Q5: Dataset truncation removal — how should the tokenizer handle non-divisible lengths?

Your spec says to remove:
```python
truncated_length = mel_length - (mel_length % TOKEN_COMPRESSION_RATIO)
```

Currently `MelTokenizer` uses stride-2 causal convolutions with left-padding. With `TOKEN_COMPRESSION_RATIO=4` (two stride-2 layers), an input of length `T` produces output of length `ceil(T/4)` due to the causal padding.

However, `MelDetokenizer` does `tf.repeat(x, repeats=2, axis=1)` per layer, so it always produces exactly `T_tok * 4` frames. If the input wasn't divisible by 4, the output will be longer than the input.

Should we:
- (a) Pad the input to the next multiple of TOKEN_COMPRESSION_RATIO inside the tokenizer (transparent to callers)
- (b) Trim the detokenizer output to match the original input length
- (c) Both — pad on encode, trim on decode

**A5:** Option (c) — both. Tokenizer pads input to next multiple of TOKEN_COMPRESSION_RATIO, detokenizer trims output to match original length. This makes the tokenizer/detokenizer pair handle arbitrary lengths transparently.

---

### Q6: Test updates — what's the expectation?

The refactor touches core interfaces:
- New `forward_tokens()` method on MelGenerator
- Changed `generate()` loop
- Changed scheduled sampling in MelGeneratorTraining
- Changed dataset (input == target, no truncation)
- Changed loss (no shifting)

There are existing test files with mock models (`SimpleMelGenerator`) that mirror the real interface (they have `forward_from_tokens`, `tokenizer`, `detokenizer` attributes). These mocks will need updating to add `forward_tokens()` and the projection head.

Should I:
- (a) Update existing tests to match the new interfaces (minimal — just fix what breaks)
- (b) Leave tests untouched — you'll handle them separately
- (c) Update tests AND add new test cases for the new behavior (e.g. verifying no re-tokenization in generation)

**A6:** Option (c) — update existing tests to match new interfaces AND add new test cases verifying the new behavior (e.g. no re-tokenization in generation loop, token-space autoregression correctness).

---

### Q7: Optional detokenizer upsampling improvement (item #6) — in or out of scope?

Your spec lists replacing `tf.repeat` with `UpSampling1D + Conv1D` or `Conv1DTranspose` as optional. Given this is a correctness-focused refactor, should we:
- (a) Include it in this refactor
- (b) Leave it out — separate follow-up task

**A7:** Option (a) — include the detokenizer upsampling improvement in this refactor. Replace `tf.repeat` with learned upsampling (Conv1DTranspose or UpSampling1D + Conv1D).

---

### Q8: Which learned upsampling approach for the detokenizer?

Two options:
- (a) `Conv1DTranspose` (transposed convolution) — single layer per upsample step, learned kernel, can introduce checkerboard artifacts
- (b) `UpSampling1D` (nearest-neighbor) + `Conv1D` — two ops per step, no artifacts, slightly more parameters

Both maintain causality with appropriate padding. Which do you prefer?

**A8:** Option (b) — `UpSampling1D` (nearest-neighbor) + `Conv1D`. No checkerboard artifacts, causal with left-padding.

---

### Q9: Inference module (`inference.py`) — any changes needed?

`MusicGenerationModel.generate()` calls `self.mel_generator.generate()` which returns a mel spectrogram, then passes it through the vocoder. Since we're changing `MelGenerator.generate()` internals (token-space autoregression, single detokenize at end), the output contract stays the same — it still returns `[num_frames, N_MELS]`.

Should `inference.py` remain untouched, or is there anything you want changed there (e.g. exposing token-space generation for debugging, or changing the normalizer flow)?

**A9:** Leave `inference.py` untouched. The output contract of `MelGenerator.generate()` doesn't change.
