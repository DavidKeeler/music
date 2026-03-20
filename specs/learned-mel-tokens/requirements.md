# Requirements — Learned Continuous Mel Tokens

## Q&A Record

*(Questions and answers will be appended below as requirements are clarified.)*

---

### Q1: Tokenizer downsampling factor

The spec mentions "stride 2 twice" as an example, giving 4x compression (matching the current `REDUCTION_FACTOR=4`). Should we treat 4x as the target compression ratio, or should this be configurable (e.g., a `TOKEN_STRIDE` config parameter)? Is matching the current 4x ratio intentional to keep sequence lengths comparable?

**A1:** Configurable. Add a config parameter for the compression ratio (default 4x to match current behavior). The number of strided conv layers and their strides should be derived from this.

---

### Q2: Causality in the tokenizer

The current system is strictly causal. Strided Conv1D layers in the tokenizer have a receptive field spanning neighboring frames. Should the tokenizer be strictly causal (each token only sees past/current mel frames), or is a non-causal (bidirectional) tokenizer acceptable since it only encodes already-observed input?

**A2:** Strictly causal. The tokenizer must use causal convolutions so each token only depends on past/current mel frames.

---

### Q3: Detokenizer architecture — Conv1DTranspose vs Upsampling+Conv1D

TF's `Conv1DTranspose` can produce checkerboard artifacts and has quirks with causal padding. The alternative — `tf.repeat` upsample + Conv1D — is simpler and more stable. Preference?

**A3:** Use `tf.repeat` (nearest-neighbor upsample) followed by regular Conv1D. Avoids checkerboard artifacts and causal padding issues with Conv1DTranspose.

---

### Q4: Detokenizer causality

The detokenizer runs after the autoregressive transformer, so the token sequence is already causal. Should the detokenizer itself be strictly causal, or is non-causal acceptable within the local upsampled window? (With 4x upsample, non-causal would let an output frame see the token representing its group — all from the same token anyway.)

**A4:** Strictly causal. The detokenizer must also preserve causality end-to-end — each output mel frame depends only on current/past tokens.

---

### Q5: Tokenizer/detokenizer training — joint or pre-trained?

Two options: (1) Joint end-to-end training — tokenizer, transformer, detokenizer all train together with MSE loss. (2) Pre-train tokenizer+detokenizer as an autoencoder first, then train the transformer. Which approach?

**A5:** Joint end-to-end training. Tokenizer, transformer, and detokenizer all train together with the same MSE loss. Simpler to start with; pre-training can be explored later if token quality is insufficient.

---

### Q6: Dataset pipeline — sequence alignment

Currently the dataset truncates to `REDUCTION_FACTOR`-divisible lengths and yields grouped frames. With learned tokens, the tokenizer handles downsampling. Should we keep `SEQ_LEN=512` and enforce divisibility by compression ratio in the dataset, or let the tokenizer handle arbitrary lengths with padding?

**A6:** Enforce divisibility by compression ratio in the dataset. Keep `SEQ_LEN=512`, truncate mel lengths to be divisible by the compression ratio. Dataset yields raw `[SEQ_LEN, N_MELS]` pairs — no grouping/reshaping.

---

### Q7: Scheduled sampling in token space

Two options for where to apply teacher forcing mixing: (1) Mel space — mix ground-truth mel with detokenized predictions, re-tokenize for pass 2. (2) Token space — mix ground-truth tokens with predicted tokens directly. More efficient. Preference?

**A7:** Token space. Mix ground-truth tokens (from tokenizing ground truth) with predicted tokens (transformer output) directly. More efficient — avoids extra tokenize/detokenize round-trip.

---

### Q8: LatentEncoder input format

Currently takes `[B, T, GROUPED_MEL_DIM]`. With this refactor, input becomes `[B, T, N_MELS]` — longer sequence, fewer channels. The Conv1D layers still work. Should we leave the architecture untouched (just update docs), or is there concern about processing 4x more timesteps?

**A8:** Leave architecture untouched, just update docstrings/comments to reflect the new `[B, T, N_MELS]` input shape. No concern about the longer sequence — the Conv1D strides handle it.

---

### Q9: Config cleanup

`REDUCTION_FACTOR`, `GROUPED_MEL_DIM`, and `EFFECTIVE_SEQ_LEN` are replaced by the new tokenizer compression ratio. Remove entirely from `config.py`, or keep commented out / deprecated for reference?

**A9:** Remove entirely. No deprecated/commented-out constants — clean break.
