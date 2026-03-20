# Implementation Plan — Learned Continuous Mel Tokens

## Checklist

- [ ] Step 1: Update config.py
- [ ] Step 2: Add MelTokenizer and MelDetokenizer
- [ ] Step 3: Rewrite MelGenerator
- [ ] Step 4: Update dataset pipeline
- [ ] Step 5: Update training wrapper
- [ ] Step 6: Rewrite generate()
- [ ] Step 7: Final cleanup and validation

---

## Step 1: Update config.py

**Objective:** Replace grouped-frame constants with learned tokenizer config.

**Implementation:**
- Remove `REDUCTION_FACTOR`, `EFFECTIVE_SEQ_LEN`, `GROUPED_MEL_DIM` and their comments
- Add `TOKEN_COMPRESSION_RATIO = 4`
- Add `TOKEN_NUM_CONV_LAYERS = int(math.log2(TOKEN_COMPRESSION_RATIO))`
- Add `TOKEN_SEQ_LEN = SEQ_LEN // TOKEN_COMPRESSION_RATIO`
- Add assertion that `TOKEN_COMPRESSION_RATIO` is a power of 2
- Update `WINDOW_SIZES` comment to reference `TOKEN_SEQ_LEN`

**Tests:**
- `TOKEN_SEQ_LEN == 128` with defaults
- `TOKEN_NUM_CONV_LAYERS == 2` with defaults
- Assertion fires for non-power-of-2 values

**Integration:** All downstream files will import from the new constants. No file should import the removed constants after this step.

**Demo:** `python -c "from src.music_generation.config import TOKEN_COMPRESSION_RATIO, TOKEN_SEQ_LEN; print(f'ratio={TOKEN_COMPRESSION_RATIO}, seq_len={TOKEN_SEQ_LEN}')"` prints `ratio=4, seq_len=128`.

---

## Step 2: Add MelTokenizer and MelDetokenizer

**Objective:** Implement the learned tokenizer and detokenizer as Keras layers in `model.py`.

**Implementation:**

`MelTokenizer`:
- `TOKEN_NUM_CONV_LAYERS` blocks, each: causal left-pad `(kernel_size - 1)` → `Conv1D(filters, 3, stride=2, padding='valid')` → `LayerNorm` → `GELU`
- Filter sizes: linearly interpolate from `N_MELS` to `D_MODEL` across layers
- Final `Dense(D_MODEL)` projection
- `call(mel)`: `[B, T, N_MELS]` → `[B, T // C, D_MODEL]`

`MelDetokenizer`:
- `TOKEN_NUM_CONV_LAYERS` blocks, each: `tf.repeat(x, 2, axis=1)` → `CausalConv1D(filters, 3)` → `LayerNorm` → `GELU`
- Filter sizes: linearly interpolate from `D_MODEL` to `N_MELS` across layers
- Final `Dense(N_MELS)` projection
- `call(tokens)`: `[B, T_tok, D_MODEL]` → `[B, T_tok * C, N_MELS]`

**Tests:**
- Shape test: tokenizer `[2, 512, 100]` → `[2, 128, 256]`
- Shape test: detokenizer `[2, 128, 256]` → `[2, 512, 100]`
- Causality test for tokenizer: two inputs identical up to frame `t`, tokens match up to `t // C`
- Causality test for detokenizer: two token seqs identical up to position `k`, mel frames match up to `k * C + (C-1)`
- Repeat with `TOKEN_COMPRESSION_RATIO=8` (3 layers)

**Integration:** These are standalone classes with no dependencies on the rest of the model. Can be tested in isolation.

**Demo:** Instantiate both, pass random tensor through tokenizer then detokenizer, print input/output shapes match.

---

## Step 3: Rewrite MelGenerator

**Objective:** Replace `input_proj`/`output_proj` with tokenizer/detokenizer. Add `forward_from_tokens` helper.

**Implementation:**
- Remove `self.input_proj = Dense(D_MODEL)` and `self.output_proj = Dense(GROUPED_MEL_DIM)`
- Add `self.tokenizer = MelTokenizer()` and `self.detokenizer = MelDetokenizer()`
- Keep `self.conv_layers`, `self.transformer1/2/3`, `self.z_proj` unchanged
- `call(mel, z, training)`:
  1. `tokens = self.tokenizer(mel)` → `[B, T_tok, D_MODEL]`
  2. Add z conditioning in token space (same broadcast pattern, using `T_tok`)
  3. Conv stack on tokens
  4. Transformer stack on tokens
  5. `output = self.detokenizer(tokens)` → `[B, T, N_MELS]`
- Add `forward_from_tokens(tokens, z, training)` — runs steps 2–5 (for scheduled sampling pass 2)
- Update docstring and all shape comments
- Remove all imports of `GROUPED_MEL_DIM`, `REDUCTION_FACTOR`, `EFFECTIVE_SEQ_LEN`

**Tests:**
- Forward pass: `[2, 512, 100]` → `[2, 512, 100]`
- `forward_from_tokens`: `[2, 128, 256]` → `[2, 512, 100]`
- Model builds without error, `model.summary()` shows tokenizer/detokenizer layers

**Integration:** `MelGenerator` now accepts `[B, T, N_MELS]` and returns `[B, T, N_MELS]`. Downstream consumers (training, inference) must pass raw mel.

**Demo:** `model = MelGenerator(); out = model(tf.random.normal([2, 512, 100])); print(out.shape)` → `(2, 512, 100)`.

---

## Step 4: Update dataset pipeline

**Objective:** Remove grouped-frame logic from `dataset.py`. Yield raw `[SEQ_LEN, N_MELS]` pairs.

**Implementation:**
- Remove imports of `REDUCTION_FACTOR`, `GROUPED_MEL_DIM`, `EFFECTIVE_SEQ_LEN`
- Import `TOKEN_COMPRESSION_RATIO`
- In `_generator()`:
  - Truncate mel to `TOKEN_COMPRESSION_RATIO`-divisible length (same logic, different constant)
  - Remove `tf.reshape(mel, [-1, GROUPED_MEL_DIM])` — work with raw mel
  - Skip if `mel.shape[0] <= SEQ_LEN` (was `grouped_length <= EFFECTIVE_SEQ_LEN`)
  - Stride = `SEQ_LEN // 2` (was `EFFECTIVE_SEQ_LEN // 2`)
  - Yield `[SEQ_LEN, N_MELS]` input/target pairs (1-frame offset on raw mel)
- Update `output_signature` to `(SEQ_LEN, N_MELS)`

**Tests:**
- Dataset yields tensors of shape `[SEQ_LEN, N_MELS]`
- Batched dataset yields `[B, SEQ_LEN, N_MELS]`
- No references to `GROUPED_MEL_DIM` or `REDUCTION_FACTOR` remain in file

**Integration:** Training script now receives `[B, 512, 100]` batches instead of `[B, 128, 400]`. Compatible with updated `MelGenerator.call()`.

**Demo:** `ds, steps = create_dataset(...); x, y = next(iter(ds)); print(x.shape)` → `(B, 512, 100)`.

---

## Step 5: Update training wrapper

**Objective:** Simplify loss computation and implement token-space scheduled sampling.

**Implementation:**

`_pure_teacher_forcing(x, y)`:
- Remove `from .config import REDUCTION_FACTOR, N_MELS` local import
- Remove reshape to `[B, T/R, R, N_MELS]`
- Direct MSE: `mel_loss = tf.reduce_mean(tf.square(preds[:, :-1] - y[:, 1:]))`

`_parallel_scheduled_sampling(x, y)`:
- Pass 1 (no gradients):
  - `preds_pass1 = self.base_model(x)` → `[B, T, N_MELS]`
  - Tokenize both: `gt_tokens = self.base_model.tokenizer(x)`, `pred_tokens = self.base_model.tokenizer(preds_pass1)`
  - Mix in token space: `[B, T_tok, D_MODEL]` with shifted pred tokens
- Pass 2 (with gradients):
  - Detokenize mixed tokens for latent encoder input
  - `self.base_model.forward_from_tokens(mixed_tokens, z, training=True)` → `[B, T, N_MELS]`
  - Direct MSE loss (no reshape)

Shape assertion in `train()`:
- Change `GROUPED_MEL_DIM` → `N_MELS` in assert
- Remove `GROUPED_MEL_DIM` from imports

**Tests:**
- Pure teacher forcing: loss is finite, gradients flow
- Scheduled sampling: loss is finite, token mixing shapes are correct
- `train_step` dispatches correctly based on `tf_ratio`

**Integration:** Works with updated dataset (Step 4) and updated model (Step 3).

**Demo:** Run 5 training steps on synthetic data, verify loss decreases and no errors.

---

## Step 6: Rewrite generate()

**Objective:** Autoregressive generation in token space.

**Implementation:**
- Remove all `REDUCTION_FACTOR`, `GROUPED_MEL_DIM` references
- Import `TOKEN_COMPRESSION_RATIO`, `TOKEN_SEQ_LEN`
- New flow:
  1. Truncate seed to `TOKEN_COMPRESSION_RATIO`-divisible length (pad if too short)
  2. `seed_tokens = self.tokenizer(seed_mel[None])` → `[1, seed_tok_len, D_MODEL]`
  3. Context = last `TOKEN_SEQ_LEN` tokens
  4. Loop `num_steps = ceil(num_frames / TOKEN_COMPRESSION_RATIO)`:
     - Forward pass on context → take last token
     - Add temperature noise to token
     - Append to context
  5. `generated_mel = self.detokenizer(all_generated_tokens[None])` → `[1, num_steps*C, N_MELS]`
  6. Return `generated_mel[0, :num_frames]`

**Tests:**
- `generate(seed, 500)` returns `[500, N_MELS]`
- `generate(seed, 1)` works (edge case)
- Short seed (< compression ratio) pads correctly
- No NaN in output

**Integration:** `MusicGenerationModel.generate()` in `inference.py` calls this unchanged — API is preserved.

**Demo:** `mel = model.generate(tf.random.normal([100, 100]), num_frames=500); print(mel.shape)` → `(500, 100)`.

---

## Step 7: Final cleanup and validation

**Objective:** Verify no grouped-mel references remain. End-to-end validation.

**Implementation:**
- `grep -r "GROUPED_MEL_DIM\|REDUCTION_FACTOR\|EFFECTIVE_SEQ_LEN" src/music_generation/` → zero matches
- Update `LatentEncoder` docstring: `[B, T, GROUPED_MEL_DIM]` → `[B, T, N_MELS]`
- Update `MelGenerator` class docstring to describe tokenizer/detokenizer architecture
- Verify `inference.py` works without changes (API encapsulated)

**Tests:**
- Full integration: dataset → model → training step → checkpoint save → load → generate
- Grep for removed constants returns nothing
- `model.summary()` shows clean architecture

**Integration:** Complete system works end-to-end.

**Demo:** Train for 1 epoch on real data, generate a sample, save as WAV.
