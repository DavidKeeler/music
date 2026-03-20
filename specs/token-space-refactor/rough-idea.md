# Token-Space Refactor: Fix Autoregressive Correctness

Refactor the mel generator implementation to fix critical correctness issues in token-space modeling and generation, while keeping the current single-level continuous token design (no hierarchical/coarse tokens).

## Objective

Ensure the model is truly autoregressive in token space, eliminate tokenization inconsistencies, and simplify training where appropriate.

## Critical Fixes

### 1. Fix generation: remove mel → token roundtrip
Current generation does `tokens → mel → tokens → next token`, breaking autoregressive correctness and introducing drift. Need a `forward_tokens()` method that stops before the detokenizer, and update the generation loop to use it. Detokenize only once at the end.

### 2. Fix scheduled sampling: stay in token space
Currently re-tokenizes model predictions via `tokenizer(preds_pass1)`, introducing tokenizer reconstruction error and train/inference mismatch. Should use model-native token predictions instead.

### 3. Remove unnecessary mel reconstruction in pass 2
Currently does `detokenizer(mixed_tokens)` forcing a tokens → mel → tokens roundtrip. Should call model directly from tokens.

## Recommended Simplifications

### 4. Simplify loss (remove shifting)
Model is already causal internally — no need for manual temporal shifting of `preds[:, :-1, :] vs y[:, 1:, :]`.

### 5. Remove dataset length truncation constraint
Delete `truncated_length = mel_length - (mel_length % TOKEN_COMPRESSION_RATIO)` — tokenizer supports arbitrary lengths.

### 6. Optional: improve detokenizer upsampling
Replace `tf.repeat` with UpSampling1D + Conv1D or Conv1DTranspose (not required for correctness).

## Invariants
- Strict causality in token space
- Input/output shapes: `[B, T, N_MELS]`, token shape: `[B, T_tok, D_MODEL]`
- No discrete quantization (continuous tokens only)
