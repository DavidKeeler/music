# Learned Continuous Mel Tokens

## Objective

Replace the fixed reduction-factor grouping (`GROUPED_MEL_DIM`, `REDUCTION_FACTOR`) in the TensorFlow mel spectrogram generator with a **learned tokenizer + detokenizer architecture**. The model operates on compressed latent tokens and reconstructs full-resolution mel spectrograms.

## Required Changes

### 1. Remove grouped frame logic

- Eliminate all dependencies on `GROUPED_MEL_DIM`, `REDUCTION_FACTOR`, and reshape logic that packs/unpacks grouped mel frames
- All inputs and outputs should be `[B, T, N_MELS]`

### 2. Add MelTokenizer module

- Input: `[B, T, N_MELS]` → Output: `[B, T_tokens, D_MODEL]`
- Strided Conv1D layers to downsample time (e.g., stride 2 twice)
- Final projection to `D_MODEL`

### 3. Add MelDetokenizer module

- Input: `[B, T_tokens, D_MODEL]` → Output: `[B, T, N_MELS]`
- Conv1DTranspose (or upsampling + Conv1D) to restore time resolution

### 4. Update MelGenerator forward pass

Old: mel → projection → conv stack → transformer → projection → grouped mel
New: mel → tokenizer → tokens → conv stack → transformer → detokenizer → mel

- Remove `input_proj` and `output_proj`
- Insert `self.tokenizer` and `self.detokenizer`
- Keep causal conv stack and transformer blocks
- Ensure causality is preserved in token space

### 5. Latent conditioning (z)

- Keep latent conditioning but apply in token space
- Project `z` to `[B, D_MODEL]`, broadcast to `[B, T_tokens, D_MODEL]`, add to token embeddings

### 6. Simplify loss

- Replace grouped-frame loss with direct `MSE(predicted_mel, target_mel)`
- Remove all reshape logic for per-frame grouping

### 7. Update training wrapper

- LatentEncoder remains unchanged (inputs still `[B, T, N_MELS]`)
- Scheduled sampling operates on full mel sequences (no grouped shifting logic)

### 8. Update generation logic

- Operate autoregressively over tokens, not grouped frames
- Steps: tokenize seed → generate next token step-by-step → detokenize full sequence
- Remove all reduction factor / grouped reshaping logic

### 9. Shape invariants

- Token sequence is shorter than mel sequence (due to downsampling)
- Final output matches `[num_frames, N_MELS]`

## Constraints

- Preserve strict causality in the generator
- Keep existing TransformerBlock and CausalConvBlock unchanged
- Keep model compatible with current training loop structure
- Avoid introducing discrete quantization (tokens must remain continuous)

## Deliverables

- New `MelTokenizer` and `MelDetokenizer` classes
- Updated `MelGenerator`
- Updated training logic with simplified loss
- Updated `generate()` using token autoregression
- Removal of all grouped mel assumptions
