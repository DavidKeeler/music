# PROMPT.md — Learned Continuous Mel Tokens

## Objective

Refactor the TensorFlow mel spectrogram generator to replace fixed reduction-factor frame grouping with a learned tokenizer/detokenizer architecture. The model operates on compressed continuous latent tokens and reconstructs full-resolution mel spectrograms.

## Spec Directory

All design artifacts are in `specs/learned-mel-tokens/`:
- `design.md` — full architecture, component interfaces, data models
- `plan.md` — 7-step incremental implementation plan
- `requirements.md` — clarified requirements (9 Q&A)

## Key Requirements

- Add `MelTokenizer`: causal strided Conv1D encoder, `[B, T, N_MELS]` → `[B, T/C, D_MODEL]`
- Add `MelDetokenizer`: `tf.repeat` upsample + causal Conv1D, `[B, T/C, D_MODEL]` → `[B, T, N_MELS]`
- Rewrite `MelGenerator` to use tokenizer/detokenizer instead of `input_proj`/`output_proj`
- Update dataset to yield raw `[SEQ_LEN, N_MELS]` pairs (no grouped frames)
- Simplify training loss to direct MSE (no reshape)
- Implement token-space scheduled sampling
- Rewrite `generate()` for token-space autoregression
- Remove `REDUCTION_FACTOR`, `GROUPED_MEL_DIM`, `EFFECTIVE_SEQ_LEN` from config
- Add configurable `TOKEN_COMPRESSION_RATIO` (default 4, must be power of 2)

## Constraints

- Strict causality in tokenizer, detokenizer, and generator
- Keep `TransformerBlock`, `CausalConvBlock`, `CausalConv1D`, `LocalWindowAttention` unchanged
- Keep `LatentEncoder` architecture unchanged (update docstrings only)
- Keep `inference.py` API unchanged (changes encapsulated in `MelGenerator`)
- No discrete quantization — tokens remain continuous
- Joint end-to-end training (no separate pre-training)

## Acceptance Criteria

- Given `[B, T, N_MELS]` input, `MelTokenizer` outputs `[B, T//C, D_MODEL]`
- Given `[B, T_tok, D_MODEL]` input, `MelDetokenizer` outputs `[B, T_tok*C, N_MELS]`
- Given two inputs identical up to frame `t`, tokenizer outputs match up to `t//C`
- Given two token seqs identical up to position `k`, detokenizer outputs match up to `k*C+(C-1)`
- `MelGenerator` forward pass: `[B, 512, 100]` → `[B, 512, 100]`
- Training completes 1 epoch without error, loss decreases
- `generate(seed, 500)` returns `[500, N_MELS]`
- `grep -r "GROUPED_MEL_DIM\|REDUCTION_FACTOR\|EFFECTIVE_SEQ_LEN" src/music_generation/` returns zero matches
- Model works with `TOKEN_COMPRESSION_RATIO=8` (3 stride-2 layers)

## Files to Modify

| File | Change |
|------|--------|
| `src/music_generation/config.py` | Remove old constants, add tokenizer config |
| `src/music_generation/model.py` | Add `MelTokenizer`, `MelDetokenizer`, rewrite `MelGenerator` |
| `src/music_generation/train.py` | Simplify loss, token-space scheduled sampling |
| `src/music_generation/dataset.py` | Yield raw mel, remove grouped-frame logic |
