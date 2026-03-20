# Scratchpad — Learned Continuous Mel Tokens

## 2026-03-19: Initial Analysis

### Understanding
Replacing fixed REDUCTION_FACTOR frame grouping with learned MelTokenizer/MelDetokenizer.
Core change: instead of manually reshaping [T,80]->[T/R,R*80], a causal conv encoder learns
the compression [T,N_MELS]->[T/C,D_MODEL] and a decoder reconstructs [T/C,D_MODEL]->[T,N_MELS].

### Files to modify (4 main):
1. config.py - Remove REDUCTION_FACTOR/GROUPED_MEL_DIM/EFFECTIVE_SEQ_LEN, add TOKEN_COMPRESSION_RATIO
2. model.py - Add MelTokenizer, MelDetokenizer, rewrite MelGenerator
3. dataset.py - Yield raw [SEQ_LEN, N_MELS] pairs, no grouped frames
4. train.py - Direct MSE loss, token-space scheduled sampling

### Key constraints:
- Keep TransformerBlock, CausalConvBlock, CausalConv1D, LocalWindowAttention unchanged
- Keep LatentEncoder architecture unchanged (docstring only)
- Keep inference.py API unchanged
- Strict causality everywhere

### Plan (following spec's 7-step plan):
1. Update config.py (remove old, add new constants)
2. Add MelTokenizer + MelDetokenizer to model.py
3. Rewrite MelGenerator to use tokenizer/detokenizer
4. Update dataset.py to yield raw mel
5. Update train.py (simplify loss, token-space sampling)
6. Rewrite generate() for token-space autoregression
7. Final cleanup + validation

### Starting with Step 1: config.py
Simplest, no dependencies, enables all downstream work.

## Step 1 Complete
- Removed REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN from config.py
- Added TOKEN_COMPRESSION_RATIO=4, TOKEN_NUM_CONV_LAYERS=2, TOKEN_SEQ_LEN=128
- D_MODEL updated to 256 (was 128, matches design spec)
- MAX_CONTEXT_FRAMES removed (superseded)
- Committed: dccbc9c
- Next: Steps 2-6 are now unblocked. Step 2 (MelTokenizer/MelDetokenizer) is next.

## Step 2 Complete
- Added MelTokenizer and MelDetokenizer to model.py
- MelTokenizer: N stride-2 causal conv blocks + Dense(D_MODEL)
- MelDetokenizer: N (tf.repeat 2x + CausalConv1D) blocks + Dense(N_MELS)
- Both verified: correct shapes for C=4 and C=8, strict causality passes
- Added backward-compat aliases in config.py so existing code doesn't break
- Pre-existing test failures: test_dataset (GROUPED_MEL_DIM vs N_MELS), test_causality (D_MODEL=256 mismatch) — both will be fixed in later steps
- Committed: 3ef7cf0
- Next: Step 3 (rewrite MelGenerator to use tokenizer/detokenizer + add forward_from_tokens)

## Step 3 Complete
- Rewrote MelGenerator: tokenizer replaces input_proj, detokenizer replaces output_proj
- Added forward_from_tokens() for scheduled sampling pass 2
- Rewrote generate() for token-space autoregression (tokenize seed, loop in token space, detokenize at end)
- Updated LatentEncoder docstring
- Removed REDUCTION_FACTOR/GROUPED_MEL_DIM/EFFECTIVE_SEQ_LEN imports from model.py
- Updated 3 test files: test_model.py, test_causality_and_windows.py, test_dilated_convolutions.py
- Key finding: causality tests need fixed z (z=None causes random sampling each call)
- All 22 model-related tests pass
- Pre-existing failures: test_dataset (GROUPED_MEL_DIM), test_train/scheduled_sampling (SimpleMelGenerator mock lacks z kwarg)
- Committed: c265ee1
- Next: Step 4 (update dataset.py) or Step 5 (update train.py)

## Step 4: Update dataset.py (in progress)

### What needs to change:
- Remove imports of REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN
- Import TOKEN_COMPRESSION_RATIO instead
- In _generator(): truncate to TOKEN_COMPRESSION_RATIO-divisible, remove tf.reshape to grouped frames, work with raw mel [T, N_MELS]
- Skip if mel.shape[0] <= SEQ_LEN (not EFFECTIVE_SEQ_LEN)
- Stride = SEQ_LEN // 2, yield [SEQ_LEN, N_MELS] pairs
- Update output_signature to (SEQ_LEN, N_MELS)

### Test file:
- test_dataset.py already uses N_MELS not GROUPED_MEL_DIM — should pass after changes

## Step 4 Complete
- Removed REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN imports from dataset.py
- Replaced with TOKEN_COMPRESSION_RATIO, N_MELS, SEQ_LEN
- Generator now yields raw [SEQ_LEN, N_MELS] pairs (no tf.reshape to grouped frames)
- output_signature updated to (SEQ_LEN, N_MELS)
- All 24 dataset+model tests pass
- Pre-existing failures in train/vocoder/body_point tests unchanged (will be fixed in Steps 5-7)
- Committed: 512f96c
- Next: Step 5 (update train.py) — simplify loss to direct MSE, token-space scheduled sampling

## Step 5 Complete
- Replaced GROUPED_MEL_DIM with N_MELS in train.py imports
- Simplified _pure_teacher_forcing: direct MSE `preds[:, :-1] - y[:, 1:]`, no reshape
- Rewrote _parallel_scheduled_sampling for token-space mixing:
  - Pass 1: tokenize ground truth and predictions, mix tokens with shifted pred_tokens
  - Pass 2: detokenize mixed tokens for latent encoder, forward_from_tokens for training
- Updated train() to use steps_per_epoch from create_dataset() tuple
- Updated shape assertion from GROUPED_MEL_DIM to N_MELS
- Updated 3 test files: mocks now have tokenizer/detokenizer/forward_from_tokens
- Key finding: mock call() must use tokenizer→detokenizer path for graph mode (model.fit)
  to ensure all variables are registered with optimizer. Passthrough mock causes
  "Unknown variable" error in tf.cond when parallel branch introduces new variables.
- All 67 model+train+dataset tests pass (0 new failures)
- Committed: d5842e5
- Next: Step 6 (rewrite generate() for token-space autoregression) — already done in Step 3!
  Step 7 (final cleanup) is the remaining work.

## Step 6 & 7 Complete (Final)

### Step 6: Already done in Step 3
- generate() was rewritten for token-space autoregression in Step 3
- Verified: test_generate_shape passes, generate(seed, 500) returns [500, N_MELS]
- Closed task without changes needed

### Step 7: Final cleanup
- Removed backward-compat aliases (REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN) from config.py
- Fixed unused imports in tests/test_dilated_convolutions.py
- Deleted obsolete tests: test_reduction_factor.py, test_inference_r4.py, test_inference_r4_logic.py
- Verified: `grep -r "GROUPED_MEL_DIM|REDUCTION_FACTOR|EFFECTIVE_SEQ_LEN" src/music_generation/` returns zero matches
- Verified: TOKEN_COMPRESSION_RATIO=8 works (3 stride-2 layers)
- Verified: Forward pass [B, 512, 100] -> [B, 512, 100]
- Verified: generate(seed, 500) returns [500, N_MELS]
- All 46 core tests pass, 163 total pass (31 failures are pre-existing/unrelated)
- Committed: d2c4452

### All 7 steps complete. Objective satisfied.
