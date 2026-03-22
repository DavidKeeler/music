# Pose-Conditioned Training — Implementation Plan

- [ ] Step 1: Temporal Pose Encoder
- [ ] Step 2: TransformerBlock Cross-Attention
- [ ] Step 3: MelGenerator Pose Integration
- [ ] Step 4: Pose Dataset Pipeline
- [ ] Step 5: Alpha Scheduling & Conditioning Dropout
- [ ] Step 6: PoseConditionedTraining Wrapper
- [ ] Step 7: Training Script (train_pose.py)
- [ ] Step 8: Alignment Losses
- [ ] Step 9: Checkpoint Auto-Detection
- [ ] Step 10: End-to-End Integration Test

---

## Step 1: Temporal Pose Encoder

**Objective:** Add `build_temporal_pose_encoder` to `src/body_point_module/encoder.py` that outputs a full temporal sequence instead of a single vector.

**Implementation guidance:**
- Add `build_temporal_pose_encoder(input_dim=85, hidden_dim=256, output_dim=128)` to `encoder.py`
- Uses `tf.keras.Input(shape=(None, input_dim))` for variable-length sequences
- TimeDistributed MLP (reuse `build_mlp_encoder`) → causal Conv1D → output full sequence (no last-timestep extraction)
- Add `POSE_EMBEDDING_DIM = 128` to `src/music_generation/config.py`

**Files modified:**
- `src/body_point_module/encoder.py`
- `src/music_generation/config.py`

**Test requirements:**
- Output shape `[B, T, 128]` for input `[B, T, 85]` with various T values
- Gradient flow through encoder
- Variable sequence length support (T=100, T=512)

**Integration notes:** No downstream dependencies yet. Existing `build_pose_encoder` unchanged.

**Demo:** Instantiate encoder, pass random `[2, 512, 85]` tensor, verify output is `[2, 512, 128]`.

---

## Step 2: TransformerBlock Cross-Attention

**Objective:** Add optional cross-attention to `TransformerBlock` in `src/music_generation/layers.py`.

**Implementation guidance:**
- Add `enable_cross_attn=False` parameter to `__init__`
- When True, create `self.cross_attn` (MultiHeadAttention) and `self.cross_attn_norm` (LayerNormalization)
- Modify `call(self, x, cross_attn_kv=None, cross_attn_alpha=None)`:
  - Self-attention (unchanged)
  - If `enable_cross_attn` and `cross_attn_kv is not None`: `x = x + alpha * cross_attn(query=norm(x), key=kv, value=kv)`
  - FFN (unchanged)
- Serialize `enable_cross_attn` in `get_config()` (already `@register_keras_serializable`)

**Files modified:**
- `src/music_generation/layers.py`

**Test requirements:**
- `TransformerBlock(enable_cross_attn=False)`: calling with no extra args produces identical output to current
- `TransformerBlock(enable_cross_attn=True)`: calling with `cross_attn_kv=None` produces same output as `enable_cross_attn=False`
- Calling with `cross_attn_kv` tensor and `alpha=0.0` produces same output as without
- Calling with `cross_attn_kv` tensor and `alpha=1.0` produces different output
- Output shape unchanged `[B, T, D_MODEL]`

**Integration notes:** Existing code calls `self.transformer1(x)` etc. with no extra args — all pass through unchanged since `cross_attn_kv` defaults to None.

**Demo:** Create block with `enable_cross_attn=True`, pass x `[2, 128, 256]` and kv `[2, 128, 256]`, verify output shape and alpha gating.

---

## Step 3: MelGenerator Pose Integration

**Objective:** Add pose conditioning path to MelGenerator. Backward compatible — `pose=None` default everywhere.

**Implementation guidance:**
- Add to `__init__`:
  - `self.pose_encoder = build_temporal_pose_encoder()`
  - `self.pose_proj = Dense(D_MODEL)`
  - `self.pose_stride_conv = Conv1D(D_MODEL, kernel_size=TOKEN_COMPRESSION_RATIO, strides=TOKEN_COMPRESSION_RATIO, padding='same')`
  - `self.pose_alpha = tf.Variable(0.0, trainable=False)`
- Change transformer2/transformer3 construction to `enable_cross_attn=True`
- Add `pose=None` to `call()`, `forward_tokens()`, `forward_from_tokens()`, `generate()`
- In `call()`: if pose is not None, encode → project → downsample to get `pose_embedded [B, T_tok, D_MODEL]`
- In `forward_tokens()`: pass `cross_attn_kv=pose_embedded, cross_attn_alpha=self.pose_alpha` to transformer2 and transformer3
- transformer1 called as before (no cross-attn args)

**Files modified:**
- `src/music_generation/model.py`

**Test requirements:**
- `model(mel)` with no pose: output identical to current model (regression test)
- `model(mel, pose=pose_tensor)` with `pose_alpha=0.0`: output equals no-pose output
- `model(mel, pose=pose_tensor)` with `pose_alpha=1.0`: output differs
- Output shape `[B, T, N_MELS]` in all cases
- `forward_tokens` and `forward_from_tokens` work with and without pose
- `generate()` works with and without pose

**Integration notes:** Existing `train.py` calls `base_model(x, z=z, training=True)` — no pose arg, works unchanged. All existing tests pass without modification.

**Demo:** Load model, run forward pass with and without pose, compare outputs at alpha=0 and alpha=1.

---

## Step 4: Pose Dataset Pipeline

**Objective:** Add `create_pose_dataset` to `src/music_generation/dataset.py` that yields `(mel, mel, pose)` triples.

**Implementation guidance:**
- Add `create_pose_dataset(data_dir, cache_dir, batch_size, shuffle=True)` function
- Extend `MusicNetDataset` or create parallel class that:
  - For each `.wav`, looks for `{stem}_pose.npy` in same directory
  - Loads pose `[T_frames, 85]`, slices to SEQ_LEN matching mel slicing
  - Skips files without matching pose file (log warning)
  - Yields `(input_mel, target_mel, pose_features)` triples
- Output signature: `(SEQ_LEN, N_MELS)`, `(SEQ_LEN, N_MELS)`, `(SEQ_LEN, 85)`
- Add `POSE_FEATURE_DIM = 85` to config

**Files modified:**
- `src/music_generation/dataset.py`
- `src/music_generation/config.py`

**Test requirements:**
- With matching pose files: yields correct shapes
- Without pose file: skips with warning, doesn't crash
- Pose sequence length mismatch: truncate/pad to match mel

**Integration notes:** Existing `create_dataset` unchanged. `create_pose_dataset` is a separate function.

**Demo:** Create temp dir with a .wav and matching _pose.npy, call `create_pose_dataset`, iterate one batch, print shapes.

---

## Step 5: Alpha Scheduling & Conditioning Dropout

**Objective:** Implement cosine alpha schedule and per-sample conditioning dropout utilities.

**Implementation guidance:**
- Add `cosine_alpha_schedule(step, alpha_start, alpha_end, total_steps)` to `train_pose.py` (or a shared utils location)
  - Returns `alpha_start + (alpha_end - alpha_start) * 0.5 * (1 - cos(pi * progress))`
- Add conditioning dropout helper:
  - Takes `pose_embedded [B, T, D]` and `dropout_rate` float
  - Generates per-sample mask `[B]` from `tf.random.uniform`
  - Returns `pose_embedded * mask[:, None, None]` and the mask `[B]` for metric logging
  - When mask is 0 for a sample, cross-attention contribution is zero (pose_embedded is all zeros → alpha * cross_attn(q, zeros, zeros) ≈ 0)

**Files modified:**
- `src/music_generation/train_pose.py` (new file, started here)

**Test requirements:**
- Cosine schedule: step=0 → alpha_start, step=total_steps → alpha_end, monotonic
- Dropout: rate=0.0 → no samples masked, rate=1.0 → all masked
- Dropout mask shape and broadcasting correct

**Integration notes:** These are pure functions, no model dependency.

**Demo:** Plot cosine schedule over 100k steps. Run dropout on batch of 16, count masked samples.

---

## Step 6: PoseConditionedTraining Wrapper

**Objective:** Create `PoseConditionedTraining` model wrapper in `train_pose.py` that handles pose-conditioned training with alpha scheduling and dropout.

**Implementation guidance:**
- Extends `tf.keras.Model`, similar pattern to `MelGeneratorTraining`
- `__init__` params: `base_model, alpha_start, alpha_end, alpha_steps, cond_dropout_rate, initial_tf_ratio, min_tf_ratio, decay_k, warmup_steps, kl_beta`
- Contains `LatentEncoder` (same as MelGeneratorTraining)
- `train_step(data)`:
  - Unpack `(x, y, pose)`
  - Update alpha via cosine schedule, assign to `base_model.pose_alpha`
  - Update tf_ratio (same exponential schedule as existing)
  - Apply conditioning dropout to pose
  - Dispatch to pure teacher forcing or scheduled sampling (same tf.cond pattern)
  - Both paths pass `pose=pose` to base_model
- Metrics: mel_loss, kl_loss, alpha, tf_ratio, cond_dropout_count
- `_pure_teacher_forcing(x, y, pose)` and `_parallel_scheduled_sampling(x, y, pose)` mirror existing but pass pose through

**Files modified:**
- `src/music_generation/train_pose.py`

**Test requirements:**
- Single training step with pose data runs without error
- Alpha updates correctly each step
- Conditioning dropout masks applied
- tf_ratio schedule works same as existing
- Gradients flow through pose encoder when not frozen

**Integration notes:** Depends on Steps 1-5. Reuses `exponential_tf_schedule` and `WarmupCosineSchedule` from `train.py`.

**Demo:** Create wrapper with small model, run 10 training steps, print alpha and loss progression.

---

## Step 7: Training Script (train_pose.py)

**Objective:** Complete `train_pose.py` with CLI, data loading, model construction, and training loop.

**Implementation guidance:**
- `main()` with argparse: `--data_dir, --cache_dir, --checkpoint_dir, --resume, --batch_size, --epochs, --lr, --alpha_start, --alpha_end, --alpha_steps, --cond_dropout, --freeze_encoder, --freeze_early`
- `train_pose()` function:
  - `configure_memory()` and `check_batch_size()` (reuse from train.py)
  - Load dataset via `create_pose_dataset`
  - Construct `MelGenerator` + `PoseConditionedTraining` wrapper
  - Load checkpoint if `--resume` (auto-detect, see Step 9)
  - Apply freezing: if `--freeze_encoder`, set `base_model.pose_encoder.trainable = False`; if `--freeze_early`, freeze tokenizer + conv_layers + transformer1
  - Compile with `WarmupCosineSchedule` optimizer
  - `model.fit()` with ModelCheckpoint and TensorBoard callbacks
- Checkpoint saved as `.keras` to `--checkpoint_dir`

**Files modified:**
- `src/music_generation/train_pose.py`

**Test requirements:**
- Script runs with `--help` without error
- Freeze flags correctly set `.trainable = False` on target layers
- Training completes one epoch on small synthetic data

**Integration notes:** Depends on Steps 1-6. Imports from `train.py` (schedule classes, utility functions).

**Demo:** Run Phase 2 command from design doc on small synthetic dataset, verify checkpoint saved.

---

## Step 8: Alignment Losses

**Objective:** Implement `PoseAudioAlignmentLoss` and `OnsetAlignmentLoss` in `losses.py`. Not wired into training.

**Implementation guidance:**
- `PoseAudioAlignmentLoss(tf.keras.layers.Layer)`:
  - `call(pose_features, mel_spectrogram)` → scalar
  - Compute pose velocity magnitude: diff along time axis of pose positions, L2 norm per frame
  - Compute spectral flux: L1 diff of mel along time axis, sum across frequency
  - L1 loss between normalized velocity and normalized spectral flux
- `OnsetAlignmentLoss(tf.keras.layers.Layer)`:
  - `call(pose_features, mel_spectrogram)` → scalar
  - Pose acceleration: second difference of pose positions
  - Onset strength: half-wave rectified spectral flux
  - Cosine similarity loss

**Files modified:**
- `src/music_generation/losses.py`

**Test requirements:**
- Both return scalar float32
- Gradient flows through both inputs
- Constant pose → low alignment loss
- Shape handling for various sequence lengths

**Integration notes:** Standalone. Can be wired into training later by adding to loss computation with a weight coefficient.

**Demo:** Compute both losses on random data, print values.

---

## Step 9: Checkpoint Auto-Detection

**Objective:** Implement partial checkpoint loading that auto-initializes missing layers when loading a Phase 1 checkpoint into the pose-conditioned model.

**Implementation guidance:**
- Add `load_checkpoint_auto(model, checkpoint_path)` to `train_pose.py`
- Strategy:
  1. Build model with a dummy forward pass to initialize all weights
  2. If `.keras` file: load saved model, iterate saved weights by name, match to new model weights by name, assign matches, log skipped
  3. If `.h5` / `.weights.h5`: use `model.load_weights(path, skip_mismatch=True)` (Keras built-in) or manual name matching
  4. Log which layers were loaded vs initialized fresh
- Handle both Phase 1 → Phase 2 (many missing) and Phase 2 → Phase 3 (all present) cases

**Files modified:**
- `src/music_generation/train_pose.py`

**Test requirements:**
- Load Phase 1 checkpoint (no pose layers): training starts, new layers randomly initialized
- Load Phase 2 checkpoint (has pose layers): all weights restored
- Mismatched shapes logged and skipped gracefully

**Integration notes:** Called in `train_pose()` before `model.fit()`. Depends on Step 7 structure.

**Demo:** Save an audio-only model checkpoint, load into pose-conditioned model, print loaded vs fresh layer counts.

---

## Step 10: End-to-End Integration Test

**Objective:** Verify the full pipeline works: dataset → model → training → checkpoint → resume.

**Implementation guidance:**
- Create synthetic test data: small .wav files + matching _pose.npy files
- Run Phase 2 training for 5 steps with small model
- Verify: loss decreases, alpha increases, checkpoint saved
- Load checkpoint, run 5 more steps (Phase 3 config)
- Verify: alpha continues from loaded value, no crashes
- Run inference with pose conditioning on trained model

**Files modified:**
- None (manual or test script)

**Test requirements:**
- Full Phase 2 → Phase 3 pipeline completes
- Checkpoint round-trip preserves all weights and optimizer state
- Inference with pose produces different output than without (alpha > 0)
- Audio-only inference still works (pose=None)

**Integration notes:** This validates all previous steps together.

**Demo:** Run the full Phase 2 → Phase 3 sequence on synthetic data, generate audio with and without pose, compare spectrograms.
