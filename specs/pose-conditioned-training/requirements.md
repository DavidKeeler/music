# Requirements

## Questions & Answers

### Q1: Pose data source during training
The current BodyPointModule processes live video frames one at a time (detect → normalize → buffer → encode). For training, what's the pose data source? Specifically:

- Will you have pre-extracted pose sequences stored alongside audio files (e.g., `.npy` files with keypoint sequences)?
- Or will training involve processing video frames on-the-fly through the full BodyPointModule pipeline?

This determines whether the training dataset needs a video pipeline or just loads pre-computed pose tensors.

**A1:** Pre-extracted pose sequences, already aligned to audio. Keypoint detection runs as a preprocessing step before training. Training loads pre-computed pose tensors (no video pipeline needed).

### Q2: Pose data format and granularity
What shape/format will the pre-extracted pose files be in? Specifically:

- Raw keypoints `[T_frames, 17, 2]` (x, y per joint) + confidence `[T_frames, 17]`?
- Or the full FeatureBuilder output `[T_frames, 85]` (x, y, dx, dy, conf per joint)?
- Or already encoded through the pose encoder as embeddings `[T_frames, 128]`?

The knowledge base says the pose encoder needs to be refactored to output a temporal sequence `[B, T_pose, embedding_dim]` instead of a single vector. If we store raw keypoints or features, the encoder runs during training and gets gradients. If we store pre-encoded embeddings, the encoder is frozen.

**A2:** Both modes needed. Store raw features (e.g., `[T_frames, 85]`) so the pose encoder can run during training. Support two modes:
- **Frozen encoder**: encoder weights fixed, no gradients flow through pose encoder
- **Trainable encoder**: encoder gets gradients during joint training

This implies storing the FeatureBuilder output `[T_frames, 85]` (or raw keypoints) so the encoder is always in the loop, with a flag to freeze/unfreeze it.

### Q3: Temporal alignment strategy
The knowledge base recommends downsampling pose to token rate using Conv1D with stride=TOKEN_COMPRESSION_RATIO (currently 4). The audio token sequence is `[B, T/4, D_MODEL]` after the MelTokenizer.

The pose sequence will be at some frame rate from the video (e.g., 30fps). The audio mel frames are at ~93.75 fps (24kHz / 256 hop). These won't match 1:1.

How should we handle this alignment?
- Option A: During preprocessing, resample pose to match mel frame rate (~93.75 fps), then let the model's Conv1D downsample to token rate
- Option B: Store pose at its native video frame rate and handle alignment in the model/dataset pipeline
- Option C: Something else?

**A3:** Option A — resample pose to mel frame rate during preprocessing. The model's Conv1D downsamples to token rate. This keeps the model simple and alignment deterministic.

### Q4: Pose encoder architecture for temporal output
The current `build_pose_encoder` takes `[B, buffer_size, 85]` and outputs `[B, embedding_dim]` (single vector via last-timestep extraction). The knowledge base says to refactor it to output `[B, T_pose, embedding_dim]`.

Since we're resampling pose to mel frame rate and then downsampling to token rate in the model, the encoder needs to process `[B, T_pose, 85]` → `[B, T_pose, embedding_dim]`. The simplest approach:

- Apply the per-frame MLP (TimeDistributed) to get `[B, T_pose, embedding_dim]`
- Keep the temporal Conv1D but remove the last-timestep extraction so it outputs the full sequence
- Then `pose_proj` (Dense to D_MODEL) + Conv1D(stride=TOKEN_COMPRESSION_RATIO) brings it to `[B, T_tok, D_MODEL]` for cross-attention

Does this approach work, or do you want a different encoder architecture for the temporal version?

**A4:** Yes, this approach works. Refactor pose encoder to output full temporal sequence. Downsampling to token rate happens in the model via Conv1D(stride=TOKEN_COMPRESSION_RATIO).

### Q5: Alpha scheduling specifics
The knowledge base describes alpha ramping across phases:
- Phase 2: alpha 0.0 → 0.3 over 50k–200k steps
- Phase 3: alpha → 1.0

How should alpha be scheduled?
- Linear ramp within each phase?
- Cosine?
- Should the phase transitions be manual (separate script invocations with different flags) or automatic (single run with step-based phase switching)?

**A5:** Separate script invocations per phase (manual transitions). Cosine schedule for alpha ramping within each phase.

### Q6: Conditioning dropout implementation
The knowledge base recommends dropping pose input with probability 0.1–0.2 during Phase 3. When pose is dropped:

- Replace with zeros `[B, T_tok, D_MODEL]`? (cross-attention sees all-zero keys/values)
- Or skip the cross-attention entirely for that sample? (equivalent to alpha=0 for that sample)

And should dropout be per-batch or per-sample within a batch?

**A6:** Skip cross-attention entirely (alpha=0 for that sample). Per-sample within a batch — some samples in a batch get pose conditioning, others don't.

### Q7: Checkpoint loading between phases
Phase 2 loads a Phase 1 (audio-only) checkpoint. Phase 3 loads a Phase 2 checkpoint. The Phase 1 model won't have cross-attention layers or pose encoder weights.

Should the script:
- Auto-detect missing keys and initialize new layers (cross-attention, pose_proj, pose encoder) from scratch when loading a Phase 1 checkpoint?
- Or require the user to specify `--phase` explicitly so the script knows what to expect?

**A7:** Auto-detect missing keys. When loading a Phase 1 checkpoint, new layers (cross-attention, pose_proj, pose encoder) get initialized from scratch automatically. No explicit `--phase` flag needed.

### Q8: Pose data file convention
The pre-extracted pose sequences will be stored alongside audio. What file naming/location convention?

- Same directory as audio, matching filenames? (e.g., `song1.wav` → `song1_pose.npy`)
- Separate pose directory with matching filenames? (e.g., `pose_dir/song1.npy`)
- Something else?

**A8:** Same directory as audio, matching filenames. e.g., `song1.wav` → `song1_pose.npy`. Shape: `[T_frames, 85]` (already resampled to mel frame rate).

### Q9: Optional alignment loss
The knowledge base recommends an optional auxiliary loss encouraging alignment between pose velocity and spectral flux/onset strength. Do you want this included in the initial implementation, or defer it to a later iteration?

**A9:** Implement the alignment loss functions (pose velocity ↔ spectral flux/onset strength) but don't wire them into the training loop yet. Available for future use, not active by default.

### Q10: Scope of model.py changes
The knowledge base says to add cross-attention to MelGenerator. The existing `train.py` and all its tests depend on the current MelGenerator API (`call(mel, z=None, training=False)`).

Two approaches:
- **Option A**: Make pose optional in MelGenerator — add `pose=None` parameter, cross-attention layers only activate when pose is provided. Existing audio-only code works unchanged.
- **Option B**: Create a separate `PoseConditionedMelGenerator` subclass that extends MelGenerator with cross-attention.

Option A is simpler and matches the knowledge base design (pose is optional conditioning). Option B keeps the classes separate but duplicates code. Which do you prefer?

**A10:** Option A — make pose optional in MelGenerator. Add `pose=None` to `call()`, `forward_tokens()`, `forward_from_tokens()`, and `generate()`. Cross-attention layers only activate when pose is provided. Existing audio-only code and tests work unchanged.

### Q11: Are we done with requirements?
I think we've covered the key decisions:
- Pre-extracted pose data `[T, 85]`, same dir as audio, resampled to mel rate
- Pose encoder refactored for temporal output, supports frozen/trainable modes
- Cross-attention in transformer2/transformer3 with gated alpha (cosine schedule)
- Per-sample conditioning dropout (skip cross-attention)
- Separate script invocations per phase, auto-detect checkpoint type
- Optional pose=None in MelGenerator API (backward compatible)
- Alignment losses implemented but not active

Is there anything else you want to clarify, or are we good to move on?

**A11:** Requirements complete. Moving to research/design.

