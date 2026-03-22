# Scratchpad — Pose-Conditioned Training

## 2026-03-22 Iteration 1

### Orientation
- Step 1 (temporal pose encoder + config) already done (closed task-1774163055-48b6)
- Step 2 (TransformerBlock cross-attn) already implemented in layers.py — closed task-1774163064-8640
- Now implementing Step 3: MelGenerator pose integration in model.py

### Step 2 — Already done
- Cross-attention code was already in layers.py from previous iteration
- Committed serialization decorators + cross-attn as 600e672
- Closed task

### Step 3 — Done
- Added pose=None to all MelGenerator methods
- Pose path: encoder -> proj -> stride_conv -> cross_attn_kv in transformer2/3
- Verified: alpha=0 identical to no-pose, alpha=1 differs
- All 23 model tests + 42 training tests pass
- Committed as 787b993

### Next: Step 4 (pose dataset), Step 5-7 (train_pose.py), Step 8 (alignment losses), Step 9 (checkpoint auto-detect)

### Step 4 — Done
- Added `_pose_generator` to MusicNetDataset + `create_pose_dataset()` function
- Yields (mel, mel, pose) triples, skips files without pose data (warning)
- Aligns pose/mel lengths via min(mel_len, pose_len)
- Smoke tested: shapes (B, 512, 100), (B, 512, 100), (B, 512, 85) correct
- All existing tests pass (19 core, 200+ total minus pre-existing vocoder failures)
- Committed as 390dc92

### Next: Step 5-7 (train_pose.py), Step 8 (alignment losses), Step 9 (checkpoint auto-detect)

### Step 5-7 — Done
- Created `src/music_generation/train_pose.py` with:
  - `cosine_alpha_schedule()`: smooth ramp, verified 0→0.15→0.3 at steps 0/50k/100k
  - `apply_conditioning_dropout()`: per-sample pose zeroing
  - `PoseConditionedTraining`: full wrapper with alpha scheduling, cond dropout, TF schedule
  - `load_checkpoint_with_autodetect()`: Phase 1/2/3 auto-detection + partial weight matching
  - `_apply_freezing()`: freeze encoder and/or early layers
  - CLI with all required args
- Smoke tested: alpha schedule, dropout, output shapes, freezing all correct
- 185 existing tests pass, 3 pre-existing failures (vocos, griffin-lim, scheduled_sampling config)
- Committed as 5828ec4

### Next: Step 8 (alignment losses), Step 9 (checkpoint auto-detect — already included in train_pose.py)

### Step 8 — Alignment Losses (in progress)
- Task: task-1774163064-dc62
- Need: PoseAudioAlignmentLoss and OnsetAlignmentLoss in losses.py
- Implemented but NOT wired into training
- Acceptance criteria: returns scalar, gradient flows, constant pose → low loss
- Existing losses.py has vocoder losses (STFT, adversarial, feature matching, sub-band)
- Will add two new classes at the end of the file

### Step 8 — Done
- PoseAudioAlignmentLoss: normalized L1 between pose velocity magnitude and spectral flux
- OnsetAlignmentLoss: normalized L1 between pose acceleration magnitude and onset strength
- Both return scalar, gradients flow, constant inputs → 0.0 loss
- 190 existing tests pass (3 pre-existing failures: vocos, griffin-lim, scheduled_sampling config)
- Committed as e5f6b92

### Next: Step 9 (checkpoint auto-detection) — task-1774163064-ecb5

### Step 9 — Checkpoint Auto-Detection (in progress)
- Task: task-1774163064-ecb5
- `load_checkpoint_with_autodetect` already implemented in train_pose.py (Step 5-7)
- `_copy_matching_weights` helper also exists
- Need: test that verifies the acceptance criterion:
  "Given Phase 1 checkpoint, when train_pose.py loads it, then new layers auto-initialize and training starts"
- Plan: Write test that saves Phase 1 (audio-only) MelGeneratorTraining checkpoint, loads into PoseConditionedTraining, verifies audio weights transferred and pose layers fresh, then runs a training step

### Step 9 — Done
- Fixed `_copy_matching_weights` to use path-based matching (was using v.name='kernel'/'bias')
- Added `_normalize_var_path` helper for cross-wrapper weight matching
- Added temporary model fallback for .weights.h5 files
- Added serialization decorators to train.py classes
- 7 new tests, all pass. 192 existing tests pass (1 pre-existing failure: scheduled_sampling config)
- Committed as d441940

### All Steps Complete
- Steps 1-9 all done. Checking for remaining tasks.
