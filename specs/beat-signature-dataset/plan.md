# Beat Signature Dataset Generator — Implementation Plan

## Checklist

- [ ] Step 1: BeatAnalyzer — beat/downbeat/meter detection wrapper
- [ ] Step 2: ConductingPattern — canonical waypoints + spline interpolation
- [ ] Step 3: SkeletonBuilder — wrist trajectory → full 17-joint skeleton
- [ ] Step 4: PatternGenerator — orchestrate pattern + skeleton + variation
- [ ] Step 5: DatasetWriter — TFRecord serialization
- [ ] Step 6: CLI entry point — end-to-end pipeline
- [ ] Step 7: Stretch time signatures (6/8, 5/4, 7/8)

---

## Step 1: BeatAnalyzer

**Objective:** Wrap BeatNet to extract beat positions, downbeats, tempo, and time signature from a .wav file.

**Implementation guidance:**
- Create `src/music_generation/beat_analyzer.py`
- `BeatAnalyzer.analyze(audio_path, time_signature_override=None) -> BeatInfo`
- Use BeatNet in offline mode with DBN inference
- Parse BeatNet output (numpy array of `[beat_time, beat_number]`) into `BeatInfo` dataclass
- Infer time signature from beat numbering pattern (max beat number before reset = beats per measure)
- If `time_signature_override` is set, use it instead of detected meter
- Handle errors: return None and log warning if BeatNet fails

**Test requirements:**
- Mock BeatNet output, verify BeatInfo fields are correctly populated
- Verify time signature override works
- Verify graceful handling of BeatNet failure

**Integration notes:** Standalone module, no dependencies on other new components.

**Demo:** Run on a single .wav file, print detected beats, tempo, and time signature.

---

## Step 2: ConductingPattern

**Objective:** Define canonical conducting waypoints and interpolate smooth wrist trajectories.

**Implementation guidance:**
- Create `src/music_generation/conducting_patterns.py`
- `ConductingPattern.PATTERNS` dict mapping `(numerator, denominator)` → list of (x, y) ictus waypoints
- Start with 2/4, 3/4, 4/4 only
- `interpolate_trajectory(waypoints, beat_times, fps, seed=None) -> np.ndarray`
  - For each measure, map waypoints to beat times
  - Use `scipy.interpolate.CubicSpline` (periodic boundary) to get smooth curve at target fps
  - Apply variation: scale amplitude (±30%), jitter ictus timing (±8%), add Gaussian noise to trajectory
  - Return `[num_frames, 2]` array of (x, y) wrist positions

**Test requirements:**
- Verify output frame count matches expected `duration * fps`
- Verify ictus frames are near beat times (within 1-2 frames)
- Verify different seeds produce different trajectories
- Verify same seed produces identical output

**Integration notes:** Pure numpy/scipy, no TF dependency.

**Demo:** Generate a wrist trajectory for a 4/4 pattern at 120 BPM, plot with matplotlib.

---

## Step 3: SkeletonBuilder

**Objective:** Expand a wrist trajectory into a full 17-joint MoveNet-format skeleton.

**Implementation guidance:**
- Create `src/music_generation/skeleton_builder.py`
- `SkeletonBuilder.build_sequence(right_wrist_xy, beat_info, fps) -> np.ndarray`
- Define rest pose for all 17 joints in normalized coordinates
- Right wrist (10): from input trajectory
- Left wrist (9): mirror right wrist (negate x)
- Elbows (7, 8): 2-link IK from shoulder to wrist with fixed upper arm length (~0.6 shoulder widths)
- Shoulders (5, 6): base position + small vertical shift proportional to wrist height
- Nose (0): center + small downward y-displacement on downbeat frames (lerp in/out over ~4 frames)
- Eyes (1, 2), Ears (3, 4): fixed offset from nose
- Hips (11, 12): fixed + tiny noise
- Knees (13, 14), Ankles (15, 16): fixed
- Confidence: 1.0 for all
- Output format: `[num_frames, 17, 3]` in (y, x, confidence) order

**Test requirements:**
- Verify output shape [N, 17, 3]
- Verify left/right arm symmetry (mirrored x)
- Verify elbow is between shoulder and wrist
- Verify lower body joints are approximately static

**Integration notes:** Depends on knowing downbeat frame indices from BeatInfo.

**Demo:** Visualize a single frame as a stick figure.

---

## Step 4: PatternGenerator

**Objective:** Orchestrate the full pipeline from BeatInfo → keypoint sequence.

**Implementation guidance:**
- Create `src/music_generation/pattern_generator.py` (or add to `conducting_patterns.py`)
- `PatternGenerator.generate(beat_info, fps, duration, seed=None) -> np.ndarray`
- Steps:
  1. Look up ConductingPattern for beat_info.time_signature
  2. Slice beat_info to requested duration
  3. Call `interpolate_trajectory()` → right wrist trajectory
  4. Call `SkeletonBuilder.build_sequence()` → full skeleton
  5. Return `[num_frames, 17, 3]`

**Test requirements:**
- End-to-end: given a BeatInfo, verify output shape and format
- Verify unsupported time signature raises clear error

**Integration notes:** Composes Steps 2 and 3.

**Demo:** Generate keypoints for a 10-second 4/4 clip, print shape and sample frame.

---

## Step 5: DatasetWriter

**Objective:** Serialize paired (mel, keypoints) samples as TFRecords.

**Implementation guidance:**
- Create `src/music_generation/dataset_writer.py`
- `DatasetWriter.write_sample(output_path, mel, keypoints, beat_info, metadata)`
- Serialize using `tf.train.Example` with:
  - `mel`: float_list, flattened [T_mel, 80]
  - `keypoints`: float_list, flattened [T_kp, 17*3]
  - `time_signature`: int64_list [num, denom]
  - `tempo`: float_list
  - `fps`: int64_list
  - `source_file`: bytes_list
  - `mel_frames`: int64 (for reshaping on read)
  - `kp_frames`: int64 (for reshaping on read)
- Also write a `read_sample()` function that parses TFRecords back into tensors
- Shard output: one TFRecord per source audio file

**Test requirements:**
- Write a sample, read it back, verify shapes and values match
- Verify all metadata fields round-trip correctly

**Integration notes:** Uses TensorFlow only for serialization. Reuses mel computation from `audio_utils.py`.

**Demo:** Write one sample, read it back, print shapes.

---

## Step 6: CLI Entry Point

**Objective:** Wire everything together as a runnable CLI tool.

**Implementation guidance:**
- Create `src/music_generation/generate_dataset.py`
- `argparse` CLI with flags: `--input_dir`, `--output_dir`, `--fps`, `--time_signature`, `--sample_duration`, `--variations_per_file`, `--seed`
- Pipeline per file:
  1. Load audio via `audio_utils.load_audio()`
  2. Compute mel via `audio_utils.audio_to_mel()`
  3. Run `BeatAnalyzer.analyze()`
  4. For each variation (different sub-seed):
     a. `PatternGenerator.generate()` → keypoints
     b. Slice mel and keypoints into `sample_duration` windows
     c. `DatasetWriter.write_sample()`
- Progress logging: file count, samples generated, any skipped files
- Skip files on error (too short, beat detection failure, unsupported meter)

**Test requirements:**
- Integration test: process a short .wav → verify TFRecord output exists and is parseable
- Verify `--seed` produces deterministic output
- Verify `--time_signature` override propagates

**Integration notes:** Depends on all previous steps. Reuses existing `audio_utils.py` for mel computation.

**Demo:** Run on a small directory of .wav files, inspect output TFRecords.

---

## Step 7: Stretch Time Signatures (6/8, 5/4, 7/8)

**Objective:** Add conducting patterns for compound and irregular meters.

**Implementation guidance:**
- Add waypoints to `ConductingPattern.PATTERNS` for (6, 8), (5, 4), (7, 8)
- 6/8: 6-point pattern with compound subdivisions
- 5/4: asymmetric 3+2 grouping
- 7/8: asymmetric 4+3 grouping
- May need to adjust spline interpolation for uneven beat durations in asymmetric meters

**Test requirements:**
- Verify waypoint count matches beats per measure
- Verify interpolation handles uneven beat spacing

**Integration notes:** Only changes `conducting_patterns.py`. Everything else works unchanged.

**Demo:** Generate and visualize a 5/4 conducting pattern.
