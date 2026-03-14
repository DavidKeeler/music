# PROMPT.md — Beat Signature Dataset Generator

## Objective

Build a CLI tool at `src/music_generation/generate_dataset.py` that generates synthetic conducting gesture datasets from `.wav` files. It pairs mel spectrograms with procedurally generated conducting keypoint sequences aligned to detected beats, outputting TFRecords compatible with the existing body-point-module pipeline.

## Key Requirements

- Detect beats, downbeats, tempo, and time signature from audio using BeatNet (offline mode, DBN inference)
- Generate canonical conducting patterns for 2/4, 3/4, 4/4 (required); 6/8, 5/4, 7/8 (stretch)
- Output 17-joint MoveNet Thunder format: (y, x, confidence), shoulder-centered normalized coordinates
- Full upper body motion: mirrored arms, IK-derived elbows, head nod on downbeat, static lower body
- Add variation per sample: amplitude ±30%, timing jitter ±8%, Gaussian trajectory noise, style variation
- Interpolate trajectories with `scipy.interpolate.CubicSpline`
- Compute mel spectrograms using existing `src/music_generation/audio_utils.py`
- Write paired (mel, keypoints) samples as TFRecords with metadata (time_signature, tempo, fps, source_file)
- CLI flags: `--input_dir`, `--output_dir`, `--fps` (default 30), `--time_signature` (override), `--sample_duration` (default 10s), `--variations_per_file` (default 3), `--seed`
- Deterministic output given same seed
- Skip files on error with warning log

## Acceptance Criteria

```
Given: A directory of .wav files
When: Running python -m src.music_generation.generate_dataset --input_dir <dir> --output_dir <dir>
Then: TFRecord files are written containing paired mel/keypoint samples

Given: Generated keypoint frames
When: Inspecting any frame
Then: Shape is [17, 3], format is (y, x, confidence), coordinates are shoulder-normalized

Given: Same inputs and --seed value
When: Running the tool twice
Then: Output is identical

Given: --time_signature 3/4 passed via CLI
When: Processing any audio file
Then: 3/4 conducting pattern is used regardless of detected meter

Given: Same audio file with different seeds
When: Comparing keypoint sequences
Then: Trajectories differ while following the same canonical pattern
```

## File Structure

```
src/music_generation/
├── generate_dataset.py      # CLI entry point
├── beat_analyzer.py         # BeatNet wrapper → BeatInfo dataclass
├── conducting_patterns.py   # Canonical waypoints, spline interpolation, PatternGenerator
├── skeleton_builder.py      # Wrist trajectory → 17-joint MoveNet skeleton
└── dataset_writer.py        # TFRecord write/read
```

## Reference

Full design and plan: `specs/beat-signature-dataset/`

## Suggested Command

```bash
ralph run --config presets/spec-driven.yml
```
