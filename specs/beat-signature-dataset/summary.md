# Beat Signature Dataset Generator — Summary

## Artifacts

| File | Description |
|------|-------------|
| `specs/beat-signature-dataset/rough-idea.md` | Original idea + agent research (flagged as partially unvetted) |
| `specs/beat-signature-dataset/requirements.md` | 13 Q&A pairs covering scope, format, and constraints |
| `specs/beat-signature-dataset/research/beat-tracking-and-meter.md` | BeatNet vs madmom vs librosa comparison; meter detection reliability |
| `specs/beat-signature-dataset/research/conducting-patterns.md` | Canonical conducting geometry for all target time signatures |
| `specs/beat-signature-dataset/research/movenet-keypoint-mapping.md` | MoveNet 17-joint mapping and synthetic generation strategy |
| `specs/beat-signature-dataset/design.md` | Full design: 5 components, data models, acceptance criteria, test strategy |
| `specs/beat-signature-dataset/plan.md` | 7-step implementation plan with demos at each step |

## Overview

A CLI tool (`python -m src.music_generation.generate_dataset`) that takes a directory of `.wav` files and produces TFRecord datasets of paired (mel spectrogram, synthetic conducting keypoints). Uses BeatNet for beat/meter detection, canonical conducting patterns with spline interpolation and variation/noise, and outputs MoveNet-compatible 17-joint skeletons directly usable by the body-point-module pipeline.

## Key Decisions

- **BeatNet offline mode** for joint beat/downbeat/meter detection
- **Canonical patterns** with procedural variation (not real mocap)
- **MoveNet Thunder format** (17 joints, y/x/confidence, shoulder-normalized)
- **TFRecord output** for TF pipeline compatibility
- **CLI-first**, configurable for repeated dataset generation during training

## Suggested Next Steps

To implement, use Ralph with the spec-driven pipeline:

```bash
ralph run --config presets/spec-driven.yml
```

Or for the full PDD-to-code flow:

```bash
ralph run --config presets/pdd-to-code-assist.yml
```
