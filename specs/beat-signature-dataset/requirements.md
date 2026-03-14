# Requirements: Beat Signature Dataset

## Questions & Answers

### Q1: Should the conducting gestures be purely synthetic/procedural (generated from canonical beat pattern templates), or do you want to incorporate real motion capture data from actual conductors?

**A1:** Purely synthetic/procedural. This is a synthetic dataset generator only. Real conducting and dance datasets will be handled separately.

### Q2: Which time signatures do you need to support? Just the common ones (2/4, 3/4, 4/4) or also compound/irregular meters (6/8, 5/4, 7/8, etc.)?

**A2:** Primarily 2/4, 3/4, 4/4. Also 6/8, 5/4, 7/8 if they're easy to add — not a hard requirement.

### Q3: For the music side of the pairing — are you planning to use existing music files (e.g. from MusicNet or another dataset) with beat detection, or generate synthetic audio as well (e.g. metronome clicks, simple MIDI sequences)?

**A3:** Existing music files.

### Q4: What's the body keypoint format you want for the output? Should it match the body-point-module's MoveNet format (17 joints, x/y/confidence), or a different skeleton representation?

**A4:** Match the body-point-module's format — MoveNet Thunder: 17 keypoints, each with (y, x, confidence). This ensures the synthetic dataset is directly compatible with the existing pipeline.

### Q5: How should beat positions be determined from the existing music files? Options:
- Use a beat tracking library (e.g. `madmom`, `librosa.beat`) for automatic detection
- Require pre-annotated beat/meter data (e.g. MusicNet has some annotations)
- Both — prefer annotations when available, fall back to beat tracking

**A5:** Use a beat tracking library (e.g. madmom, librosa.beat) for automatic detection.

### Q6: For the canonical conducting patterns — should only the right arm/hand move (as in traditional conducting), or do you want full upper body motion (both arms, torso sway, head movement)?

**A6:** Full upper body motion (both arms, torso, head).

### Q7: What frame rate should the keypoint sequences be generated at? The body-point-module doesn't specify a fixed rate. Common options:
- Match the mel spectrogram frame rate from the music generation system (hop-based, typically ~86 fps at hop=256/sr=22050)
- Standard video frame rates (24, 30, or 60 fps)
- Configurable

**A7:** Standard video frame rates (24, 30, or 60 fps), configurable.

### Q8: Should the generator add variation/noise to the canonical patterns (e.g. slight randomization of trajectory, amplitude, timing) to make the dataset more diverse, or keep them clean and deterministic?

**A8:** Yes, add variation/noise to the canonical patterns for diversity.

### Q9: What's the output format for the dataset? Options:
- TFRecord files (native to your TF pipeline)
- NumPy `.npz` files (audio mel + keypoints arrays)
- JSON/CSV for keypoints + separate audio files
- Something else

**A9:** TFRecord files — plugs into the existing TF pipeline and is efficient. (Current pipeline uses cached .npy mels with tf.data.Dataset, so TFRecord is a natural extension.)

### Q10: Roughly how large should the dataset be? Are we talking hundreds, thousands, or tens of thousands of paired samples? And what duration per sample — a few measures, 30 seconds, full tracks?

**A10:** Configurable. This is a dataset generation tool, not a one-off dataset. Will be used to build multiple datasets over the course of training. Size and sample duration should be parameters.

### Q11: Where do the input music files come from? The existing MusicNet data at `~/data/music/musicnet/train_data/`, or should the tool accept any directory of .wav files?

**A11:** Any directory of .wav files. Not tied to MusicNet specifically.

### Q12: Should the tool also handle time signature detection automatically (from the beat tracker output), or will the time signature be provided as a parameter per run (e.g. "generate 4/4 conducting for all files in this directory")?

**A12:** Auto-detect time signature from the audio rather than relying on annotations or manual parameters. (Libraries like madmom can estimate meter from beat tracking output.)

### Q13: Last question — should this be a CLI tool (e.g. `python -m src.music_generation.generate_dataset --input_dir ... --output_dir ... --fps 30`), a Python API, or both?

**A13:** CLI tool.
