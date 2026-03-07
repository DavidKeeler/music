# Vocoder Status

## Current Situation

**TensorFlowTTS is broken and cannot be installed.**

Error: `tensorflow-gpu` dependency has invalid syntax (`python_version>"3.7"` should be `python_version>="3.7"`). The package is unmaintained since 2021.

## What Works

✅ **Mel generator training** - Train the transformer model that generates mel spectrograms
✅ **Model architecture** - All layers and attention mechanisms work
✅ **Dataset pipeline** - Audio loading and mel spectrogram computation
✅ **Testing** - All unit tests pass

## What Doesn't Work

❌ **Vocoder training** - `train_vocoder.py` fails because TensorFlowTTS can't be installed
❌ **Audio generation** - Can't convert mel spectrograms back to audio
❌ **End-to-end inference** - Can generate mels but not audio

## Recommended Path Forward

### Short term: Train without vocoder

```bash
# This works fine
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet \
  --checkpoint_dir ./checkpoints \
  --epochs 100 \
  --batch_size 16
```

The mel generator doesn't need a vocoder for training. You can:
- Train and save mel generator checkpoints
- Evaluate using mel spectrogram metrics (MSE, spectral loss)
- Save generated mel spectrograms as `.npy` files

### Medium term: Add PyTorch vocoder for inference only

Install Vocos (PyTorch) just for converting mels to audio:

```bash
pip install torch vocos
```

Use it only in inference scripts, keep training pure TensorFlow.

### Long term: Find or build TensorFlow vocoder

Options:
1. Wait for community to fix TensorFlowTTS
2. Port Vocos to TensorFlow (significant work)
3. Use Griffin-Lim algorithm (lower quality but no dependencies)

## Quick Test

Verify your setup works for training:

```bash
# Should work
python -m pytest tests/test_model.py
python -m pytest tests/test_dataset.py
python -m pytest tests/test_training_smoke.py

# Will fail (vocoder not available)
python -m pytest tests/test_vocoder.py
python -m src.music_generation.train_vocoder --help
```

## Questions?

- **Can I train the model?** Yes, vocoder not needed for training
- **Can I generate audio?** Not currently, only mel spectrograms
- **Should I wait for a fix?** No, proceed with mel generator training
- **Will this affect model quality?** No, vocoder is separate from mel generation
