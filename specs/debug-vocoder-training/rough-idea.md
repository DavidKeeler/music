# Debug Vocoder Training

## Problem Statement

The vocoder training script fails with a missing dependency error:

```
ModuleNotFoundError: No module named 'huggingface_hub'
```

## Command That Failed

```bash
python3 -m src.music_generation.train_vocoder \
    --data_dir ~/data/music/musicnet \
    --checkpoint_dir ~/models/conducting/vocoder_checkpoints \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0002
```

## Error Details

- The script attempts to load a pretrained HiFiGAN vocoder
- `vocoder.HiFiGANVocoder.from_pretrained()` requires `huggingface_hub` package
- The package is not installed in the current venv
- May be additional issues after this is resolved

## Goal

Get the vocoder training running successfully end-to-end, resolving all dependency and runtime issues.

## Environment

- Python 3.12
- macOS with Apple M2 (Metal GPU)
- TensorFlow with Metal plugin
- Virtual environment: `venv`
