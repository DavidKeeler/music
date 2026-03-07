# Rough Idea: Vocoder Replacement

## Problem

The current TensorFlow music generation system cannot perform vocoder training or audio generation because:

1. **TensorFlowTTS is broken** - Has invalid dependency syntax (`tensorflow-gpu` requirement) that prevents installation on modern Python/pip
2. **Package is unmaintained** - Last updated in 2021, no fixes expected
3. **Blocks critical functionality**:
   - Cannot train vocoder on custom music data
   - Cannot convert generated mel spectrograms to audio
   - Cannot perform end-to-end inference (mel generation → audio)

## Current State

**What works:**
- Mel generator training (transformer-based mel spectrogram generation)
- Dataset pipeline (audio → mel conversion)
- All model architecture and tests

**What's broken:**
- `train_vocoder.py` - fails on import
- Audio generation/inference
- End-to-end pipeline testing

## Goal

Implement a working vocoder solution that:

1. **Enables vocoder training** - Fine-tune on MusicNet dataset (22,050 Hz music)
2. **Supports inference** - Convert mel spectrograms to audio waveforms
3. **Maintains compatibility** - Works with existing mel generator (80-dim mels, 22,050 Hz)
4. **Provides good quality** - Suitable for music (not just speech)
5. **Is maintainable** - Uses actively maintained libraries

## Constraints

- Must work with existing mel spectrogram format (80 bins, 22,050 Hz)
- Should support training on custom data (MusicNet)
- Prefer pure TensorFlow, but PyTorch acceptable if necessary
- Must integrate with existing `train_vocoder.py` and `inference.py` structure

## Options to Explore

1. **Vocos (PyTorch)** - Modern, fast, high-quality, actively maintained
2. **Griffin-Lim** - No training needed, but lower quality
3. **TensorFlow port** - Build our own or find alternative TF vocoder
4. **Hybrid approach** - PyTorch vocoder with TensorFlow mel generator

## Success Criteria

- Can run `train_vocoder.py` successfully
- Can generate audio from mel spectrograms
- Audio quality is acceptable for music
- Integration tests pass
