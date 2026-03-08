# Rough Idea: Debug Training

## Problem Statement

Training is currently working but encountering issues. We just fixed a TensorFlow graph execution error where `tf.Tensor` was being used as a Python boolean in the `train_step` method.

## Context

The music generation system uses:
- TensorFlow/Keras for the mel spectrogram generator
- Transformer-based architecture with teacher forcing
- Autoregressive training approach
- MelGAN vocoder for audio synthesis

## Current Error (Fixed)

```
tensorflow.python.framework.errors_impl.OperatorNotAllowedInGraphError: Using a symbolic `tf.Tensor` as a Python `bool` is not allowed.
```

This occurred in the NaN detection code: `if tf.math.is_nan(loss) or tf.math.is_inf(loss):`

## Goal

Create a comprehensive debugging plan that Ralph can execute autonomously to:
1. Identify remaining training issues
2. Add proper monitoring and diagnostics
3. Ensure stable training convergence
4. Validate model outputs at each stage
5. Handle edge cases and failure modes gracefully

## Training Command

```bash
python3 -m src.music_generation.train \
    --data_dir ~/data/music/musicnet/train_data \
    --checkpoint_dir ~/models/conducting/mel_checkpoints \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.0001
```

## Expected Outcome

A robust training pipeline with:
- Comprehensive error handling
- Real-time monitoring of training metrics
- Early detection of training failures (NaN, divergence, etc.)
- Automated recovery mechanisms where possible
- Clear diagnostic output for debugging
