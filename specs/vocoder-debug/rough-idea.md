# Vocoder Training Debugging

## Problem Statement

The vocoder training script (`python -m src.music_generation.train_vocoder`) needs debugging and validation to ensure it works correctly for finetuning the pretrained HiFi-GAN/MelGAN vocoder on music data.

## Current Command
```bash
python -m src.music_generation.train_vocoder \
  --data_dir ~/data/music/musicnet/train_data \
  --checkpoint_dir ./vocoder_checkpoints \
  --epochs 50 \
  --batch_size 8
```

## Potential Issues to Investigate

### 1. Pretrained Model Loading
**Location:** `src/music_generation/vocoder.py` - `load_pretrained_vocoder()`

**Concerns:**
- Model loading may fail or return None
- TensorFlowTTS integration may have compatibility issues
- Default model name may not be appropriate for music (trained on speech)
- Error handling is minimal

### 2. Dataset Pipeline
**Location:** `src/music_generation/vocoder.py` - `VocoderDataset`

**Concerns:**
- Uses Python generator instead of efficient tf.data operations
- Random cropping happens in eager mode (slow)
- No caching of mel spectrograms
- Shuffle happens in Python, not TensorFlow
- May have memory issues with large datasets
- `audio_to_mel_ljspeech()` parameters may not match generator expectations

### 3. Training Loop
**Location:** `src/music_generation/train_vocoder.py` - `VocoderTraining`

**Concerns:**
- STFT loss configuration may not be optimal for music
- No discriminator training (only generator finetuning)
- Learning rate may be too high/low
- No learning rate scheduling
- Gradient norm tracking but no gradient clipping
- Length trimming may cause issues with mismatched shapes

### 4. Checkpoint Management
**Location:** `src/music_generation/train_vocoder.py` - `train_vocoder()`

**Concerns:**
- Saves full model instead of just weights (large files)
- No best model tracking (only epoch-based)
- No validation dataset or metrics
- Checkpoint path format may cause issues

### 5. Audio Processing Mismatch
**Location:** `src/music_generation/audio_utils.py`

**Concerns:**
- `audio_to_mel_ljspeech()` uses speech parameters (22050 Hz, specific hop/win lengths)
- May not match pretrained vocoder's expected input format
- No verification that mel → audio → mel roundtrip works
- Normalization may be inconsistent

### 6. Missing Features
- No validation loop or metrics
- No audio generation during training (can't hear quality)
- No early stopping
- No gradient clipping (may cause training instability)
- No mixed precision training
- No progress logging beyond TensorBoard

## Desired Outcome

A robust vocoder training script that:
1. Successfully loads pretrained vocoder
2. Efficiently processes music data
3. Trains stably with appropriate hyperparameters
4. Saves checkpoints correctly
5. Provides validation metrics and audio samples
6. Has proper error handling and logging
7. Works on resource-constrained hardware (MacBook Air)

## Questions to Answer

1. Does the pretrained model load successfully?
2. Are the mel spectrogram parameters compatible with the vocoder?
3. Does training converge (loss decreases)?
4. Does generated audio quality improve over epochs?
5. Are there any NaN/Inf issues during training?
6. Is the dataset pipeline efficient enough?
7. Should we use discriminator training or just generator finetuning?
8. What are appropriate hyperparameters (lr, batch_size, epochs)?
