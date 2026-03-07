# Research: HiFi-GAN TensorFlow Implementation

## Overview

HiFi-GAN is a GAN-based vocoder that achieves high-fidelity audio synthesis efficiently. It's the modern standard for mel-to-audio conversion.

**Key advantages:**
- High quality (near-human MOS scores)
- Fast inference (167x real-time on GPU, 13x on CPU)
- Pretrained weights available
- Fine-tunable on custom data

## Architecture

```
mel spectrogram [B, T, 80]
    ↓
conv (initial projection)
    ↓
upsample stack (transposed convs)
    ↓
residual blocks (multi-receptive field)
    ↓
final conv
    ↓
audio waveform [B, samples]
```

**Key components:**
- Multi-receptive field fusion (MRF): parallel residual blocks with different kernel sizes
- Multi-period discriminator (MPD): evaluates audio at different periods
- Multi-scale discriminator (MSD): evaluates at different resolutions

## TensorFlow Implementations

### Option 1: TensorFlowTTS (TensorSpeech)

**Repository:** https://github.com/TensorSpeech/TensorFlowTTS

**Status:** ⚠️ Unmaintained since 2021, broken dependencies

**What it includes:**
- HiFi-GAN generator and discriminators
- Pretrained weights (LJSpeech)
- Full training pipeline
- Multi-band MelGAN alternative

**Problem:** Cannot install due to invalid `tensorflow-gpu` dependency syntax

**Potential workaround:** Clone repo and use code directly without pip install

```bash
git clone https://github.com/TensorSpeech/TensorFlowTTS.git
# Use tensorflow_tts/ directory as local module
```

### Option 2: Extract from TensorFlowTTS

Since the package won't install but the code is valid, we can:

1. Copy HiFi-GAN implementation files directly
2. Remove broken setup.py dependencies
3. Adapt to our codebase

**Files needed:**
- `tensorflow_tts/models/hifigan.py` - Generator architecture
- `tensorflow_tts/models/hifigan_discriminator.py` - Discriminators
- `tensorflow_tts/losses/` - GAN losses
- Pretrained weights from Hugging Face

### Option 3: Minimal HiFi-GAN Implementation

Build a simplified HiFi-GAN from scratch (~250 lines):

**Generator:**
```python
class HiFiGANGenerator(tf.keras.Model):
    def __init__(self):
        # Initial conv: mel_dim → upsample_initial_channel
        # Upsample stack: transposed convs (256 → 128 → 64 → 32)
        # MRF blocks: parallel residual blocks
        # Output conv: channels → 1 (audio)
    
    def call(self, mel):
        # mel: [B, T, 80]
        # return: [B, T*256, 1] → squeeze to [B, samples]
```

**Discriminator:**
```python
class MultiPeriodDiscriminator(tf.keras.Model):
    # Evaluate at periods [2, 3, 5, 7, 11]
    
class MultiScaleDiscriminator(tf.keras.Model):
    # Evaluate at 3 scales with avg pooling
```

## Pretrained Weights

### LJSpeech (Speech, 22,050 Hz)

**Source:** Hugging Face - tensorspeech/tts-hifigan-ljspeech-en

**Specs:**
- Sample rate: 22,050 Hz
- Mel bins: 80
- FFT: 1024
- Hop: 256
- fmax: 8000 Hz (speech range)

**Note:** fmax is too low for music - needs fine-tuning

### Universal HiFi-GAN (Multi-speaker)

Some implementations provide universal models trained on multiple datasets.

## Fine-tuning for Music

### ⚠️ Critical: Mel Spectrogram Format

**HiFi-GAN expects log-mel spectrograms**, not linear mel values.

Your mel generator must produce:
```python
# Correct format
mel = librosa.feature.melspectrogram(audio, ...)
log_mel = librosa.power_to_db(mel)  # ← Log compression
log_mel_normalized = (log_mel - mean) / std  # ← Normalization
```

**Common mistake:**
```python
# Wrong - linear mel values
mel = librosa.feature.melspectrogram(audio, ...)
# Missing log compression!
```

**Verify your pipeline:**
1. Check `audio_utils.py` - ensure log compression is applied
2. Verify mel normalization is consistent between training and inference
3. Test with pretrained vocoder before fine-tuning

**If mismatch occurs:**
- Audio will sound distorted or silent
- Vocoder may produce noise
- Training will not converge

### Configuration Changes

**Critical for music:**
```python
# Audio processing
sample_rate = 22050
n_fft = 1024
hop_length = 256
win_length = 1024
n_mels = 80
fmin = 0
fmax = 11000  # ← IMPORTANT: increase from 8000 (speech) to 11000 (music)
```

**Why fmax matters:**
- Speech: most energy below 8 kHz
- Music: harmonics extend to 11+ kHz
- Clipping high frequencies degrades music quality

### Training Strategy

**Option A: Full fine-tuning**
```
1. Load pretrained generator + discriminators
2. Train on MusicNet with GAN losses
3. 50k-150k steps (depends on dataset size)
```

**Option B: Generator-only fine-tuning**
```
1. Load pretrained generator
2. Freeze discriminators (or use pretrained)
3. Train generator with reconstruction loss only
4. Faster, more stable
```

**Option C: Two-stage fine-tuning**
```
1. Stage 1: Generator only with L1 + spectral loss (10k steps)
2. Stage 2: Full GAN training (50k steps)
```

### Loss Functions

**Generator losses:**
- L1 mel reconstruction: `|mel_pred - mel_true|`
- Multi-scale spectral loss: STFT at multiple resolutions
- Adversarial loss: fool discriminators
- Feature matching: match discriminator features

**Discriminator losses:**
- Real/fake classification
- Hinge loss or least-squares GAN

**Typical weights:**
```python
lambda_mel = 45.0
lambda_feature = 2.0
lambda_adversarial = 1.0
```

## Implementation Recommendation

**Best path for this project:**

1. **Extract TensorFlowTTS HiFi-GAN code**
   - Clone TensorFlowTTS repo
   - Copy `tensorflow_tts/models/hifigan*.py` to our codebase
   - Remove package dependencies, use as local module

2. **Load pretrained weights**
   - Download from Hugging Face
   - Convert checkpoint format if needed

3. **Update mel config for music**
   - Set fmax = 11000
   - Regenerate all mel spectrograms with new config

4. **Fine-tune on MusicNet**
   - Start with generator-only (stable)
   - Add GAN training if quality insufficient
   - 50k-100k steps should be enough

5. **Integration**
   - Replace broken `vocoder.py` imports
   - Update `train_vocoder.py` to use new HiFi-GAN
   - Keep same interface for `inference.py`

## Alternative: PyTorch Vocos

If TensorFlow proves too difficult:

**Vocos advantages:**
- Modern, actively maintained
- Faster than HiFi-GAN
- Better quality
- Easy to install: `pip install vocos`

**Integration approach:**
- Keep TensorFlow for mel generator
- Use PyTorch only for vocoder
- Convert tensors at boundary: `torch.from_numpy(mel.numpy())`

**Tradeoff:** Mixed framework, but vocoder is separate component anyway

## References

- [HiFi-GAN paper](https://arxiv.org/abs/2010.05646) - Original paper
- [TensorFlowTTS repo](https://github.com/TensorSpeech/TensorFlowTTS) - TF implementation (unmaintained)
- [HiFi-GAN demo](https://jik876.github.io/hifi-gan-demo/) - Audio samples
- [Vocos](https://github.com/charactr-platform/vocos) - Modern alternative (PyTorch)

Content was rephrased for compliance with licensing restrictions.
