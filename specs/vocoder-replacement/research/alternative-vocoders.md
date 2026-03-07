# Research: Alternative Vocoder Options

## WaveRNN (TensorFlow)

### Overview

Autoregressive vocoder that generates audio sample-by-sample.

**Architecture:**
```
mel frame → GRU → sample prediction → next GRU input
```

### Why Not to Use

- **Autoregressive = very slow** - generates one sample at a time
- **Worse audio quality** than modern GAN vocoders
- Outdated approach (2018 era)
- Limited TensorFlow support

**Verdict:** ❌ Skip entirely - not worth considering for any modern use case

---

## DiffWave (TensorFlow)

### Overview

Diffusion-based vocoder - iteratively denoises to generate audio.

**Architecture:**
```
noise → denoising steps (20-50 iterations) → audio
```

### Why Not Suitable

- **20-50 inference steps required** - too slow for real-time conducting
- High compute requirements
- Overkill for this use case
- Very limited TensorFlow support

### Quality vs Speed Tradeoff

- Quality: Excellent (sometimes better than HiFi-GAN)
- Speed: Too slow for real-time music generation
- Training: Stable but computationally expensive

**Verdict:** ❌ Not ideal for conducting use case - inference too slow despite excellent quality

---

## Griffin-Lim Algorithm

### Overview

Classical phase reconstruction algorithm - no neural network needed.

**Process:**
```
mel spectrogram → inverse mel filterbank → STFT magnitude
→ iterative phase estimation → audio
```

### Use Cases

**Good for:**
- Debugging mel generation
- Dataset verification
- Early listening tests
- Quick prototyping

**Not suitable for:**
- Final output - quality too low
- Production use
- High-fidelity music

### Quality Issues

- Noticeable artifacts
- "Phasiness" and metallic sound
- Missing harmonics
- Acceptable for verification, unusable for final audio

### Implementation

Already available:

```python
import librosa
import numpy as np

def griffin_lim_vocoder(mel_spec, n_iter=32):
    """Convert mel to audio using Griffin-Lim."""
    # Inverse mel filterbank
    stft_matrix = librosa.feature.inverse.mel_to_stft(
        mel_spec,
        sr=22050,
        n_fft=1024,
        fmin=0,
        fmax=11000
    )
    
    # Griffin-Lim phase reconstruction
    audio = librosa.griffinlim(
        stft_matrix,
        n_iter=n_iter,
        hop_length=256,
        win_length=1024
    )
    
    return audio
```

**Verdict:** ✓ Useful for debugging and early testing only - must replace with neural vocoder for final output

---

## Vocos (PyTorch)

### Overview

Modern Fourier-based vocoder - generates STFT coefficients directly instead of time-domain samples.

**Architecture:**
```
mel → ConvNeXt backbone → ISTFT head → audio
```

### Key Advantages

- **Extremely fast** - 10x faster than HiFi-GAN
- **Simpler architecture** - easier to understand and modify
- **Very stable training** - no GAN collapse issues
- **Excellent music quality** - designed for high-fidelity audio
- **Actively maintained** - 2023-2024 development

### Tradeoffs

- **PyTorch only** - no TensorFlow version exists
- Requires mixed framework approach
- Must run outside TensorFlow pipeline

### Why Consider Despite PyTorch

For a conducting/music generation system:
- Vocoder is separate component (runs after mel generation)
- Speed matters for real-time performance
- Quality critical for music (not just speech)
- Tensor conversion overhead is minimal

### Integration with TensorFlow

**Hybrid approach:**

```python
# TensorFlow mel generator
mel = mel_generator(input)  # TF tensor

# Convert to PyTorch for vocoder
mel_torch = torch.from_numpy(mel.numpy())

# Vocos vocoder
audio = vocos.decode(mel_torch)  # PyTorch tensor

# Convert back if needed
audio_np = audio.numpy()
```

**Overhead:** Negligible - tensor conversion is fast, vocoder is architecturally separate

### Pretrained Models

Available on Hugging Face:
- `charactr/vocos-mel-24khz` - 24 kHz
- `charactr/vocos-mel-22khz` - 22 kHz (matches our sample rate)

### Fine-tuning

Simple process:
```python
from vocos import Vocos

vocos = Vocos.from_pretrained("charactr/vocos-mel-22khz")

# Fine-tune on music
for mel, audio in music_dataset:
    loss = vocos.train_step(mel, audio)
```

**Verdict:** ⭐ Excellent option - best quality/speed/stability tradeoff. Worth considering despite PyTorch requirement.

---

## Comparison Matrix

| Vocoder | Quality | Speed | TF Support | Maintenance | Recommendation |
|---------|---------|-------|------------|-------------|----------------|
| HiFi-GAN (TF) | ⭐⭐⭐⭐ | Fast | ⚠️ Unmaintained | Poor | Primary option |
| Vocos (PyTorch) | ⭐⭐⭐⭐⭐ | Very Fast | None | Excellent | Strong alternative |
| Griffin-Lim | ⭐⭐ | Fast | Built-in | N/A | Debug/testing only |
| WaveRNN | ⭐⭐ | Very Slow | Limited | Poor | ❌ Skip |
| DiffWave | ⭐⭐⭐⭐⭐ | Very Slow | None | N/A | ❌ Too slow |

---

## Recommendation

**Primary: HiFi-GAN (TensorFlow)**
- Extract from TensorFlowTTS repo
- Use code directly (skip broken pip install)
- Load pretrained weights
- Fine-tune on music
- Keeps stack pure TensorFlow

**Strong Alternative: Vocos (PyTorch)**
- If HiFi-GAN extraction proves difficult
- Best quality, speed, and stability
- Simpler architecture
- Worth the mixed-framework tradeoff

**Temporary: Griffin-Lim**
- For debugging and early testing only
- No training needed
- Must replace with neural vocoder for final output

**Skip:**
- ❌ WaveRNN - outdated, slow, poor quality
- ❌ DiffWave - too slow for real-time conducting

Content was rephrased for compliance with licensing restrictions.
