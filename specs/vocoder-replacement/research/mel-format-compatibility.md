# Research: Mel Spectrogram Format Compatibility

## Current Implementation Status

### ✅ Log Compression Already Applied

Checked `src/music_generation/audio_utils.py`:

```python
def audio_to_mel(waveform: tf.Tensor) -> tf.Tensor:
    # ... STFT computation ...
    mel = tf.matmul(magnitude, mel_matrix)
    log_mel = tf.math.log(mel + 1e-8)  # ← Log compression present
    return log_mel
```

**Good news:** The pipeline already produces log-mel spectrograms, which is what HiFi-GAN expects.

## Current Configuration

From `config.py`:
```python
SAMPLE_RATE = 22050
N_FFT = 1024
FRAME_LENGTH = 1024
FRAME_STEP = 256
N_MELS = 80
```

**Missing:** `fmax` parameter - currently defaults to `SAMPLE_RATE / 2.0 = 11025 Hz`

## Required Changes for Music

### Update fmax for Music

Current code:
```python
mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=N_MELS,
    num_spectrogram_bins=FRAME_LENGTH // 2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=0.0,
    upper_edge_hertz=SAMPLE_RATE / 2.0  # ← 11025 Hz (Nyquist)
)
```

**For music, should be:**
```python
upper_edge_hertz=11000  # Explicit music range (vs 8000 for speech)
```

**Why this matters:**
- Speech: most energy below 8 kHz
- Music: harmonics extend to 11 kHz
- Using full Nyquist (11025) is actually fine for music
- Pretrained speech models use 8000, so we need to fine-tune anyway

### Normalization

Current implementation does NOT normalize:
```python
log_mel = tf.math.log(mel + 1e-8)
return log_mel  # Raw log-mel values
```

**HiFi-GAN typically expects normalized mels:**
```python
log_mel = tf.math.log(mel + 1e-8)
normalized = (log_mel - mean) / std
return normalized
```

**Two approaches:**

**Option 1: Normalize during preprocessing**
- Compute dataset statistics
- Apply normalization in `audio_to_mel()`
- Store mean/std in config

**Option 2: Normalize in model**
- Keep raw log-mels in dataset
- Add normalization layer in mel generator
- More flexible for fine-tuning

## Compatibility Check

### What HiFi-GAN Expects

From TensorFlowTTS HiFi-GAN config:
```yaml
sampling_rate: 22050
hop_size: 256
win_size: 1024
n_fft: 1024
n_mels: 80
fmin: 0
fmax: 8000  # ← Speech range
```

**Our config matches except:**
- ✅ sampling_rate: 22050 (match)
- ✅ hop_size: 256 (match)
- ✅ win_size: 1024 (match)
- ✅ n_fft: 1024 (match)
- ✅ n_mels: 80 (match)
- ✅ fmin: 0 (match)
- ⚠️ fmax: 11025 vs 8000 (ours is higher - better for music)

**Conclusion:** Our mels are compatible but have more high-frequency content. This is GOOD for music, but means we MUST fine-tune the pretrained vocoder.

## Action Items

### Before Vocoder Integration

1. **Verify log compression** - ✅ Already done
2. **Add normalization** - Need to implement
3. **Compute dataset statistics** - mean/std of log-mels
4. **Update config** - Add FMAX constant

### During Fine-tuning

1. **Start with pretrained weights** - trained on fmax=8000
2. **Fine-tune on music** - adapt to fmax=11025 and music characteristics
3. **Monitor spectral loss** - ensure high frequencies are learned

### Testing

```python
def test_mel_format():
    """Verify mel format is compatible with HiFi-GAN."""
    audio = load_audio("test.wav")
    mel = audio_to_mel(audio)
    
    # Check log scale (values should be negative)
    assert tf.reduce_max(mel) < 10, "Mel values too large - missing log?"
    assert tf.reduce_min(mel) < 0, "Mel values all positive - missing log?"
    
    # Check shape
    assert mel.shape[-1] == 80, "Wrong number of mel bins"
    
    # Check range (typical log-mel range)
    assert tf.reduce_min(mel) > -20, "Mel values too negative"
    assert tf.reduce_max(mel) < 5, "Mel values too positive"
```

## Summary

**Current status:**
- ✅ Log compression applied
- ✅ Correct mel parameters (80 bins, 22050 Hz, etc.)
- ⚠️ Missing normalization
- ⚠️ fmax slightly higher than pretrained model (good for music)

**Required changes:**
1. Add mel normalization (compute mean/std from dataset)
2. Fine-tune vocoder on music data (adapt to our fmax and music characteristics)

**Risk level:** Low - format is compatible, just needs fine-tuning
