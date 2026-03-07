# Vocoder Replacement Design

## Overview

Replace the broken TensorFlowTTS dependency with a working vocoder solution that enables:
1. Fine-tuning on MusicNet dataset (22,050 Hz music)
2. Converting mel spectrograms to audio waveforms
3. End-to-end music generation pipeline

## Detailed Requirements

### Functional Requirements

**FR1: Vocoder Inference**
- Input: Log-mel spectrogram [batch, time, 80]
- Output: Audio waveform [batch, samples]
- Sample rate: 22,050 Hz
- Quality: High-fidelity suitable for music

**FR2: Vocoder Training**
- Load pretrained weights (speech model)
- Fine-tune on MusicNet music dataset
- Support GAN training with discriminators
- Configurable training parameters

**FR3: Integration**
- Compatible with existing mel generator
- Works with current mel format (log-mel, 80 bins)
- Maintains existing API in `vocoder.py` and `inference.py`
- Supports checkpoint save/load

**FR4: Audio Quality**
- Preserve high-frequency content (up to 11 kHz)
- Minimal artifacts
- Suitable for music (not just speech)

### Non-Functional Requirements

**NFR1: Performance**
- Inference: Real-time or faster on GPU
- Training: Converge within 50k-150k steps

**NFR2: Maintainability**
- Use actively maintained libraries where possible
- Clear documentation
- Testable components

**NFR3: Compatibility**
- Python 3.9-3.12
- TensorFlow 2.13+ (preferred) or PyTorch (acceptable)
- macOS Apple Silicon support

## Architecture Overview

### Two-Path Solution

```
┌─────────────────────────────────────────────────────┐
│                 Vocoder Replacement                  │
├─────────────────────────────────────────────────────┤
│                                                       │
│  Primary Path: HiFi-GAN (TensorFlow)                │
│  ┌──────────────────────────────────────┐           │
│  │ Extract from TensorFlowTTS repo      │           │
│  │ ↓                                     │           │
│  │ Load pretrained weights              │           │
│  │ ↓                                     │           │
│  │ Fine-tune on MusicNet                │           │
│  └──────────────────────────────────────┘           │
│                                                       │
│  Fallback Path: Vocos (PyTorch)                     │
│  ┌──────────────────────────────────────┐           │
│  │ Install via pip                      │           │
│  │ ↓                                     │           │
│  │ TF→PyTorch tensor conversion         │           │
│  │ ↓                                     │           │
│  │ Vocos inference                      │           │
│  └──────────────────────────────────────┘           │
│                                                       │
│  Debug Path: Griffin-Lim                            │
│  ┌──────────────────────────────────────┐           │
│  │ librosa.griffinlim()                 │           │
│  │ (testing/debugging only)             │           │
│  └──────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

### Integration with Existing System

```
┌──────────────────┐
│  Mel Generator   │ (TensorFlow)
│  (Transformer)   │
└────────┬─────────┘
         │ mel [B, T, 80]
         ↓
┌──────────────────┐
│  Normalization   │ (new component)
│  (mean/std)      │
└────────┬─────────┘
         │ normalized mel
         ↓
┌──────────────────┐
│  Vocoder         │ (HiFi-GAN or Vocos)
│  (mel → audio)   │
└────────┬─────────┘
         │ audio [B, samples]
         ↓
┌──────────────────┐
│  Output          │
│  (22,050 Hz)     │
└──────────────────┘
```

## Components and Interfaces

### Component 1: Mel Normalization

**Purpose:** Normalize log-mel spectrograms to match vocoder training distribution

**Location:** `src/music_generation/audio_utils.py`

**Interface:**
```python
class MelNormalizer:
    """Normalize mel spectrograms using dataset statistics."""
    
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def normalize(self, mel: tf.Tensor) -> tf.Tensor:
        """Normalize mel spectrogram."""
        return (mel - self.mean) / self.std
    
    def denormalize(self, mel: tf.Tensor) -> tf.Tensor:
        """Denormalize mel spectrogram."""
        return mel * self.std + self.mean
    
    @classmethod
    def from_dataset(cls, dataset_path: str) -> 'MelNormalizer':
        """Compute statistics from dataset."""
        # Compute mean/std from cached mels
        pass
```

### Component 2: HiFi-GAN Vocoder (Primary)

**Purpose:** TensorFlow-based neural vocoder

**Location:** `src/music_generation/vocoder/hifigan/`

**Structure:**
```
vocoder/
├── __init__.py
├── hifigan_generator.py      # Generator architecture
├── hifigan_discriminator.py  # MPD + MSD discriminators
├── losses.py                  # GAN losses
└── config.py                  # HiFi-GAN config
```

**Interface:**
```python
class HiFiGANVocoder(tf.keras.Model):
    """HiFi-GAN vocoder for mel-to-audio conversion."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.generator = TFHifiGANGenerator(config)
    
    def call(self, mel: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Convert mel [B, T, 80] to audio [B, samples]."""
        # Transpose: [B, T, 80] → [B, 80, T]
        mel_transposed = tf.transpose(mel, [0, 2, 1])
        audio = self.generator(mel_transposed, training=training)
        return tf.squeeze(audio, axis=-1)
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> 'HiFiGANVocoder':
        """Load pretrained weights."""
        pass
```

### Component 3: Vocos Wrapper (Fallback)

**Purpose:** PyTorch Vocos vocoder with TensorFlow compatibility

**Location:** `src/music_generation/vocoder/vocos_wrapper.py`

**Interface:**
```python
class VocosWrapper:
    """PyTorch Vocos vocoder for TensorFlow pipeline."""
    
    def __init__(self, model_name: str = "charactr/vocos-mel-22khz"):
        import torch
        from vocos import Vocos
        
        self.vocos = Vocos.from_pretrained(model_name)
        self.vocos.eval()
    
    def __call__(self, mel: tf.Tensor) -> tf.Tensor:
        """Convert TF mel to audio."""
        import torch
        
        # TF → NumPy → PyTorch
        mel_np = mel.numpy()
        mel_torch = torch.from_numpy(mel_np)
        
        # Generate audio
        with torch.no_grad():
            audio_torch = self.vocos.decode(mel_torch)
        
        # PyTorch → NumPy → TF
        audio_np = audio_torch.numpy()
        return tf.constant(audio_np)
```

### Component 4: Unified Vocoder Interface

**Purpose:** Abstract vocoder selection

**Location:** `src/music_generation/vocoder.py`

**Interface:**
```python
def load_pretrained_vocoder(
    backend: str = "hifigan",  # "hifigan", "vocos", or "griffin-lim"
    checkpoint_path: Optional[str] = None
) -> Union[HiFiGANVocoder, VocosWrapper, GriffinLimVocoder]:
    """Load pretrained vocoder.
    
    Args:
        backend: Vocoder type ("hifigan", "vocos", "griffin-lim")
        checkpoint_path: Path to checkpoint (None = download pretrained)
    
    Returns:
        Vocoder instance with __call__(mel) -> audio interface
    """
    if backend == "hifigan":
        from .vocoder.hifigan import HiFiGANVocoder
        if checkpoint_path is None:
            checkpoint_path = download_hifigan_pretrained()
        return HiFiGANVocoder.from_pretrained(checkpoint_path)
    
    elif backend == "vocos":
        from .vocoder.vocos_wrapper import VocosWrapper
        return VocosWrapper()
    
    elif backend == "griffin-lim":
        from .vocoder.griffin_lim import GriffinLimVocoder
        return GriffinLimVocoder()
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

### Component 5: Vocoder Training

**Purpose:** Fine-tune vocoder on MusicNet

**Location:** `src/music_generation/train_vocoder.py`

**Interface:**
```python
def train_vocoder(
    data_dir: str,
    checkpoint_dir: str,
    pretrained_path: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 2e-4
):
    """Fine-tune HiFi-GAN vocoder on music data.
    
    Training strategy:
    1. Phase 1 (10k steps): Generator-only with L1 + spectral loss
    2. Phase 2 (50k steps): Full GAN training
    """
    # Load pretrained generator
    generator = HiFiGANVocoder.from_pretrained(pretrained_path)
    
    # Create discriminators
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    
    # Phase 1: Generator-only
    train_generator_only(generator, dataset, steps=10000)
    
    # Phase 2: Full GAN
    train_gan(generator, mpd, msd, dataset, steps=50000)
```

## Data Models

### Mel Spectrogram Format

```python
# Shape: [batch, time, n_mels]
# Values: Log-mel spectrogram, normalized
# Range: Approximately [-3, 3] after normalization

mel_spec = {
    "shape": [B, T, 80],
    "dtype": tf.float32,
    "format": "log-mel",
    "normalized": True,
    "sample_rate": 22050,
    "hop_length": 256,
    "n_fft": 1024,
    "fmax": 11025
}
```

### Audio Waveform Format

```python
# Shape: [batch, samples]
# Values: Audio samples in range [-1, 1]
# Sample rate: 22,050 Hz

audio = {
    "shape": [B, T * 256],  # hop_length = 256
    "dtype": tf.float32,
    "range": [-1.0, 1.0],
    "sample_rate": 22050
}
```

### Vocoder Configuration

```python
hifigan_config = {
    "sampling_rate": 22050,
    "hop_size": 256,
    "win_size": 1024,
    "n_fft": 1024,
    "n_mels": 80,
    "fmin": 0,
    "fmax": 11000,  # Music range (vs 8000 for speech)
    
    # Generator
    "upsample_rates": [8, 8, 2, 2],  # Total: 256x upsampling
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    
    # Discriminator
    "mpd_periods": [2, 3, 5, 7, 11],
    "msd_scales": 3
}
```

## Error Handling

### Vocoder Loading Errors

```python
try:
    vocoder = load_pretrained_vocoder(backend="hifigan")
except FileNotFoundError:
    logger.warning("HiFi-GAN checkpoint not found, falling back to Vocos")
    vocoder = load_pretrained_vocoder(backend="vocos")
except ImportError:
    logger.warning("Vocos not installed, using Griffin-Lim")
    vocoder = load_pretrained_vocoder(backend="griffin-lim")
```

### Mel Format Errors

```python
def validate_mel_format(mel: tf.Tensor):
    """Validate mel spectrogram format."""
    # Check shape
    if len(mel.shape) != 3:
        raise ValueError(f"Expected 3D tensor, got shape {mel.shape}")
    
    if mel.shape[-1] != 80:
        raise ValueError(f"Expected 80 mel bins, got {mel.shape[-1]}")
    
    # Check log scale (values should be negative)
    if tf.reduce_min(mel) > 0:
        raise ValueError("Mel values all positive - missing log compression?")
    
    # Check range
    if tf.reduce_max(mel) > 10:
        raise ValueError("Mel values too large - missing log compression?")
```

### Training Errors

```python
# GAN training instability
if discriminator_loss < 0.1:
    logger.warning("Discriminator too strong, reducing learning rate")
    discriminator_lr *= 0.5

if generator_loss > 100:
    logger.error("Generator diverging, stopping training")
    raise TrainingDivergenceError()
```

## Acceptance Criteria

### AC1: Vocoder Inference
**Given** a mel spectrogram of shape [1, 100, 80]  
**When** passed to the vocoder  
**Then** output audio has shape [1, 25600] (100 * 256)  
**And** audio values are in range [-1, 1]  
**And** audio is not silent (max absolute value > 0.01)

### AC2: Pretrained Model Loading
**Given** no local checkpoint exists  
**When** loading pretrained vocoder  
**Then** weights are downloaded from Hugging Face  
**And** vocoder can generate audio immediately  
**And** audio quality is reasonable (no major artifacts)

### AC3: Fine-tuning Convergence
**Given** pretrained HiFi-GAN vocoder  
**When** fine-tuned on MusicNet for 50k steps  
**Then** mel reconstruction loss decreases below 0.5  
**And** generated audio is high-fidelity  
**And** high frequencies (8-11 kHz) are preserved

### AC4: End-to-End Generation
**Given** trained mel generator and vocoder  
**When** generating music from seed audio  
**Then** mel generator produces mel spectrogram  
**And** vocoder converts mel to audio  
**And** output audio is saved successfully  
**And** audio is listenable and recognizable as music

### AC5: Backend Switching
**Given** multiple vocoder backends available  
**When** switching between "hifigan", "vocos", "griffin-lim"  
**Then** each backend produces audio from same mel input  
**And** API remains consistent across backends  
**And** no code changes required in inference.py

### AC6: Error Recovery
**Given** HiFi-GAN fails to load  
**When** vocoder initialization is attempted  
**Then** system falls back to Vocos  
**And** if Vocos unavailable, falls back to Griffin-Lim  
**And** user is notified of fallback via logging

## Testing Strategy

### Unit Tests

**Test 1: Mel Normalization**
```python
def test_mel_normalization():
    normalizer = MelNormalizer(mean=-5.0, std=2.0)
    mel = tf.constant([[-5.0, -3.0, -7.0]])
    normalized = normalizer.normalize(mel)
    assert tf.reduce_mean(normalized) < 0.1  # Approximately zero mean
```

**Test 2: HiFi-GAN Shape**
```python
def test_hifigan_shape():
    vocoder = HiFiGANVocoder(config)
    mel = tf.random.normal([2, 100, 80])
    audio = vocoder(mel)
    assert audio.shape == [2, 25600]  # 100 * 256
```

**Test 3: Vocos Wrapper**
```python
def test_vocos_wrapper():
    vocoder = VocosWrapper()
    mel = tf.random.normal([1, 50, 80])
    audio = vocoder(mel)
    assert isinstance(audio, tf.Tensor)
    assert audio.shape[0] == 1
```

### Integration Tests

**Test 4: End-to-End Pipeline**
```python
def test_end_to_end_generation():
    # Load models
    mel_gen = load_mel_generator("checkpoints/mel_gen.h5")
    vocoder = load_pretrained_vocoder(backend="hifigan")
    
    # Generate
    seed = load_audio("test_seed.wav")
    mel = mel_gen.generate(seed, num_frames=200)
    audio = vocoder(mel)
    
    # Verify
    assert audio.shape[1] == 200 * 256
    assert tf.reduce_max(tf.abs(audio)) > 0.01
    assert not tf.reduce_any(tf.math.is_nan(audio))
```

**Test 5: Fine-tuning Smoke Test**
```python
def test_vocoder_finetuning():
    # Load pretrained
    vocoder = HiFiGANVocoder.from_pretrained("pretrained.h5")
    
    # Create dummy dataset
    dataset = create_dummy_music_dataset(num_samples=100)
    
    # Train for 10 steps
    train_vocoder(vocoder, dataset, steps=10)
    
    # Verify still generates audio
    mel = tf.random.normal([1, 50, 80])
    audio = vocoder(mel)
    assert audio.shape == [1, 12800]
```

### Manual Testing

**Test 6: Audio Quality Assessment**
- Generate audio from test mel spectrograms
- Listen to outputs from each vocoder backend
- Compare to ground truth audio
- Verify no major artifacts (clicks, distortion, silence)

**Test 7: Music Fidelity**
- Generate music using fine-tuned vocoder
- Verify high frequencies are present (spectrogram analysis)
- Compare to Griffin-Lim baseline
- Confirm improvement in quality

## Appendices

### Appendix A: Technology Choices

**Primary: HiFi-GAN (TensorFlow)**
- Proven architecture for high-fidelity audio
- Pretrained weights available
- Pure TensorFlow stack
- Moderate complexity

**Fallback: Vocos (PyTorch)**
- State-of-the-art quality and speed
- Actively maintained
- Simpler architecture
- Requires mixed framework

**Debug: Griffin-Lim**
- No training required
- Built into librosa
- Useful for testing mel generation
- Not suitable for production

### Appendix B: Research Findings Summary

See `research/` directory for detailed findings:
- `hifigan-tensorflow.md` - HiFi-GAN implementation details
- `alternative-vocoders.md` - Comparison of vocoder options
- `implementation-strategy.md` - Step-by-step implementation plan
- `mel-format-compatibility.md` - Mel spectrogram format verification

**Key insights:**
- Current mel format is compatible (log compression already applied)
- Need to add normalization layer
- fmax=11025 is good for music (better than speech's 8000)
- Fine-tuning required to adapt pretrained model to music

### Appendix C: Alternative Approaches Considered

**Approach 1: Fix TensorFlowTTS**
- Fork repo and fix broken dependencies
- **Rejected:** Package has deeper issues, unmaintained

**Approach 2: Build HiFi-GAN from scratch**
- Implement HiFi-GAN in pure TensorFlow
- **Rejected:** Too much work, extracting existing code is faster

**Approach 3: Use only Griffin-Lim**
- Skip neural vocoder entirely
- **Rejected:** Quality too low for music

**Approach 4: PyTorch for entire pipeline**
- Port mel generator to PyTorch
- **Rejected:** Too much rework, mel generator already works

### Appendix D: Limitations

**Known limitations:**
1. HiFi-GAN code extraction may require debugging
2. Pretrained weights are speech-focused (need fine-tuning)
3. Vocos requires PyTorch dependency
4. Fine-tuning requires GPU and time (5-10 hours)
5. No real-time streaming support (batch processing only)

**Future improvements:**
1. Streaming vocoder for real-time generation
2. Lighter-weight model for CPU inference
3. Multi-speaker/multi-instrument support
4. Higher sample rates (44.1 kHz, 48 kHz)
