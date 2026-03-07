# Research: Implementation Strategy

## Problem Summary

Need a working vocoder that:
1. Runs in TensorFlow (preferred) or PyTorch (acceptable)
2. Can be fine-tuned on MusicNet (22,050 Hz music)
3. Integrates with existing mel generator
4. Provides good quality for music (not just speech)

## Recommended Approach: Extract TensorFlowTTS HiFi-GAN

### Why This Works

TensorFlowTTS package is broken, but the **code itself is valid**. We can:
- Clone the repo
- Copy relevant files to our codebase
- Use as local module (no pip install needed)
- Load pretrained weights from Hugging Face

### Step-by-Step Plan

**1. Clone TensorFlowTTS**
```bash
cd /tmp
git clone https://github.com/TensorSpeech/TensorFlowTTS.git
```

**2. Extract HiFi-GAN files**

Copy to `src/music_generation/vocoder/`:
- `tensorflow_tts/models/hifigan.py` → Generator
- `tensorflow_tts/models/hifigan_discriminator.py` → Discriminators  
- `tensorflow_tts/losses/` → Loss functions
- `tensorflow_tts/configs/hifigan.yaml` → Config template

**3. Create wrapper**

```python
# src/music_generation/vocoder/hifigan_wrapper.py

from .hifigan import TFHifiGANGenerator
import tensorflow as tf

class HiFiGANVocoder(tf.keras.Model):
    """Wrapper for HiFi-GAN vocoder."""
    
    def __init__(self, config_path=None):
        super().__init__()
        self.generator = TFHifiGANGenerator(...)
    
    def call(self, mel_spectrogram):
        """Convert mel [B, T, 80] to audio [B, samples]."""
        # Transpose if needed: [B, T, 80] → [B, 80, T]
        mel = tf.transpose(mel_spectrogram, [0, 2, 1])
        audio = self.generator(mel)
        return tf.squeeze(audio, axis=-1)
    
    @classmethod
    def from_pretrained(cls, checkpoint_path):
        """Load pretrained weights."""
        model = cls()
        model.load_weights(checkpoint_path)
        return model
```

**4. Download pretrained weights**

From Hugging Face: `tensorspeech/tts-hifigan-ljspeech-en`

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="tensorspeech/tts-hifigan-ljspeech-en",
    filename="generator.h5"
)
```

**5. Update vocoder.py**

```python
# src/music_generation/vocoder.py

from .vocoder.hifigan_wrapper import HiFiGANVocoder

def load_pretrained_vocoder(checkpoint_path=None):
    """Load HiFi-GAN vocoder."""
    if checkpoint_path is None:
        # Download from Hugging Face
        checkpoint_path = download_pretrained_hifigan()
    
    return HiFiGANVocoder.from_pretrained(checkpoint_path)
```

**6. Update train_vocoder.py**

```python
# src/music_generation/train_vocoder.py

from .vocoder.hifigan_wrapper import HiFiGANVocoder
from .vocoder.hifigan_discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator

def train_vocoder(args):
    # Load pretrained generator
    generator = HiFiGANVocoder.from_pretrained(args.pretrained_path)
    
    # Create discriminators
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    
    # Training loop with GAN losses
    ...
```

### Configuration for Music

**Update mel spectrogram config:**

```python
# src/music_generation/config.py

# Audio processing
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0
FMAX = 11000  # ← Changed from 8000 (speech) to 11000 (music)
```

**Important:** Must regenerate all cached mel spectrograms with new fmax

### Fine-tuning Strategy

**Phase 1: Generator-only (10k steps)**
- Freeze discriminators
- Train generator with L1 + spectral loss
- Learning rate: 2e-4
- Stable, fast convergence

**Phase 2: Full GAN (50k steps)**
- Unfreeze discriminators
- Add adversarial loss
- Learning rate: 1e-4 (generator), 2e-4 (discriminators)
- Better quality, but less stable

**Loss weights:**
```python
lambda_mel = 45.0
lambda_spectral = 1.0
lambda_feature_matching = 2.0
lambda_adversarial = 1.0
```

### Expected Results

**After Phase 1 (generator-only):**
- Decent quality, some artifacts
- Good enough for prototyping
- ~10k steps = 1-2 hours on GPU

**After Phase 2 (full GAN):**
- High quality, minimal artifacts
- Production-ready
- ~50k steps = 5-10 hours on GPU

### Fallback: Vocos (PyTorch)

If TensorFlowTTS extraction fails:

**1. Install Vocos**
```bash
pip install torch vocos
```

**2. Create wrapper**
```python
# src/music_generation/vocoder/vocos_wrapper.py

import torch
from vocos import Vocos
import tensorflow as tf

class VocosWrapper:
    """PyTorch Vocos vocoder for TensorFlow pipeline."""
    
    def __init__(self):
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-22khz")
        self.vocos.eval()
    
    def __call__(self, mel_spectrogram):
        """Convert TF mel to audio."""
        # TF → NumPy → PyTorch
        mel_np = mel_spectrogram.numpy()
        mel_torch = torch.from_numpy(mel_np)
        
        # Generate audio
        with torch.no_grad():
            audio_torch = self.vocos.decode(mel_torch)
        
        # PyTorch → NumPy → TF
        audio_np = audio_torch.numpy()
        return tf.constant(audio_np)
```

**3. Update vocoder.py**
```python
def load_pretrained_vocoder(use_pytorch=False):
    if use_pytorch:
        from .vocoder.vocos_wrapper import VocosWrapper
        return VocosWrapper()
    else:
        from .vocoder.hifigan_wrapper import HiFiGANVocoder
        return HiFiGANVocoder.from_pretrained()
```

### Testing Strategy

**Unit tests:**
```python
def test_vocoder_shape():
    vocoder = load_pretrained_vocoder()
    mel = tf.random.normal([1, 100, 80])
    audio = vocoder(mel)
    assert audio.shape == [1, 100 * 256]  # hop_length = 256

def test_vocoder_quality():
    # Load real audio
    audio_true = load_audio("test.wav")
    mel = audio_to_mel(audio_true)
    
    # Reconstruct
    audio_pred = vocoder(mel)
    
    # Check spectral similarity
    mel_pred = audio_to_mel(audio_pred)
    mse = tf.reduce_mean((mel - mel_pred) ** 2)
    assert mse < 0.1  # Reasonable reconstruction
```

**Integration test:**
```python
def test_end_to_end():
    # Load models
    mel_generator = load_mel_generator("checkpoints/mel_gen.h5")
    vocoder = load_pretrained_vocoder()
    
    # Generate
    seed = load_audio("seed.wav")
    mel = mel_generator.generate(seed, num_frames=500)
    audio = vocoder(mel)
    
    # Save
    sf.write("output.wav", audio.numpy()[0], 22050)
    
    # Verify
    assert audio.shape[1] > 0
    assert not tf.reduce_any(tf.math.is_nan(audio))
```

## Timeline Estimate

**Option 1: TensorFlowTTS extraction**
- Extract files: 1 hour
- Create wrapper: 2 hours
- Load pretrained weights: 1 hour
- Test inference: 1 hour
- Fine-tune on music: 5-10 hours (GPU time)
- **Total: ~1 day dev + 10 hours GPU**

**Option 2: Vocos (PyTorch)**
- Install: 5 minutes
- Create wrapper: 1 hour
- Test inference: 30 minutes
- Fine-tune (optional): 2-5 hours
- **Total: ~2 hours dev + 5 hours GPU**

## Risk Assessment

**TensorFlowTTS extraction risks:**
- Code may have internal dependencies we miss
- Checkpoint format may be incompatible
- Config may need adjustments

**Mitigation:** Start with Vocos as backup plan

**Vocos risks:**
- Mixed framework (TF + PyTorch)
- Tensor conversion overhead

**Mitigation:** Overhead is minimal, vocoder is separate component

## Recommendation

**Primary path:** Extract TensorFlowTTS HiFi-GAN
- Keeps stack pure TensorFlow
- Proven architecture
- Pretrained weights available

**Backup path:** Vocos (PyTorch)
- If extraction fails or takes too long
- Better quality and speed
- Easier integration

**Temporary solution:** Griffin-Lim
- For immediate prototyping
- No training needed
- Replace later with neural vocoder

Content was rephrased for compliance with licensing restrictions.
