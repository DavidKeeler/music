# MelGAN Vocoder Setup & Testing Plan

## ⚠️ CURRENT STATUS: VOCODER NOT AVAILABLE

**Issue:** TensorFlowTTS is unmaintained and has broken dependencies (invalid `tensorflow-gpu` requirement syntax). It cannot be installed on modern Python/pip versions.

**Impact:**
- ✅ **Mel generator training works** - vocoder not needed for training
- ❌ **Vocoder training not available** - `train_vocoder.py` will fail
- ❌ **Audio generation not available** - cannot convert mel spectrograms to audio

**Workarounds:**
1. **For training:** Continue training mel generator without vocoder - save mel spectrograms only
2. **For inference:** Use PyTorch-based vocoder (Vocos) or Griffin-Lim algorithm
3. **For evaluation:** Compare mel spectrograms directly instead of audio

## Alternative Solutions

### Option 1: Use PyTorch Vocos (Recommended)

Vocos is a modern, maintained vocoder that's faster and higher quality than MelGAN.

```bash
pip install torch vocos
```

Update `vocoder.py` to use Vocos:
```python
import torch
from vocos import Vocos

class VocosWrapper:
    def __init__(self):
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    
    def __call__(self, mel_spectrogram):
        # Convert TF tensor to PyTorch
        mel_torch = torch.from_numpy(mel_spectrogram.numpy())
        audio = self.vocos.decode(mel_torch)
        return audio.numpy()
```

### Option 2: Use Griffin-Lim (No dependencies)

Fast but lower quality reconstruction using only scipy/numpy:

```python
import librosa

def griffin_lim_vocoder(mel_spectrogram, n_iter=32):
    """Convert mel to audio using Griffin-Lim algorithm."""
    # Inverse mel filterbank
    linear_spec = librosa.feature.inverse.mel_to_stft(
        mel_spectrogram, sr=22050, n_fft=1024
    )
    # Griffin-Lim phase reconstruction
    audio = librosa.griffinlim(linear_spec, n_iter=n_iter, hop_length=256)
    return audio
```

### Option 3: Wait for TensorFlow vocoder

Monitor these projects for TensorFlow-native vocoders:
- Vocos TensorFlow port (not yet available)
- TensorFlowTTS fork/fix (community may fix it)

---

## Original Documentation (For Reference)

The sections below describe the original TensorFlowTTS setup, which is currently non-functional.

## Setup Instructions

### 1. Install TensorFlowTTS

```bash
pip install TensorFlowTTS
```

**Library:** TensorFlowTTS provides pretrained TTS models including MelGAN vocoder.

**Source repo:** https://github.com/TensorSpeech/TensorFlowTTS

### 2. Pretrained Model

**Model:** MelGAN trained on LJSpeech dataset (22,050 Hz)

**Hugging Face:** https://huggingface.co/tensorspeech/tts-melgan-ljspeech-en

The model will auto-download on first use via `TFAutoModel.from_pretrained()`.

## Implementation Tasks

### Task 1: Update vocoder.py to use MelGAN

Modify `src/music_generation/vocoder.py`:

```python
import tensorflow as tf
from tensorflow_tts.inference import TFAutoModel

class HiFiGANVocoder(tf.keras.Model):
    """Wrapper for MelGAN vocoder from TensorFlowTTS."""
    
    def __init__(self, pretrained_model=None):
        super().__init__()
        if pretrained_model is None:
            # Load pretrained MelGAN
            self.generator = TFAutoModel.from_pretrained(
                "tensorspeech/tts-melgan-ljspeech-en"
            )
        else:
            self.generator = pretrained_model
    
    def call(self, mel_spectrogram, training=False):
        """Convert mel spectrogram to audio.
        
        Args:
            mel_spectrogram: [batch, time, mel_bins] or [batch, mel_bins, time]
            
        Returns:
            Audio waveform [batch, samples]
        """
        # MelGAN expects [batch, time, mel_bins]
        if mel_spectrogram.shape[1] == 80:  # [batch, 80, time]
            mel_spectrogram = tf.transpose(mel_spectrogram, [0, 2, 1])
        
        # MelGAN inference returns [batch, samples, 1]
        audio = self.generator.inference(mel_spectrogram)
        
        # Squeeze to [batch, samples]
        return tf.squeeze(audio, axis=-1)
```

Update `load_pretrained_vocoder()`:

```python
def load_pretrained_vocoder(model_url: Optional[str] = None) -> HiFiGANVocoder:
    """Load pretrained MelGAN vocoder.
    
    Args:
        model_url: Ignored (kept for compatibility). Uses MelGAN by default.
        
    Returns:
        HiFiGANVocoder with pretrained MelGAN
    """
    return HiFiGANVocoder()
```

### Task 2: Update requirements.txt

Add to `requirements.txt`:

```
TensorFlowTTS>=1.0.0
```

### Task 3: Create test script

Create `tests/test_vocoder_melgan.py`:

```python
"""Test MelGAN vocoder with dummy mel spectrogram."""

import tensorflow as tf
import soundfile as sf
from pathlib import Path
from music_generation.vocoder import load_pretrained_vocoder

def test_melgan_inference():
    """Test MelGAN can convert mel to audio."""
    
    # Load vocoder
    print("Loading MelGAN vocoder...")
    vocoder = load_pretrained_vocoder()
    
    # Create dummy mel spectrogram [batch=1, time=100, mel_bins=80]
    # Normalized to roughly match LJSpeech range
    mel = tf.random.normal([1, 100, 80]) * 0.5
    
    print(f"Input mel shape: {mel.shape}")
    
    # Generate audio
    print("Generating audio...")
    audio = vocoder(mel, training=False)
    
    print(f"Output audio shape: {audio.shape}")
    print(f"Audio range: [{tf.reduce_min(audio):.3f}, {tf.reduce_max(audio):.3f}]")
    
    # Save to file
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "melgan_test.wav"
    
    sf.write(str(output_path), audio.numpy()[0], 22050, "PCM_16")
    print(f"Saved audio to: {output_path}")
    
    # Verify audio is not silent
    assert tf.reduce_max(tf.abs(audio)) > 0.01, "Audio is too quiet"
    
    print("✓ Test passed")

if __name__ == "__main__":
    test_melgan_inference()
```

## Testing Plan

### Step 1: Install dependencies

```bash
cd /Users/davidkeeler/code/conducting3
pip install TensorFlowTTS soundfile
```

### Step 2: Run vocoder test

```bash
python tests/test_vocoder_melgan.py
```

**Expected output:**
- Downloads MelGAN model on first run (~50MB)
- Generates audio from dummy mel spectrogram
- Saves `test_outputs/melgan_test.wav`
- Audio should be audible (noise/random tones from random mel input)

### Step 3: Integration test with actual model

Once the vocoder works standalone, test with the full music generation pipeline:

```python
from music_generation.model import MusicGenerationModel
from music_generation.vocoder import load_pretrained_vocoder

# Load model + vocoder
model = MusicGenerationModel()
vocoder = load_pretrained_vocoder()

# Generate mel from model
mel = model.generate(...)  # Your generation logic

# Convert to audio
audio = vocoder(mel)
```

## Notes

- **Sample rate:** MelGAN is trained at 22,050 Hz
- **Mel bins:** Expects 80-dimensional mel spectrograms
- **Normalization:** Ensure your mel spectrograms match LJSpeech normalization range
- **Pure TensorFlow:** No PyTorch dependencies needed

### ⚠️ Critical: Verify Mel Model Compatibility

**Before integrating the vocoder, ensure your mel generation model uses matching parameters:**

1. **Sample rate:** Your audio processing must use 22,050 Hz
   - Check `audio_utils.py` or wherever audio is loaded/processed
   - Update `SAMPLE_RATE` constant if needed

2. **Mel bins:** Your model must output 80-dimensional mel spectrograms
   - Check the output shape of your mel generator
   - Verify `n_mels=80` in mel spectrogram computation

3. **FFT parameters:** Should match LJSpeech standard
   - `n_fft=1024`
   - `hop_length=256` (frame step)
   - `win_length=1024` (frame length)

4. **Normalization:** Mel values should be in LJSpeech range
   - Typically log-mel with mean/std normalization
   - Check your mel computation matches the vocoder's training data

**Action:** Review `src/music_generation/audio_utils.py` and model configuration to ensure compatibility. Mismatched parameters will cause poor audio quality or runtime errors.

## Troubleshooting

**Issue:** Model download fails
- **Solution:** Check internet connection, try manual download from Hugging Face

**Issue:** Audio quality is poor
- **Solution:** Check mel spectrogram normalization matches training data
- **Alternative:** Try MultiBand MelGAN for better quality

**Issue:** Shape mismatch errors
- **Solution:** Verify mel shape is [batch, time, 80] before passing to vocoder

## References

- TensorFlowTTS repo: https://github.com/TensorSpeech/TensorFlowTTS
- MelGAN model: https://huggingface.co/tensorspeech/tts-melgan-ljspeech-en
- Examples: https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples
