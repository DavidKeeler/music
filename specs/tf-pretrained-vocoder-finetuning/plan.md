# Implementation Plan: TensorFlow Pretrained Vocoder Finetuning

## Overview

Enhance the existing TensorFlow vocoder implementation with three features: unified inference model, gradient norm logging, and SavedModel checkpoints. Each step builds incrementally and results in testable functionality.

## Checklist

- [ ] Step 1: Add gradient norm logging to VocoderTraining
- [ ] Step 2: Update checkpoint saving to SavedModel format
- [ ] Step 3: Update MelGAN loading and output handling
- [ ] Step 4: Create MusicGenerationModel inference class
- [ ] Step 5: Add unit tests
- [ ] Step 6: Add smoke tests

---

## Step 1: Add Gradient Norm Logging to VocoderTraining

**Objective:** Add gradient norm as a custom metric in the vocoder training loop.

**Files to modify:**
- `src/music_generation/train_vocoder.py`

**Implementation:**

1. Add `grad_norm_tracker` metric to `VocoderTraining.__init__()`:
```python
self.grad_norm_tracker = tf.keras.metrics.Mean(name="grad_norm")
```

2. Compute gradient norm in `train_step()` after computing gradients:
```python
grad_norm = tf.sqrt(sum([tf.reduce_sum(g**2) for g in grads if g is not None]))
```

3. Update metric and return in result dict:
```python
self.grad_norm_tracker.update_state(grad_norm)
return {"loss": loss, "grad_norm": self.grad_norm_tracker.result()}
```

4. Add `metrics` property:
```python
@property
def metrics(self):
    return [self.grad_norm_tracker]
```

**Test:**
- Run training for 1 step
- Verify `grad_norm` appears in logs
- Check TensorBoard shows grad_norm metric

**Integration:**
- Gradient norm automatically logged to TensorBoard via existing callback
- No changes needed to training script

**Demo:**
```bash
python -m src.music_generation.train_vocoder \
  --data_dir ~/data/music/musicnet/train_data \
  --epochs 1 --batch_size 2
# Check logs show: loss: X.XXX - grad_norm: Y.YYY
```

---

## Step 2: Update Checkpoint Saving to SavedModel Format

**Objective:** Save full model instead of weights-only for easier loading and finetuning.

**Files to modify:**
- `src/music_generation/train_vocoder.py`
- `src/music_generation/vocoder.py`

**Implementation:**

1. In `train_vocoder()`, change ModelCheckpoint parameter:
```python
tf.keras.callbacks.ModelCheckpoint(
    str(checkpoint_path),
    save_weights_only=False,  # Changed from True
    save_freq='epoch'
)
```

2. Update `load_vocoder_from_checkpoint()` in `vocoder.py`:
```python
def load_vocoder_from_checkpoint(checkpoint_path: str) -> HiFiGANVocoder:
    """Load vocoder from finetuned checkpoint.
    
    Args:
        checkpoint_path: Path to SavedModel directory
        
    Returns:
        Loaded vocoder model
    """
    vocoder = tf.keras.models.load_model(checkpoint_path)
    return vocoder
```

3. Remove the `.h5` file handling logic (no longer needed)

**Test:**
- Train for 1 epoch
- Verify checkpoint is a directory (not .h5 file)
- Load checkpoint with `load_vocoder_from_checkpoint()`
- Verify loaded model works for inference

**Integration:**
- Checkpoints now include optimizer state for resuming training
- Directory structure: `checkpoint_epoch_01/` instead of `checkpoint_epoch_01.h5`

**Demo:**
```bash
# Train and save
python -m src.music_generation.train_vocoder --epochs 1

# Verify checkpoint directory exists
ls checkpoints/vocoder/checkpoint_epoch_01/
# Should show: saved_model.pb, variables/, assets/
```

---

## Step 3: Update MelGAN Loading and Output Handling

**Objective:** Load MelGAN from TensorFlowTTS and handle its output shape correctly.

**Files to modify:**
- `src/music_generation/vocoder.py`
- `requirements.txt`

**Implementation:**

1. Add dependency to `requirements.txt`:
```
TensorFlowTTS>=1.0.0
```

2. Update `load_pretrained_vocoder()`:
```python
def load_pretrained_vocoder(model_url: Optional[str] = None) -> HiFiGANVocoder:
    """Load pretrained MelGAN vocoder from TensorFlowTTS.
    
    Args:
        model_url: Ignored (kept for compatibility)
        
    Returns:
        HiFiGANVocoder with pretrained MelGAN
    """
    from tensorflow_tts.inference import TFAutoModel
    
    melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")
    return HiFiGANVocoder(pretrained_model=melgan)
```

3. Update `HiFiGANVocoder.call()` to use MelGAN's `inference()` method:
```python
def call(self, mel_spectrogram, training=False):
    """Convert mel spectrogram to audio.
    
    Args:
        mel_spectrogram: [batch, time, 80]
        training: Whether in training mode
        
    Returns:
        Audio waveform [batch, samples]
    """
    if self.generator is None:
        raise ValueError("No generator model loaded")
    
    # Ensure [batch, time, 80] shape
    if mel_spectrogram.shape[1] == 80:
        mel_spectrogram = tf.transpose(mel_spectrogram, [0, 2, 1])
    
    # MelGAN returns [batch, samples, 1]
    audio = self.generator.inference(mel_spectrogram)
    
    # Squeeze to [batch, samples]
    return tf.squeeze(audio, axis=-1)
```

**Test:**
- Load pretrained vocoder
- Create dummy mel `[1, 100, 80]`
- Run inference
- Verify output shape is `[1, samples]` (not `[1, samples, 1]`)

**Integration:**
- First run downloads MelGAN model (~50MB)
- Existing training code works without changes

**Demo:**
```python
from src.music_generation.vocoder import load_pretrained_vocoder
import tensorflow as tf

vocoder = load_pretrained_vocoder()
mel = tf.random.normal([1, 100, 80])
audio = vocoder(mel)
print(audio.shape)  # Should be (1, samples)
```

---

## Step 4: Create MusicGenerationModel Inference Class

**Objective:** Create unified inference model combining MelGenerator and Vocoder.

**Files to create:**
- `src/music_generation/inference.py`

**Implementation:**

1. Create new file with `MusicGenerationModel` class:
```python
import tensorflow as tf
from .vocoder import HiFiGANVocoder

class MusicGenerationModel(tf.keras.Model):
    """End-to-end music generation combining MelGenerator and Vocoder."""
    
    def __init__(self, mel_generator, vocoder):
        super().__init__()
        self.mel_generator = mel_generator
        self.vocoder = vocoder
    
    def call(self, inputs, training=False):
        mel = self.mel_generator(inputs, training=training)
        audio = self.vocoder(mel, training=training)
        return audio
    
    def generate(self, seed_mel, num_frames, temperature=1.0):
        """Generate audio from seed mel spectrogram.
        
        Args:
            seed_mel: [time, 80]
            num_frames: Number of frames to generate
            temperature: Sampling temperature
            
        Returns:
            Audio waveform [samples]
        """
        # Generate mel
        generated_mel = self.mel_generator.generate(
            seed_mel, num_frames, temperature
        )
        
        # Add batch dim: [time, 80] -> [1, time, 80]
        mel_batch = tf.expand_dims(generated_mel, 0)
        
        # Vocode: [1, time, 80] -> [1, samples]
        audio = self.vocoder(mel_batch, training=False)
        
        # Remove batch dim: [1, samples] -> [samples]
        return tf.squeeze(audio, axis=0)
    
    @classmethod
    def from_checkpoints(cls, mel_checkpoint, vocoder_checkpoint):
        """Load from checkpoint files.
        
        Args:
            mel_checkpoint: Path to MelGenerator SavedModel
            vocoder_checkpoint: Path to Vocoder SavedModel
            
        Returns:
            MusicGenerationModel instance
        """
        mel_generator = tf.keras.models.load_model(mel_checkpoint)
        vocoder = tf.keras.models.load_model(vocoder_checkpoint)
        return cls(mel_generator, vocoder)
```

**Test:**
- Create mock MelGenerator and Vocoder
- Initialize MusicGenerationModel
- Call `generate()` with dummy seed
- Verify output is 1D audio tensor

**Integration:**
- Used for inference scripts
- Models trained separately, combined only for generation

**Demo:**
```python
from src.music_generation.inference import MusicGenerationModel
from src.music_generation.audio_utils import load_audio, audio_to_mel_ljspeech
import soundfile as sf

# Load models
model = MusicGenerationModel.from_checkpoints(
    'checkpoints/mel_generator',
    'checkpoints/vocoder/checkpoint_epoch_10'
)

# Load seed
seed_audio = load_audio('seed.wav')
seed_mel = audio_to_mel_ljspeech(seed_audio)

# Generate
audio = model.generate(seed_mel, num_frames=200)

# Save
sf.write('generated.wav', audio.numpy(), 22050)
```

---

## Step 5: Add Unit Tests

**Objective:** Test individual components in isolation.

**Files to create:**
- `tests/test_inference.py`

**Files to modify:**
- `tests/test_train_vocoder.py`

**Implementation:**

1. Create `tests/test_inference.py`:
```python
import tensorflow as tf
from src.music_generation.inference import MusicGenerationModel

def test_music_generation_model_init():
    """Test initialization."""
    mel_gen = tf.keras.Sequential([tf.keras.layers.Dense(80)])
    vocoder = tf.keras.Sequential([tf.keras.layers.Dense(1000)])
    
    model = MusicGenerationModel(mel_gen, vocoder)
    
    assert model.mel_generator is mel_gen
    assert model.vocoder is vocoder

def test_generate_returns_1d_audio():
    """Test generate method returns 1D audio."""
    # Create simple mock models
    mel_gen = create_mock_mel_generator()
    vocoder = create_mock_vocoder()
    
    model = MusicGenerationModel(mel_gen, vocoder)
    
    seed_mel = tf.random.normal([10, 80])
    audio = model.generate(seed_mel, num_frames=20)
    
    assert len(audio.shape) == 1  # 1D tensor
    assert audio.shape[0] > 0  # Has samples
```

2. Add to `tests/test_train_vocoder.py`:
```python
def test_vocoder_training_has_grad_norm():
    """Test gradient norm metric exists."""
    from src.music_generation.train_vocoder import VocoderTraining
    from src.music_generation.losses import MultiResolutionSTFTLoss
    
    generator = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    stft_loss = MultiResolutionSTFTLoss()
    
    model = VocoderTraining(generator, stft_loss)
    
    assert hasattr(model, 'grad_norm_tracker')
    assert 'grad_norm' in [m.name for m in model.metrics]
```

**Test:**
```bash
pytest tests/test_inference.py -v
pytest tests/test_train_vocoder.py::test_vocoder_training_has_grad_norm -v
```

**Integration:**
- Run with existing test suite
- All tests should pass

---

## Step 6: Add Smoke Tests

**Objective:** Verify training and loading work end-to-end on tiny dataset.

**Files to create:**
- `tests/test_training_smoke.py`

**Implementation:**

```python
import tensorflow as tf
import tempfile
from pathlib import Path
from src.music_generation.train_vocoder import train_vocoder
from src.music_generation.vocoder import load_pretrained_vocoder, load_vocoder_from_checkpoint

def test_vocoder_training_smoke():
    """Smoke test: train for 1 step."""
    # Create tiny dataset
    def dummy_generator():
        for _ in range(2):
            mel = tf.random.normal([50, 80])
            audio = tf.random.normal([8192])
            yield mel, audio
    
    dataset = tf.data.Dataset.from_generator(
        dummy_generator,
        output_signature=(
            tf.TensorSpec(shape=[50, 80], dtype=tf.float32),
            tf.TensorSpec(shape=[8192], dtype=tf.float32)
        )
    ).batch(1)
    
    # Train
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = load_pretrained_vocoder().generator
        train_vocoder(generator, dataset, epochs=1, lr=1e-4, checkpoint_dir=tmpdir)
        
        # Verify checkpoint exists
        checkpoints = list(Path(tmpdir).glob("checkpoint_epoch_*"))
        assert len(checkpoints) > 0

def test_checkpoint_save_and_load():
    """Smoke test: save and load checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save model
        vocoder = load_pretrained_vocoder()
        checkpoint_path = Path(tmpdir) / "test_checkpoint"
        vocoder.save(str(checkpoint_path))
        
        # Load checkpoint
        loaded = load_vocoder_from_checkpoint(str(checkpoint_path))
        
        # Verify it works
        mel = tf.random.normal([1, 50, 80])
        audio = loaded(mel)
        assert audio.shape[0] == 1
```

**Test:**
```bash
pytest tests/test_training_smoke.py -v -s
```

**Integration:**
- May take a few minutes (downloads MelGAN on first run)
- Verifies entire training pipeline works

---

## Summary

**Implementation order:**
1. Gradient norm logging (easiest, immediate value)
2. SavedModel checkpoints (improves checkpoint management)
3. MelGAN loading (enables pretrained model use)
4. Inference model (ties everything together)
5. Unit tests (validates components)
6. Smoke tests (validates end-to-end)

**Each step:**
- Builds on previous steps
- Results in working, testable functionality
- Can be demoed independently
- Integrates cleanly with existing code

**Total estimated changes:**
- ~150 lines of new code
- ~50 lines of modified code
- ~100 lines of test code
- Minimal, focused implementation
