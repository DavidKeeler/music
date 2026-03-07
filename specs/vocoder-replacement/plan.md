# Implementation Plan: Vocoder Replacement

## Checklist

- [ ] Step 1: Add mel normalization
- [ ] Step 2: Implement Griffin-Lim vocoder (debug path)
- [ ] Step 3: Extract HiFi-GAN from TensorFlowTTS
- [ ] Step 4: Create HiFi-GAN wrapper
- [ ] Step 5: Implement Vocos wrapper (fallback)
- [ ] Step 6: Update unified vocoder interface
- [ ] Step 7: Update train_vocoder.py
- [ ] Step 8: Add vocoder tests
- [ ] Step 9: Update inference.py integration
- [ ] Step 10: Fine-tune on MusicNet

---

## Step 1: Add Mel Normalization

**Objective:** Compute dataset statistics and add normalization layer for mel spectrograms

**Implementation:**

1. Create `MelNormalizer` class in `audio_utils.py`:
```python
class MelNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def normalize(self, mel: tf.Tensor) -> tf.Tensor:
        return (mel - self.mean) / self.std
    
    def denormalize(self, mel: tf.Tensor) -> tf.Tensor:
        return mel * self.std + self.mean
    
    @classmethod
    def from_dataset(cls, cache_dir: str):
        # Load cached mels, compute mean/std
        pass
```

2. Add script to compute statistics: `scripts/compute_mel_stats.py`

3. Save statistics to `config.py` or JSON file

**Tests:**
- `test_mel_normalizer()` - verify normalization/denormalization
- `test_compute_stats()` - verify statistics computation

**Integration:**
- Mel normalizer used before vocoder inference
- Statistics computed once, saved to config

**Demo:** Run `compute_mel_stats.py` on cached mels, verify mean ≈ 0, std ≈ 1 after normalization

---

## Step 2: Implement Griffin-Lim Vocoder

**Objective:** Add Griffin-Lim as debug/testing vocoder (no training needed)

**Implementation:**

1. Create `src/music_generation/vocoder/griffin_lim.py`:
```python
class GriffinLimVocoder:
    def __init__(self, n_iter: int = 32):
        self.n_iter = n_iter
    
    def __call__(self, mel: tf.Tensor) -> tf.Tensor:
        # Convert to numpy, use librosa
        mel_np = mel.numpy()
        audio_list = []
        for mel_frame in mel_np:
            stft = librosa.feature.inverse.mel_to_stft(mel_frame, ...)
            audio = librosa.griffinlim(stft, n_iter=self.n_iter, ...)
            audio_list.append(audio)
        return tf.constant(np.array(audio_list))
```

2. Add to `vocoder.py` interface

**Tests:**
- `test_griffin_lim_shape()` - verify output shape
- `test_griffin_lim_not_silent()` - verify audio generated

**Integration:**
- Available as `backend="griffin-lim"` in `load_pretrained_vocoder()`

**Demo:** Generate audio from test mel, save as WAV, verify it's audible (though low quality)

---

## Step 3: Extract HiFi-GAN from TensorFlowTTS

**Objective:** Copy HiFi-GAN implementation files from TensorFlowTTS repo

**Implementation:**

1. Clone TensorFlowTTS:
```bash
cd /tmp
git clone https://github.com/TensorSpeech/TensorFlowTTS.git
```

2. Copy files to `src/music_generation/vocoder/hifigan/`:
   - `tensorflow_tts/models/hifigan.py` → `generator.py`
   - `tensorflow_tts/models/hifigan_discriminator.py` → `discriminator.py`
   - `tensorflow_tts/losses/` → `losses.py` (relevant parts)
   - `tensorflow_tts/configs/hifigan.yaml` → `config.py`

3. Remove internal TensorFlowTTS dependencies:
   - Replace imports with local equivalents
   - Extract only needed loss functions
   - Adapt config to our format

4. Create `__init__.py` to expose main classes

**Tests:**
- `test_hifigan_import()` - verify modules import without errors
- `test_generator_instantiation()` - create generator instance

**Integration:**
- Files are local to our codebase
- No pip install needed

**Demo:** Import HiFi-GAN classes, instantiate generator, verify no import errors

---

## Step 4: Create HiFi-GAN Wrapper

**Objective:** Wrap HiFi-GAN with consistent interface for our pipeline

**Implementation:**

1. Create `src/music_generation/vocoder/hifigan_wrapper.py`:
```python
class HiFiGANVocoder(tf.keras.Model):
    def __init__(self, config: dict):
        super().__init__()
        from .hifigan.generator import TFHifiGANGenerator
        self.generator = TFHifiGANGenerator(config)
    
    def call(self, mel: tf.Tensor, training: bool = False) -> tf.Tensor:
        # [B, T, 80] → [B, 80, T]
        mel_t = tf.transpose(mel, [0, 2, 1])
        audio = self.generator(mel_t, training=training)
        return tf.squeeze(audio, axis=-1)
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str):
        model = cls(default_config)
        model.load_weights(checkpoint_path)
        return model
```

2. Add pretrained weight download:
```python
def download_hifigan_pretrained():
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="tensorspeech/tts-hifigan-ljspeech-en",
        filename="generator.h5"
    )
```

**Tests:**
- `test_hifigan_wrapper_shape()` - verify input/output shapes
- `test_hifigan_pretrained_load()` - load pretrained weights

**Integration:**
- Used in `load_pretrained_vocoder(backend="hifigan")`

**Demo:** Load pretrained HiFi-GAN, generate audio from random mel, verify shape and non-silence

---

## Step 5: Implement Vocos Wrapper

**Objective:** Add Vocos (PyTorch) as fallback vocoder option

**Implementation:**

1. Add to `requirements.txt`:
```
# Optional: PyTorch vocoder (fallback)
torch>=2.0.0
vocos>=0.1.0
```

2. Create `src/music_generation/vocoder/vocos_wrapper.py`:
```python
class VocosWrapper:
    def __init__(self, model_name: str = "charactr/vocos-mel-22khz"):
        try:
            import torch
            from vocos import Vocos
        except ImportError:
            raise ImportError("Vocos requires: pip install torch vocos")
        
        self.vocos = Vocos.from_pretrained(model_name)
        self.vocos.eval()
    
    def __call__(self, mel: tf.Tensor) -> tf.Tensor:
        import torch
        
        mel_np = mel.numpy()
        mel_torch = torch.from_numpy(mel_np)
        
        with torch.no_grad():
            audio_torch = self.vocos.decode(mel_torch)
        
        return tf.constant(audio_torch.numpy())
```

**Tests:**
- `test_vocos_wrapper()` - verify TF→PyTorch→TF conversion
- `test_vocos_quality()` - compare to HiFi-GAN output

**Integration:**
- Available as `backend="vocos"` option
- Graceful fallback if torch not installed

**Demo:** Generate audio using Vocos, compare quality to HiFi-GAN and Griffin-Lim

---

## Step 6: Update Unified Vocoder Interface

**Objective:** Provide single entry point for all vocoder backends

**Implementation:**

1. Update `src/music_generation/vocoder.py`:
```python
def load_pretrained_vocoder(
    backend: str = "hifigan",
    checkpoint_path: Optional[str] = None
):
    """Load vocoder with automatic fallback."""
    
    if backend == "hifigan":
        try:
            from .vocoder.hifigan_wrapper import HiFiGANVocoder
            if checkpoint_path is None:
                checkpoint_path = download_hifigan_pretrained()
            return HiFiGANVocoder.from_pretrained(checkpoint_path)
        except Exception as e:
            logger.warning(f"HiFi-GAN failed: {e}, falling back to Vocos")
            backend = "vocos"
    
    if backend == "vocos":
        try:
            from .vocoder.vocos_wrapper import VocosWrapper
            return VocosWrapper()
        except ImportError:
            logger.warning("Vocos not installed, using Griffin-Lim")
            backend = "griffin-lim"
    
    if backend == "griffin-lim":
        from .vocoder.griffin_lim import GriffinLimVocoder
        return GriffinLimVocoder()
    
    raise ValueError(f"Unknown backend: {backend}")
```

2. Remove old TensorFlowTTS imports

**Tests:**
- `test_vocoder_backend_selection()` - verify each backend loads
- `test_vocoder_fallback()` - verify fallback chain works

**Integration:**
- Used by `inference.py` and `train_vocoder.py`

**Demo:** Load vocoder with each backend, verify consistent API

---

## Step 7: Update train_vocoder.py

**Objective:** Enable vocoder fine-tuning on MusicNet

**Implementation:**

1. Update `src/music_generation/train_vocoder.py`:
```python
def train_vocoder(args):
    # Load pretrained generator
    generator = HiFiGANVocoder.from_pretrained(args.pretrained_path)
    
    # Create discriminators
    from .vocoder.hifigan.discriminator import (
        MultiPeriodDiscriminator,
        MultiScaleDiscriminator
    )
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    
    # Phase 1: Generator-only (10k steps)
    logger.info("Phase 1: Generator-only training")
    train_generator_only(
        generator, dataset,
        steps=10000,
        lr=2e-4
    )
    
    # Phase 2: Full GAN (50k steps)
    logger.info("Phase 2: Full GAN training")
    train_gan(
        generator, mpd, msd, dataset,
        steps=50000,
        gen_lr=1e-4,
        disc_lr=2e-4
    )
```

2. Implement training loops with GAN losses

3. Add checkpoint saving

**Tests:**
- `test_train_vocoder_smoke()` - train for 10 steps, verify no errors

**Integration:**
- CLI command: `python -m src.music_generation.train_vocoder`

**Demo:** Run training for 100 steps, verify loss decreases, checkpoint saved

---

## Step 8: Add Vocoder Tests

**Objective:** Comprehensive test coverage for vocoder components

**Implementation:**

1. Create `tests/test_vocoder.py`:
```python
def test_vocoder_shape():
    vocoder = load_pretrained_vocoder(backend="griffin-lim")
    mel = tf.random.normal([2, 100, 80])
    audio = vocoder(mel)
    assert audio.shape == [2, 25600]

def test_vocoder_not_silent():
    vocoder = load_pretrained_vocoder(backend="griffin-lim")
    mel = tf.random.normal([1, 50, 80])
    audio = vocoder(mel)
    assert tf.reduce_max(tf.abs(audio)) > 0.01

def test_mel_format_validation():
    mel = tf.random.normal([1, 100, 80])
    validate_mel_format(mel)  # Should not raise

def test_backend_fallback():
    # Mock HiFi-GAN failure
    vocoder = load_pretrained_vocoder(backend="hifigan")
    # Should fall back to Vocos or Griffin-Lim
    assert vocoder is not None
```

2. Create `tests/test_vocoder_integration.py`:
```python
def test_end_to_end_generation():
    mel_gen = load_mel_generator("checkpoints/mel_gen.h5")
    vocoder = load_pretrained_vocoder()
    
    seed = load_audio("test_seed.wav")
    mel = mel_gen.generate(seed, num_frames=200)
    audio = vocoder(mel)
    
    assert audio.shape[1] == 200 * 256
    assert not tf.reduce_any(tf.math.is_nan(audio))
```

**Tests:**
- All tests pass with each vocoder backend

**Integration:**
- Run with `pytest tests/test_vocoder*.py`

**Demo:** Run full test suite, verify all tests pass

---

## Step 9: Update inference.py Integration

**Objective:** Enable end-to-end music generation with vocoder

**Implementation:**

1. Update `src/music_generation/inference.py`:
```python
class MusicGenerationModel:
    def __init__(self, mel_generator, vocoder):
        self.mel_generator = mel_generator
        self.vocoder = vocoder
        self.normalizer = MelNormalizer.from_config()
    
    def generate(self, seed_mel, num_frames=500):
        # Generate mel
        mel = self.mel_generator.generate(seed_mel, num_frames)
        
        # Normalize
        mel_normalized = self.normalizer.normalize(mel)
        
        # Convert to audio
        audio = self.vocoder(mel_normalized)
        
        return audio
    
    @classmethod
    def from_checkpoints(cls, mel_checkpoint, vocoder_backend="hifigan"):
        mel_gen = load_mel_generator(mel_checkpoint)
        vocoder = load_pretrained_vocoder(backend=vocoder_backend)
        return cls(mel_gen, vocoder)
```

2. Update CLI interface

**Tests:**
- `test_inference_end_to_end()` - full generation pipeline

**Integration:**
- Used for music generation scripts

**Demo:** Generate music from seed, save as WAV, listen to output

---

## Step 10: Fine-tune on MusicNet

**Objective:** Adapt pretrained vocoder to music domain

**Implementation:**

1. Prepare MusicNet dataset:
```bash
python -m src.music_generation.dataset \
  --data_dir ~/data/music/musicnet \
  --cache_dir ./cache
```

2. Run vocoder training:
```bash
python -m src.music_generation.train_vocoder \
  --data_dir ~/data/music/musicnet \
  --checkpoint_dir ./vocoder_checkpoints \
  --pretrained_path ./pretrained/hifigan.h5 \
  --epochs 50 \
  --batch_size 16
```

3. Monitor training:
   - Mel reconstruction loss should decrease
   - Listen to validation samples every 5k steps
   - Compare to Griffin-Lim baseline

4. Evaluate fine-tuned model:
   - Generate test audio samples
   - Measure spectral similarity
   - Subjective quality assessment

**Tests:**
- `test_finetuned_quality()` - compare to pretrained baseline

**Integration:**
- Fine-tuned checkpoint used in production inference

**Demo:** Generate music with fine-tuned vocoder, compare to pretrained and Griffin-Lim

---

## Summary

**Total steps:** 10  
**Estimated time:** 2-3 days development + 5-10 hours GPU training  
**Dependencies:** TensorFlow, librosa, huggingface_hub, optional (torch, vocos)

**Critical path:**
1. Steps 1-2: Quick wins (normalization, Griffin-Lim)
2. Steps 3-4: HiFi-GAN extraction (main work)
3. Step 6: Integration
4. Step 10: Fine-tuning (GPU time)

**Parallel work:**
- Step 5 (Vocos) can be done independently
- Step 8 (tests) can be written alongside implementation
