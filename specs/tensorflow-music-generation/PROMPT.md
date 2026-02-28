# PROMPT for Ralph: TensorFlow Music Generation System

## Objective

Port the PyTorch music generation system from `/Users/davidkeeler/code/conducting2` to TensorFlow/Keras in `/Users/davidkeeler/code/conducting3`, following the patterns established in `/Users/davidkeeler/code/conducting`.

## Context

**Source codebase (PyTorch):** `/Users/davidkeeler/code/conducting2/src/music_generation/`

Key files to port:
- `model.py` - MelGenerator (Transformer-based)
- `model_components.py` - Attention, FFN, positional encoding
- `train.py` - Training loop with teacher forcing
- `train_vocoder.py` - Vocoder finetuning
- `vocoder_losses.py` - STFT loss
- `dataset.py` - MusicNet dataset
- `audio_utils.py` - Audio preprocessing
- `inference.py` - End-to-end generation
- `config.py` - All hyperparameters

**Reference TensorFlow patterns:** `/Users/davidkeeler/code/conducting/src/main/python/complex_music_model/`
- `train_audio.py` - Shows Keras custom training with teacher forcing
- `model.py` - Shows Keras model structure
- `audio_config.py` - Shows configuration pattern

## Requirements

### 1. Project Structure

Create in `/Users/davidkeeler/code/conducting3/`:
```
src/
  music_generation/
    __init__.py
    config.py          # All hyperparameters
    model.py           # MelGenerator (Keras Model)
    layers.py          # Custom Keras layers
    train.py           # Training with model.fit()
    train_vocoder.py   # Vocoder training
    vocoder.py         # Vocoder model
    losses.py          # STFT and other losses
    dataset.py         # tf.data.Dataset pipeline
    audio_utils.py     # TensorFlow audio ops
    inference.py       # End-to-end generation
tests/
  test_*.py          # Unit tests
requirements.txt
README.md
```

### 2. Mel Generator Model

Convert PyTorch MelGenerator to Keras:
- Input: mel spectrogram frames [batch, time, mel_bins]
- Architecture: Transformer encoder-decoder
- Components:
  - Positional encoding (sinusoidal)
  - Multi-head self-attention
  - Feed-forward networks
  - Layer normalization
  - Residual connections
- Output: predicted next mel frame [batch, time, mel_bins]

**Key conversions:**
- `nn.Module` → `tf.keras.Model`
- `nn.Linear` → `tf.keras.layers.Dense`
- `nn.LayerNorm` → `tf.keras.layers.LayerNormalization`
- `nn.MultiheadAttention` → `tf.keras.layers.MultiHeadAttention`
- `F.relu` → `tf.nn.relu`
- Tensor shapes: PyTorch [B, C, T] → TensorFlow [B, T, C]

### 3. Training with Teacher Forcing

Implement custom `train_step()` following `/Users/davidkeeler/code/conducting/src/main/python/complex_music_model/train_audio.py`:

```python
class MelGeneratorTraining(tf.keras.Model):
    def __init__(self, base_model, initial_tf_ratio=1.0, decay_k=1e-5, min_ratio=0.05):
        super().__init__()
        self.base_model = base_model
        self.initial_tf_ratio = initial_tf_ratio
        self.decay_k = decay_k
        self.min_ratio = min_ratio
    
    def compute_tf_ratio(self):
        step = tf.cast(self.optimizer.iterations, tf.float32)
        ratio = self.initial_tf_ratio * tf.exp(-self.decay_k * step)
        return tf.maximum(self.min_ratio, ratio)
    
    def train_step(self, data):
        x, y = data
        tf_ratio = self.compute_tf_ratio()
        
        with tf.GradientTape() as tape:
            # Autoregressive generation with teacher forcing
            # ... (implement logic from PyTorch train.py)
            loss = # compute loss
        
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {"loss": loss, "tf_ratio": tf_ratio}
```

Use `model.fit()` with:
- Custom learning rate schedule (warmup + cosine decay)
- ModelCheckpoint callback
- TensorBoard callback
- EarlyStopping callback

### 4. Vocoder

**Option A (Recommended):** Use TensorFlow Hub pretrained vocoder
- Search for HiFi-GAN or MelGAN on TF Hub
- Finetune on music data

**Option B:** Convert PyTorch HiFi-GAN weights
- Load PyTorch checkpoint
- Rebuild architecture in Keras
- Transfer weights manually

**Option C:** Train from scratch
- Implement HiFi-GAN generator in Keras
- Train with STFT loss

Implement STFT loss in TensorFlow:
```python
def stft_loss(pred_audio, target_audio, fft_sizes, hop_sizes, win_sizes):
    total_loss = 0.0
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        pred_spec = tf.signal.stft(pred_audio, fft_size, hop_size, win_size)
        target_spec = tf.signal.stft(target_audio, fft_size, hop_size, win_size)
        
        pred_mag = tf.abs(pred_spec)
        target_mag = tf.abs(target_spec)
        
        total_loss += tf.reduce_mean(tf.abs(pred_mag - target_mag))
    
    return total_loss / len(fft_sizes)
```

### 5. Data Pipeline

Convert PyTorch Dataset to `tf.data.Dataset`:

```python
def create_dataset(data_dir, cache_dir, batch_size, shuffle=True):
    # Load audio files
    audio_files = glob.glob(f"{data_dir}/*.wav")
    
    def load_and_process(file_path):
        # Load audio
        audio, sr = tf.audio.decode_wav(tf.io.read_file(file_path))
        # Convert to mel
        mel = audio_to_mel(audio)
        # Create input/target pairs
        return mel[:-1], mel[1:]
    
    ds = tf.data.Dataset.from_tensor_slices(audio_files)
    ds = ds.map(load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(1000)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds
```

### 6. Audio Preprocessing

Convert torchaudio operations to TensorFlow:
- `torchaudio.load()` → `tf.audio.decode_wav()`
- `torchaudio.transforms.MelSpectrogram()` → `tf.signal.stft()` + `tf.signal.linear_to_mel_weight_matrix()`
- Match LJSpeech parameters from PyTorch config

### 7. Inference

End-to-end generation:
```python
class MusicGenerationModel(tf.keras.Model):
    def __init__(self, mel_generator, vocoder):
        super().__init__()
        self.mel_generator = mel_generator
        self.vocoder = vocoder
    
    def generate(self, initial_input, num_frames):
        # Autoregressive mel generation
        mel = self.mel_generator.generate(initial_input, num_frames)
        # Convert mel to audio
        audio = self.vocoder(mel)
        return audio
```

### 8. Configuration

Port all hyperparameters from `conducting2/src/music_generation/config.py`:
- Model dimensions (d_model, num_heads, num_layers, etc.)
- Training parameters (batch_size, learning_rate, epochs, etc.)
- Audio parameters (sample_rate, n_fft, hop_length, n_mels, etc.)
- Teacher forcing parameters (initial_ratio, decay_k, min_ratio)

### 9. Testing

Port all tests from `conducting2/tests/`:
- Model architecture tests
- Training step tests
- Data pipeline tests
- Audio processing tests
- Inference tests

Use `tf.test.TestCase` for TensorFlow-specific assertions.

## Acceptance Criteria

**Given** the TensorFlow implementation is complete  
**When** I run training with `python -m src.music_generation.train --data_dir ~/data/music/musicnet/train_data`  
**Then** the model trains successfully and saves checkpoints

**Given** a trained mel generator checkpoint  
**When** I run inference with `MusicGenerationModel.generate()`  
**Then** it produces audio waveforms matching the quality of the PyTorch version

**Given** all tests are ported  
**When** I run `pytest tests/`  
**Then** all tests pass

## Implementation Notes

1. **Tensor shapes**: TensorFlow uses [batch, time, features] while PyTorch uses [batch, features, time]. Adjust all operations accordingly.

2. **Checkpointing**: Use `tf.keras.callbacks.ModelCheckpoint` with SavedModel format or `.h5` format.

3. **Device placement**: TensorFlow handles GPU automatically, but you can use `with tf.device('/GPU:0'):` if needed.

4. **Eager execution**: Keep eager execution enabled for debugging. Use `@tf.function` for performance-critical functions.

5. **Reference the working TensorFlow code** in `/Users/davidkeeler/code/conducting/` for patterns and best practices.

## References

- PyTorch source: `/Users/davidkeeler/code/conducting2/src/music_generation/`
- TensorFlow reference: `/Users/davidkeeler/code/conducting/src/main/python/complex_music_model/`
- Spec directory: `/Users/davidkeeler/code/conducting3/specs/tensorflow-music-generation/`
