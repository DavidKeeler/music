# Music Generation with TensorFlow

TensorFlow/Keras port of the PyTorch music generation system. Generates music using a Transformer-based mel spectrogram generator and MelGAN vocoder.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9-3.12
- TensorFlow 2.13+
- CUDA 11.8+ (for GPU training)

## Project Structure

```
src/music_generation/
├── config.py          # Hyperparameters
├── model.py           # MelGenerator (Keras Model)
├── layers.py          # Custom Keras layers
├── train.py           # Training with teacher forcing
├── train_vocoder.py   # Vocoder finetuning
├── vocoder.py         # MelGAN vocoder wrapper
├── losses.py          # STFT and adversarial losses
├── dataset.py         # tf.data.Dataset pipeline
├── audio_utils.py     # Audio preprocessing
└── inference.py       # End-to-end generation
```

## Training

### 1. Prepare Data

Download MusicNet dataset and extract to `~/data/music/musicnet/train_data/`.

### 2. Train Mel Generator

```bash
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints \
  --epochs 100 \
  --batch_size 16
```

**Key parameters:**
- `--initial_tf_ratio`: Initial teacher forcing ratio (default: 1.0)
- `--tf_decay_k`: Teacher forcing decay rate (default: 1e-5)
- `--min_tf_ratio`: Minimum teacher forcing ratio (default: 0.05)

### 3. Finetune Vocoder (Optional)

```bash
python -m src.music_generation.train_vocoder \
  --data_dir ~/data/music/musicnet/train_data \
  --checkpoint_dir ./vocoder_checkpoints \
  --epochs 50 \
  --batch_size 8
```

## Inference

```python
from src.music_generation.inference import MusicGenerationModel
from src.music_generation.audio_utils import load_audio, audio_to_mel
import soundfile as sf

# Load model
model = MusicGenerationModel.from_checkpoints(
    mel_checkpoint='./checkpoints/mel_generator.h5',
    vocoder_checkpoint='./vocoder_checkpoints/vocoder.h5'
)

# Load seed audio
seed_audio = load_audio('seed.wav')
seed_mel = audio_to_mel(seed_audio)

# Generate
audio = model.generate(seed_mel, num_frames=500)

# Save
sf.write('generated.wav', audio.numpy(), 22050)
```

## Architecture

**Mel Generator:**
- Transformer encoder-decoder with causal attention
- Input: mel spectrogram frames [batch, time, 80]
- Output: predicted next frame [batch, time, 80]
- Training: autoregressive with teacher forcing

**Vocoder:**
- MelGAN from TensorFlowTTS
- Input: mel spectrogram [batch, time, 80]
- Output: audio waveform [batch, samples]

## Testing

```bash
pytest tests/
```

## References

- Original PyTorch implementation: `conducting2/src/music_generation/`
- TensorFlow patterns: `conducting/src/main/python/complex_music_model/`
