#!/usr/bin/env python3
"""Test single training step for vocoder."""

import tensorflow as tf
from pathlib import Path
from src.music_generation.vocoder import VocoderDataset
from src.music_generation.train_vocoder import VocoderTraining
from src.music_generation.hifigan.generator import TFHifiGANGenerator
from src.music_generation.hifigan.config import get_default_config
from src.music_generation.losses import MultiResolutionSTFTLoss

# Configure memory
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Parameters
data_dir = Path.home() / 'data' / 'music' / 'musicnet'
batch_size = 2  # Small batch for testing

print("Creating HiFiGAN generator from scratch...")
hifigan_config = get_default_config()
generator = TFHifiGANGenerator(hifigan_config)
print(f"✓ Generator created")

print("\nCreating dataset...")
dataset = VocoderDataset(data_dir)
train_ds = dataset.create_dataset(batch_size=batch_size, shuffle=True)
print(f"✓ Dataset created")

print("\nCreating training wrapper...")
stft_loss = MultiResolutionSTFTLoss(
    fft_sizes=[512, 1024, 2048],
    hop_sizes=[128, 256, 512],
    win_sizes=[512, 1024, 2048]
)
training_model = VocoderTraining(generator, stft_loss)

# Build model with dummy input
dummy_mel = tf.random.normal([1, 100, 80])
_ = training_model(dummy_mel, training=False)

training_model.compile(optimizer=tf.keras.optimizers.Adam(0.0002))
print(f"✓ Training model compiled")
print(f"  Trainable variables: {len(training_model.trainable_variables)}")

print("\nRunning single training step...")
batch = next(iter(train_ds))
mel, audio = batch
print(f"  Batch shapes: mel={mel.shape}, audio={audio.shape}")

result = training_model.train_step((mel, audio))
print(f"✓ Training step completed")
print(f"  Loss: {result['loss']:.6f}")
print(f"  Gradient norm: {result['grad_norm']:.6f}")

# Verify loss is finite
assert tf.math.is_finite(result['loss']), "Loss is not finite!"
print("\n✓ All checks passed!")
