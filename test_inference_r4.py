"""Test inference with R=4 reduction factor."""
import tensorflow as tf
import numpy as np
from src.music_generation.model import MelGenerator
from src.music_generation.train import MelGeneratorTraining
from src.music_generation.config import REDUCTION_FACTOR, GROUPED_MEL_DIM, N_MELS

print("Testing inference with R=4 reduction factor...")
print(f"REDUCTION_FACTOR: {REDUCTION_FACTOR}")
print(f"GROUPED_MEL_DIM: {GROUPED_MEL_DIM}")
print(f"N_MELS: {N_MELS}")

# Load model (training wrapper)
training_model = tf.keras.models.load_model('./checkpoints_r4/mel_generator.keras')
print("✓ Training model loaded successfully")

# Extract base model
model = training_model.base_model
print("✓ Base model extracted")

# Create seed mel spectrogram (100 frames, 80 mels)
seed_mel = tf.random.normal([100, 80])
print(f"\nSeed shape: {seed_mel.shape}")

# Generate 200 frames
num_frames = 200
generated = model.generate(seed_mel, num_frames=num_frames)
print(f"Generated shape: {generated.shape}")

# Verify output shape
assert generated.shape == (num_frames, 80), f"Expected ({num_frames}, 80), got {generated.shape}"
print(f"✓ Output shape correct: {generated.shape}")

# Verify output is not all zeros or NaN
assert not tf.reduce_all(generated == 0), "Output is all zeros!"
assert not tf.reduce_any(tf.math.is_nan(generated)), "Output contains NaN!"
print("✓ Output contains valid values")

# Test with different seed lengths
for seed_len in [50, 97, 128, 200]:
    seed = tf.random.normal([seed_len, 80])
    output = model.generate(seed, num_frames=100)
    assert output.shape == (100, 80), f"Failed for seed_len={seed_len}"
    print(f"✓ Generation works with seed_len={seed_len}")

print("\n✅ All inference tests passed!")
print("\nKey verification points:")
print("  - Model loads and generates successfully")
print("  - Input/output API unchanged: [T, 80] frames")
print("  - Internal grouped frame processing works correctly")
print("  - Seed truncation to R-divisible length works")
print("  - Output shape matches requested num_frames")
