"""Test dataset loading and preprocessing for vocoder training."""

import os
import tensorflow as tf
from pathlib import Path
from src.music_generation import vocoder
from src.music_generation import config

def test_dataset_loading():
    """Test that VocoderDataset loads and preprocesses data correctly."""
    
    # Use MusicNet dataset
    data_dir = os.path.expanduser("~/data/music/musicnet/train_data")
    
    print(f"Testing dataset loading from: {data_dir}")
    
    # Check directory exists
    if not Path(data_dir).exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return False
    
    # Create dataset
    print("Creating VocoderDataset...")
    dataset = vocoder.VocoderDataset(
        data_dir,
        segment_length=config.VOCODER_SEGMENT_LENGTH
    )
    
    # Check audio files found
    num_files = len(dataset.audio_files)
    print(f"Found {num_files} audio files")
    
    if num_files == 0:
        print("ERROR: No audio files found")
        return False
    
    # Create tf.data.Dataset
    print("Creating tf.data.Dataset with batch_size=2...")
    train_dataset = dataset.create_dataset(batch_size=2, shuffle=True)
    
    # Test loading one batch
    print("Loading first batch...")
    for mel, audio in train_dataset.take(1):
        print(f"  Mel shape: {mel.shape}")
        print(f"  Audio shape: {audio.shape}")
        
        # Validate shapes
        batch_size = mel.shape[0]
        mel_time = mel.shape[1]
        mel_bins = mel.shape[2]
        audio_len = audio.shape[1]
        
        print(f"\nValidation:")
        print(f"  Batch size: {batch_size} (expected: 2)")
        print(f"  Mel bins: {mel_bins} (expected: 80)")
        print(f"  Audio length: {audio_len} (expected: {config.VOCODER_SEGMENT_LENGTH})")
        
        # Check for valid values
        mel_finite = tf.reduce_all(tf.math.is_finite(mel))
        audio_finite = tf.reduce_all(tf.math.is_finite(audio))
        
        print(f"  Mel finite: {mel_finite.numpy()}")
        print(f"  Audio finite: {audio_finite.numpy()}")
        
        # Check for non-silent audio
        audio_mean = tf.reduce_mean(tf.abs(audio))
        print(f"  Audio mean abs: {audio_mean.numpy():.6f}")
        
        # Assertions
        assert batch_size == 2, f"Expected batch_size=2, got {batch_size}"
        assert mel_bins == 80, f"Expected mel_bins=80, got {mel_bins}"
        assert audio_len == config.VOCODER_SEGMENT_LENGTH, f"Expected audio_len={config.VOCODER_SEGMENT_LENGTH}, got {audio_len}"
        assert mel_finite, "Mel contains non-finite values"
        assert audio_finite, "Audio contains non-finite values"
        assert audio_mean > 0.0, "Audio is silent"
        
        print("\n✅ All validations passed!")
        return True
    
    print("ERROR: No batches loaded")
    return False


if __name__ == '__main__':
    success = test_dataset_loading()
    exit(0 if success else 1)
