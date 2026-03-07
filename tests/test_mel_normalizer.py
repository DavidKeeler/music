"""Tests for mel normalization."""

import tensorflow as tf
import numpy as np
import pytest
from pathlib import Path
from src.music_generation.audio_utils import MelNormalizer


def test_mel_normalizer_normalize():
    """Test that normalization works correctly."""
    normalizer = MelNormalizer(mean=-5.0, std=2.0)
    
    # Create test mel with known values
    mel = tf.constant([[-5.0, -3.0, -7.0]], dtype=tf.float32)
    
    # Normalize
    normalized = normalizer.normalize(mel)
    
    # Expected: ([-5, -3, -7] - (-5)) / 2 = [0, 1, -1]
    expected = tf.constant([[0.0, 1.0, -1.0]], dtype=tf.float32)
    
    tf.debugging.assert_near(normalized, expected, atol=1e-6)


def test_mel_normalizer_denormalize():
    """Test that denormalization works correctly."""
    normalizer = MelNormalizer(mean=-5.0, std=2.0)
    
    # Create normalized mel
    normalized = tf.constant([[0.0, 1.0, -1.0]], dtype=tf.float32)
    
    # Denormalize
    mel = normalizer.denormalize(normalized)
    
    # Expected: [0, 1, -1] * 2 + (-5) = [-5, -3, -7]
    expected = tf.constant([[-5.0, -3.0, -7.0]], dtype=tf.float32)
    
    tf.debugging.assert_near(mel, expected, atol=1e-6)


def test_mel_normalizer_roundtrip():
    """Test that normalize -> denormalize is identity."""
    normalizer = MelNormalizer(mean=-4.5, std=1.8)
    
    # Create random mel
    original = tf.random.normal([10, 80], mean=-4.5, stddev=1.8)
    
    # Normalize then denormalize
    normalized = normalizer.normalize(original)
    reconstructed = normalizer.denormalize(normalized)
    
    # Should be close to original
    tf.debugging.assert_near(reconstructed, original, atol=1e-5)


def test_mel_normalizer_from_dataset(tmp_path):
    """Test computing statistics from cached mels."""
    # Create dummy mel files
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    # Create 3 mel files with known statistics
    # Mean should be -5.0, std should be 2.0
    mel1 = np.array([[-5.0, -3.0, -7.0], [-5.0, -5.0, -5.0]])  # shape [2, 3]
    mel2 = np.array([[-5.0, -7.0, -3.0]])  # shape [1, 3]
    mel3 = np.array([[-5.0, -5.0, -5.0], [-3.0, -7.0, -5.0]])  # shape [2, 3]
    
    np.save(cache_dir / "mel1.npy", mel1)
    np.save(cache_dir / "mel2.npy", mel2)
    np.save(cache_dir / "mel3.npy", mel3)
    
    # Compute statistics
    normalizer = MelNormalizer.from_dataset(str(cache_dir))
    
    # Check mean is approximately -5.0
    assert abs(normalizer.mean - (-5.0)) < 0.1
    
    # Check std is approximately 1.41 (std of [-7, -5, -3] repeated)
    assert abs(normalizer.std - 1.41) < 0.2


def test_mel_normalizer_from_dataset_empty(tmp_path):
    """Test that from_dataset raises error on empty directory."""
    cache_dir = tmp_path / "empty_cache"
    cache_dir.mkdir()
    
    with pytest.raises(ValueError, match="No cached mel files found"):
        MelNormalizer.from_dataset(str(cache_dir))


def test_mel_normalizer_batch():
    """Test normalization works with batched input."""
    normalizer = MelNormalizer(mean=-5.0, std=2.0)
    
    # Create batched mel [batch, time, mels]
    mel = tf.random.normal([4, 100, 80], mean=-5.0, stddev=2.0)
    
    # Normalize
    normalized = normalizer.normalize(mel)
    
    # Check shape preserved
    assert normalized.shape == mel.shape
    
    # Check approximately zero mean and unit std
    assert abs(tf.reduce_mean(normalized).numpy()) < 0.5
    assert abs(tf.math.reduce_std(normalized).numpy() - 1.0) < 0.5
