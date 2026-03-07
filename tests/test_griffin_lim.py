"""Tests for Griffin-Lim vocoder."""

import tensorflow as tf
import numpy as np
import pytest

from src.music_generation.vocoder import GriffinLimVocoder, load_pretrained_vocoder


def test_griffin_lim_initialization():
    """Test GriffinLimVocoder initialization."""
    vocoder = GriffinLimVocoder()
    assert vocoder.n_iter == 32
    assert vocoder.hop_length == 256
    assert vocoder.n_fft == 1024


def test_griffin_lim_shape():
    """Test Griffin-Lim output shape."""
    vocoder = GriffinLimVocoder(n_iter=8)  # Fewer iterations for speed
    
    # Create test mel spectrogram [batch, time, mel_bins]
    mel = tf.random.normal([2, 100, 80])
    
    # Generate audio
    audio = vocoder(mel)
    
    # Check shape: [batch, samples]
    assert audio.shape[0] == 2
    assert audio.ndim == 2
    # Expected samples: approximately time * hop_length
    # Griffin-Lim may produce slightly different lengths due to windowing
    expected_samples = 100 * 256
    assert abs(audio.shape[1] - expected_samples) < 500, \
        f"Audio length {audio.shape[1]} too far from expected {expected_samples}"


def test_griffin_lim_shape_transposed():
    """Test Griffin-Lim with transposed input [batch, mel_bins, time]."""
    vocoder = GriffinLimVocoder(n_iter=8)
    
    # Create test mel spectrogram [batch, mel_bins, time]
    mel = tf.random.normal([2, 80, 100])
    
    # Generate audio
    audio = vocoder(mel)
    
    # Check shape: [batch, samples]
    assert audio.shape[0] == 2
    assert audio.ndim == 2
    expected_samples = 100 * 256
    assert abs(audio.shape[1] - expected_samples) < 500


def test_griffin_lim_not_silent():
    """Test that Griffin-Lim generates non-silent audio."""
    vocoder = GriffinLimVocoder(n_iter=8)
    
    # Create test mel spectrogram with some energy
    mel = tf.random.normal([1, 100, 80], mean=-5.0, stddev=2.0)
    
    # Generate audio
    audio = vocoder(mel)
    
    # Check that audio is not silent
    max_abs = tf.reduce_max(tf.abs(audio))
    assert max_abs > 0.01, f"Audio is too quiet: max_abs={max_abs}"


def test_griffin_lim_values_in_range():
    """Test that Griffin-Lim output values are reasonable."""
    vocoder = GriffinLimVocoder(n_iter=8)
    
    # Create test mel spectrogram
    mel = tf.random.normal([1, 100, 80], mean=-5.0, stddev=2.0)
    
    # Generate audio
    audio = vocoder(mel)
    
    # Check that values are in reasonable range (not NaN or Inf)
    assert not tf.reduce_any(tf.math.is_nan(audio))
    assert not tf.reduce_any(tf.math.is_inf(audio))
    
    # Audio should be roughly in reasonable range (may exceed [-1, 1] slightly)
    max_abs = tf.reduce_max(tf.abs(audio))
    assert max_abs < 20.0, f"Audio values too large: max_abs={max_abs}"


def test_load_pretrained_vocoder_griffin_lim():
    """Test loading Griffin-Lim via load_pretrained_vocoder."""
    vocoder = load_pretrained_vocoder(backend="griffin-lim")
    
    assert isinstance(vocoder, GriffinLimVocoder)
    
    # Test that it works
    mel = tf.random.normal([1, 50, 80])
    audio = vocoder(mel)
    assert audio.shape[0] == 1
    expected_samples = 50 * 256
    assert abs(audio.shape[1] - expected_samples) < 500


def test_load_pretrained_vocoder_invalid_backend():
    """Test that invalid backend raises error."""
    with pytest.raises(ValueError, match="Unknown backend"):
        load_pretrained_vocoder(backend="invalid")
