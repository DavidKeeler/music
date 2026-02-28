"""Tests for audio processing utilities."""

import pytest
import tensorflow as tf
import tempfile
import os
import soundfile as sf
import numpy as np

from src.music_generation.audio_utils import (
    load_audio,
    audio_to_mel,
    normalize_mel,
    denormalize_mel,
)
from src.music_generation.config import SAMPLE_RATE, N_MELS


def create_test_audio(sample_rate=22050, duration=1.0, num_channels=1):
    """Create a test audio waveform (sine wave)."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    waveform = np.sin(2 * np.pi * 440 * t)
    if num_channels == 2:
        waveform = np.stack([waveform, waveform], axis=1)
    return waveform, sample_rate


class TestAudioToMel:
    """Tests for audio_to_mel function."""

    def test_mel_shape(self):
        """Test that mel spectrogram has correct shape."""
        waveform, _ = create_test_audio(duration=2.0)
        waveform = tf.constant(waveform, dtype=tf.float32)
        
        mel = audio_to_mel(waveform)
        
        assert len(mel.shape) == 2, "Mel should be 2D [time, n_mels]"
        assert mel.shape[1] == N_MELS, f"Expected {N_MELS} mel bins"
        assert mel.shape[0] > 0, "Should have time frames"

    def test_mel_no_nan_or_inf(self):
        """Test that mel spectrogram contains no NaN or Inf."""
        waveform, _ = create_test_audio(duration=1.0)
        waveform = tf.constant(waveform, dtype=tf.float32)
        
        mel = audio_to_mel(waveform)
        
        assert not tf.reduce_any(tf.math.is_nan(mel)), "Mel should not contain NaN"
        assert not tf.reduce_any(tf.math.is_inf(mel)), "Mel should not contain Inf"


class TestNormalization:
    """Tests for normalize_mel and denormalize_mel functions."""

    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize → denormalize is identity."""
        original = tf.random.normal([100, 80]) * 10 + 5
        mean = 5.0
        std = 10.0
        
        normalized = normalize_mel(original, mean, std)
        recovered = denormalize_mel(normalized, mean, std)
        
        assert tf.reduce_all(tf.abs(original - recovered) < 1e-5), \
            "Roundtrip should recover original values"
