"""Tests for MelGenerator model."""

import tensorflow as tf
import pytest
from src.music_generation.model import MelGenerator


def test_forward_shape():
    """Test forward pass preserves shape."""
    model = MelGenerator()
    x = tf.random.normal([2, 512, 80])
    y = model(x, training=False)
    assert y.shape == (2, 512, 80)


def test_forward_different_seq_lengths():
    """Test forward works with various sequence lengths."""
    model = MelGenerator()
    for seq_len in [64, 128, 256, 512]:
        x = tf.random.normal([1, seq_len, 80])
        y = model(x, training=False)
        assert y.shape == (1, seq_len, 80)


def test_generate_shape():
    """Test autoregressive generation produces correct shape."""
    model = MelGenerator()
    seed = tf.random.normal([64, 80])
    generated = model.generate(seed, num_frames=100, temperature=1.0)
    assert generated.shape == (100, 80)


def test_no_nan_inf():
    """Test model doesn't produce NaN or Inf."""
    model = MelGenerator()
    x = tf.random.normal([2, 256, 80])
    y = model(x, training=False)
    
    assert not tf.reduce_any(tf.math.is_nan(y))
    assert not tf.reduce_any(tf.math.is_inf(y))
