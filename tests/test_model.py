"""Tests for MelGenerator model."""

import tensorflow as tf
import pytest
from src.music_generation.model import MelGenerator
from src.music_generation.config import N_MELS, SEQ_LEN, D_MODEL, TOKEN_SEQ_LEN


def test_forward_shape():
    """Test forward pass with raw mel frames."""
    model = MelGenerator()
    x = tf.random.normal([2, SEQ_LEN, N_MELS])
    y = model(x, training=False)
    assert y.shape == (2, SEQ_LEN, N_MELS)


def test_forward_different_seq_lengths():
    """Test forward works with various sequence lengths divisible by C."""
    model = MelGenerator()
    for seq_len in [64, 128, 256, 512]:
        x = tf.random.normal([1, seq_len, N_MELS])
        y = model(x, training=False)
        assert y.shape == (1, seq_len, N_MELS)


def test_forward_arbitrary_lengths():
    """Test forward handles lengths not divisible by TOKEN_COMPRESSION_RATIO."""
    model = MelGenerator()
    for seq_len in [1, 5, 13, 63, 100, 510]:
        x = tf.random.normal([2, seq_len, N_MELS])
        y = model(x, training=False)
        assert y.shape == (2, seq_len, N_MELS), f"Failed for seq_len={seq_len}"


def test_forward_from_tokens():
    """Test forward_from_tokens produces correct shape."""
    model = MelGenerator()
    tokens = tf.random.normal([2, TOKEN_SEQ_LEN, D_MODEL])
    y = model.forward_from_tokens(tokens, training=False)
    assert y.shape == (2, SEQ_LEN, N_MELS)


def test_generate_shape():
    """Test autoregressive generation produces correct shape."""
    model = MelGenerator()
    seed = tf.random.normal([64, N_MELS])
    generated = model.generate(seed, num_frames=100, temperature=1.0)
    assert generated.shape == (100, N_MELS)


def test_no_nan_inf():
    """Test model doesn't produce NaN or Inf."""
    model = MelGenerator()
    x = tf.random.normal([2, 256, N_MELS])
    y = model(x, training=False)
    assert not tf.reduce_any(tf.math.is_nan(y))
    assert not tf.reduce_any(tf.math.is_inf(y))
