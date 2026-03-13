"""Tests for MelGenerator model."""

import tensorflow as tf
import pytest
from src.music_generation.model import MelGenerator
from src.music_generation.config import GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN, N_MELS


def test_forward_shape():
    """Test forward pass with grouped frames."""
    model = MelGenerator()
    # Input: [B, T/R, R*80] = [2, 128, 320]
    x = tf.random.normal([2, EFFECTIVE_SEQ_LEN, GROUPED_MEL_DIM])
    y = model(x, training=False)
    assert y.shape == (2, EFFECTIVE_SEQ_LEN, GROUPED_MEL_DIM)


def test_forward_different_seq_lengths():
    """Test forward works with various sequence lengths."""
    model = MelGenerator()
    for seq_len in [16, 32, 64, 128]:  # Grouped sequence lengths
        x = tf.random.normal([1, seq_len, GROUPED_MEL_DIM])
        y = model(x, training=False)
        assert y.shape == (1, seq_len, GROUPED_MEL_DIM)


def test_generate_shape():
    """Test autoregressive generation produces correct shape."""
    model = MelGenerator()
    # Seed with individual frames [64, 80]
    seed = tf.random.normal([64, N_MELS])
    # Generate 100 individual frames
    generated = model.generate(seed, num_frames=100, temperature=1.0)
    assert generated.shape == (100, N_MELS)


def test_no_nan_inf():
    """Test model doesn't produce NaN or Inf."""
    model = MelGenerator()
    # Input: [B, T/R, R*80] = [2, 64, 320]
    x = tf.random.normal([2, 64, GROUPED_MEL_DIM])
    y = model(x, training=False)
    
    assert not tf.reduce_any(tf.math.is_nan(y))
    assert not tf.reduce_any(tf.math.is_inf(y))
