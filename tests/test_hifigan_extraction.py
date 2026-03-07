"""Tests for HiFi-GAN extraction."""

import pytest
import tensorflow as tf
from src.music_generation.hifigan import TFHifiGANGenerator
from src.music_generation.hifigan.config import get_default_config


def test_hifigan_import():
    """Test that HiFi-GAN modules can be imported."""
    from src.music_generation.hifigan import (
        TFHifiGANGenerator,
        TFReflectionPad1d,
        TFConvTranspose1d,
        WeightNormalization,
    )
    assert TFHifiGANGenerator is not None
    assert TFReflectionPad1d is not None
    assert TFConvTranspose1d is not None
    assert WeightNormalization is not None


def test_hifigan_generator_instantiation():
    """Test that HiFi-GAN generator can be instantiated."""
    config = get_default_config()
    generator = TFHifiGANGenerator(config)
    assert generator is not None


def test_hifigan_generator_forward():
    """Test that HiFi-GAN generator can perform forward pass."""
    config = get_default_config()
    generator = TFHifiGANGenerator(config)
    
    # Create fake mel input [batch, time, mel_bins]
    fake_mel = tf.random.uniform(shape=[1, 100, 80], dtype=tf.float32)
    
    # Forward pass
    output = generator(fake_mel)
    
    # Check output shape: [batch, time * 256, 1]
    assert output.shape[0] == 1
    assert output.shape[1] == 100 * 256  # 256 = 8*8*2*2 (upsample_scales)
    assert output.shape[2] == 1


def test_hifigan_output_range():
    """Test that HiFi-GAN output is in valid range."""
    config = get_default_config()
    generator = TFHifiGANGenerator(config)
    
    fake_mel = tf.random.uniform(shape=[1, 50, 80], dtype=tf.float32)
    output = generator(fake_mel)
    
    # With tanh activation, output should be in [-1, 1]
    assert tf.reduce_min(output) >= -1.0
    assert tf.reduce_max(output) <= 1.0
