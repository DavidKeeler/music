"""Tests for HiFi-GAN vocoder wrapper."""

import pytest
import tensorflow as tf
import numpy as np
from src.music_generation.vocoder import HiFiGANVocoder, load_pretrained_vocoder


def test_hifigan_wrapper_initialization():
    """Test HiFi-GAN wrapper can be initialized."""
    from src.music_generation.hifigan.generator import TFHifiGANGenerator
    from src.music_generation.hifigan.config import get_default_config
    
    config = get_default_config()
    generator = TFHifiGANGenerator(config)
    
    vocoder = HiFiGANVocoder(generator=generator)
    assert vocoder.generator is not None


def test_hifigan_wrapper_shape():
    """Test HiFi-GAN wrapper handles correct input/output shapes."""
    from src.music_generation.hifigan.generator import TFHifiGANGenerator
    from src.music_generation.hifigan.config import get_default_config
    
    config = get_default_config()
    generator = TFHifiGANGenerator(config)
    vocoder = HiFiGANVocoder(generator=generator)
    
    # Input: [batch, time, mel_bins]
    mel = tf.random.normal([2, 100, 80])
    audio = vocoder(mel, training=False)
    
    # Output: [batch, samples]
    assert audio.shape[0] == 2
    assert len(audio.shape) == 2
    # With upsample_scales=[8,8,2,2], 100 frames -> 100*256 = 25600 samples
    assert audio.shape[1] == 25600


def test_hifigan_wrapper_not_silent():
    """Test HiFi-GAN wrapper produces non-silent audio."""
    from src.music_generation.hifigan.generator import TFHifiGANGenerator
    from src.music_generation.hifigan.config import get_default_config
    
    config = get_default_config()
    generator = TFHifiGANGenerator(config)
    vocoder = HiFiGANVocoder(generator=generator)
    
    mel = tf.random.normal([1, 100, 80])
    audio = vocoder(mel, training=False)
    
    # Audio should not be silent (random weights produce some output)
    assert tf.reduce_max(tf.abs(audio)) > 0.01


def test_hifigan_wrapper_value_range():
    """Test HiFi-GAN wrapper produces audio in valid range."""
    from src.music_generation.hifigan.generator import TFHifiGANGenerator
    from src.music_generation.hifigan.config import get_default_config
    
    config = get_default_config()
    generator = TFHifiGANGenerator(config)
    vocoder = HiFiGANVocoder(generator=generator)
    
    mel = tf.random.normal([1, 100, 80])
    audio = vocoder(mel, training=False)
    
    # Audio should be in reasonable range (tanh output is [-1, 1])
    assert tf.reduce_min(audio) >= -1.5
    assert tf.reduce_max(audio) <= 1.5


def test_load_pretrained_hifigan_backend():
    """Test loading HiFi-GAN via load_pretrained_vocoder()."""
    # This test will fail if huggingface_hub is not installed or download fails
    # We'll just test that the function exists and returns correct type
    try:
        vocoder = load_pretrained_vocoder(backend="hifigan")
        assert isinstance(vocoder, HiFiGANVocoder)
        assert vocoder.generator is not None
    except (ImportError, RuntimeError) as e:
        # Expected if huggingface_hub not installed or download fails
        pytest.skip(f"Skipping pretrained test: {e}")


def test_hifigan_from_pretrained_without_checkpoint():
    """Test HiFi-GAN.from_pretrained() creates model structure."""
    # Test that from_pretrained() at least creates the model structure
    # even if download fails
    try:
        vocoder = HiFiGANVocoder.from_pretrained()
        assert isinstance(vocoder, HiFiGANVocoder)
        assert vocoder.generator is not None
    except (ImportError, RuntimeError) as e:
        # Expected if huggingface_hub not installed or download fails
        pytest.skip(f"Skipping pretrained test: {e}")


def test_hifigan_wrapper_no_generator_error():
    """Test HiFi-GAN wrapper raises error when no generator loaded."""
    vocoder = HiFiGANVocoder(generator=None)
    mel = tf.random.normal([1, 100, 80])
    
    with pytest.raises(ValueError, match="No generator model loaded"):
        vocoder(mel)
