"""Tests for Vocos vocoder wrapper."""

import pytest
import tensorflow as tf
import numpy as np


def _has_vocos_deps():
    """Check if torch and vocos are installed."""
    try:
        import torch
        import vocos
        return True
    except ImportError:
        return False


def test_vocos_import_error():
    """Test that VocosWrapper raises ImportError when dependencies missing."""
    # This test will pass if torch/vocos are not installed
    # If they are installed, it will skip
    try:
        import torch
        import vocos
        pytest.skip("torch and vocos are installed")
    except ImportError:
        from src.music_generation.vocos_wrapper import VocosWrapper
        
        with pytest.raises(ImportError, match="Vocos requires PyTorch"):
            VocosWrapper()


@pytest.mark.skipif(
    not _has_vocos_deps(),
    reason="Requires torch and vocos: pip install torch vocos"
)
def test_vocos_initialization():
    """Test Vocos wrapper initialization."""
    from src.music_generation.vocos_wrapper import VocosWrapper
    
    vocoder = VocosWrapper()
    assert vocoder.vocos is not None


@pytest.mark.skipif(
    not _has_vocos_deps(),
    reason="Requires torch and vocos: pip install torch vocos"
)
def test_vocos_shape_time_last():
    """Test Vocos with [batch, time, 80] input."""
    from src.music_generation.vocos_wrapper import VocosWrapper
    
    vocoder = VocosWrapper()
    mel = tf.random.normal([2, 100, 80])
    audio = vocoder(mel)
    
    # Output should be [batch, samples]
    assert audio.shape[0] == 2
    assert len(audio.shape) == 2
    assert audio.shape[1] > 0  # Has audio samples


@pytest.mark.skipif(
    not _has_vocos_deps(),
    reason="Requires torch and vocos: pip install torch vocos"
)
def test_vocos_shape_mel_middle():
    """Test Vocos with [batch, 80, time] input."""
    from src.music_generation.vocos_wrapper import VocosWrapper
    
    vocoder = VocosWrapper()
    mel = tf.random.normal([2, 80, 100])
    audio = vocoder(mel)
    
    # Output should be [batch, samples]
    assert audio.shape[0] == 2
    assert len(audio.shape) == 2
    assert audio.shape[1] > 0


@pytest.mark.skipif(
    not _has_vocos_deps(),
    reason="Requires torch and vocos: pip install torch vocos"
)
def test_vocos_not_silent():
    """Test that Vocos generates non-silent audio."""
    from src.music_generation.vocos_wrapper import VocosWrapper
    
    vocoder = VocosWrapper()
    mel = tf.random.normal([1, 100, 80])
    audio = vocoder(mel)
    
    # Audio should not be silent
    max_abs = tf.reduce_max(tf.abs(audio))
    assert max_abs > 0.01


@pytest.mark.skipif(
    not _has_vocos_deps(),
    reason="Requires torch and vocos: pip install torch vocos"
)
def test_vocos_value_range():
    """Test that Vocos output is in valid audio range."""
    from src.music_generation.vocos_wrapper import VocosWrapper
    
    vocoder = VocosWrapper()
    mel = tf.random.normal([1, 100, 80])
    audio = vocoder(mel)
    
    # Audio should be in [-1, 1] range (or close)
    assert tf.reduce_min(audio) >= -2.0
    assert tf.reduce_max(audio) <= 2.0


@pytest.mark.skipif(
    not _has_vocos_deps(),
    reason="Requires torch and vocos: pip install torch vocos"
)
def test_load_vocos_vocoder():
    """Test load_vocos_vocoder helper function."""
    from src.music_generation.vocos_wrapper import load_vocos_vocoder
    
    vocoder = load_vocos_vocoder()
    assert vocoder is not None
    
    # Test it works
    mel = tf.random.normal([1, 50, 80])
    audio = vocoder(mel)
    assert audio.shape[0] == 1


def test_vocoder_backend_vocos():
    """Test loading Vocos via load_pretrained_vocoder."""
    from src.music_generation.vocoder import load_pretrained_vocoder, GriffinLimVocoder
    
    if not _has_vocos_deps():
        # Should fall back to Griffin-Lim (or raise if fallback disabled)
        vocoder = load_pretrained_vocoder(backend="vocos", enable_fallback=True)
        # With fallback enabled, should get Griffin-Lim
        assert isinstance(vocoder, GriffinLimVocoder)
        
        # Without fallback, should raise
        with pytest.raises(ImportError):
            load_pretrained_vocoder(backend="vocos", enable_fallback=False)
    else:
        # Should load successfully
        vocoder = load_pretrained_vocoder(backend="vocos")
        assert vocoder is not None
        
        # Test inference
        mel = tf.random.normal([1, 50, 80])
        audio = vocoder(mel)
        assert audio.shape[0] == 1

