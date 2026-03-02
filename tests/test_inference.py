"""Tests for inference module."""

import tensorflow as tf
import pytest
from src.music_generation.inference import MusicGenerationModel
from src.music_generation.model import MelGenerator


def test_music_generation_model_generate():
    """Test MusicGenerationModel.generate() produces audio."""
    mel_generator = MelGenerator()
    
    # Create minimal vocoder for testing
    class SimpleVocoder(tf.keras.Model):
        def call(self, mel, training=False):
            # mel: [batch, time, 80] -> audio: [batch, samples]
            batch_size = tf.shape(mel)[0]
            return tf.random.normal([batch_size, 22050])
    
    vocoder = SimpleVocoder()
    
    model = MusicGenerationModel(mel_generator, vocoder)
    
    # Generate from seed
    seed_mel = tf.random.normal([64, 80])
    audio = model.generate(seed_mel, num_frames=100, temperature=1.0)
    
    # Check output is 1D audio
    assert len(audio.shape) == 1
    assert audio.shape[0] > 0
    assert not tf.reduce_any(tf.math.is_nan(audio))


def test_music_generation_model_initialization():
    """Test MusicGenerationModel can be initialized with mel_generator and vocoder."""
    mel_generator = MelGenerator()
    
    class SimpleVocoder(tf.keras.Model):
        def call(self, mel, training=False):
            batch_size = tf.shape(mel)[0]
            return tf.random.normal([batch_size, 22050])
    
    vocoder = SimpleVocoder()
    
    model = MusicGenerationModel(mel_generator, vocoder)
    
    assert isinstance(model, MusicGenerationModel)
    assert model.mel_generator is not None
    assert model.vocoder is not None


def test_music_generation_model_call():
    """Test MusicGenerationModel.call() forward pass."""
    mel_generator = MelGenerator(d_model=64, num_heads=2, num_layers=2, d_ff=128)
    
    class SimpleVocoder(tf.keras.Model):
        def call(self, mel, training=False):
            # mel: [batch, time, 80] -> audio: [batch, samples]
            batch_size = tf.shape(mel)[0]
            time_steps = tf.shape(mel)[1]
            return tf.random.normal([batch_size, time_steps * 256])
    
    vocoder = SimpleVocoder()
    
    model = MusicGenerationModel(mel_generator, vocoder)
    
    # Test forward pass
    inputs = tf.random.normal([2, 10, 80])
    audio = model(inputs, training=False)
    
    # Check output shape
    assert len(audio.shape) == 2  # [batch, samples]
    assert audio.shape[0] == 2  # batch size
    assert audio.shape[1] > 0  # has samples
    assert not tf.reduce_any(tf.math.is_nan(audio))
