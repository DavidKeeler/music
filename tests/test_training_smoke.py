"""Smoke tests for training scripts."""
import tensorflow as tf
import numpy as np
from src.music_generation.train import MelGeneratorTraining
from src.music_generation.train_vocoder import VocoderTraining
from src.music_generation.model import MelGenerator


def test_mel_generator_training_smoke():
    """Smoke test: MelGeneratorTraining can run one training step."""
    mel_gen = MelGenerator(d_model=64, num_heads=2, num_layers=2, d_ff=128)
    training_model = MelGeneratorTraining(mel_gen, initial_tf_ratio=1.0, tf_decay_k=1e-5, min_tf_ratio=0.05)
    
    # Synthetic batch
    batch_size, seq_len, mel_dim = 2, 10, 80
    x = tf.random.normal([batch_size, seq_len, mel_dim])
    
    # Run one training step
    result = training_model.train_step(x)
    
    assert "loss" in result
    assert not tf.math.is_nan(result["loss"])


def test_vocoder_training_smoke():
    """Smoke test: VocoderTraining can run one training step."""
    # Minimal mock vocoder
    class MockVocoder(tf.keras.Model):
        def call(self, mel):
            batch_size = tf.shape(mel)[0]
            time_steps = tf.shape(mel)[1]
            return tf.random.normal([batch_size, time_steps * 256])
    
    vocoder = MockVocoder()
    training_model = VocoderTraining(vocoder)
    
    # Synthetic batch: (mel, audio)
    batch_size, mel_time, mel_dim = 2, 50, 80
    audio_samples = mel_time * 256
    mel = tf.random.normal([batch_size, mel_time, mel_dim])
    audio = tf.random.normal([batch_size, audio_samples])
    
    # Run one training step
    result = training_model.train_step((mel, audio))
    
    assert "loss" in result
    assert "grad_norm" in result
    assert not tf.math.is_nan(result["loss"])
    assert not tf.math.is_nan(result["grad_norm"])
