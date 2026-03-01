"""Tests for vocoder training."""

import tensorflow as tf
import pytest
from src.music_generation.train_vocoder import VocoderTraining
from src.music_generation.losses import MultiResolutionSTFTLoss


def test_vocoder_training_gradient_norm():
    """Test that VocoderTraining tracks gradient norm."""
    # Create minimal vocoder that accepts [batch, time, 80] and outputs [batch, samples]
    class SimpleGenerator(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(22050)
        
        def call(self, mel, training=False):
            # mel: [batch, time, 80] -> audio: [batch, samples]
            # Use first frame to generate audio
            return self.dense(mel[:, 0, :])
    
    generator = SimpleGenerator()
    
    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[512],
        hop_sizes=[128],
        win_sizes=[512]
    )
    model = VocoderTraining(generator, stft_loss)
    model.compile(optimizer='adam')
    
    # Create dummy data
    mel = tf.random.normal([2, 100, 80])
    audio = tf.random.normal([2, 22050])
    
    # Run one training step
    metrics = model.train_step((mel, audio))
    
    # Check that grad_norm is tracked
    assert "grad_norm" in metrics
    assert metrics["grad_norm"] > 0
    assert not tf.math.is_nan(metrics["grad_norm"])
