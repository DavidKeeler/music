"""Tests for parallel training implementation."""

import tensorflow as tf
import pytest
from src.music_generation.train import MelGeneratorTraining
from src.music_generation.config import N_MELS


class SimpleMelGenerator(tf.keras.Model):
    """Minimal model for testing."""
    
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(N_MELS)
    
    def call(self, inputs, training=False):
        return self.dense(inputs)


def test_train_step_shapes():
    """Verify train_step produces correct shapes."""
    base_model = SimpleMelGenerator()
    model = MelGeneratorTraining(base_model)
    model.compile(optimizer='adam')
    
    batch_size, seq_len = 2, 32
    x = tf.random.normal([batch_size, seq_len, N_MELS])
    y = tf.random.normal([batch_size, seq_len, N_MELS])
    
    result = model.train_step((x, y))
    
    assert 'loss' in result
    assert result['loss'].shape == ()


def test_single_forward_pass():
    """Verify single forward pass (not autoregressive loop)."""
    base_model = SimpleMelGenerator()
    model = MelGeneratorTraining(base_model)
    
    batch_size, seq_len = 2, 32
    x = tf.random.normal([batch_size, seq_len, N_MELS])
    
    preds = model(x, training=True)
    
    assert preds.shape == x.shape


def test_loss_decreases():
    """Verify loss decreases over training steps."""
    base_model = SimpleMelGenerator()
    model = MelGeneratorTraining(base_model)
    model.compile(optimizer='adam')
    
    batch_size, seq_len = 4, 64
    x = tf.random.normal([batch_size, seq_len, N_MELS])
    y = tf.random.normal([batch_size, seq_len, N_MELS])
    
    losses = []
    for _ in range(10):
        result = model.train_step((x, y))
        losses.append(float(result['loss']))
    
    # Loss should decrease (at least final < initial)
    assert losses[-1] < losses[0]
