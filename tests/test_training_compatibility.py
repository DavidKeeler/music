"""Test training compatibility with updated 3-layer architecture."""
import tensorflow as tf
import numpy as np
from src.music_generation.model import MelGenerator
from src.music_generation.train import MelGeneratorTraining
from src.music_generation.config import N_MELS


def test_training_step_no_errors():
    """Verify training step runs without errors on 3-layer model."""
    model = MelGenerator()
    training_model = MelGeneratorTraining(model)
    training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    # Create small batch
    batch_size, seq_len = 2, 64
    x = tf.random.normal([batch_size, seq_len, N_MELS])
    y = x  # Target is same as input for autoregressive
    
    # Run one training step
    result = training_model.train_step((x, y))
    
    # Verify loss is valid
    assert not tf.math.is_nan(result["loss"])
    assert not tf.math.is_inf(result["loss"])
    assert result["loss"] > 0


def test_training_multiple_steps():
    """Verify model can train for multiple steps without diverging."""
    model = MelGenerator()
    training_model = MelGeneratorTraining(model)
    training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    batch_size, seq_len = 2, 128
    losses = []
    
    for _ in range(5):
        x = tf.random.normal([batch_size, seq_len, N_MELS])
        y = x
        result = training_model.train_step((x, y))
        losses.append(result["loss"].numpy())
    
    # Verify all losses are valid
    for loss in losses:
        assert not np.isnan(loss)
        assert not np.isinf(loss)
        assert loss > 0


def test_teacher_forcing_ratio_decay():
    """Verify teacher forcing ratio decays over training steps."""
    model = MelGenerator()
    training_model = MelGeneratorTraining(
        model, initial_tf_ratio=1.0, decay_k=1e-3, min_ratio=0.05
    )
    training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    batch_size, seq_len = 2, 64
    x = tf.random.normal([batch_size, seq_len, N_MELS])
    y = x
    
    ratios = []
    for _ in range(10):
        result = training_model.train_step((x, y))
        ratios.append(result["tf_ratio"].numpy())
    
    # Verify ratio decreases
    assert ratios[0] > ratios[-1]
    # Verify ratio stays above minimum
    assert all(r >= 0.05 for r in ratios)
