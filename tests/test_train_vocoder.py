"""Tests for vocoder training script."""

import pytest
import tensorflow as tf
import numpy as np
from src.music_generation.train_vocoder import VocoderTraining, train_vocoder


class SimpleGenerator(tf.keras.Model):
    """Minimal generator for testing."""
    
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(256)
    
    def call(self, mel_spectrogram, training=False):
        # mel: [B, T, 80] -> audio: [B, T*256]
        batch_size = tf.shape(mel_spectrogram)[0]
        time_steps = tf.shape(mel_spectrogram)[1]
        
        # Simple projection
        x = self.dense(mel_spectrogram)  # [B, T, 256]
        x = tf.reshape(x, [batch_size, time_steps * 256])  # [B, T*256]
        
        return x


def test_vocoder_training_initialization():
    """Test VocoderTraining wrapper initialization."""
    from src.music_generation.losses import MultiResolutionSTFTLoss
    
    generator = SimpleGenerator()
    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048]
    )
    
    training_model = VocoderTraining(generator, stft_loss)
    
    assert training_model.generator is generator
    assert training_model.stft_loss is stft_loss
    assert hasattr(training_model, 'grad_norm_tracker')


def test_vocoder_training_forward_pass():
    """Test VocoderTraining forward pass."""
    from src.music_generation.losses import MultiResolutionSTFTLoss
    
    generator = SimpleGenerator()
    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048]
    )
    training_model = VocoderTraining(generator, stft_loss)
    
    # Create dummy input
    mel = tf.random.normal([2, 100, 80])
    
    # Forward pass
    audio = training_model(mel, training=False)
    
    # Check output shape
    assert audio.shape[0] == 2
    assert audio.shape[1] == 100 * 256


def test_vocoder_training_train_step():
    """Test VocoderTraining train_step."""
    from src.music_generation.losses import MultiResolutionSTFTLoss
    
    generator = SimpleGenerator()
    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048]
    )
    training_model = VocoderTraining(generator, stft_loss)
    
    # Compile with optimizer
    training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    
    # Create dummy data
    mel = tf.random.normal([2, 100, 80])
    target_audio = tf.random.normal([2, 25600])
    
    # Train step
    result = training_model.train_step((mel, target_audio))
    
    # Check result contains loss and grad_norm
    assert 'loss' in result
    assert 'grad_norm' in result
    assert not tf.math.is_nan(result['loss'])
    assert not tf.math.is_nan(result['grad_norm'])


def test_vocoder_training_smoke():
    """Smoke test: train for a few steps."""
    from src.music_generation.losses import MultiResolutionSTFTLoss
    
    generator = SimpleGenerator()
    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048]
    )
    training_model = VocoderTraining(generator, stft_loss)
    
    # Compile
    training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    
    # Create dummy dataset
    def data_generator():
        for _ in range(10):
            mel = np.random.randn(2, 100, 80).astype(np.float32)
            audio = np.random.randn(2, 25600).astype(np.float32)
            yield mel, audio
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(2, 100, 80), dtype=tf.float32),
            tf.TensorSpec(shape=(2, 25600), dtype=tf.float32)
        )
    )
    
    # Train for 10 steps
    history = training_model.fit(dataset, epochs=1, verbose=0)
    
    # Check training completed
    assert 'loss' in history.history
    assert len(history.history['loss']) == 1
    assert not np.isnan(history.history['loss'][0])


def test_train_vocoder_function(tmp_path):
    """Test train_vocoder function."""
    from src.music_generation.losses import MultiResolutionSTFTLoss
    
    generator = SimpleGenerator()
    
    # Create dummy dataset
    def data_generator():
        for _ in range(5):
            mel = np.random.randn(2, 100, 80).astype(np.float32)
            audio = np.random.randn(2, 25600).astype(np.float32)
            yield mel, audio
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(2, 100, 80), dtype=tf.float32),
            tf.TensorSpec(shape=(2, 25600), dtype=tf.float32)
        )
    )
    
    # Train for 1 epoch
    checkpoint_dir = str(tmp_path / "checkpoints")
    train_vocoder(
        generator=generator,
        train_dataset=dataset,
        epochs=1,
        lr=1e-4,
        checkpoint_dir=checkpoint_dir
    )
    
    # Check checkpoint directory was created
    import os
    assert os.path.exists(checkpoint_dir)
