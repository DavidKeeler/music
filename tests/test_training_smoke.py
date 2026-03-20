"""Smoke tests for training scripts."""
import tensorflow as tf
import numpy as np
import tempfile
from pathlib import Path
from src.music_generation.train import MelGeneratorTraining
from src.music_generation.train_vocoder import VocoderTraining
from src.music_generation.model import MelGenerator
from src.music_generation.inference import MusicGenerationModel
from src.music_generation.config import N_MELS


def test_mel_generator_training_smoke():
    """Smoke test: MelGeneratorTraining can run one training step."""
    mel_gen = MelGenerator()
    training_model = MelGeneratorTraining(mel_gen)
    training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    # Synthetic batch — use 12 frames (divisible by TOKEN_COMPRESSION_RATIO=4)
    batch_size, seq_len = 2, 12
    x = tf.random.normal([batch_size, seq_len, N_MELS])
    y = x
    
    # Run one training step
    result = training_model.train_step((x, y))
    
    assert "loss" in result
    assert not tf.math.is_nan(result["loss"])


def test_vocoder_training_smoke():
    """Smoke test: VocoderTraining can run one training step."""
    # Minimal mock vocoder
    class MockVocoder(tf.keras.Model):
        def __init__(self):
            super().__init__()
            # Add a dummy layer with trainable weights
            self.dense = tf.keras.layers.Dense(256, use_bias=False)
        
        def call(self, mel):
            batch_size = tf.shape(mel)[0]
            time_steps = tf.shape(mel)[1]
            # Use the dense layer so gradients flow through
            features = self.dense(mel)  # [batch, time, 256]
            # Reshape to audio samples
            audio = tf.reshape(features, [batch_size, time_steps * 256])
            return audio
    
    # Minimal mock STFT loss
    class MockSTFTLoss(tf.keras.layers.Layer):
        def call(self, pred_audio, target_audio):
            return tf.reduce_mean(tf.abs(pred_audio - target_audio))
    
    vocoder = MockVocoder()
    stft_loss = MockSTFTLoss()
    training_model = VocoderTraining(vocoder, stft_loss)
    training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    
    # Synthetic batch: (mel, audio)
    batch_size, mel_time, mel_dim = 2, 50, 80
    audio_samples = mel_time * 256
    mel = tf.random.normal([batch_size, mel_time, mel_dim])
    audio = tf.random.normal([batch_size, audio_samples])
    
    # Build the model by calling it once
    _ = training_model(mel)
    
    # Run one training step
    result = training_model.train_step((mel, audio))
    
    assert "loss" in result
    assert "grad_norm" in result
    assert not tf.math.is_nan(result["loss"])
    assert not tf.math.is_nan(result["grad_norm"])


def test_checkpoint_save_and_load():
    """Smoke test: save and load checkpoint with .h5 extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from src.music_generation.model import MelGenerator
        
        mel_gen = MelGenerator()
        
        # Build model — use length divisible by TOKEN_COMPRESSION_RATIO
        mel_gen(tf.random.normal([1, 12, N_MELS]))
        
        mel_checkpoint = Path(tmpdir) / "mel_gen.weights.h5"
        mel_gen.save_weights(str(mel_checkpoint))
        
        mel_gen_loaded = MelGenerator()
        mel_gen_loaded(tf.random.normal([1, 12, N_MELS]))
        mel_gen_loaded.load_weights(str(mel_checkpoint))
        
        test_input = tf.random.normal([1, 12, N_MELS])
        output = mel_gen_loaded(test_input)
        
        assert output.shape == (1, 12, N_MELS)
        assert not tf.reduce_any(tf.math.is_nan(output))
