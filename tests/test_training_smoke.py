"""Smoke tests for training scripts."""
import tensorflow as tf
import numpy as np
import tempfile
from pathlib import Path
from src.music_generation.train import MelGeneratorTraining
from src.music_generation.train_vocoder import VocoderTraining
from src.music_generation.model import MelGenerator
from src.music_generation.inference import MusicGenerationModel


def test_mel_generator_training_smoke():
    """Smoke test: MelGeneratorTraining can run one training step."""
    mel_gen = MelGenerator(d_model=64, num_heads=2, num_layers=2, d_ff=128)
    training_model = MelGeneratorTraining(mel_gen)
    training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    # Synthetic batch
    batch_size, seq_len, mel_dim = 2, 10, 80
    x = tf.random.normal([batch_size, seq_len, mel_dim])
    y = x
    
    # Run one training step
    result = training_model.train_step((x, y))
    
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


def test_checkpoint_save_and_load():
    """Smoke test: save and load SavedModel checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal models
        class SimpleGenerator(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = tf.keras.layers.Dense(80)
            
            def call(self, x, training=False):
                return self.dense(x)
        
        class SimpleVocoder(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = tf.keras.layers.Dense(22050)
            
            def call(self, mel, training=False):
                return self.dense(mel[:, 0, :])
        
        # Create and save models
        mel_gen = SimpleGenerator()
        vocoder = SimpleVocoder()
        
        mel_checkpoint = Path(tmpdir) / "mel_gen"
        vocoder_checkpoint = Path(tmpdir) / "vocoder"
        
        mel_gen.save(str(mel_checkpoint))
        vocoder.save(str(vocoder_checkpoint))
        
        # Load via MusicGenerationModel
        model = MusicGenerationModel.from_checkpoints(
            str(mel_checkpoint),
            str(vocoder_checkpoint)
        )
        
        # Verify it works
        seed_mel = tf.random.normal([10, 80])
        audio = model.generate(seed_mel, num_frames=5)
        
        assert len(audio.shape) == 1
        assert audio.shape[0] > 0
