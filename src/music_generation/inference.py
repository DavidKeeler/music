"""Inference model for end-to-end music generation."""

import tensorflow as tf
from pathlib import Path
from .model import MelGenerator
from .vocoder import HiFiGANVocoder, load_vocoder_from_checkpoint


class MusicGenerationModel(tf.keras.Model):
    """End-to-end music generation: mel generation + vocoding."""
    
    def __init__(self, mel_generator: tf.keras.Model, vocoder: tf.keras.Model):
        """Initialize with mel generator and vocoder.
        
        Args:
            mel_generator: Trained MelGenerator model
            vocoder: Finetuned MelGAN vocoder
        """
        super().__init__()
        self.mel_generator = mel_generator
        self.vocoder = vocoder
    
    def call(self, inputs, training=False):
        """Forward pass: mel generation + vocoding.
        
        Args:
            inputs: Input tensor for mel generator
            training: Whether in training mode
            
        Returns:
            Audio waveform
        """
        mel = self.mel_generator(inputs, training=training)
        audio = self.vocoder(mel, training=training)
        return audio
    
    def generate(
        self,
        seed_mel: tf.Tensor,
        num_frames: int,
        temperature: float = 1.0
    ) -> tf.Tensor:
        """Generate audio from seed mel spectrogram.
        
        Args:
            seed_mel: Seed mel spectrogram [time, 80]
            num_frames: Number of mel frames to generate
            temperature: Sampling temperature
            
        Returns:
            Generated audio waveform [samples]
        """
        # Generate mel frames autoregressively
        generated_mel = self.mel_generator.generate(
            seed_mel, num_frames, temperature
        )  # [num_frames, 80]
        
        # Transform to vocoder format: [num_frames, 80] -> [1, num_frames, 80]
        mel_for_vocoder = tf.expand_dims(generated_mel, 0)
        
        # Decode to audio
        audio = self.vocoder(mel_for_vocoder, training=False)  # [1, samples]
        
        # Return 1D waveform
        return tf.squeeze(audio)
    
    @classmethod
    def from_checkpoints(
        cls,
        mel_checkpoint: str,
        vocoder_checkpoint: str
    ) -> "MusicGenerationModel":
        """Load from checkpoint files.
        
        Args:
            mel_checkpoint: Path to mel generator checkpoint (.h5 or SavedModel)
            vocoder_checkpoint: Path to vocoder checkpoint (.h5 or SavedModel)
            
        Returns:
            Initialized MusicGenerationModel
        """
        # Load mel generator
        mel_checkpoint_path = Path(mel_checkpoint)
        if mel_checkpoint_path.suffix == '.h5':
            mel_generator = MelGenerator()
            mel_generator.load_weights(str(mel_checkpoint))
        else:
            mel_generator = tf.keras.models.load_model(str(mel_checkpoint))
        
        # Load vocoder
        vocoder = load_vocoder_from_checkpoint(vocoder_checkpoint)
        
        return cls(mel_generator, vocoder)
