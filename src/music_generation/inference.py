"""Inference model for end-to-end music generation."""

import tensorflow as tf
from pathlib import Path
from .model import MelGenerator, LatentEncoder
from .vocoder import load_pretrained_vocoder, load_vocoder_from_checkpoint
from .audio_utils import MelNormalizer
from .config import LATENT_DIM


class MusicGenerationModel(tf.keras.Model):
    """End-to-end music generation: mel generation + vocoding."""
    
    def __init__(
        self,
        mel_generator: tf.keras.Model,
        vocoder: tf.keras.Model,
        normalizer: MelNormalizer = None
    ):
        """Initialize with mel generator and vocoder.
        
        Args:
            mel_generator: Trained MelGenerator model
            vocoder: Vocoder model (HiFi-GAN, Vocos, or Griffin-Lim)
            normalizer: Optional mel normalizer (if None, no normalization)
        """
        super().__init__()
        self.mel_generator = mel_generator
        self.vocoder = vocoder
        self.normalizer = normalizer
    
    def call(self, inputs, training=False):
        """Forward pass: mel generation + vocoding.
        
        Args:
            inputs: Input tensor for mel generator
            training: Whether in training mode
            
        Returns:
            Audio waveform
        """
        mel = self.mel_generator(inputs, training=training)
        
        # Denormalize if normalizer provided (model outputs normalized mels)
        if self.normalizer is not None:
            mel = self.normalizer.denormalize(mel)
        
        audio = self.vocoder(mel, training=training)
        return audio
    
    def generate(
        self,
        seed_mel: tf.Tensor,
        num_frames: int,
        temperature: float = 1.0,
        z: tf.Tensor = None
    ) -> tf.Tensor:
        """Generate audio from seed mel spectrogram.
        
        Args:
            seed_mel: Seed mel spectrogram [time, 80]
            num_frames: Number of mel frames to generate
            temperature: Sampling temperature
            z: Optional latent vector [1, latent_dim]. If None, sampled from N(0,1).
            
        Returns:
            Generated audio waveform [samples]
        """
        if z is None:
            z = tf.random.normal([1, LATENT_DIM])
        
        # Generate mel frames autoregressively
        generated_mel = self.mel_generator.generate(
            seed_mel, num_frames, temperature, z=z
        )  # [num_frames, 80]
        
        # Transform to vocoder format: [num_frames, 80] -> [1, num_frames, 80]
        mel_for_vocoder = tf.expand_dims(generated_mel, 0)
        
        # Denormalize if normalizer provided (model outputs normalized mels)
        if self.normalizer is not None:
            mel_for_vocoder = self.normalizer.denormalize(mel_for_vocoder)
        
        # Decode to audio
        audio = self.vocoder(mel_for_vocoder, training=False)  # [1, samples]
        
        # Return 1D waveform
        return tf.squeeze(audio)
    
    @classmethod
    def from_checkpoints(
        cls,
        mel_checkpoint: str,
        vocoder_checkpoint: str = None,
        vocoder_backend: str = "vocos",
        normalizer: MelNormalizer = None,
        enable_fallback: bool = True,
        use_pose: bool = False
    ) -> "MusicGenerationModel":
        """Load from checkpoint files.
        
        Args:
            mel_checkpoint: Path to mel generator checkpoint (.h5 or SavedModel)
            vocoder_checkpoint: Path to finetuned vocoder checkpoint (optional)
            vocoder_backend: Vocoder backend ("hifigan", "vocos", "griffin-lim")
            normalizer: Optional mel normalizer
            enable_fallback: Enable automatic fallback to other backends
            use_pose: Whether model was trained with pose conditioning
            
        Returns:
            Initialized MusicGenerationModel
        """
        # Load mel generator
        mel_checkpoint_path = Path(mel_checkpoint)
        if mel_checkpoint_path.suffix == '.h5':
            mel_generator = MelGenerator(use_pose=use_pose)
            mel_generator.load_weights(str(mel_checkpoint))
        else:
            mel_generator = tf.keras.models.load_model(str(mel_checkpoint))
        
        # Load vocoder
        if vocoder_checkpoint is not None:
            # Load finetuned vocoder from checkpoint
            vocoder = load_vocoder_from_checkpoint(vocoder_checkpoint)
        else:
            # Load pretrained vocoder with backend selection
            vocoder = load_pretrained_vocoder(
                backend=vocoder_backend,
                enable_fallback=enable_fallback
            )
        
        return cls(mel_generator, vocoder, normalizer)
