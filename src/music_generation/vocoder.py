"""Vocoder model for mel-to-audio conversion."""

import tensorflow as tf
import numpy as np
import librosa
from pathlib import Path
from typing import Optional

try:
    from TensorFlowTTS.tensorflow_tts.inference import TFAutoModel
except ImportError:
    TFAutoModel = None


class GriffinLimVocoder:
    """Griffin-Lim vocoder for mel-to-audio conversion (debug/testing only)."""
    
    def __init__(self, n_iter: int = 32, hop_length: int = 256, n_fft: int = 1024):
        """Initialize Griffin-Lim vocoder.
        
        Args:
            n_iter: Number of Griffin-Lim iterations
            hop_length: STFT hop length in samples
            n_fft: FFT size
        """
        self.n_iter = n_iter
        self.hop_length = hop_length
        self.n_fft = n_fft
    
    def __call__(self, mel: tf.Tensor) -> tf.Tensor:
        """Convert mel spectrogram to audio using Griffin-Lim.
        
        Args:
            mel: Mel spectrogram [batch, time, mel_bins] or [batch, mel_bins, time]
            
        Returns:
            Audio waveform [batch, samples]
        """
        # Convert to numpy
        mel_np = mel.numpy()
        
        # Handle shape: ensure [batch, mel_bins, time]
        if mel_np.ndim == 3 and mel_np.shape[2] == 100:
            # [batch, time, 80] -> [batch, 80, time]
            mel_np = np.transpose(mel_np, (0, 2, 1))
        
        # Convert mel to linear spectrogram and apply Griffin-Lim
        audio_list = []
        for mel_frame in mel_np:
            # mel_frame is [mel_bins, time]
            # Convert from log mel to linear mel
            mel_linear = np.exp(mel_frame)
            
            # Convert mel to STFT magnitude
            stft = librosa.feature.inverse.mel_to_stft(
                mel_linear,
                sr=24000,
                n_fft=self.n_fft,
                power=1.0
            )
            
            # Apply Griffin-Lim
            audio = librosa.griffinlim(
                stft,
                n_iter=self.n_iter,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            audio_list.append(audio)
        
        return tf.constant(np.array(audio_list), dtype=tf.float32)


class HiFiGANVocoder(tf.keras.Model):
    """Wrapper for HiFi-GAN vocoder model."""
    
    def __init__(self, generator=None):
        """Initialize vocoder.
        
        Args:
            generator: HiFi-GAN generator model (TFHifiGANGenerator or TF Hub model)
        """
        super().__init__()
        self.generator = generator
    
    def call(self, mel_spectrogram, training=False):
        """Convert mel spectrogram to audio.
        
        Args:
            mel_spectrogram: Mel spectrogram [batch, time, mel_bins]
            training: Whether in training mode
            
        Returns:
            Audio waveform [batch, samples]
        """
        if self.generator is None:
            raise ValueError("No generator model loaded")
        
        # Handle different generator types
        if hasattr(self.generator, 'inference'):
            # TF Hub MelGAN: expects [batch, time, mel_bins], outputs [batch, samples, 1]
            audio = self.generator.inference(mel_spectrogram)
            audio = tf.squeeze(audio, axis=-1)
        else:
            # HiFi-GAN: expects [batch, time, mel_bins], outputs [batch, samples, 1]
            audio = self.generator(mel_spectrogram, training=training)
            audio = tf.squeeze(audio, axis=-1)
        
        return audio
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: Optional[str] = None):
        """Load HiFi-GAN from pretrained weights.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, downloads from Hugging Face.
            
        Returns:
            HiFiGANVocoder instance with loaded weights
        """
        from .hifigan.generator import TFHifiGANGenerator
        from .hifigan.config import get_default_config
        
        # Create generator with default config
        config = get_default_config()
        generator = TFHifiGANGenerator(config)
        
        # Build generator with dummy input [batch, time, mel_bins]
        dummy_mel = tf.random.normal([1, 100, 80])
        _ = generator(dummy_mel, training=False)
        
        # Load weights if checkpoint provided
        if checkpoint_path is not None:
            generator.load_weights(checkpoint_path)
        else:
            # Download from Hugging Face
            try:
                from huggingface_hub import hf_hub_download
                checkpoint_path = hf_hub_download(
                    repo_id="tensorspeech/tts-hifigan-ljspeech-en",
                    filename="generator.h5"
                )
                generator.load_weights(checkpoint_path)
            except ImportError:
                raise ImportError(
                    "huggingface_hub not installed. "
                    "Run: pip install huggingface_hub"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download pretrained weights: {e}")
        
        return cls(generator=generator)


def load_pretrained_vocoder(
    backend: str = "hifigan",
    model_name: Optional[str] = None,
    enable_fallback: bool = True
):
    """Load pretrained vocoder with automatic fallback.
    
    Args:
        backend: Vocoder backend ("hifigan", "vocos", "melgan", or "griffin-lim")
        model_name: Model name for TFAutoModel (melgan) or Vocos. If None, uses defaults.
        enable_fallback: If True, automatically fall back to next available backend on failure
        
    Returns:
        Vocoder instance (HiFiGANVocoder, VocosWrapper, or GriffinLimVocoder)
        
    Note:
        Fallback chain (if enable_fallback=True):
        1. HiFi-GAN (best quality, requires huggingface_hub)
        2. Vocos (excellent quality, requires torch)
        3. Griffin-Lim (debug only, always available)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Try requested backend
    if backend == "hifigan":
        try:
            return HiFiGANVocoder.from_pretrained()
        except Exception as e:
            if not enable_fallback:
                raise
            logger.warning(f"HiFi-GAN failed ({e}), falling back to Vocos")
            backend = "vocos"
    
    if backend == "vocos":
        try:
            from .vocos_wrapper import load_vocos_vocoder
            if model_name is None:
                model_name = "charactr/vocos-mel-24khz"
            return load_vocos_vocoder(model_name=model_name)
        except (ImportError, Exception) as e:
            if not enable_fallback:
                raise
            logger.warning(f"Vocos failed ({e}), falling back to Griffin-Lim")
            backend = "griffin-lim"
    
    if backend == "griffin-lim":
        logger.info("Using Griffin-Lim vocoder (debug quality)")
        return GriffinLimVocoder()
    
    if backend == "melgan":
        if TFAutoModel is None:
            if enable_fallback:
                logger.warning("TensorFlowTTS not installed, falling back to HiFi-GAN")
                return load_pretrained_vocoder("hifigan", enable_fallback=enable_fallback)
            raise ImportError("TensorFlowTTS not installed. Run: pip install TensorFlowTTS")
        
        try:
            if model_name is None:
                model_name = "tensorspeech/tts-melgan-ljspeech-en"
            pretrained_model = TFAutoModel.from_pretrained(model_name)
            return HiFiGANVocoder(generator=pretrained_model)
        except Exception as e:
            if not enable_fallback:
                raise
            logger.warning(f"MelGAN failed ({e}), falling back to HiFi-GAN")
            return load_pretrained_vocoder("hifigan", enable_fallback=enable_fallback)
    
    raise ValueError(f"Unknown backend: {backend}. Choose 'hifigan', 'vocos', 'melgan', or 'griffin-lim'")



def load_vocoder_from_checkpoint(checkpoint_path: str) -> HiFiGANVocoder:
    """Load vocoder from finetuned checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory or .h5 file
        
    Returns:
        Loaded HiFiGANVocoder
    """
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.suffix == '.h5':
        # Load from .h5 file
        vocoder = HiFiGANVocoder()
        vocoder.load_weights(str(checkpoint_path))
    else:
        # Load from SavedModel directory
        vocoder = tf.keras.models.load_model(str(checkpoint_path))
    
    return vocoder


class VocoderDataset:
    """Dataset for vocoder finetuning."""
    
    def __init__(self, data_dir: str, segment_length: int = 8192):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing .wav files
            segment_length: Audio segment length in samples
        """
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.audio_files = list(self.data_dir.rglob("*.wav"))
    
    def __len__(self):
        return len(self.audio_files)
    
    def _load_audio_segment(self, file_path):
        """Load and process audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            mel: Mel spectrogram [time, mel_bins]
            audio: Audio segment [segment_length]
        """
        from . import audio_utils
        
        # Load audio
        audio = audio_utils.load_audio(str(file_path))
        
        # Random crop to segment length
        audio_len = tf.shape(audio)[0]
        if audio_len > self.segment_length:
            max_start = audio_len - self.segment_length
            start = tf.random.uniform([], 0, max_start, dtype=tf.int32)
            audio_segment = audio[start:start + self.segment_length]
        else:
            # Pad if too short
            padding = self.segment_length - audio_len
            audio_segment = tf.pad(audio, [[0, padding]])
        
        # Compute mel using LJSpeech parameters
        mel = audio_utils.audio_to_mel_ljspeech(audio_segment)
        
        return mel, audio_segment
    
    def create_dataset(self, batch_size: int, shuffle: bool = True, segments_per_file: int = 100):
        """Create tf.data.Dataset for training.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            segments_per_file: Number of random crops per audio file per epoch
            
        Returns:
            tf.data.Dataset yielding (mel, audio) pairs
        """
        def generator():
            indices = list(range(len(self.audio_files)))
            if shuffle:
                import random
                random.shuffle(indices)
            
            for idx in indices:
                file_path = self.audio_files[idx]
                for _ in range(segments_per_file):
                    mel, audio = self._load_audio_segment(file_path)
                    yield mel.numpy(), audio.numpy()
        
        # Infer output signature from first sample
        mel, audio = self._load_audio_segment(self.audio_files[0])
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=mel.shape, dtype=tf.float32),
                tf.TensorSpec(shape=(self.segment_length,), dtype=tf.float32)
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(100)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2)
        
        return dataset
