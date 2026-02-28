"""Vocoder model for mel-to-audio conversion."""

import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from typing import Optional


class HiFiGANVocoder(tf.keras.Model):
    """Wrapper for HiFi-GAN vocoder model."""
    
    def __init__(self, pretrained_model=None):
        """Initialize vocoder.
        
        Args:
            pretrained_model: Pretrained TF Hub model or custom generator
        """
        super().__init__()
        self.generator = pretrained_model
    
    def call(self, mel_spectrogram, training=False):
        """Convert mel spectrogram to audio.
        
        Args:
            mel_spectrogram: Mel spectrogram [batch, mel_bins, time] or [batch, time, mel_bins]
            training: Whether in training mode
            
        Returns:
            Audio waveform [batch, samples]
        """
        if self.generator is None:
            raise ValueError("No generator model loaded")
        
        # Ensure shape is [batch, time, mel_bins] for TensorFlow convention
        if len(tf.shape(mel_spectrogram)) == 3:
            if mel_spectrogram.shape[1] == 80:  # [batch, 80, time]
                mel_spectrogram = tf.transpose(mel_spectrogram, [0, 2, 1])
        
        return self.generator(mel_spectrogram, training=training)


def load_pretrained_vocoder(model_url: Optional[str] = None) -> HiFiGANVocoder:
    """Load pretrained vocoder from TensorFlow Hub.
    
    Args:
        model_url: TensorFlow Hub model URL. If None, returns placeholder.
        
    Returns:
        HiFiGANVocoder instance
        
    Note:
        Common TF Hub vocoder models:
        - "https://tfhub.dev/google/soundstream/mel/decoder/music/1"
        - Custom HiFi-GAN models if available
    """
    if model_url is None:
        # Return vocoder with no pretrained model (must be loaded separately)
        return HiFiGANVocoder(pretrained_model=None)
    
    # Load from TensorFlow Hub
    pretrained_model = hub.load(model_url)
    return HiFiGANVocoder(pretrained_model=pretrained_model)


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
    
    def create_dataset(self, batch_size: int, shuffle: bool = True):
        """Create tf.data.Dataset for training.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            
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
            dataset = dataset.shuffle(1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
