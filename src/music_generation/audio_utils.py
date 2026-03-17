"""Audio processing utilities for music generation."""

import tensorflow as tf
import soundfile as sf
import librosa
import numpy as np

from .config import (
    SAMPLE_RATE,
    N_MELS,
    FRAME_LENGTH,
    FRAME_STEP,
    VOCOS_SAMPLE_RATE,
    VOCOS_HOP_LENGTH,
    VOCOS_N_FFT,
    VOCOS_N_MELS,
    VOCOS_F_MIN,
    VOCOS_F_MAX,
    # Legacy aliases
    LJSPEECH_SAMPLE_RATE,
    LJSPEECH_HOP_LENGTH,
    LJSPEECH_N_FFT,
    LJSPEECH_N_MELS,
    LJSPEECH_F_MIN,
    LJSPEECH_F_MAX,
)


def load_audio(path: str) -> tf.Tensor:
    """Load audio file, resample to target rate, convert to mono.
    
    Args:
        path: Path to audio file
        
    Returns:
        1D tensor of audio samples at SAMPLE_RATE
        
    Raises:
        ValueError: If audio file is empty
    """
    waveform, sr = sf.read(path, dtype='float32')
    
    if len(waveform) == 0:
        raise ValueError("Audio file is empty")
    
    # Convert to TensorFlow tensor
    waveform = tf.constant(waveform, dtype=tf.float32)
    
    # Convert stereo to mono
    if len(waveform.shape) > 1:
        waveform = tf.reduce_mean(waveform, axis=1)
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = tf.numpy_function(
            lambda x: librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE),
            [waveform],
            tf.float32
        )
    
    return waveform


def audio_to_mel(waveform: tf.Tensor) -> tf.Tensor:
    """Convert audio waveform to log-mel spectrogram.
    
    Args:
        waveform: 1D tensor of audio samples
        
    Returns:
        2D tensor of shape [time, n_mels] containing log-mel spectrogram
    """
    stft = tf.signal.stft(waveform, FRAME_LENGTH, FRAME_STEP, pad_end=True)
    magnitude = tf.abs(stft)
    
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=FRAME_LENGTH // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=0.0,
        upper_edge_hertz=12000.0
    )
    
    mel = tf.matmul(magnitude, mel_matrix)
    log_mel = tf.math.log(mel + 1e-8)
    
    return log_mel


def normalize_mel(mel: tf.Tensor, mean: float, std: float) -> tf.Tensor:
    """Normalize mel spectrogram using mean and std.
    
    Args:
        mel: Mel spectrogram tensor
        mean: Mean value for normalization
        std: Standard deviation for normalization
        
    Returns:
        Normalized mel spectrogram
        
    Raises:
        ValueError: If mel contains NaN or Inf values
    """
    if tf.reduce_any(tf.math.is_nan(mel)):
        raise ValueError("Mel spectrogram contains NaN values")
    if tf.reduce_any(tf.math.is_inf(mel)):
        raise ValueError("Mel spectrogram contains Inf values")
    
    return (mel - mean) / (std + 1e-8)


def denormalize_mel(mel: tf.Tensor, mean: float, std: float) -> tf.Tensor:
    """Denormalize mel spectrogram using mean and std.
    
    Args:
        mel: Normalized mel spectrogram tensor
        mean: Mean value used in normalization
        std: Standard deviation used in normalization
        
    Returns:
        Denormalized mel spectrogram
    """
    return mel * std + mean


def audio_to_mel_ljspeech(waveform: tf.Tensor) -> tf.Tensor:
    """Convert audio to mel using Vocos-compatible parameters.
    
    Uses TensorFlow signal processing with Vocos 24kHz parameters
    for compatibility with pretrained Vocos vocoder.
    
    Args:
        waveform: 1D tensor of audio samples at 24000 Hz
        
    Returns:
        2D tensor [time, n_mels] (time_frames x 100)
    """
    stft = tf.signal.stft(
        waveform,
        frame_length=VOCOS_N_FFT,
        frame_step=VOCOS_HOP_LENGTH,
        pad_end=True
    )
    magnitude = tf.abs(stft)
    
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=VOCOS_N_MELS,
        num_spectrogram_bins=VOCOS_N_FFT // 2 + 1,
        sample_rate=VOCOS_SAMPLE_RATE,
        lower_edge_hertz=VOCOS_F_MIN,
        upper_edge_hertz=VOCOS_F_MAX
    )
    
    mel = tf.matmul(magnitude, mel_matrix)
    log_mel = tf.math.log(mel + 1e-8)
    
    return log_mel


class MelNormalizer:
    """Normalize mel spectrograms using dataset statistics.
    
    This class provides normalization and denormalization of mel spectrograms
    using precomputed mean and standard deviation from the training dataset.
    """
    
    def __init__(self, mean: float, std: float):
        """Initialize normalizer with statistics.
        
        Args:
            mean: Mean value for normalization
            std: Standard deviation for normalization
        """
        self.mean = mean
        self.std = std
    
    def normalize(self, mel: tf.Tensor) -> tf.Tensor:
        """Normalize mel spectrogram.
        
        Args:
            mel: Mel spectrogram tensor
            
        Returns:
            Normalized mel spectrogram
        """
        return normalize_mel(mel, self.mean, self.std)
    
    def denormalize(self, mel: tf.Tensor) -> tf.Tensor:
        """Denormalize mel spectrogram.
        
        Args:
            mel: Normalized mel spectrogram tensor
            
        Returns:
            Denormalized mel spectrogram
        """
        return denormalize_mel(mel, self.mean, self.std)
    
    @classmethod
    def from_dataset(cls, cache_dir: str) -> 'MelNormalizer':
        """Compute statistics from cached mel spectrograms.
        
        Args:
            cache_dir: Directory containing cached .npy mel files
            
        Returns:
            MelNormalizer instance with computed statistics
            
        Raises:
            ValueError: If no cached mel files found
        """
        from pathlib import Path
        
        cache_path = Path(cache_dir)
        mel_files = list(cache_path.glob("*.npy"))
        
        if not mel_files:
            raise ValueError(f"No cached mel files found in {cache_dir}")
        
        # Compute statistics using streaming approach
        total_sum = 0.0
        total_sum_sq = 0.0
        total_count = 0
        
        for mel_file in mel_files:
            mel = np.load(mel_file)
            total_sum += np.sum(mel)
            total_sum_sq += np.sum(mel ** 2)
            total_count += mel.size
        
        mean = total_sum / total_count
        variance = (total_sum_sq / total_count) - (mean ** 2)
        std = np.sqrt(variance)
        
        return cls(mean=float(mean), std=float(std))
