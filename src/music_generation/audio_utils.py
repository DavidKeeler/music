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
        upper_edge_hertz=SAMPLE_RATE / 2.0
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
    """Convert audio to mel using LJSpeech-compatible parameters.
    
    Uses TensorFlow signal processing with LJSpeech parameters
    for compatibility with pretrained HiFi-GAN vocoder.
    
    Args:
        waveform: 1D tensor of audio samples at 22050 Hz
        
    Returns:
        2D tensor [time, n_mels] (time_frames x 80)
    """
    stft = tf.signal.stft(
        waveform,
        frame_length=LJSPEECH_N_FFT,
        frame_step=LJSPEECH_HOP_LENGTH,
        pad_end=True
    )
    magnitude = tf.abs(stft)
    
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=LJSPEECH_N_MELS,
        num_spectrogram_bins=LJSPEECH_N_FFT // 2 + 1,
        sample_rate=LJSPEECH_SAMPLE_RATE,
        lower_edge_hertz=LJSPEECH_F_MIN,
        upper_edge_hertz=LJSPEECH_F_MAX
    )
    
    mel = tf.matmul(magnitude, mel_matrix)
    log_mel = tf.math.log(mel + 1e-8)
    
    return log_mel
