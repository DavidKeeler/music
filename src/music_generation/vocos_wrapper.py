"""Vocos vocoder wrapper (PyTorch-based fallback)."""

import tensorflow as tf
import numpy as np
from typing import Optional


class VocosWrapper:
    """Wrapper for Vocos vocoder (PyTorch-based).
    
    Provides fallback option when TensorFlow vocoders fail.
    Requires: pip install torch vocos
    """
    
    def __init__(self, model_name: str = "charactr/vocos-mel-24khz"):
        """Initialize Vocos vocoder.
        
        Args:
            model_name: Hugging Face model ID for Vocos
        
        Raises:
            ImportError: If torch or vocos not installed
        """
        try:
            import torch
            from vocos import Vocos
        except ImportError as e:
            raise ImportError(
                "Vocos requires PyTorch and vocos package. "
                "Install with: pip install torch vocos"
            ) from e
        
        self.vocos = Vocos.from_pretrained(model_name)
        self.vocos.eval()
        self.torch = torch
    
    def __call__(self, mel: tf.Tensor) -> tf.Tensor:
        """Convert mel spectrogram to audio.
        
        Args:
            mel: Mel spectrogram [batch, time, 80] or [batch, 80, time]
        
        Returns:
            Audio waveform [batch, samples]
        """
        # Convert to numpy
        mel_np = mel.numpy()
        
        # Vocos expects [batch, 100, time]
        if mel_np.ndim == 3 and mel_np.shape[1] != 100 and mel_np.shape[2] == 100:
            mel_np = np.transpose(mel_np, (0, 2, 1))
        
        # Convert to PyTorch
        mel_torch = self.torch.from_numpy(mel_np).float()
        
        # Generate audio
        with self.torch.no_grad():
            audio_torch = self.vocos.decode(mel_torch)
        
        # Convert back to TensorFlow
        audio_np = audio_torch.cpu().numpy()
        return tf.constant(audio_np, dtype=tf.float32)


def load_vocos_vocoder(model_name: str = "charactr/vocos-mel-24khz") -> VocosWrapper:
    """Load pretrained Vocos vocoder.
    
    Args:
        model_name: Hugging Face model ID
    
    Returns:
        VocosWrapper instance
    
    Raises:
        ImportError: If dependencies not installed
    """
    return VocosWrapper(model_name=model_name)
