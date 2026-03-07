"""Tests for vocoder backend selection and fallback."""

import pytest
import tensorflow as tf
from unittest.mock import patch, MagicMock
from src.music_generation.vocoder import load_pretrained_vocoder, GriffinLimVocoder


def test_griffin_lim_backend():
    """Test loading Griffin-Lim backend."""
    vocoder = load_pretrained_vocoder(backend="griffin-lim")
    assert isinstance(vocoder, GriffinLimVocoder)


def test_fallback_disabled_raises():
    """Test that fallback can be disabled."""
    with patch('src.music_generation.vocoder.HiFiGANVocoder.from_pretrained') as mock_hifigan:
        mock_hifigan.side_effect = RuntimeError("HiFi-GAN failed")
        
        with pytest.raises(RuntimeError, match="HiFi-GAN failed"):
            load_pretrained_vocoder(backend="hifigan", enable_fallback=False)


def test_hifigan_to_vocos_fallback():
    """Test fallback from HiFi-GAN to Vocos."""
    with patch('src.music_generation.vocoder.HiFiGANVocoder.from_pretrained') as mock_hifigan:
        mock_hifigan.side_effect = RuntimeError("HiFi-GAN failed")
        
        with patch('src.music_generation.vocos_wrapper.load_vocos_vocoder') as mock_vocos:
            mock_vocos.return_value = MagicMock()
            
            vocoder = load_pretrained_vocoder(backend="hifigan", enable_fallback=True)
            
            # Should have tried HiFi-GAN first
            mock_hifigan.assert_called_once()
            # Then fallen back to Vocos
            mock_vocos.assert_called_once()


def test_vocos_to_griffin_lim_fallback():
    """Test fallback from Vocos to Griffin-Lim."""
    with patch('src.music_generation.vocos_wrapper.load_vocos_vocoder') as mock_vocos:
        mock_vocos.side_effect = ImportError("torch not installed")
        
        vocoder = load_pretrained_vocoder(backend="vocos", enable_fallback=True)
        
        # Should have tried Vocos first
        mock_vocos.assert_called_once()
        # Then fallen back to Griffin-Lim
        assert isinstance(vocoder, GriffinLimVocoder)


def test_full_fallback_chain():
    """Test complete fallback chain: HiFi-GAN -> Vocos -> Griffin-Lim."""
    with patch('src.music_generation.vocoder.HiFiGANVocoder.from_pretrained') as mock_hifigan:
        mock_hifigan.side_effect = RuntimeError("HiFi-GAN failed")
        
        with patch('src.music_generation.vocos_wrapper.load_vocos_vocoder') as mock_vocos:
            mock_vocos.side_effect = ImportError("torch not installed")
            
            vocoder = load_pretrained_vocoder(backend="hifigan", enable_fallback=True)
            
            # Should have tried both before falling back to Griffin-Lim
            mock_hifigan.assert_called_once()
            mock_vocos.assert_called_once()
            assert isinstance(vocoder, GriffinLimVocoder)


def test_melgan_fallback_to_hifigan():
    """Test MelGAN falls back to HiFi-GAN when TensorFlowTTS not installed."""
    with patch('src.music_generation.vocoder.TFAutoModel', None):
        with patch('src.music_generation.vocoder.HiFiGANVocoder.from_pretrained') as mock_hifigan:
            mock_hifigan.return_value = MagicMock()
            
            vocoder = load_pretrained_vocoder(backend="melgan", enable_fallback=True)
            
            # Should have fallen back to HiFi-GAN
            mock_hifigan.assert_called_once()


def test_invalid_backend_raises():
    """Test that invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        load_pretrained_vocoder(backend="invalid")
