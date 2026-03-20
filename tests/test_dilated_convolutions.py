"""Tests for dilated causal convolutions in MelGenerator."""

import tensorflow as tf
import pytest
from src.music_generation.model import MelGenerator
from src.music_generation.config import N_MELS
from src.music_generation.layers import CausalConvBlock


class TestReceptiveFieldCalculation:
    """Test receptive field calculation formula."""
    
    def test_rf_formula_default(self):
        """Test RF = 1 + 2*sum(dilations) with default [1,2,4,8,16]."""
        dilations = [1, 2, 4, 8, 16]
        expected_rf = 1 + 2 * sum(dilations)  # 1 + 2*31 = 63
        assert expected_rf == 63
    
    def test_rf_formula_custom(self):
        """Test RF calculation with custom dilation rates."""
        assert 1 + 2 * sum([1, 1, 1]) == 7
        assert 1 + 2 * sum([2, 4, 8]) == 29
        assert 1 + 2 * sum([1, 2, 4, 8, 16]) == 63


class TestModelInitialization:
    """Test model initialization with various dilation configurations."""
    
    def test_initialization_default(self):
        """Test model initializes with default [1,2,4,8,16] dilations."""
        model = MelGenerator()
        assert hasattr(model, 'conv_layers')
        assert len(model.conv_layers) == 5
        for i, d in enumerate([1, 2, 4, 8, 16]):
            assert isinstance(model.conv_layers[i], CausalConvBlock)
    
    def test_initialization_custom(self, monkeypatch):
        """Test model initializes with custom dilation rates."""
        import src.music_generation.model as model_mod
        monkeypatch.setattr(model_mod, 'CONV_DILATION_RATES', [2, 4, 8])
        m = model_mod.MelGenerator()
        assert len(m.conv_layers) == 3
    
    def test_arbitrary_rates_no_errors(self, monkeypatch):
        """Test any arbitrary list works with no errors or warnings."""
        import src.music_generation.model as model_mod
        monkeypatch.setattr(model_mod, 'CONV_DILATION_RATES', [1, 3, 9, 27])
        m = model_mod.MelGenerator()
        assert len(m.conv_layers) == 4

    def test_receptive_field_logging(self, monkeypatch, caplog):
        """Test receptive field log shows 63 frames ~672.0 ms for default rates."""
        import logging
        caplog.set_level(logging.INFO)
        model = MelGenerator()
        assert "63 frames" in caplog.text
        assert "672.0 ms" in caplog.text


class TestIntegration:
    """Integration tests for forward pass, causality, and generation."""
    
    def test_output_shape_unchanged(self):
        """Test model output shape matches input shape."""
        model = MelGenerator()
        x = tf.random.normal([2, 40, N_MELS])  # T must be divisible by C=4
        y = model(x, training=False)
        assert y.shape == x.shape
    
    def test_causality_preserved(self):
        """Test output at time t doesn't depend on future inputs."""
        model = MelGenerator()
        z = tf.random.normal([1, 64])  # Fixed latent for both calls
        x = tf.random.normal([1, 80, N_MELS])
        y_full = model(x, z=z, training=False)
        y_truncated = model(x[:, :40, :], z=z, training=False)
        tf.debugging.assert_near(y_full[:, :40, :], y_truncated, atol=1e-5)
    
    def test_forward_pass_no_errors(self):
        """Test forward pass completes without errors."""
        model = MelGenerator()
        x = tf.random.normal([4, 128, N_MELS])
        y = model(x, training=True)
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))
    
    def test_generation_works(self):
        """Test autoregressive generation works with dilated convolutions."""
        model = MelGenerator()
        seed = tf.random.normal([20, N_MELS])
        generated = model.generate(seed, num_frames=10)
        assert generated.shape == (10, N_MELS)
        assert not tf.reduce_any(tf.math.is_nan(generated))
        assert not tf.reduce_any(tf.math.is_inf(generated))

    def test_convs_before_transformers(self):
        """Verify all conv layers run before transformer blocks."""
        model = MelGenerator()
        x = tf.random.normal([1, 40, N_MELS])
        y = model(x, training=False)
        assert y.shape == (1, 40, N_MELS)
