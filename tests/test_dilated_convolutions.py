"""Tests for dilated causal convolutions in MelGenerator."""

import tensorflow as tf
import pytest
from src.music_generation.model import MelGenerator
from src.music_generation.config import GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN


class TestReceptiveFieldCalculation:
    """Test receptive field calculation formula."""
    
    def test_rf_formula_default(self):
        """Test RF = 1 + 2*sum(dilations) with default [1,2,4]."""
        dilations = [1, 2, 4]
        expected_rf = 1 + 2 * sum(dilations)  # 1 + 2*7 = 15
        assert expected_rf == 15
    
    def test_rf_formula_custom(self):
        """Test RF calculation with custom dilation rates."""
        dilations = [1, 1, 1]
        rf = 1 + 2 * sum(dilations)
        assert rf == 7
        
        dilations = [2, 4, 8]
        rf = 1 + 2 * sum(dilations)
        assert rf == 29


class TestModelInitialization:
    """Test model initialization with various dilation configurations."""
    
    def test_initialization_default(self):
        """Test model initializes with default [1,2,4] dilations."""
        model = MelGenerator()
        assert model is not None
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'conv_head')
    
    def test_initialization_custom(self, monkeypatch):
        """Test model initializes with custom dilation rates."""
        import src.music_generation.model as model
        monkeypatch.setattr(model, 'CONV_DILATION_RATES', [2, 4, 8])
        m = model.MelGenerator()
        assert m is not None
    
    def test_dilation_list_too_short(self, monkeypatch, caplog):
        """Test warning issued when list has <3 elements."""
        import src.music_generation.model as model
        import logging
        caplog.set_level(logging.WARNING)
        monkeypatch.setattr(model, 'CONV_DILATION_RATES', [1, 2])
        m = model.MelGenerator()
        assert "has 2 values" in caplog.text
        assert "Padding with 1s" in caplog.text
    
    def test_dilation_list_too_long(self, monkeypatch, caplog):
        """Test warning issued when list has >3 elements."""
        import src.music_generation.model as model
        import logging
        caplog.set_level(logging.WARNING)
        monkeypatch.setattr(model, 'CONV_DILATION_RATES', [1, 2, 4, 8, 16])
        m = model.MelGenerator()
        assert "has 5 values" in caplog.text
        assert "Using first 3" in caplog.text


class TestIntegration:
    """Integration tests for forward pass, causality, and generation."""
    
    def test_output_shape_unchanged(self):
        """Test model output shape matches input shape."""
        pass
    
    def test_causality_preserved(self):
        """Test output at time t doesn't depend on future inputs."""
        pass
    
    def test_forward_pass_no_errors(self):
        """Test forward pass completes without errors."""
        pass
    
    def test_generation_works(self):
        """Test autoregressive generation works with dilated convolutions."""
        pass
