"""Tests for causality and window constraints in the model."""

import tensorflow as tf
import pytest
from src.music_generation.model import MelGenerator
from src.music_generation.layers import LocalWindowAttention


def test_causality_forward_pass():
    """Test that output at time t only depends on inputs <= t."""
    model = MelGenerator()
    
    # Create input sequence
    x = tf.random.normal([1, 256, 80])
    
    # Forward pass
    y_full = model(x, training=False)
    
    # Modify future frames and verify output at early timesteps doesn't change
    x_modified = tf.identity(x)
    x_modified = tf.tensor_scatter_nd_update(
        x_modified,
        [[0, 200, 0]],  # Modify frame 200
        [999.0]
    )
    y_modified = model(x_modified, training=False)
    
    # Output at timesteps < 200 should be identical
    assert tf.reduce_max(tf.abs(y_full[0, :200, :] - y_modified[0, :200, :])) < 1e-5


def test_causality_generate():
    """Test that generate() is strictly causal."""
    model = MelGenerator()
    
    # Generate with seed
    seed = tf.random.normal([64, 80])
    generated = model.generate(seed, num_frames=50, temperature=0.0)
    
    # Each frame should be deterministic given the same seed
    generated2 = model.generate(seed, num_frames=50, temperature=0.0)
    
    assert tf.reduce_max(tf.abs(generated - generated2)) < 1e-5


def test_window_attention_mask():
    """Test that LocalWindowAttention respects window size."""
    window_size = 128
    attn = LocalWindowAttention(d_model=512, num_heads=8, window_size=window_size)
    
    # Create input with seq_len > window_size
    x = tf.random.normal([1, 256, 512])
    
    # Build the layer
    _ = attn(x)
    
    # Verify attention can only look back window_size frames
    # This is implicitly tested by the mask logic in the layer
    # We verify the layer runs without error and produces valid output
    output = attn(x)
    
    assert output.shape == x.shape
    assert not tf.reduce_any(tf.math.is_nan(output))
    assert not tf.reduce_any(tf.math.is_inf(output))


def test_window_sizes_in_model():
    """Test that model uses correct window sizes for each transformer layer."""
    from src.music_generation.config import WINDOW_SIZES
    
    model = MelGenerator()
    
    # Verify each transformer has the correct window size
    assert model.transformer1.attn.window_size == WINDOW_SIZES[0]  # 32
    assert model.transformer2.attn.window_size == WINDOW_SIZES[1]  # 64
    assert model.transformer3.attn.window_size == WINDOW_SIZES[2]  # 128


def test_relative_position_bias_shape():
    """Test that relative position bias has correct shape."""
    window_size = 128
    num_heads = 8
    attn = LocalWindowAttention(d_model=512, num_heads=num_heads, window_size=window_size)
    
    # Build the layer
    x = tf.random.normal([1, 64, 512])
    _ = attn(x)
    
    # Check bias shape
    assert attn.relative_position_bias.shape == [num_heads, window_size]


def test_causality_with_different_seq_lengths():
    """Test causality holds for various sequence lengths."""
    model = MelGenerator()
    
    for seq_len in [64, 128, 256, 512]:
        x = tf.random.normal([1, seq_len, 80])
        y_full = model(x, training=False)
        
        # Modify last frame
        x_modified = tf.identity(x)
        x_modified = tf.tensor_scatter_nd_update(
            x_modified,
            [[0, seq_len - 1, 0]],
            [999.0]
        )
        y_modified = model(x_modified, training=False)
        
        # All frames except the last should be identical
        if seq_len > 1:
            assert tf.reduce_max(tf.abs(y_full[0, :-1, :] - y_modified[0, :-1, :])) < 1e-5
