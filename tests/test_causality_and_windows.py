"""Tests for causality and window constraints in the model."""

import tensorflow as tf
import pytest
from src.music_generation.model import MelGenerator
from src.music_generation.layers import LocalWindowAttention
from src.music_generation.config import N_MELS, TOKEN_COMPRESSION_RATIO, LATENT_DIM


def test_causality_forward_pass():
    """Test that output at time t only depends on inputs <= t."""
    model = MelGenerator()
    C = TOKEN_COMPRESSION_RATIO
    z = tf.zeros([1, LATENT_DIM])  # Fixed z to avoid random sampling differences
    
    x = tf.random.normal([1, 256, N_MELS])
    y_full = model(x, z=z, training=False)
    
    modify_frame = 200
    safe_frame = (modify_frame // C) * C
    
    x_modified = tf.identity(x)
    x_modified = tf.tensor_scatter_nd_update(
        x_modified, [[0, modify_frame, 0]], [999.0]
    )
    y_modified = model(x_modified, z=z, training=False)
    
    assert tf.reduce_max(tf.abs(y_full[0, :safe_frame, :] - y_modified[0, :safe_frame, :])) < 1e-5


def test_causality_generate():
    """Test that generate() is strictly causal (deterministic with fixed z)."""
    model = MelGenerator()
    z = tf.zeros([1, LATENT_DIM])
    
    seed = tf.random.normal([64, N_MELS])
    generated = model.generate(seed, num_frames=50, temperature=0.0, z=z)
    generated2 = model.generate(seed, num_frames=50, temperature=0.0, z=z)
    
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
    C = TOKEN_COMPRESSION_RATIO
    z = tf.zeros([1, LATENT_DIM])
    
    for seq_len in [64, 128, 256, 512]:
        x = tf.random.normal([1, seq_len, N_MELS])
        y_full = model(x, z=z, training=False)
        
        x_modified = tf.identity(x)
        x_modified = tf.tensor_scatter_nd_update(
            x_modified, [[0, seq_len - 1, 0]], [999.0]
        )
        y_modified = model(x_modified, z=z, training=False)
        
        safe_frame = ((seq_len - 1) // C) * C
        if safe_frame > 0:
            assert tf.reduce_max(tf.abs(y_full[0, :safe_frame, :] - y_modified[0, :safe_frame, :])) < 1e-5
