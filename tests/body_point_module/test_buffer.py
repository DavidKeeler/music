"""Unit tests for HistoryBuffer."""

import tensorflow as tf
from src.body_point_module.buffer import HistoryBuffer


def test_buffer_init():
    """Test buffer initialization."""
    buffer = HistoryBuffer(buffer_size=8, feature_dim=85)
    assert buffer.buffer_size == 8
    assert buffer.feature_dim == 85
    assert len(buffer) == 0


def test_buffer_add_single():
    """Test adding single frame."""
    buffer = HistoryBuffer(buffer_size=8, feature_dim=10)
    features = tf.ones([10])
    buffer.add(features)
    assert len(buffer) == 1


def test_buffer_fifo():
    """Test FIFO behavior when buffer is full."""
    buffer = HistoryBuffer(buffer_size=4, feature_dim=10)
    
    # Add 6 frames
    for i in range(6):
        buffer.add(tf.ones([10]) * i)
    
    # Buffer should only contain last 4 frames
    assert len(buffer) == 4
    tensor = buffer.get_tensor()
    
    # Should contain frames 2, 3, 4, 5
    assert tf.reduce_all(tensor[0] == 2.0)
    assert tf.reduce_all(tensor[1] == 3.0)
    assert tf.reduce_all(tensor[2] == 4.0)
    assert tf.reduce_all(tensor[3] == 5.0)


def test_buffer_padding():
    """Test left-padding when buffer not full."""
    buffer = HistoryBuffer(buffer_size=8, feature_dim=10)
    
    buffer.add(tf.ones([10]))
    buffer.add(tf.ones([10]) * 2)
    
    tensor = buffer.get_tensor()
    assert tensor.shape == (8, 10)
    
    # First 6 should be zeros (padding)
    assert tf.reduce_all(tensor[:6] == 0.0)
    # Last 2 should be data
    assert tf.reduce_all(tensor[6] == 1.0)
    assert tf.reduce_all(tensor[7] == 2.0)


def test_buffer_full_no_padding():
    """Test no padding when buffer is full."""
    buffer = HistoryBuffer(buffer_size=4, feature_dim=10)
    
    for i in range(4):
        buffer.add(tf.ones([10]) * i)
    
    tensor = buffer.get_tensor()
    assert tensor.shape == (4, 10)
    
    # All frames should be data, no padding
    for i in range(4):
        assert tf.reduce_all(tensor[i] == float(i))


def test_buffer_reset():
    """Test buffer reset."""
    buffer = HistoryBuffer(buffer_size=4, feature_dim=10)
    buffer.add(tf.ones([10]))
    buffer.add(tf.ones([10]) * 2)
    
    assert len(buffer) == 2
    
    buffer.reset()
    assert len(buffer) == 0
    
    # After reset, should be empty and pad fully
    tensor = buffer.get_tensor()
    assert tensor.shape == (4, 10)
    assert tf.reduce_all(tensor == 0.0)


def test_buffer_shape_consistency():
    """Test tensor shape is always [buffer_size, feature_dim]."""
    buffer = HistoryBuffer(buffer_size=8, feature_dim=85)
    
    # Empty buffer
    tensor = buffer.get_tensor()
    assert tensor.shape == (8, 85)
    
    # Partially filled
    for i in range(3):
        buffer.add(tf.random.normal([85]))
    tensor = buffer.get_tensor()
    assert tensor.shape == (8, 85)
    
    # Full buffer
    for i in range(5):
        buffer.add(tf.random.normal([85]))
    tensor = buffer.get_tensor()
    assert tensor.shape == (8, 85)


def test_buffer_with_realistic_features():
    """Test buffer with realistic feature vectors from FeatureBuilder."""
    buffer = HistoryBuffer(buffer_size=8, feature_dim=85)
    
    # Simulate 10 frames of features
    for i in range(10):
        features = tf.random.normal([85])
        buffer.add(features)
    
    # Should contain last 8 frames
    assert len(buffer) == 8
    tensor = buffer.get_tensor()
    assert tensor.shape == (8, 85)
    
    # Verify all values are finite
    assert tf.reduce_all(tf.math.is_finite(tensor))
