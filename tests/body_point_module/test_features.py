"""Unit tests for FeatureBuilder."""
import tensorflow as tf
import pytest
from src.body_point_module.features import FeatureBuilder


def test_first_frame_zero_velocity():
    """First frame should have zero velocity."""
    builder = FeatureBuilder()
    keypoints = tf.random.normal([17, 2])
    confidence = tf.random.uniform([17], 0.5, 1.0)
    
    features = builder.build_features(keypoints, confidence)
    
    # Extract velocity components (indices 2, 3 of each 5-element group)
    velocities_x = features[2::5]  # dx values
    velocities_y = features[3::5]  # dy values
    
    assert tf.reduce_all(velocities_x == 0.0)
    assert tf.reduce_all(velocities_y == 0.0)


def test_second_frame_velocity():
    """Second frame should compute velocity correctly."""
    builder = FeatureBuilder()
    
    kpts1 = tf.ones([17, 2])
    conf = tf.ones([17])
    features1 = builder.build_features(kpts1, conf)
    
    kpts2 = tf.ones([17, 2]) * 2.0
    features2 = builder.build_features(kpts2, conf)
    
    # Velocity should be 1.0 for both x and y
    velocities_x = features2[2::5]
    velocities_y = features2[3::5]
    
    assert tf.reduce_all(tf.abs(velocities_x - 1.0) < 0.01)
    assert tf.reduce_all(tf.abs(velocities_y - 1.0) < 0.01)


def test_feature_shape():
    """Features should have shape [85]."""
    builder = FeatureBuilder()
    keypoints = tf.random.normal([17, 2])
    confidence = tf.random.uniform([17])
    
    features = builder.build_features(keypoints, confidence)
    assert features.shape == (85,)


def test_feature_layout():
    """Features should be laid out as [x, y, dx, dy, conf] per joint."""
    builder = FeatureBuilder()
    
    # Create known keypoints
    keypoints = tf.constant([[1.0, 2.0]] * 17, dtype=tf.float32)
    confidence = tf.constant([0.9] * 17, dtype=tf.float32)
    
    features = builder.build_features(keypoints, confidence)
    
    # Check first joint (indices 0-4)
    assert tf.abs(features[0] - 1.0) < 0.01  # x
    assert tf.abs(features[1] - 2.0) < 0.01  # y
    assert tf.abs(features[2] - 0.0) < 0.01  # dx (first frame)
    assert tf.abs(features[3] - 0.0) < 0.01  # dy (first frame)
    assert tf.abs(features[4] - 0.9) < 0.01  # confidence


def test_confidence_preserved():
    """Confidence values should be preserved in features."""
    builder = FeatureBuilder()
    
    keypoints = tf.random.normal([17, 2])
    confidence = tf.constant([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                              0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25], 
                             dtype=tf.float32)
    
    features = builder.build_features(keypoints, confidence)
    
    # Extract confidence values (every 5th element starting from index 4)
    extracted_conf = features[4::5]
    
    assert tf.reduce_all(tf.abs(extracted_conf - confidence) < 0.01)


def test_reset():
    """Reset should clear previous keypoints."""
    builder = FeatureBuilder()
    
    # First sequence
    kpts1 = tf.ones([17, 2])
    conf = tf.ones([17])
    features1 = builder.build_features(kpts1, conf)
    
    kpts2 = tf.ones([17, 2]) * 2.0
    features2 = builder.build_features(kpts2, conf)
    
    # Velocity should be non-zero
    velocities = features2[2::5]
    assert tf.reduce_any(velocities != 0.0)
    
    # Reset
    builder.reset()
    
    # Next frame should have zero velocity again
    kpts3 = tf.ones([17, 2]) * 3.0
    features3 = builder.build_features(kpts3, conf)
    
    velocities_after_reset = features3[2::5]
    assert tf.reduce_all(velocities_after_reset == 0.0)


def test_negative_velocity():
    """Should handle negative velocity correctly."""
    builder = FeatureBuilder()
    
    kpts1 = tf.ones([17, 2]) * 2.0
    conf = tf.ones([17])
    features1 = builder.build_features(kpts1, conf)
    
    kpts2 = tf.ones([17, 2])  # Move backwards
    features2 = builder.build_features(kpts2, conf)
    
    # Velocity should be -1.0
    velocities_x = features2[2::5]
    velocities_y = features2[3::5]
    
    assert tf.reduce_all(tf.abs(velocities_x - (-1.0)) < 0.01)
    assert tf.reduce_all(tf.abs(velocities_y - (-1.0)) < 0.01)


def test_multiple_frames_sequence():
    """Test velocity computation over multiple frames."""
    builder = FeatureBuilder()
    conf = tf.ones([17])
    
    # Frame 1: position 0
    kpts1 = tf.zeros([17, 2])
    features1 = builder.build_features(kpts1, conf)
    vel1 = features1[2::5]
    assert tf.reduce_all(vel1 == 0.0)  # First frame
    
    # Frame 2: position 1
    kpts2 = tf.ones([17, 2])
    features2 = builder.build_features(kpts2, conf)
    vel2 = features2[2::5]
    assert tf.reduce_all(tf.abs(vel2 - 1.0) < 0.01)  # Velocity = 1
    
    # Frame 3: position 3 (velocity = 2)
    kpts3 = tf.ones([17, 2]) * 3.0
    features3 = builder.build_features(kpts3, conf)
    vel3 = features3[2::5]
    assert tf.reduce_all(tf.abs(vel3 - 2.0) < 0.01)  # Velocity = 2
    
    # Frame 4: position 3 (velocity = 0, no movement)
    kpts4 = tf.ones([17, 2]) * 3.0
    features4 = builder.build_features(kpts4, conf)
    vel4 = features4[2::5]
    assert tf.reduce_all(tf.abs(vel4) < 0.01)  # Velocity = 0
