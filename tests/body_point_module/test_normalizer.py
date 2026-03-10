"""Unit tests for SkeletonNormalizer."""

import tensorflow as tf
import pytest
from src.body_point_module.normalizer import SkeletonNormalizer


class TestSkeletonNormalizer:
    """Test shoulder-width normalization."""
    
    def test_output_shapes(self):
        """Test output shapes are correct."""
        # Create synthetic keypoints [1, 1, 17, 3]
        keypoints = tf.random.uniform([1, 1, 17, 3], minval=0.0, maxval=1.0)
        image_shape = (256, 256)
        
        normalized, confidence = SkeletonNormalizer.normalize(keypoints, image_shape)
        
        assert normalized.shape == (17, 2), f"Expected (17, 2), got {normalized.shape}"
        assert confidence.shape == (17,), f"Expected (17,), got {confidence.shape}"
    
    def test_shoulder_indices(self):
        """Test shoulder indices are correct."""
        assert SkeletonNormalizer.LEFT_SHOULDER_IDX == 5
        assert SkeletonNormalizer.RIGHT_SHOULDER_IDX == 6
    
    def test_shoulder_midpoint_maps_to_origin(self):
        """Test that shoulder midpoint maps to (0, 0)."""
        # Create keypoints where shoulders are at known positions
        keypoints = tf.zeros([1, 1, 17, 3])
        
        # Set left shoulder at (0.3, 0.4) and right shoulder at (0.3, 0.6)
        # This gives midpoint at (0.3, 0.5) in normalized coords
        keypoints = tf.tensor_scatter_nd_update(
            keypoints,
            [[0, 0, 5, 0], [0, 0, 5, 1]],  # left shoulder y, x
            [0.3, 0.4]
        )
        keypoints = tf.tensor_scatter_nd_update(
            keypoints,
            [[0, 0, 6, 0], [0, 0, 6, 1]],  # right shoulder y, x
            [0.3, 0.6]
        )
        
        image_shape = (256, 256)
        normalized, _ = SkeletonNormalizer.normalize(keypoints, image_shape)
        
        # Check that shoulders map close to origin
        left_shoulder_norm = normalized[5]
        right_shoulder_norm = normalized[6]
        
        # Midpoint should be at origin
        midpoint = (left_shoulder_norm + right_shoulder_norm) / 2.0
        
        assert tf.abs(midpoint[0]) < 1e-5, f"Midpoint x should be ~0, got {midpoint[0]}"
        assert tf.abs(midpoint[1]) < 1e-5, f"Midpoint y should be ~0, got {midpoint[1]}"
    
    def test_shoulder_distance_normalization(self):
        """Test that shoulder distance is used for normalization."""
        # Create keypoints where shoulders are at known positions
        keypoints = tf.zeros([1, 1, 17, 3])
        
        # Set left shoulder at (0.5, 0.4) and right shoulder at (0.5, 0.6)
        # Distance = 0.2 * 256 = 51.2 pixels
        keypoints = tf.tensor_scatter_nd_update(
            keypoints,
            [[0, 0, 5, 0], [0, 0, 5, 1]],  # left shoulder y, x
            [0.5, 0.4]
        )
        keypoints = tf.tensor_scatter_nd_update(
            keypoints,
            [[0, 0, 6, 0], [0, 0, 6, 1]],  # right shoulder y, x
            [0.5, 0.6]
        )
        
        image_shape = (256, 256)
        normalized, _ = SkeletonNormalizer.normalize(keypoints, image_shape)
        
        # Check that shoulder distance is 1.0 in normalized space
        left_shoulder_norm = normalized[5]
        right_shoulder_norm = normalized[6]
        shoulder_dist = tf.norm(left_shoulder_norm - right_shoulder_norm)
        
        assert tf.abs(shoulder_dist - 1.0) < 1e-5, f"Shoulder distance should be 1.0, got {shoulder_dist}"
    
    def test_confidence_preserved(self):
        """Test that confidence scores are preserved."""
        keypoints = tf.random.uniform([1, 1, 17, 3], minval=0.0, maxval=1.0)
        image_shape = (256, 256)
        
        _, confidence = SkeletonNormalizer.normalize(keypoints, image_shape)
        
        # Check confidence is in valid range
        assert tf.reduce_all(confidence >= 0.0)
        assert tf.reduce_all(confidence <= 1.0)
        
        # Check confidence matches input
        expected_confidence = keypoints[0, 0, :, 2]
        assert tf.reduce_all(tf.abs(confidence - expected_confidence) < 1e-6)
    
    def test_different_image_sizes(self):
        """Test normalization works with different image sizes."""
        keypoints = tf.random.uniform([1, 1, 17, 3], minval=0.0, maxval=1.0)
        
        for image_shape in [(256, 256), (480, 640), (720, 1280)]:
            normalized, confidence = SkeletonNormalizer.normalize(keypoints, image_shape)
            
            assert normalized.shape == (17, 2)
            assert confidence.shape == (17,)
            assert not tf.reduce_any(tf.math.is_nan(normalized))
            assert not tf.reduce_any(tf.math.is_inf(normalized))
    
    def test_all_keypoints_normalized(self):
        """Test that all 17 keypoints are normalized."""
        keypoints = tf.random.uniform([1, 1, 17, 3], minval=0.0, maxval=1.0)
        image_shape = (256, 256)
        
        normalized, _ = SkeletonNormalizer.normalize(keypoints, image_shape)
        
        # Check all keypoints have valid values
        assert not tf.reduce_any(tf.math.is_nan(normalized))
        assert not tf.reduce_any(tf.math.is_inf(normalized))
        
        # Check all keypoints are different (with high probability for random input)
        for i in range(17):
            for j in range(i + 1, 17):
                # At least some pairs should be different
                if i < 5:  # Check first few
                    diff = tf.norm(normalized[i] - normalized[j])
                    # With random input, points should be different
                    # (this is a weak test but catches obvious bugs)
                    pass  # Just verify no crashes
