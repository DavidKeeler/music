import pytest
import tensorflow as tf
from src.body_point_module.pose_detector import PoseDetector


class TestPoseDetector:
    """Test suite for PoseDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create PoseDetector instance."""
        return PoseDetector()
    
    def test_load_model(self, detector):
        """Test that MoveNet model loads successfully."""
        assert detector.model is not None
        assert detector.movenet is not None
    
    def test_detect_shape(self, detector):
        """Test that detect returns correct shape."""
        frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.float32)
        frame = tf.cast(frame, tf.uint8)
        
        keypoints = detector.detect(frame)
        
        assert keypoints.shape == (1, 1, 17, 3)
    
    def test_detect_coordinate_range(self, detector):
        """Test that coordinates are in range [0, 1]."""
        frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.float32)
        frame = tf.cast(frame, tf.uint8)
        
        keypoints = detector.detect(frame)
        
        # Extract x, y coordinates (indices 0, 1)
        coords = keypoints[..., :2]
        
        assert tf.reduce_all(coords >= 0.0)
        assert tf.reduce_all(coords <= 1.0)
    
    def test_detect_confidence_range(self, detector):
        """Test that confidence scores are in range [0, 1]."""
        frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.float32)
        frame = tf.cast(frame, tf.uint8)
        
        keypoints = detector.detect(frame)
        
        # Extract confidence (index 2)
        confidence = keypoints[..., 2]
        
        assert tf.reduce_all(confidence >= 0.0)
        assert tf.reduce_all(confidence <= 1.0)
    
    def test_detect_variable_input_size(self, detector):
        """Test that detector handles variable input sizes."""
        # Test with different input sizes
        for h, w in [(128, 128), (256, 256), (480, 640), (720, 1280)]:
            frame = tf.random.uniform([h, w, 3], 0, 255, dtype=tf.float32)
            frame = tf.cast(frame, tf.uint8)
            
            keypoints = detector.detect(frame)
            
            assert keypoints.shape == (1, 1, 17, 3)
