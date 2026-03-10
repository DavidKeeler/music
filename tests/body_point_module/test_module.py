"""Unit tests for BodyPointModule."""

import tensorflow as tf
import pytest
from src.body_point_module import BodyPointModule


def create_test_frame(height, width):
    """Create a test frame with uint8 dtype."""
    return tf.cast(tf.random.uniform([height, width, 3], 0, 256, dtype=tf.int32), tf.uint8)


class TestBodyPointModuleInitialization:
    """Test module initialization."""
    
    def test_default_initialization(self):
        """Test module initializes with default parameters."""
        module = BodyPointModule()
        assert module.buffer_size == 8
        assert module.embedding_dim == 128
        assert module.output_mode == 'embedding'
        
    def test_custom_initialization(self):
        """Test module initializes with custom parameters."""
        module = BodyPointModule(buffer_size=4, embedding_dim=64, output_mode='with_confidence')
        assert module.buffer_size == 4
        assert module.embedding_dim == 64
        assert module.output_mode == 'with_confidence'
        
    def test_components_initialized(self):
        """Test all components are initialized."""
        module = BodyPointModule()
        assert module.pose_detector is not None
        assert module.normalizer is not None
        assert module.feature_builder is not None
        assert module.buffer is not None
        assert module.encoder is not None


class TestBodyPointModuleProcessing:
    """Test frame processing."""
    
    def test_process_single_frame(self):
        """Test processing a single frame."""
        module = BodyPointModule()
        frame = create_test_frame(256, 256)
        
        embedding = module.process_frame(frame)
        
        assert embedding.shape == (1, 128)
        assert embedding.dtype == tf.float32
        
    def test_process_multiple_frames(self):
        """Test processing multiple frames."""
        module = BodyPointModule()
        
        for i in range(10):
            frame = create_test_frame(256, 256)
            embedding = module.process_frame(frame)
            assert embedding.shape == (1, 128)
            
    def test_variable_frame_sizes(self):
        """Test processing frames of different sizes."""
        module = BodyPointModule()
        
        sizes = [(256, 256), (480, 640), (720, 1280)]
        for h, w in sizes:
            frame = create_test_frame(h, w)
            embedding = module.process_frame(frame)
            assert embedding.shape == (1, 128)


class TestBodyPointModuleOutputModes:
    """Test different output modes."""
    
    def test_embedding_mode(self):
        """Test embedding-only output mode."""
        module = BodyPointModule(output_mode='embedding')
        frame = create_test_frame(256, 256)
        
        result = module.process_frame(frame)
        
        assert isinstance(result, tf.Tensor)
        assert result.shape == (1, 128)
        
    def test_with_confidence_mode(self):
        """Test output with confidence scores."""
        module = BodyPointModule(output_mode='with_confidence')
        frame = create_test_frame(256, 256)
        
        embedding, confidence = module.process_frame(frame)
        
        assert embedding.shape == (1, 128)
        assert confidence.shape == (17,)
        assert tf.reduce_all(confidence >= 0.0)
        assert tf.reduce_all(confidence <= 1.0)
        
    def test_with_keypoints_mode(self):
        """Test output with raw keypoints."""
        module = BodyPointModule(output_mode='with_keypoints')
        frame = create_test_frame(256, 256)
        
        embedding, keypoints = module.process_frame(frame)
        
        assert embedding.shape == (1, 128)
        assert keypoints.shape == (1, 1, 17, 3)
        
    def test_invalid_output_mode(self):
        """Test invalid output mode raises error."""
        module = BodyPointModule(output_mode='invalid')
        frame = create_test_frame(256, 256)
        
        with pytest.raises(ValueError, match="Unknown output_mode"):
            module.process_frame(frame)


class TestBodyPointModuleState:
    """Test state management."""
    
    def test_buffer_fills_correctly(self):
        """Test buffer fills up to buffer_size."""
        module = BodyPointModule(buffer_size=4)
        
        # Process frames and check buffer state
        for i in range(6):
            frame = create_test_frame(256, 256)
            module.process_frame(frame)
            
            state = module.get_state()
            expected_length = min(i + 1, 4)
            assert state['buffer_length'] == expected_length
            assert state['buffer_size'] == 4
            
    def test_reset_clears_state(self):
        """Test reset clears internal state."""
        module = BodyPointModule()
        
        # Process some frames
        for _ in range(5):
            frame = create_test_frame(256, 256)
            module.process_frame(frame)
            
        # Check buffer is filled
        state = module.get_state()
        assert state['buffer_length'] == 5
        
        # Reset
        module.reset()
        
        # Check buffer is empty
        state = module.get_state()
        assert state['buffer_length'] == 0
        
    def test_reset_affects_velocity(self):
        """Test reset clears velocity computation."""
        module = BodyPointModule()
        
        # Process frames
        frame1 = create_test_frame(256, 256)
        frame2 = create_test_frame(256, 256)
        
        module.process_frame(frame1)
        module.process_frame(frame2)
        
        # Reset
        module.reset()
        
        # Process new frame - should have zero velocity
        frame3 = create_test_frame(256, 256)
        embedding = module.process_frame(frame3)
        
        # Should succeed without error
        assert embedding.shape == (1, 128)


class TestBodyPointModuleEndToEnd:
    """Test end-to-end integration."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from frame to embedding."""
        module = BodyPointModule()
        
        # Create realistic frame
        frame = create_test_frame(480, 640)
        
        # Process
        embedding = module.process_frame(frame)
        
        # Verify output
        assert embedding.shape == (1, 128)
        assert embedding.dtype == tf.float32
        assert tf.reduce_all(tf.math.is_finite(embedding))
        
    def test_multiple_videos(self):
        """Test processing multiple video streams with reset."""
        module = BodyPointModule()
        
        # Video 1
        for _ in range(5):
            frame = create_test_frame(256, 256)
            module.process_frame(frame)
            
        # Reset for new video
        module.reset()
        
        # Video 2
        for _ in range(5):
            frame = create_test_frame(256, 256)
            embedding = module.process_frame(frame)
            assert embedding.shape == (1, 128)
            
    def test_custom_embedding_dimension(self):
        """Test custom embedding dimension."""
        module = BodyPointModule(embedding_dim=256)
        frame = create_test_frame(256, 256)
        
        embedding = module.process_frame(frame)
        
        assert embedding.shape == (1, 256)
