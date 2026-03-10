"""Main BodyPointModule for pose embedding extraction."""

import tensorflow as tf
from .pose_detector import PoseDetector
from .normalizer import SkeletonNormalizer
from .features import FeatureBuilder
from .buffer import HistoryBuffer
from .encoder import build_pose_encoder


class BodyPointModule:
    """Main body point module for pose embedding extraction."""
    
    def __init__(self, buffer_size=8, embedding_dim=128, output_mode='embedding'):
        """
        Initialize body point module.
        
        Args:
            buffer_size: Number of frames in history buffer
            embedding_dim: Output embedding dimension
            output_mode: 'embedding', 'with_confidence', or 'with_keypoints'
        """
        self.buffer_size = buffer_size
        self.embedding_dim = embedding_dim
        self.output_mode = output_mode
        
        # Initialize components
        self.pose_detector = PoseDetector()
        self.normalizer = SkeletonNormalizer()
        self.feature_builder = FeatureBuilder()
        self.buffer = HistoryBuffer(buffer_size, feature_dim=85)
        self.encoder = build_pose_encoder(
            input_dim=85,
            hidden_dim=256,
            output_dim=embedding_dim,
            buffer_size=buffer_size
        )
        
    def process_frame(self, frame):
        """
        Process single video frame.
        
        Args:
            frame: [H, W, 3] uint8 tensor
            
        Returns:
            embedding: [1, embedding_dim] float32 tensor
            (or tuple with additional outputs based on output_mode)
        """
        # 1. Detect keypoints
        keypoints = self.pose_detector.detect(frame)
        
        # 2. Normalize skeleton
        image_shape = tf.shape(frame)[:2]
        normalized, confidence = self.normalizer.normalize(keypoints, image_shape)
        
        # 3. Build features with velocity
        features = self.feature_builder.build_features(normalized, confidence)
        
        # 4. Update buffer
        self.buffer.add(features)
        
        # 5. Get buffer tensor
        buffer_tensor = self.buffer.get_tensor()
        buffer_tensor = tf.expand_dims(buffer_tensor, axis=0)  # Add batch dim
        
        # 6. Encode to embedding
        embedding = self.encoder(buffer_tensor)  # [1, embedding_dim]
        
        # 7. Return based on output mode
        if self.output_mode == 'embedding':
            return embedding
        elif self.output_mode == 'with_confidence':
            return embedding, confidence
        elif self.output_mode == 'with_keypoints':
            return embedding, keypoints
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
    
    def reset(self):
        """Reset internal state for new video stream."""
        self.buffer.reset()
        self.feature_builder.reset()
    
    def get_state(self):
        """Get current buffer state (for debugging)."""
        return {
            'buffer_length': len(self.buffer),
            'buffer_size': self.buffer_size
        }
