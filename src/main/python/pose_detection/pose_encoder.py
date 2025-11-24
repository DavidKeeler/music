import tensorflow as tf
import numpy as np
import sys
import os

# Add parent directory to path to import complex_layers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from complex_layers import ComplexDense, ComplexTransformerBlock, ComplexLayerNorm

class ComplexPoseEncoder(tf.keras.Model):
    """
    Complex-valued pose encoder that processes keypoint sequences to complex embeddings.
    
    Input: [batch, time_steps, 14, 2] (MediaPipe arms + torso keypoints)
    Output: [batch, time_steps, embedding_dim] (complex64)
    """
    
    def __init__(self, embedding_dim=256, num_transformer_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_transformer_layers = num_transformer_layers
        
        # Keypoint feature extractor (real-valued)
        self.keypoint_dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.keypoint_dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        
        # Normalization before complex domain
        self.pre_norm = tf.keras.layers.LayerNormalization()
        
        # Convert to complex domain
        self.to_complex = ComplexDense(embedding_dim, activation=None)
        
        # Complex transformer blocks for temporal processing
        self.transformer_blocks = [
            ComplexTransformerBlock(embedding_dim, num_heads=8, dff=512)
            for _ in range(num_transformer_layers)
        ]
        
        # Output projection
        self.output_dense = ComplexDense(embedding_dim)
        
    def call(self, keypoint_sequence, training=False):
        """
        Args:
            keypoint_sequence: [batch, time_steps, 14, 2] (normalized keypoints)
        Returns:
            embeddings: [batch, time_steps, embedding_dim] (complex64)
        """
        batch_size = tf.shape(keypoint_sequence)[0]
        time_steps = tf.shape(keypoint_sequence)[1]
        
        # Reshape to process all frames at once
        keypoints = tf.reshape(keypoint_sequence, [-1, 14, 2])
        
        # Process keypoints (real-valued)
        x = self.keypoint_dense1(keypoints)  # [batch*time, 14, 128]
        x = self.keypoint_dense2(x)          # [batch*time, 14, 64]
        x = self.flatten(x)                  # [batch*time, 14*64]
        
        # Reshape back to sequence
        x = tf.reshape(x, [batch_size, time_steps, -1])
        
        # Normalize before complex processing
        x = self.pre_norm(x)
        
        # Convert to complex domain
        x = tf.cast(x, tf.complex64)
        x = self.to_complex(x)
        
        # Complex transformer processing
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        
        # Output embedding
        embeddings = self.output_dense(x)
        
        return embeddings

def preprocess_video(video_path, target_fps=30):
    """
    Extract normalized keypoints from video using MediaPipe.
    
    Args:
        video_path: Path to video file
        target_fps: Target frame rate for processing
    
    Returns:
        keypoints: [time_steps, 14, 2] tensor
    """
    try:
        from .mediapipe_processor import run_pose_sequence
    except ImportError:
        from mediapipe_processor import run_pose_sequence
    
    keypoints = run_pose_sequence(video_path)
    if keypoints is None:
        return None
    
    return tf.convert_to_tensor(keypoints, dtype=tf.float32)
