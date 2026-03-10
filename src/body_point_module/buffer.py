"""History buffer for maintaining sliding window of frame features."""

import tensorflow as tf


class HistoryBuffer:
    """Sliding window buffer for frame features."""
    
    def __init__(self, buffer_size, feature_dim):
        """
        Initialize history buffer.
        
        Args:
            buffer_size: Maximum number of frames to store
            feature_dim: Dimension of each feature vector
        """
        self.buffer_size = buffer_size
        self.feature_dim = feature_dim
        self.buffer = []
        
    def add(self, features):
        """
        Add frame features to buffer.
        
        Args:
            features: [feature_dim] tensor
        """
        self.buffer.append(features)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
    def get_tensor(self):
        """
        Get buffer as tensor with left-padding if needed.
        
        Returns:
            buffer_tensor: [buffer_size, feature_dim]
        """
        if len(self.buffer) < self.buffer_size:
            # Left-pad with zeros
            padding_size = self.buffer_size - len(self.buffer)
            padding = [tf.zeros(self.feature_dim)] * padding_size
            full_buffer = padding + self.buffer
        else:
            full_buffer = self.buffer
            
        return tf.stack(full_buffer, axis=0)
    
    def reset(self):
        """Clear buffer."""
        self.buffer = []
    
    def __len__(self):
        """Return current buffer length."""
        return len(self.buffer)
