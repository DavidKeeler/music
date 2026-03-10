"""Feature builder with velocity computation."""
import tensorflow as tf


class FeatureBuilder:
    """Build feature vectors with velocity from normalized keypoints."""
    
    def __init__(self):
        self.prev_keypoints = None
        
    def build_features(self, keypoints, confidence):
        """
        Build feature vector with velocity.
        
        Args:
            keypoints: [17, 2] (x_norm, y_norm)
            confidence: [17] confidence scores
            
        Returns:
            features: [85] (x, y, dx, dy, conf) × 17
        """
        # Compute velocity
        if self.prev_keypoints is None:
            velocity = tf.zeros_like(keypoints)  # [17, 2]
        else:
            velocity = keypoints - self.prev_keypoints
            
        # Update previous
        self.prev_keypoints = keypoints
        
        # Build feature vector [17, 5]
        features = tf.concat([
            keypoints,  # [17, 2]
            velocity,   # [17, 2]
            tf.expand_dims(confidence, axis=1)  # [17, 1]
        ], axis=1)
        
        # Flatten to [85]
        features = tf.reshape(features, [-1])
        
        return features
    
    def reset(self):
        """Reset velocity computation."""
        self.prev_keypoints = None
