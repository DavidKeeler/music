"""Skeleton normalization using shoulder-width normalization."""

import tensorflow as tf


class SkeletonNormalizer:
    """Normalize skeleton keypoints to body-relative coordinates."""
    
    LEFT_SHOULDER_IDX = 5
    RIGHT_SHOULDER_IDX = 6
    
    @staticmethod
    def normalize(keypoints, image_shape):
        """
        Apply shoulder-width normalization.
        
        Args:
            keypoints: [1, 1, 17, 3] (y, x, confidence)
            image_shape: (height, width)
            
        Returns:
            normalized: [17, 2] (x_norm, y_norm)
            confidence: [17] confidence scores
        """
        # Extract keypoints [17, 3]
        kpts = keypoints[0, 0]
        
        # Convert to pixel coordinates
        height, width = image_shape
        y_pixels = kpts[:, 0] * height
        x_pixels = kpts[:, 1] * width
        confidence = kpts[:, 2]
        
        # Get shoulder positions
        left_shoulder = tf.stack([x_pixels[SkeletonNormalizer.LEFT_SHOULDER_IDX],
                                   y_pixels[SkeletonNormalizer.LEFT_SHOULDER_IDX]])
        right_shoulder = tf.stack([x_pixels[SkeletonNormalizer.RIGHT_SHOULDER_IDX],
                                    y_pixels[SkeletonNormalizer.RIGHT_SHOULDER_IDX]])
        
        # Compute center and scale
        center = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = tf.norm(left_shoulder - right_shoulder)
        
        # Normalize all keypoints
        x_norm = (x_pixels - center[0]) / shoulder_width
        y_norm = (y_pixels - center[1]) / shoulder_width
        
        normalized = tf.stack([x_norm, y_norm], axis=1)  # [17, 2]
        
        return normalized, confidence
