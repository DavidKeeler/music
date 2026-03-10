import tensorflow as tf
import tensorflow_hub as hub


class PoseDetector:
    """Wrapper for MoveNet Thunder pose detection model."""
    
    MOVENET_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    INPUT_SIZE = 256
    
    def __init__(self):
        """Load MoveNet Thunder from TF Hub."""
        self.model = hub.load(self.MOVENET_URL)
        self.movenet = self.model.signatures['serving_default']
        
    def detect(self, frame):
        """
        Detect keypoints in frame.
        
        Args:
            frame: [H, W, 3] uint8 tensor
            
        Returns:
            keypoints: [1, 1, 17, 3] float32 (y, x, confidence)
        """
        # Resize and pad to 256x256
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(
            input_image, 
            self.INPUT_SIZE, 
            self.INPUT_SIZE
        )
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Run inference
        outputs = self.movenet(input_image)
        keypoints = outputs['output_0']
        
        return keypoints
