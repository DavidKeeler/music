"""Configuration constants for body point module."""

# Model dimensions
EMBEDDING_DIM = 128  # Match audio model D_MODEL
HIDDEN_DIM = 256
FEATURE_DIM = 85  # 17 joints × 5 features

# Buffer
BUFFER_SIZE = 8

# Temporal convolution
TEMPORAL_KERNEL = 3

# MoveNet
MOVENET_VARIANT = 'thunder'
MOVENET_INPUT_SIZE = 256

# Normalization
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6

# Output modes
OUTPUT_MODE_EMBEDDING = 'embedding'
OUTPUT_MODE_WITH_CONFIDENCE = 'with_confidence'
OUTPUT_MODE_WITH_KEYPOINTS = 'with_keypoints'
