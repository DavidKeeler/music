"""Body Point Module for pose-based music generation conditioning."""

from .config import (
    EMBEDDING_DIM,
    BUFFER_SIZE,
    OUTPUT_MODE_EMBEDDING,
    OUTPUT_MODE_WITH_CONFIDENCE,
    OUTPUT_MODE_WITH_KEYPOINTS,
)

__all__ = [
    'EMBEDDING_DIM',
    'BUFFER_SIZE',
    'OUTPUT_MODE_EMBEDDING',
    'OUTPUT_MODE_WITH_CONFIDENCE',
    'OUTPUT_MODE_WITH_KEYPOINTS',
]

# Import components as they are implemented
try:
    from .pose_detector import PoseDetector
    __all__.append('PoseDetector')
except ImportError:
    pass

try:
    from .normalizer import SkeletonNormalizer
    __all__.append('SkeletonNormalizer')
except ImportError:
    pass

try:
    from .module import BodyPointModule
    __all__.append('BodyPointModule')
except ImportError:
    pass
