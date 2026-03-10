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

# Import BodyPointModule only when module.py exists
try:
    from .module import BodyPointModule
    __all__.append('BodyPointModule')
except ImportError:
    pass
