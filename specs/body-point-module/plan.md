# Body Point Module - Implementation Plan

## Implementation Checklist

- [ ] Step 1: Project structure and dependencies
- [ ] Step 2: MoveNet pose detector wrapper
- [ ] Step 3: Skeleton normalization
- [ ] Step 4: Feature builder with velocity
- [ ] Step 5: History buffer
- [ ] Step 6: MLP encoder model
- [ ] Step 7: Temporal convolution layer
- [ ] Step 8: Main module integration
- [ ] Step 9: Output mode configuration
- [ ] Step 10: Unit tests

---

## Step 1: Project Structure and Dependencies

**Objective**: Set up module directory structure and install required dependencies.

**Implementation**:
1. Create directory: `src/body_point_module/`
2. Create files:
   - `__init__.py`
   - `config.py` - Configuration constants
   - `pose_detector.py` - MoveNet wrapper
   - `normalizer.py` - Skeleton normalization
   - `features.py` - Feature builder
   - `buffer.py` - History buffer
   - `encoder.py` - Pose encoder model
   - `module.py` - Main BodyPointModule class
3. Create test directory: `tests/body_point_module/`
4. Update `requirements.txt`:
   ```
   tensorflow>=2.13
   tensorflow-hub>=0.14
   ```

**Test Requirements**:
- Import all modules without errors
- TensorFlow and TF Hub installed correctly

**Integration**:
- Module structure mirrors existing `src/music_generation/`
- Follows same patterns as audio model

**Demo**:
```python
import src.body_point_module as bpm
print("Module structure created successfully")
```

---

## Step 2: MoveNet Pose Detector Wrapper

**Objective**: Load MoveNet Thunder and detect keypoints from frames.

**Implementation**:

Create `src/body_point_module/pose_detector.py`:

```python
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
```

**Test Requirements**:
- Load model successfully
- Process dummy frame [256, 256, 3]
- Output shape is [1, 1, 17, 3]
- Coordinates in range [0, 1]
- Confidence in range [0, 1]

**Integration**:
- Standalone component, no dependencies on other modules

**Demo**:
```python
detector = PoseDetector()
frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.int32)
keypoints = detector.detect(frame)
print(f"Detected keypoints shape: {keypoints.shape}")
```

---

## Step 3: Skeleton Normalization

**Objective**: Normalize keypoints using shoulder-width normalization.

**Implementation**:

Create `src/body_point_module/normalizer.py`:

```python
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
```

**Test Requirements**:
- Extract shoulders correctly (indices 5, 6)
- Compute shoulder midpoint
- Compute shoulder width (Euclidean distance)
- Normalize all 17 keypoints
- Output shape [17, 2] and [17]
- Shoulder midpoint maps to (0, 0)
- Shoulder distance maps to 1.0

**Integration**:
- Takes output from PoseDetector
- Feeds into FeatureBuilder

**Demo**:
```python
detector = PoseDetector()
normalizer = SkeletonNormalizer()

frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.int32)
keypoints = detector.detect(frame)
normalized, confidence = normalizer.normalize(keypoints, (256, 256))
print(f"Normalized shape: {normalized.shape}, Confidence shape: {confidence.shape}")
```

---

## Step 4: Feature Builder with Velocity

**Objective**: Compute velocity features and build feature vectors.

**Implementation**:

Create `src/body_point_module/features.py`:

```python
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
```

**Test Requirements**:
- First frame: velocity is zero
- Second frame: velocity is difference
- Feature shape is [85]
- Feature layout: [x, y, dx, dy, conf] per joint
- Confidence values preserved
- Reset clears previous keypoints

**Integration**:
- Takes output from SkeletonNormalizer
- Feeds into HistoryBuffer

**Demo**:
```python
builder = FeatureBuilder()

# Frame 1
kpts1 = tf.random.normal([17, 2])
conf1 = tf.random.uniform([17], 0.5, 1.0)
features1 = builder.build_features(kpts1, conf1)
print(f"Frame 1 features shape: {features1.shape}")

# Frame 2
kpts2 = kpts1 + 0.1  # Simulate movement
features2 = builder.build_features(kpts2, conf1)
print(f"Frame 2 features shape: {features2.shape}")
```

---

## Step 5: History Buffer

**Objective**: Maintain sliding window of feature vectors.

**Implementation**:

Create `src/body_point_module/buffer.py`:

```python
import tensorflow as tf

class HistoryBuffer:
    """Sliding window buffer for frame features."""
    
    def __init__(self, buffer_size, feature_dim):
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
        return len(self.buffer)
```

**Test Requirements**:
- Add features to buffer
- Buffer size limit enforced (FIFO)
- Get tensor with correct shape [buffer_size, feature_dim]
- Left-padding when buffer not full
- Reset clears buffer
- Length property works

**Integration**:
- Receives features from FeatureBuilder
- Provides tensor to PoseEncoder

**Demo**:
```python
buffer = HistoryBuffer(buffer_size=8, feature_dim=85)

for i in range(10):
    features = tf.random.normal([85])
    buffer.add(features)
    tensor = buffer.get_tensor()
    print(f"Frame {i+1}: buffer length={len(buffer)}, tensor shape={tensor.shape}")
```

---

## Step 6: MLP Encoder Model

**Objective**: Build trainable MLP encoder for pose features.

**Implementation**:

Create `src/body_point_module/encoder.py` (Part 1 - MLP):

```python
import tensorflow as tf

def build_mlp_encoder(input_dim=85, hidden_dim=256, output_dim=128):
    """
    Build MLP encoder for single frame.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        
    Returns:
        Keras Sequential model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_dim),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation('gelu'),
        tf.keras.layers.Dense(hidden_dim),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation('gelu'),
        tf.keras.layers.Dense(output_dim)
    ], name='mlp_encoder')
```

**Test Requirements**:
- Model builds successfully
- Input shape: [85]
- Output shape: [128]
- Forward pass executes
- Parameters are trainable
- Approximate parameter count: ~88K

**Integration**:
- Used within PoseEncoder (next step)
- Applied per-frame via TimeDistributed

**Demo**:
```python
mlp = build_mlp_encoder()
mlp.build(input_shape=(None, 85))
print(f"MLP parameters: {mlp.count_params()}")

# Test forward pass
features = tf.random.normal([1, 85])
embedding = mlp(features)
print(f"Output shape: {embedding.shape}")
```

---

## Step 7: Temporal Convolution Layer

**Objective**: Add temporal modeling to aggregate frame embeddings.

**Implementation**:

Update `src/body_point_module/encoder.py` (Part 2 - Full Encoder):

```python
def build_pose_encoder(input_dim=85, hidden_dim=256, output_dim=128, buffer_size=8):
    """
    Build complete pose encoder with temporal modeling.
    
    Args:
        input_dim: Input feature dimension per frame
        hidden_dim: MLP hidden dimension
        output_dim: Output embedding dimension
        buffer_size: Number of frames in buffer
        
    Returns:
        Keras Model
    """
    # Build per-frame MLP
    mlp = build_mlp_encoder(input_dim, hidden_dim, output_dim)
    
    # Full model
    inputs = tf.keras.Input(shape=(buffer_size, input_dim))
    
    # Apply MLP to each frame
    x = tf.keras.layers.TimeDistributed(mlp)(inputs)  # [buffer_size, output_dim]
    
    # Temporal convolution
    x = tf.keras.layers.Conv1D(
        filters=output_dim,
        kernel_size=3,
        padding='causal',
        activation='relu'
    )(x)  # [buffer_size, output_dim]
    
    # Extract last timestep
    outputs = x[:, -1, :]  # [output_dim]
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='pose_encoder')
    return model
```

**Test Requirements**:
- Model builds successfully
- Input shape: [buffer_size, 85]
- Output shape: [128]
- Forward pass executes
- Temporal conv has kernel_size=3, causal padding
- Last timestep extraction works
- Total parameters include MLP + conv

**Integration**:
- Takes buffer tensor from HistoryBuffer
- Outputs final embedding for BodyPointModule

**Demo**:
```python
encoder = build_pose_encoder(buffer_size=8)
print(encoder.summary())

# Test forward pass
buffer_tensor = tf.random.normal([1, 8, 85])
embedding = encoder(buffer_tensor)
print(f"Output shape: {embedding.shape}")
```

---

## Step 8: Main Module Integration

**Objective**: Integrate all components into BodyPointModule.

**Implementation**:

Create `src/body_point_module/module.py`:

```python
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
        embedding = self.encoder(buffer_tensor)
        embedding = tf.expand_dims(embedding, axis=0)  # [1, embedding_dim]
        
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
```

**Test Requirements**:
- Initialize module successfully
- Process single frame
- Process multiple frames (buffer fills)
- Output shape [1, 128]
- Reset clears state
- Different output modes work
- get_state returns correct info

**Integration**:
- Complete end-to-end pipeline
- Ready for use in music generation system

**Demo**:
```python
module = BodyPointModule(buffer_size=8, embedding_dim=128)

# Process frames
for i in range(10):
    frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.uint8)
    embedding = module.process_frame(frame)
    state = module.get_state()
    print(f"Frame {i+1}: embedding shape={embedding.shape}, buffer={state['buffer_length']}")

# Reset
module.reset()
print(f"After reset: {module.get_state()}")
```

---

## Step 9: Output Mode Configuration

**Objective**: Support different output modes for debugging and flexibility.

**Implementation**:

Update `src/body_point_module/config.py`:

```python
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
```

**Test Requirements**:
- All constants accessible
- Output modes defined
- Values match design document

**Integration**:
- Used throughout module for consistency
- Easy to modify configuration

**Demo**:
```python
from src.body_point_module import config

print(f"Embedding dim: {config.EMBEDDING_DIM}")
print(f"Buffer size: {config.BUFFER_SIZE}")
print(f"Output modes: {config.OUTPUT_MODE_EMBEDDING}, {config.OUTPUT_MODE_WITH_CONFIDENCE}")
```

---

## Step 10: Unit Tests

**Objective**: Comprehensive unit tests for all components.

**Implementation**:

Create test files in `tests/body_point_module/`:

**`test_pose_detector.py`**:
```python
import tensorflow as tf
from src.body_point_module.pose_detector import PoseDetector

def test_load_model():
    detector = PoseDetector()
    assert detector.model is not None

def test_detect_shape():
    detector = PoseDetector()
    frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.uint8)
    keypoints = detector.detect(frame)
    assert keypoints.shape == (1, 1, 17, 3)

def test_detect_ranges():
    detector = PoseDetector()
    frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.uint8)
    keypoints = detector.detect(frame)
    
    coords = keypoints[0, 0, :, :2]
    confidence = keypoints[0, 0, :, 2]
    
    assert tf.reduce_all(coords >= 0.0) and tf.reduce_all(coords <= 1.0)
    assert tf.reduce_all(confidence >= 0.0) and tf.reduce_all(confidence <= 1.0)
```

**`test_normalizer.py`**:
```python
import tensorflow as tf
from src.body_point_module.normalizer import SkeletonNormalizer

def test_normalize_shape():
    keypoints = tf.random.uniform([1, 1, 17, 3], 0, 1)
    normalized, confidence = SkeletonNormalizer.normalize(keypoints, (256, 256))
    assert normalized.shape == (17, 2)
    assert confidence.shape == (17,)

def test_shoulder_center():
    # Create keypoints with known shoulder positions
    keypoints = tf.zeros([1, 1, 17, 3])
    keypoints = tf.tensor_scatter_nd_update(
        keypoints,
        [[0, 0, 5, 1], [0, 0, 6, 1]],  # left and right shoulder x
        [0.4, 0.6]  # Centered at 0.5
    )
    
    normalized, _ = SkeletonNormalizer.normalize(keypoints, (256, 256))
    
    # Shoulder midpoint should be near origin
    left_shoulder = normalized[5]
    right_shoulder = normalized[6]
    midpoint = (left_shoulder + right_shoulder) / 2.0
    
    assert tf.reduce_all(tf.abs(midpoint) < 0.01)
```

**`test_features.py`**:
```python
import tensorflow as tf
from src.body_point_module.features import FeatureBuilder

def test_first_frame_zero_velocity():
    builder = FeatureBuilder()
    keypoints = tf.random.normal([17, 2])
    confidence = tf.random.uniform([17], 0.5, 1.0)
    
    features = builder.build_features(keypoints, confidence)
    
    # Extract velocity components (indices 2, 3 of each 5-element group)
    velocities = features[2::5]  # dx values
    assert tf.reduce_all(velocities == 0.0)

def test_second_frame_velocity():
    builder = FeatureBuilder()
    
    kpts1 = tf.ones([17, 2])
    conf = tf.ones([17])
    features1 = builder.build_features(kpts1, conf)
    
    kpts2 = tf.ones([17, 2]) * 2.0
    features2 = builder.build_features(kpts2, conf)
    
    # Velocity should be 1.0
    velocities = features2[2::5]
    assert tf.reduce_all(tf.abs(velocities - 1.0) < 0.01)

def test_feature_shape():
    builder = FeatureBuilder()
    keypoints = tf.random.normal([17, 2])
    confidence = tf.random.uniform([17])
    
    features = builder.build_features(keypoints, confidence)
    assert features.shape == (85,)
```

**`test_buffer.py`**:
```python
import tensorflow as tf
from src.body_point_module.buffer import HistoryBuffer

def test_buffer_fifo():
    buffer = HistoryBuffer(buffer_size=4, feature_dim=10)
    
    for i in range(6):
        buffer.add(tf.ones([10]) * i)
    
    assert len(buffer) == 4
    tensor = buffer.get_tensor()
    
    # Should contain frames 2, 3, 4, 5
    assert tf.reduce_all(tensor[0] == 2.0)
    assert tf.reduce_all(tensor[3] == 5.0)

def test_buffer_padding():
    buffer = HistoryBuffer(buffer_size=8, feature_dim=10)
    
    buffer.add(tf.ones([10]))
    buffer.add(tf.ones([10]) * 2)
    
    tensor = buffer.get_tensor()
    assert tensor.shape == (8, 10)
    
    # First 6 should be zeros (padding)
    assert tf.reduce_all(tensor[:6] == 0.0)
    # Last 2 should be data
    assert tf.reduce_all(tensor[6] == 1.0)
    assert tf.reduce_all(tensor[7] == 2.0)

def test_buffer_reset():
    buffer = HistoryBuffer(buffer_size=4, feature_dim=10)
    buffer.add(tf.ones([10]))
    buffer.reset()
    assert len(buffer) == 0
```

**`test_encoder.py`**:
```python
import tensorflow as tf
from src.body_point_module.encoder import build_mlp_encoder, build_pose_encoder

def test_mlp_encoder():
    mlp = build_mlp_encoder(input_dim=85, hidden_dim=256, output_dim=128)
    mlp.build(input_shape=(None, 85))
    
    features = tf.random.normal([1, 85])
    embedding = mlp(features)
    
    assert embedding.shape == (1, 128)
    assert mlp.count_params() > 80000  # Approximately 88K

def test_pose_encoder():
    encoder = build_pose_encoder(buffer_size=8)
    
    buffer_tensor = tf.random.normal([1, 8, 85])
    embedding = encoder(buffer_tensor)
    
    assert embedding.shape == (1, 128)

def test_encoder_trainable():
    encoder = build_pose_encoder()
    trainable_params = sum([tf.size(w).numpy() for w in encoder.trainable_weights])
    assert trainable_params > 0
```

**`test_module.py`**:
```python
import tensorflow as tf
from src.body_point_module.module import BodyPointModule

def test_module_init():
    module = BodyPointModule()
    assert module.buffer_size == 8
    assert module.embedding_dim == 128

def test_process_frame():
    module = BodyPointModule()
    frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.uint8)
    
    embedding = module.process_frame(frame)
    assert embedding.shape == (1, 128)

def test_multiple_frames():
    module = BodyPointModule()
    
    for i in range(10):
        frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.uint8)
        embedding = module.process_frame(frame)
        assert embedding.shape == (1, 128)
    
    state = module.get_state()
    assert state['buffer_length'] == 8

def test_reset():
    module = BodyPointModule()
    
    frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.uint8)
    module.process_frame(frame)
    
    module.reset()
    state = module.get_state()
    assert state['buffer_length'] == 0

def test_output_modes():
    # Embedding only
    module1 = BodyPointModule(output_mode='embedding')
    frame = tf.random.uniform([256, 256, 3], 0, 255, dtype=tf.uint8)
    result1 = module1.process_frame(frame)
    assert isinstance(result1, tf.Tensor)
    
    # With confidence
    module2 = BodyPointModule(output_mode='with_confidence')
    result2 = module2.process_frame(frame)
    assert isinstance(result2, tuple)
    assert len(result2) == 2
    
    # With keypoints
    module3 = BodyPointModule(output_mode='with_keypoints')
    result3 = module3.process_frame(frame)
    assert isinstance(result3, tuple)
    assert len(result3) == 2
```

**Test Requirements**:
- All tests pass
- Coverage for each component
- Edge cases tested
- Integration test for full pipeline

**Integration**:
- Run with pytest: `pytest tests/body_point_module/`
- CI/CD integration ready

**Demo**:
```bash
# Run all tests
pytest tests/body_point_module/ -v

# Run specific test file
pytest tests/body_point_module/test_module.py -v

# Run with coverage
pytest tests/body_point_module/ --cov=src.body_point_module
```

---

## Summary

This implementation plan provides:

1. **Incremental steps**: Each step builds on previous work
2. **Working demos**: Every step ends with testable functionality
3. **TDD approach**: Tests defined alongside implementation
4. **Integration points**: Clear dependencies between components
5. **End-to-end functionality**: Step 8 delivers complete working module

**Total estimated steps**: 10
**Core functionality available**: After Step 8
**Production ready**: After Step 10 (with tests)

Each step is independently testable and integrates into the final system.
