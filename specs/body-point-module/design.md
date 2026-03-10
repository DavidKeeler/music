# Body Point Module - Detailed Design

## Overview

The Body Point Module is a TensorFlow-based component that extracts pose embeddings from video frames for integration with a music generation system. It processes video frame-by-frame in real-time, detecting body keypoints and encoding them into embeddings suitable for cross-attention mechanisms.

**Purpose**: Transform video frames into pose embeddings that capture body movement and position for conditioning music generation.

**Integration Context**: Designed to work alongside the existing TensorFlow music generation system (D_MODEL=128), providing pose embeddings as additional conditioning signals.

## Detailed Requirements

### Functional Requirements

**FR1: Video Input Processing**
- Accept single video frames as input
- Support variable resolution and frame rates
- Process single person in frame
- Real-time streaming capability (frame-by-frame)

**FR2: Pose Detection**
- Use pretrained MoveNet Thunder model (frozen weights)
- Detect 17 body keypoints per frame
- Output keypoints with (y, x, confidence) format
- Accept all confidence thresholds (no filtering)

**FR3: Pose Normalization**
- Apply shoulder-width normalization to keypoints
- Center coordinates at shoulder midpoint
- Scale by shoulder distance
- Convert to body-relative coordinates

**FR4: Feature Extraction**
- Compute velocity features (dx, dy) from frame history
- Build feature vector: [x_norm, y_norm, dx, dy, confidence] per joint
- Maintain sliding history buffer (4-8 frames)
- Handle first frame with zero velocity

**FR5: Embedding Generation**
- Encode pose features to 128-dimensional embeddings
- Use trainable MLP encoder + temporal convolution
- Output single embedding vector per frame
- Match audio model's embedding dimension (D_MODEL=128)

**FR6: State Management**
- Maintain internal buffer state between frames
- Stateful module interface (single frame input)
- Provide reset method for new video streams
- No state persistence across sessions

**FR7: Output Format**
- Default: embedding vector only [128]
- Configurable: include confidence scores or raw keypoints for debugging
- Output shape: [batch=1, embedding_dim=128]

### Non-Functional Requirements

**NFR1: Performance**
- CPU deployment (Mac compatibility)
- Target: Process frames faster than real-time
- No specific latency requirements defined

**NFR2: Training**
- MoveNet weights frozen (pretrained)
- MLP encoder and temporal layer trainable
- Train end-to-end with music generation model
- No separate pre-training phase

**NFR3: Error Handling**
- Edge cases handled externally (outside module scope)
- Module assumes valid input
- No internal error recovery mechanisms

**NFR4: Testing**
- Unit tests for each component
- Test normalization, feature building, encoding, buffer management
- No integration tests or visual debugging tools required

## Architecture Overview

### High-Level Pipeline

```mermaid
graph LR
    A[Video Frame] --> B[MoveNet Thunder]
    B --> C[Keypoints 17x3]
    C --> D[Normalization]
    D --> E[Feature Builder]
    E --> F[Buffer Update]
    F --> G[MLP Encoder]
    G --> H[Temporal Conv]
    H --> I[Embedding 128]
```

### Component Hierarchy

```
BodyPointModule (Python wrapper - stateful)
├── PoseDetector (MoveNet Thunder - frozen)
├── SkeletonNormalizer (shoulder-width normalization)
├── FeatureBuilder (velocity computation)
├── HistoryBuffer (sliding window - 4-8 frames)
└── PoseEncoder (Keras Model - trainable)
    ├── MLP (3 layers)
    └── TemporalConv (1D convolution)
```

### Data Flow

```
Input: Frame [H, W, 3]
  ↓
MoveNet: [1, 1, 17, 3] (y, x, conf)
  ↓
Normalize: [17, 2] (x_norm, y_norm)
  ↓
Add Velocity: [17, 5] (x, y, dx, dy, conf)
  ↓
Flatten: [85]
  ↓
Update Buffer: [8, 85]
  ↓
MLP per frame: [8, 128]
  ↓
Temporal Conv: [8, 128]
  ↓
Take last: [128]
  ↓
Output: Embedding [128]
```

## Components and Interfaces

### 1. BodyPointModule (Main Interface)

**Purpose**: Stateful wrapper managing the entire pipeline

**Interface**:
```python
class BodyPointModule:
    def __init__(
        self,
        buffer_size: int = 8,
        embedding_dim: int = 128,
        output_mode: str = 'embedding'  # 'embedding', 'with_confidence', 'with_keypoints'
    )
    
    def process_frame(self, frame: tf.Tensor) -> tf.Tensor:
        """
        Process single video frame.
        
        Args:
            frame: [H, W, 3] uint8 tensor
            
        Returns:
            embedding: [1, 128] float32 tensor
        """
    
    def reset(self) -> None:
        """Reset internal buffer for new video stream."""
    
    def get_state(self) -> Dict:
        """Return current buffer state (for debugging)."""
```

**State**:
- `buffer`: List of feature vectors (length ≤ buffer_size)
- `pose_detector`: MoveNet model instance
- `pose_encoder`: Trainable encoder model

**Responsibilities**:
- Coordinate all components
- Manage buffer state
- Handle frame-by-frame processing

### 2. PoseDetector

**Purpose**: Wrapper for MoveNet Thunder model

**Interface**:
```python
class PoseDetector:
    def __init__(self):
        """Load MoveNet Thunder from TF Hub."""
    
    def detect(self, frame: tf.Tensor) -> tf.Tensor:
        """
        Detect keypoints in frame.
        
        Args:
            frame: [H, W, 3] uint8 tensor
            
        Returns:
            keypoints: [1, 1, 17, 3] float32 (y, x, confidence)
        """
```

**Implementation Details**:
- Load from: `https://tfhub.dev/google/movenet/singlepose/thunder/4`
- Input size: 256x256 (resize with padding)
- Weights frozen (trainable=False)
- Output coordinates normalized [0, 1]

### 3. SkeletonNormalizer

**Purpose**: Normalize keypoints to body-relative coordinates

**Interface**:
```python
class SkeletonNormalizer:
    @staticmethod
    def normalize(keypoints: tf.Tensor, image_shape: Tuple[int, int]) -> tf.Tensor:
        """
        Apply shoulder-width normalization.
        
        Args:
            keypoints: [1, 1, 17, 3] (y, x, confidence)
            image_shape: (height, width)
            
        Returns:
            normalized: [17, 2] (x_norm, y_norm)
        """
```

**Algorithm**:
1. Convert normalized coords to pixels: `pixel = coord * image_dim`
2. Extract shoulders: `left_shoulder (idx=5)`, `right_shoulder (idx=6)`
3. Compute center: `center = (left + right) / 2`
4. Compute scale: `shoulder_width = distance(left, right)`
5. Normalize: `norm = (keypoint - center) / shoulder_width`

**Keypoint Indices** (MoveNet):
```
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
5: left_shoulder, 6: right_shoulder
7: left_elbow, 8: right_elbow, 9: left_wrist, 10: right_wrist
11: left_hip, 12: right_hip
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
```

### 4. FeatureBuilder

**Purpose**: Compute velocity and construct feature vectors

**Interface**:
```python
class FeatureBuilder:
    def __init__(self):
        self.prev_keypoints = None
    
    def build_features(
        self,
        keypoints: tf.Tensor,
        confidence: tf.Tensor
    ) -> tf.Tensor:
        """
        Build feature vector with velocity.
        
        Args:
            keypoints: [17, 2] (x_norm, y_norm)
            confidence: [17] confidence scores
            
        Returns:
            features: [85] flattened (x, y, dx, dy, conf) × 17
        """
```

**Feature Construction**:
```
For each joint i in [0, 16]:
    features[i*5 + 0] = x_normalized[i]
    features[i*5 + 1] = y_normalized[i]
    features[i*5 + 2] = dx[i]  # x[t] - x[t-1]
    features[i*5 + 3] = dy[i]  # y[t] - y[t-1]
    features[i*5 + 4] = confidence[i]
```

**Velocity Computation**:
- First frame: `dx = 0, dy = 0`
- Subsequent frames: `dx = x[t] - x[t-1], dy = y[t] - y[t-1]`
- Optional: EMA smoothing with α=0.3

### 5. PoseEncoder (Keras Model)

**Purpose**: Encode pose features to embeddings with temporal modeling

**Architecture**:
```python
def build_pose_encoder(input_dim=85, hidden_dim=256, output_dim=128, buffer_size=8):
    """
    Build trainable pose encoder.
    
    Input: [buffer_size, input_dim]
    Output: [output_dim]
    """
    return tf.keras.Sequential([
        # Per-frame MLP encoding
        tf.keras.layers.TimeDistributed(
            tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation('gelu'),
                tf.keras.layers.Dense(hidden_dim),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation('gelu'),
                tf.keras.layers.Dense(output_dim)
            ])
        ),
        # Temporal convolution
        tf.keras.layers.Conv1D(
            filters=output_dim,
            kernel_size=3,
            padding='causal',
            activation='relu'
        ),
        # Extract last timestep
        tf.keras.layers.Lambda(lambda x: x[:, -1, :])
    ])
```

**Layer Details**:

**MLP Encoder** (per frame):
- Input: [85] features
- Layer 1: Dense(256) → LayerNorm → GELU
- Layer 2: Dense(256) → LayerNorm → GELU
- Layer 3: Dense(128)
- Output: [128] frame embedding
- Parameters: ~88K

**Temporal Convolution**:
- Input: [buffer_size, 128]
- Conv1D: kernel=3, filters=128, padding='causal'
- Output: [buffer_size, 128]
- Receptive field: 3 frames

**Output Extraction**:
- Take last timestep: `output[:, -1, :]`
- Shape: [128]

### 6. HistoryBuffer

**Purpose**: Maintain sliding window of recent frames

**Interface**:
```python
class HistoryBuffer:
    def __init__(self, buffer_size: int, feature_dim: int):
        self.buffer_size = buffer_size
        self.buffer = []
    
    def add(self, features: tf.Tensor) -> None:
        """Add frame features to buffer."""
        self.buffer.append(features)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def get_tensor(self) -> tf.Tensor:
        """
        Get buffer as tensor with padding if needed.
        
        Returns:
            buffer_tensor: [buffer_size, feature_dim]
        """
    
    def reset(self) -> None:
        """Clear buffer."""
        self.buffer = []
```

**Padding Strategy**:
- If buffer not full: left-pad with zeros
- Example: buffer=[f1, f2, f3], size=8 → [0, 0, 0, 0, 0, f1, f2, f3]

## Data Models

### Keypoint Format (MoveNet Output)

```python
keypoints: tf.Tensor  # [1, 1, 17, 3]
# Dimensions:
#   [0]: batch (always 1)
#   [1]: instance (always 1, single person)
#   [2]: keypoint index (0-16)
#   [3]: (y, x, confidence)

# Coordinate format:
#   y, x: normalized [0.0, 1.0] relative to image
#   confidence: [0.0, 1.0] detection confidence
```

### Normalized Keypoints

```python
normalized_keypoints: tf.Tensor  # [17, 2]
# Dimensions:
#   [0]: keypoint index (0-16)
#   [1]: (x_norm, y_norm)

# Coordinate format:
#   x_norm, y_norm: body-relative coordinates
#   - Origin at shoulder midpoint
#   - Scaled by shoulder width
#   - Typical range: [-2, 2]
```

### Feature Vector

```python
features: tf.Tensor  # [85]
# Structure: [x, y, dx, dy, conf] × 17 joints
# Layout:
#   [0:5]:   joint 0 (nose)
#   [5:10]:  joint 1 (left_eye)
#   ...
#   [80:85]: joint 16 (right_ankle)
```

### Embedding Output

```python
embedding: tf.Tensor  # [1, 128]
# Dimensions:
#   [0]: batch (always 1 for streaming)
#   [1]: embedding dimension (matches D_MODEL)
```

## Error Handling

Per requirements (A11), edge cases are handled externally. The module assumes:

**Assumptions**:
- Valid video frame provided (not None, correct shape)
- Frame contains detectable person
- Shoulder keypoints are present
- Shoulder width > 0

**No Internal Handling For**:
- Missing person in frame
- Zero or very small shoulder width
- Invalid frame format
- Buffer initialization edge cases

**Module Behavior**:
- Processes all inputs as-is
- No validation or error checking
- Caller responsible for input quality

## Acceptance Criteria

### Given-When-Then Format

**AC1: Frame Processing**
```
Given: A valid video frame [H, W, 3]
When: process_frame() is called
Then: Returns embedding [1, 128] within acceptable latency
```

**AC2: Keypoint Detection**
```
Given: A frame with visible person
When: MoveNet processes the frame
Then: Returns 17 keypoints with coordinates and confidence
```

**AC3: Normalization**
```
Given: Keypoints with detectable shoulders
When: Normalization is applied
Then: Coordinates are centered at shoulder midpoint and scaled by shoulder width
```

**AC4: Velocity Computation**
```
Given: Two consecutive frames in buffer
When: Features are built
Then: Velocity (dx, dy) is computed as frame difference
```

**AC5: First Frame Handling**
```
Given: Empty buffer (first frame)
When: Features are built
Then: Velocity is set to zero [0, 0] for all joints
```

**AC6: Buffer Management**
```
Given: Buffer size = 8
When: 10 frames are processed
Then: Buffer contains only the last 8 frames
```

**AC7: State Reset**
```
Given: Buffer with N frames
When: reset() is called
Then: Buffer is empty and next frame starts fresh
```

**AC8: Embedding Dimension**
```
Given: Any valid input frame
When: Embedding is generated
Then: Output shape is [1, 128] matching D_MODEL
```

**AC9: Trainable Components**
```
Given: PoseEncoder model
When: Gradients are computed
Then: MLP and temporal conv weights are trainable, MoveNet weights are frozen
```

**AC10: Output Modes**
```
Given: output_mode='embedding'
When: process_frame() is called
Then: Returns only embedding tensor [1, 128]

Given: output_mode='with_confidence'
When: process_frame() is called
Then: Returns (embedding, confidence_scores)
```

## Testing Strategy

### Unit Tests

**Test Suite 1: PoseDetector**
- Load MoveNet model successfully
- Process frame and return correct shape [1, 1, 17, 3]
- Coordinates in range [0, 1]
- Confidence scores in range [0, 1]

**Test Suite 2: SkeletonNormalizer**
- Extract shoulder keypoints correctly (indices 5, 6)
- Compute shoulder midpoint
- Compute shoulder width (Euclidean distance)
- Normalize all keypoints relative to shoulders
- Handle coordinate conversion (normalized → pixel → normalized)

**Test Suite 3: FeatureBuilder**
- Build features with correct shape [85]
- First frame: velocity is zero
- Subsequent frames: velocity is frame difference
- Feature layout: [x, y, dx, dy, conf] per joint
- Confidence values preserved

**Test Suite 4: HistoryBuffer**
- Add frames to buffer
- Buffer size limit enforced (FIFO)
- Get tensor with correct padding
- Reset clears buffer

**Test Suite 5: PoseEncoder**
- Model builds with correct architecture
- Input shape: [buffer_size, 85]
- Output shape: [128]
- Trainable parameters count ~88K
- Forward pass executes without error

**Test Suite 6: BodyPointModule Integration**
- Initialize module
- Process single frame
- Process multiple frames (buffer fills)
- Reset state
- Output shape [1, 128]
- Different output modes work

### Test Data

**Synthetic Test Cases**:
- Dummy frames (random noise)
- Known keypoint patterns (T-pose, arms up, etc.)
- Edge cases (person at image boundary)

**No Real Video Required**: Unit tests use synthetic data

## Appendices

### Appendix A: Technology Choices

**MoveNet Thunder**
- **Rationale**: TensorFlow-native, fast, accurate, 17 keypoints sufficient
- **Alternatives**: OpenPose (more keypoints but slower), MediaPipe (33 keypoints)
- **Trade-offs**: Thunder slower than Lightning but more accurate

**Shoulder-Width Normalization**
- **Rationale**: Stable reference, preserves proportions, upper-body focused
- **Alternatives**: Hip-based (less stable), bounding box (doesn't preserve proportions)
- **Trade-offs**: Requires reliable shoulder detection

**Simple MLP Encoder**
- **Rationale**: Fast, sufficient for pose encoding, CPU-friendly
- **Alternatives**: Graph Neural Network (captures skeleton structure better)
- **Trade-offs**: GNN more expressive but slower and more complex

**1D Temporal Convolution**
- **Rationale**: Lightweight, captures local motion, streaming-compatible
- **Alternatives**: Transformer (more expressive), GRU (stateful)
- **Trade-offs**: Limited receptive field (3 frames) vs more complex alternatives

**Python Wrapper for State**
- **Rationale**: Simple, flexible, debuggable, separates concerns
- **Alternatives**: Stateful Keras layers, custom layers with tf.Variable
- **Trade-offs**: State not saved with model, but easier to manage

### Appendix B: Research Findings Summary

**Pose Estimation**
- MoveNet outputs (y, x, confidence) format - note coordinate order
- 17 keypoints cover full body skeleton
- Confidence threshold 0.11 typical for visualization, we accept all

**Normalization**
- Two-stage normalization (global + local) improves accuracy in research
- Single-stage sufficient for streaming with latency constraints
- Shoulder-width more stable than hip-based for varied poses

**Temporal Modeling**
- Kernel size 3 appropriate for 4-8 frame buffer
- Causal padding essential for streaming (no future information)
- Dilated convolution unnecessary for small buffers

**Velocity Features**
- Simple frame difference adequate for baseline
- EMA smoothing (α=0.3) reduces jitter if needed
- Kalman filtering optimal but requires bidirectional processing

**Stateful Inference**
- Keras stateful LSTM requires fixed batch size and manual reset
- Python wrapper more flexible for custom buffer logic
- State management explicit and controllable

**MLP Design**
- 2-4x hidden dimension standard (256 for output 128)
- GELU activation matches transformer style
- LayerNorm improves training stability

### Appendix C: Alternative Approaches

**Alternative 1: 3D Pose Estimation**
- Use models like VIBE for 3D keypoints
- **Pros**: More realistic motion modeling
- **Cons**: Slower, more complex, requires depth estimation
- **Decision**: 2D sufficient for initial version

**Alternative 2: Pose Transformer (PoseBERT)**
- Pre-trained transformer on motion capture data
- **Pros**: Better temporal dynamics, transfer learning
- **Cons**: Requires large model, pre-training data, slower
- **Decision**: Too complex for initial version

**Alternative 3: Graph Convolutional Network**
- Model skeleton as graph with joints as nodes
- **Pros**: Captures body structure explicitly
- **Cons**: More complex, slower, harder to implement
- **Decision**: MLP sufficient, can upgrade later

**Alternative 4: Batch Processing**
- Process multiple frames in parallel
- **Pros**: Better GPU utilization, higher throughput
- **Cons**: Introduces latency, not true streaming
- **Decision**: Frame-by-frame for real-time requirement

### Appendix D: Future Enhancements

**Enhancement 1: Multi-Person Support**
- Track multiple people in frame
- Output embeddings per person
- Requires person tracking/association

**Enhancement 2: Confidence-Based Weighting**
- Weight keypoints by confidence in feature vector
- Mask low-confidence joints in attention
- Improves robustness to occlusion

**Enhancement 3: Adaptive Buffer Size**
- Dynamically adjust buffer based on motion speed
- Larger buffer for slow motion, smaller for fast
- Optimizes latency vs temporal context trade-off

**Enhancement 4: Pose Smoothing**
- Apply temporal smoothing to keypoints
- Reduce jitter from pose detector
- EMA or Kalman filtering

**Enhancement 5: GPU Optimization**
- Optimize for GPU when available
- Batch processing mode for offline use
- TensorRT conversion for inference

**Enhancement 6: Attention Pooling**
- Replace "take last timestep" with attention pooling
- Learn to weight temporal features
- More expressive than simple extraction

### Appendix E: Configuration Parameters

**Module Configuration**:
```python
CONFIG = {
    # Buffer
    'buffer_size': 8,  # frames
    
    # Model dimensions
    'embedding_dim': 128,  # matches D_MODEL
    'hidden_dim': 256,    # MLP hidden size
    'feature_dim': 85,    # 17 joints × 5 features
    
    # Temporal convolution
    'temporal_kernel': 3,
    'temporal_filters': 128,
    
    # MoveNet
    'movenet_variant': 'thunder',
    'movenet_input_size': 256,
    
    # Normalization
    'normalization_method': 'shoulder_width',
    
    # Velocity
    'velocity_smoothing': None,  # or 'ema' with alpha=0.3
    
    # Output
    'output_mode': 'embedding',  # 'embedding', 'with_confidence', 'with_keypoints'
}
```

**Tunable Hyperparameters**:
- `buffer_size`: 4-16 frames (trade-off: temporal context vs latency)
- `hidden_dim`: 128-512 (trade-off: capacity vs speed)
- `temporal_kernel`: 3-7 (trade-off: receptive field vs parameters)
- `velocity_smoothing`: None or EMA α=0.1-0.5 (trade-off: smoothness vs lag)

### Appendix F: Integration Notes

**Integration with Music Generation Model**:

The body point module outputs embeddings that can be used in cross-attention:

```python
# In music generation model
pose_embeddings = body_point_module.process_frame(video_frame)  # [1, 128]

# Use in cross-attention
audio_output = cross_attention(
    query=audio_tokens,        # [batch, audio_time, 128]
    key=pose_embeddings,       # [batch, 1, 128]
    value=pose_embeddings      # [batch, 1, 128]
)
```

**Temporal Alignment**:
- Video frame rate (e.g., 30fps) may differ from audio frame rate
- Resample or interpolate pose embeddings to match audio timeline
- Consider buffering pose embeddings for temporal alignment

**Training Integration**:
- Freeze MoveNet weights during training
- Train MLP encoder and temporal conv end-to-end with music model
- Gradients flow through cross-attention back to pose encoder

**Inference Pipeline**:
```
Video Stream → Body Point Module → Pose Embeddings
                                         ↓
Audio Stream → Music Generator ← Cross-Attention
                     ↓
              Generated Audio
```
