# Body Point Module Implementation

## Objective

Implement a TensorFlow-based body point module that extracts pose embeddings from video frames for music generation conditioning. The module processes frames in real-time, detecting body keypoints and encoding them into 128-dimensional embeddings suitable for cross-attention.

## Key Requirements

**Input/Output**:
- Input: Video frames (variable resolution, single person)
- Processing: Frame-by-frame streaming with 4-8 frame history buffer
- Output: 128-dimensional embeddings (matches audio model D_MODEL)

**Architecture**:
- Pretrained MoveNet Thunder for pose detection (frozen weights)
- Shoulder-width normalization for body-relative coordinates
- Velocity features (dx, dy) from frame differences
- MLP encoder (85 → 256 → 256 → 128) with LayerNorm and GELU
- Temporal 1D convolution (kernel=3, causal padding)
- Stateful Python wrapper managing internal buffer

**Components**:
1. PoseDetector - MoveNet Thunder wrapper
2. SkeletonNormalizer - Shoulder-width normalization
3. FeatureBuilder - Velocity computation and feature vector construction
4. HistoryBuffer - Sliding window (FIFO with left-padding)
5. PoseEncoder - Trainable MLP + temporal conv
6. BodyPointModule - Main stateful interface

**Training**:
- MoveNet weights frozen
- MLP encoder and temporal conv trainable
- Train end-to-end with music generation model

**Deployment**:
- CPU-optimized (Mac compatibility)
- No error handling (edge cases handled externally)

## Acceptance Criteria

```gherkin
Given: A valid video frame [H, W, 3]
When: process_frame() is called
Then: Returns embedding [1, 128]

Given: Two consecutive frames in buffer
When: Features are built
Then: Velocity (dx, dy) computed as frame difference

Given: Empty buffer (first frame)
When: Features are built
Then: Velocity set to zero for all joints

Given: Buffer size = 8 and 10 frames processed
When: Buffer state checked
Then: Contains only last 8 frames

Given: Buffer with frames
When: reset() is called
Then: Buffer is empty

Given: PoseEncoder model
When: Gradients computed
Then: MLP and temporal conv trainable, MoveNet frozen
```

## Implementation Reference

Complete specifications in `specs/body-point-module/`:
- `design.md` - Detailed architecture, components, interfaces, data models
- `plan.md` - 10-step implementation plan with code examples and tests
- `research/` - Technical research on MoveNet, normalization, temporal conv, etc.

## Module Structure

```
src/body_point_module/
├── __init__.py
├── config.py          # Constants (EMBEDDING_DIM=128, BUFFER_SIZE=8, etc.)
├── pose_detector.py   # MoveNet wrapper
├── normalizer.py      # Shoulder-width normalization
├── features.py        # Velocity computation and feature building
├── buffer.py          # History buffer management
├── encoder.py         # MLP + temporal conv models
└── module.py          # Main BodyPointModule class

tests/body_point_module/
├── test_pose_detector.py
├── test_normalizer.py
├── test_features.py
├── test_buffer.py
├── test_encoder.py
└── test_module.py
```

## Key Implementation Details

**Feature Vector**: [x_norm, y_norm, dx, dy, confidence] × 17 joints = 85 dimensions

**MoveNet Output**: [1, 1, 17, 3] with (y, x, confidence) - note coordinate order

**Shoulder Indices**: left_shoulder=5, right_shoulder=6

**Normalization**: center = (left + right) / 2, scale = distance(left, right)

**Buffer Padding**: Left-pad with zeros when buffer not full

**Temporal Conv**: Extract last timestep after convolution for current frame embedding

## Testing Requirements

Unit tests for:
- MoveNet integration and output shape/ranges
- Shoulder-width normalization correctness
- Velocity computation (zero for first frame, difference for subsequent)
- Buffer FIFO behavior and padding
- Encoder architecture and trainability
- Full module end-to-end processing

Run: `pytest tests/body_point_module/ -v`

## Dependencies

```
tensorflow>=2.13
tensorflow-hub>=0.14
```

## Success Criteria

- All unit tests pass
- Module processes frames and outputs [1, 128] embeddings
- Buffer management works correctly (FIFO, padding, reset)
- MLP encoder and temporal conv are trainable
- MoveNet weights remain frozen
- CPU inference works on Mac
