# Body Point Module Implementation - Scratchpad

## Objective
Implement a TensorFlow-based body point module that extracts pose embeddings from video frames for music generation conditioning.

## Understanding
The module needs to:
1. Process video frames frame-by-frame in real-time
2. Detect 17 body keypoints using pretrained MoveNet Thunder (frozen)
3. Normalize keypoints using shoulder-width normalization
4. Compute velocity features from frame history
5. Maintain a sliding history buffer (4-8 frames)
6. Encode features to 128-dimensional embeddings using trainable MLP + temporal conv
7. Provide stateful interface with reset capability

## Architecture Components
1. **PoseDetector** - MoveNet Thunder wrapper (frozen weights)
2. **SkeletonNormalizer** - Shoulder-width normalization
3. **FeatureBuilder** - Velocity computation and feature vector construction
4. **HistoryBuffer** - Sliding window (FIFO with left-padding)
5. **PoseEncoder** - Trainable MLP + temporal conv (Keras Model)
6. **BodyPointModule** - Main stateful interface

## Implementation Plan
Following the 10-step plan from specs/body-point-module/plan.md:
1. Project structure and dependencies
2. MoveNet pose detector wrapper
3. Skeleton normalization
4. Feature builder with velocity
5. History buffer
6. MLP encoder model
7. Temporal convolution layer
8. Main module integration
9. Output mode configuration
10. Unit tests

## Current Status
Starting fresh - no existing body_point_module code.

✅ **Step 1 Complete**: Project structure and config created
- Created src/body_point_module/ with __init__.py and config.py
- Created tests/body_point_module/ directory
- All configuration constants defined (EMBEDDING_DIM=128, BUFFER_SIZE=8, etc.)
- Module imports successfully
- Committed: 3a99afb

✅ **Step 2 Complete**: PoseDetector wrapper implemented
- Created pose_detector.py with MoveNet Thunder wrapper
- Loads model from TF Hub (movenet/singlepose/thunder/4)
- Handles frame resizing and padding to 256x256
- Returns keypoints [1,1,17,3] format (y,x,confidence)
- All 5 unit tests pass (shape, ranges, variable input sizes)
- Committed: 8edec30

✅ **Step 3 Complete**: SkeletonNormalizer implemented
- Created normalizer.py with shoulder-width normalization
- Extracts shoulders (indices 5, 6), computes center and scale
- Normalizes all 17 keypoints to body-relative coordinates
- Returns [17, 2] normalized coords and [17] confidence scores
- All 7 unit tests pass (shapes, midpoint, distance, confidence)
- Committed: 5c89677

✅ **Step 4 Complete**: FeatureBuilder implemented
- Created features.py with velocity computation
- Computes velocity from frame-to-frame keypoint differences
- Builds [x,y,dx,dy,conf]×17 feature vectors (85 dims)
- Zero velocity for first frame, then tracks differences
- Reset method clears previous keypoints
- All 8 unit tests pass (zero velocity, velocity computation, layout, reset)
- Integration test with PoseDetector + SkeletonNormalizer verified
- Committed: 76c73c8

✅ **Step 5 Complete**: HistoryBuffer implemented
- Created buffer.py with sliding window buffer
- FIFO behavior maintains fixed-size buffer (8 frames)
- Left-padding with zeros when buffer not full
- add() appends features, enforces size limit
- get_tensor() returns [buffer_size, feature_dim] tensor
- reset() clears buffer state
- All 8 unit tests pass (FIFO, padding, reset, shape consistency)
- Committed: 9b3568b

✅ **Step 6 Complete**: PoseEncoder implemented
- Created encoder.py with build_mlp_encoder() and build_pose_encoder()
- MLP: 85→256→256→128 with LayerNorm and GELU activation
- Temporal modeling: Conv1D with kernel_size=3, causal padding
- TimeDistributed MLP applied per-frame, extracts last timestep
- ~120K trainable parameters
- All 15 unit tests pass (shapes, trainability, temporal conv, gradient flow)
- Integrated into __init__.py exports
- Committed: 50beb23

## Next Steps
Continue with Step 7: Implement BodyPointModule


## Step 7: Implementing BodyPointModule

**Task**: Integrate all components into the main BodyPointModule class.

**Components to integrate**:
1. PoseDetector - MoveNet wrapper (✅ implemented)
2. SkeletonNormalizer - Shoulder-width normalization (✅ implemented)
3. FeatureBuilder - Velocity computation (✅ implemented)
4. HistoryBuffer - Sliding window (✅ implemented)
5. PoseEncoder - MLP + temporal conv (✅ implemented)

**Implementation approach**:
- Create module.py with BodyPointModule class
- Initialize all components in __init__
- Implement process_frame() for end-to-end pipeline
- Support output_mode parameter (embedding, with_confidence, with_keypoints)
- Implement reset() to clear state
- Implement get_state() for debugging

**Pipeline flow**:
frame → PoseDetector → SkeletonNormalizer → FeatureBuilder → HistoryBuffer → PoseEncoder → embedding
