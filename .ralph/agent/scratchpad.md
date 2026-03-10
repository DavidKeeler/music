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

## Next Steps
Continue with Step 3: Implement SkeletonNormalizer
