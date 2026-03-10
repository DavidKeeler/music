# Rough Idea: Body Point Module

## Overview

Design a body point module for the music generation system that:
1. Takes in video as input
2. Runs a pretrained body point annotation model to extract keypoints
3. Uses the body points as input to a body embedding module
4. Outputs a sequence of embeddings that can be used for cross-attention

## Key Considerations

The module should separate three concerns:
- **Pose extraction**: video → body keypoints
- **Pose representation**: keypoints → normalized skeleton features  
- **Pose embedding**: skeleton → embedding sequence usable for cross-attention

## Context

This module is being designed for integration with the existing TensorFlow-based music generation system (see README.md). The output embeddings will eventually be used in cross-attention mechanisms, but this design focuses solely on the body point module itself, not the integration.

## Initial Technical Direction

**Pretrained Model Options:**
- MoveNet (17 keypoints, TF-native, fast)
- OpenPose (25 body keypoints, very detailed)
- MediaPipe Pose (33 keypoints, stable tracking)
- RTMPose/MMPose (17-133 keypoints, SOTA accuracy)

**Recommended**: MoveNet Thunder for TensorFlow compatibility, speed, and sufficient detail (17 joints)

**Processing Pipeline:**
1. Video → Frame sampling
2. Frames → Pose model (MoveNet) → Keypoints (x, y, confidence)
3. Keypoints → Normalization (body-relative coordinates)
4. Normalized keypoints → Feature builder (add velocity: dx, dy)
5. Features → Pose Encoder (MLP or GNN)
6. Encoded poses → Temporal Transformer
7. Output: Pose embeddings [batch, time, dim]

**Key Design Principles:**
- Normalize skeleton to body-relative coordinates (remove camera position/scale)
- Add temporal features (velocity) for motion capture
- Consider skeleton as a graph structure (joints connected by bones)
- Use temporal modeling (transformer) to capture motion over time
- Output format: [B, T, D] where T=time steps, D=embedding dimension (256-512)
