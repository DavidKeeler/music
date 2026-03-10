# MoveNet Thunder Specifications

## Overview

MoveNet is an ultra-fast and accurate pose detection model from TensorFlow Hub that detects 17 keypoints of a body. Two variants exist:
- **Lightning**: Optimized for latency-critical applications
- **Thunder**: Optimized for high accuracy (recommended for this project)

Both run faster than real-time (30+ FPS) on most modern hardware.

## Model Details

**TF Hub URL**: `https://tfhub.dev/google/movenet/singlepose/thunder/4`

**Input**:
- Shape: `[1, height, width, 3]`
- Type: `int32` (for SavedModel format)
- Expected size: 256x256 pixels (Thunder variant)
- Images should be resized with padding to maintain aspect ratio

**Output**:
- Shape: `[1, 1, 17, 3]`
- Format: `[batch, instance, keypoint_index, (y, x, confidence)]`
- Coordinates are normalized (0.0 to 1.0) relative to image dimensions
- Confidence scores range from 0.0 to 1.0

## 17 Keypoint Structure

The model detects the following keypoints (index → name):

```python
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}
```

## Skeleton Connectivity

The body skeleton is defined by these bone connections:

```
Face:
- nose → left_eye, nose → right_eye
- left_eye → left_ear, right_eye → right_ear

Upper body:
- nose → left_shoulder, nose → right_shoulder
- left_shoulder ↔ right_shoulder (horizontal)
- left_shoulder → left_elbow → left_wrist
- right_shoulder → right_elbow → right_wrist

Torso:
- left_shoulder → left_hip
- right_shoulder → right_hip
- left_hip ↔ right_hip (horizontal)

Lower body:
- left_hip → left_knee → left_ankle
- right_hip → right_knee → right_ankle
```

## Key Observations for Module Design

1. **Coordinate Format**: Output is (y, x, confidence) not (x, y, confidence) - important for normalization
2. **Normalized Coordinates**: Values are 0.0-1.0, need to be converted to pixel coordinates before normalization
3. **Shoulder Keypoints**: Indices 5 (left_shoulder) and 6 (right_shoulder) are critical for shoulder-width normalization
4. **Confidence Threshold**: TensorFlow tutorial uses 0.11 as minimum threshold for visualization, but we accept all values per requirements
5. **Single Person**: Model is designed for single-pose detection, matching our requirements

## Integration Notes

**Loading the model**:
```python
import tensorflow_hub as hub
module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = module.signatures['serving_default']
```

**Running inference**:
```python
# Input: [1, 256, 256, 3] int32 tensor
outputs = movenet(input_image)
keypoints_with_scores = outputs['output_0'].numpy()  # [1, 1, 17, 3]
```

## References

- [MoveNet TensorFlow Tutorial](https://tensorflow.google.cn/hub/tutorials/movenet?hl=en)
- [TF Hub Model Page](https://tfhub.dev/google/movenet/singlepose/thunder/4)
- [GitHub: TensorFlow.js Models - MoveNet README](https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/movenet/README.md)

*Content was rephrased for compliance with licensing restrictions*
