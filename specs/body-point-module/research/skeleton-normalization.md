# Skeleton Normalization Techniques

## Overview

Skeleton normalization is critical for pose estimation systems to handle variations in camera position, distance, and body size. This document explores normalization strategies with focus on shoulder-width normalization.

## Common Normalization Approaches

### 1. Root-Relative Normalization
- **Method**: Subtract a root point (typically hip midpoint), scale by torso length
- **Pros**: Removes camera position and scale variations
- **Cons**: Requires reliable hip detection, torso length can vary with pose

### 2. Bounding Box Normalization
- **Method**: Normalize coordinates to 0-1 based on detected person's bounding box
- **Pros**: Simple, handles any pose
- **Cons**: Doesn't preserve body proportions, sensitive to cropping

### 3. Shoulder-Width Normalization (Selected)
- **Method**: Subtract center point, scale by shoulder distance
- **Pros**: 
  - Stable reference (shoulders less affected by pose than hips)
  - Preserves body proportions
  - Works well for upper-body focused applications
- **Cons**: Requires reliable shoulder detection

## Shoulder-Width Normalization Implementation

### Algorithm

Given keypoints from MoveNet (17 keypoints with indices):
- `left_shoulder`: index 5, coordinates (y, x, confidence)
- `right_shoulder`: index 6, coordinates (y, x, confidence)

**Step 1: Calculate shoulder midpoint (center)**
```
center_x = (left_shoulder_x + right_shoulder_x) / 2
center_y = (left_shoulder_y + right_shoulder_y) / 2
```

**Step 2: Calculate shoulder width (scale factor)**
```
shoulder_width = sqrt((left_shoulder_x - right_shoulder_x)² + 
                      (left_shoulder_y - right_shoulder_y)²)
```

**Step 3: Normalize all keypoints**
```
For each keypoint i:
    normalized_x[i] = (x[i] - center_x) / shoulder_width
    normalized_y[i] = (y[i] - center_y) / shoulder_width
```

### Properties

After normalization:
- Shoulder midpoint is at origin (0, 0)
- Shoulder distance is 1.0
- All other joints are scaled proportionally
- Coordinates are translation and scale invariant

### Two-Stage Normalization (Advanced)

Research shows that two-stage normalization can improve accuracy:

1. **Global (Body) Normalization**: Rotate entire skeleton to upright position, then apply shoulder-width normalization
2. **Local (Limb) Normalization**: Further normalize individual limbs to consistent orientations

For our frame-by-frame streaming application, we'll use single-stage shoulder-width normalization for simplicity and low latency.

## Handling Edge Cases

Per requirements (A11), edge cases are handled externally. However, for reference:

**Small shoulder width** (< threshold):
- Could use fallback to bounding box normalization
- Or use previous frame's shoulder width

**Missing shoulder keypoints**:
- Could use neck-to-hip distance as alternative scale
- Or skip normalization and pass through raw coordinates

## Coordinate System Considerations

**MoveNet output format**: (y, x, confidence) with normalized coordinates [0, 1]

**Conversion to pixel coordinates** (before normalization):
```
pixel_x = x * image_width
pixel_y = y * image_height
```

Then apply shoulder-width normalization in pixel space.

## References

- Sun et al., "Human Pose Estimation Using Global and Local Normalization," ICCV 2017
- Two-stage normalization reduces pose variation and improves spatial model learning
- Body normalization particularly effective for shoulder/hip accuracy
- Limb normalization refines distal joints (wrist, ankle)

*Content was rephrased for compliance with licensing restrictions*
