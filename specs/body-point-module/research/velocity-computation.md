# Velocity Computation and Feature Building

## Overview

Velocity features (dx, dy) capture motion information from pose sequences. For frame-by-frame streaming with a history buffer, we compute velocity from frame differences.

## Basic Velocity Computation

### Simple Frame Difference

**First-order velocity** (current frame - previous frame):
```python
dx[i] = x[t][i] - x[t-1][i]
dy[i] = y[t][i] - y[t-1][i]
```

Where:
- `t` = current frame
- `t-1` = previous frame
- `i` = keypoint index (0-16 for MoveNet)

**Properties**:
- Simple and fast
- Noisy due to pose detection jitter
- Sensitive to frame rate

### Central Difference (Smoother)

**Central difference** (uses frames before and after):
```python
dx[i] = (x[t+1][i] - x[t-1][i]) / 2
dy[i] = (y[t+1][i] - y[t-1][i]) / 2
```

**Trade-off**: Smoother but requires future frame (not suitable for real-time streaming)

## Smoothing Techniques

### 1. Exponential Moving Average (EMA)

**Smooth velocity over time**:
```python
velocity_smoothed[t] = α * velocity[t] + (1-α) * velocity_smoothed[t-1]
```

Where α ∈ [0, 1] is smoothing factor:
- α = 1.0: No smoothing (use raw velocity)
- α = 0.5: Equal weight to current and history
- α = 0.2: Heavy smoothing

**Pros**: Simple, low memory, streaming-compatible
**Cons**: Introduces lag

### 2. Moving Average

**Average over window**:
```python
velocity_smoothed[t] = mean(velocity[t-w:t])
```

Where w = window size (e.g., 3-5 frames)

**Pros**: Reduces noise effectively
**Cons**: Requires buffer, introduces lag

### 3. Kalman Filtering

**Optimal linear filter** for noisy measurements:
- Predicts next state based on motion model
- Updates prediction with measurement
- Balances prediction and measurement uncertainty

**Pros**: Optimal for linear motion, handles missing data
**Cons**: More complex, requires tuning

Research shows Kalman smoothing substantially reduces velocity error in pose estimation but requires bidirectional processing (not suitable for real-time streaming).

## Recommended Approach

For frame-by-frame streaming with 4-8 frame buffer:

### Option 1: Simple Frame Difference (Baseline)
```python
# Compute velocity from consecutive frames
if len(buffer) >= 2:
    dx = current_pose[:, 0] - buffer[-1][:, 0]  # x velocity
    dy = current_pose[:, 1] - buffer[-1][:, 1]  # y velocity
else:
    dx = zeros_like(current_pose[:, 0])
    dy = zeros_like(current_pose[:, 1])
```

### Option 2: EMA Smoothing (Recommended)
```python
# Smooth velocity with exponential moving average
alpha = 0.3  # Smoothing factor
velocity_raw = current_pose - buffer[-1]
velocity_smoothed = alpha * velocity_raw + (1-alpha) * velocity_prev
```

### Option 3: Buffer Average
```python
# Average velocity over last N frames
if len(buffer) >= 3:
    velocities = [buffer[i] - buffer[i-1] for i in range(-2, 0)]
    velocity = mean(velocities, axis=0)
```

**Recommendation**: Start with Option 1 (simple), add Option 2 (EMA) if jitter is problematic.

## Feature Vector Construction

### Per-Frame Feature Vector

For each frame with 17 keypoints:

**Input from MoveNet**: `[17, 3]` → (y, x, confidence)

**After normalization**: `[17, 2]` → (x_norm, y_norm)

**With velocity**: `[17, 5]` → (x_norm, y_norm, dx, dy, confidence)

**Flattened**: `[85]` features

### Feature Ordering

```python
features = []
for joint_idx in range(17):
    features.extend([
        x_normalized[joint_idx],
        y_normalized[joint_idx],
        dx[joint_idx],
        dy[joint_idx],
        confidence[joint_idx]
    ])
# Result: [85] feature vector
```

## Handling First Frame

When buffer is empty or has only one frame:

**Option 1**: Zero velocity
```python
dx = zeros(17)
dy = zeros(17)
```

**Option 2**: Skip velocity features initially
```python
if len(buffer) < 2:
    features = [x_norm, y_norm, confidence]  # [17, 3] → [51]
else:
    features = [x_norm, y_norm, dx, dy, confidence]  # [17, 5] → [85]
```

**Recommendation**: Option 1 (zero velocity) for consistent feature dimension.

## Implementation Notes

### Coordinate System
- MoveNet outputs (y, x, confidence)
- Convert to (x, y) for consistency
- Compute velocity in normalized space (after shoulder-width normalization)

### Velocity Magnitude
Optional: Include velocity magnitude as additional feature
```python
velocity_mag = sqrt(dx² + dy²)
features = [x_norm, y_norm, dx, dy, velocity_mag, confidence]  # [17, 6] → [102]
```

### Frame Rate Normalization
If frame rate varies, normalize velocity by time delta:
```python
dt = timestamp[t] - timestamp[t-1]
dx = (x[t] - x[t-1]) / dt
dy = (y[t] - y[t-1]) / dt
```

For fixed frame rate (e.g., 30fps), dt is constant and can be omitted.

## Summary

**Recommended configuration**:
- Simple frame difference for velocity (dx, dy)
- Optional EMA smoothing (α=0.3) if jitter is problematic
- Feature vector: [x_norm, y_norm, dx, dy, confidence] per joint
- Total features: 17 joints × 5 = 85 dimensions
- Zero velocity for first frame

This provides motion information while maintaining low latency for streaming.

## References

- Kalman smoothing reduces velocity error but requires bidirectional processing
- EMA is standard for real-time smoothing in streaming applications
- Central difference provides smoother estimates but introduces 1-frame lag

*Content was rephrased for compliance with licensing restrictions*
