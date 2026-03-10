# Temporal Convolution for Pose Sequences

## Overview

Temporal convolution applies 1D convolution along the time axis to capture motion patterns in pose sequences. For our frame-by-frame streaming application with a small history buffer (4-8 frames), we need lightweight temporal modeling.

## 1D Temporal Convolution Basics

**Input shape**: `[batch, time, features]`
- For pose: `[batch, buffer_size, feature_dim]`
- Example: `[1, 8, 85]` (8 frames, 85 features per frame)

**Operation**: Slide kernel along time dimension
```
output[t] = sum(input[t-k:t+k] * kernel)
```

**Output shape**: `[batch, time, channels]`

## Kernel Size Selection

For buffer size of 4-8 frames:

| Kernel Size | Receptive Field | Use Case |
|-------------|-----------------|----------|
| 3 | 3 frames | Local motion (velocity, acceleration) |
| 5 | 5 frames | Medium-range patterns |
| 7 | 7 frames | Covers most of buffer |

**Recommendation**: Kernel size 3 or 5
- Kernel 3: Captures immediate temporal context (frame t-1, t, t+1)
- Kernel 5: Captures broader motion patterns within buffer
- Larger kernels (7+) approach buffer size, may overfit

## Causal vs Non-Causal Convolution

**Causal convolution**: Only looks at past frames
- Padding: left-pad with zeros
- Ensures no future information leakage
- Required for real-time streaming

**Non-causal convolution**: Looks at past and future
- Padding: symmetric
- Better for offline processing
- Not suitable for frame-by-frame streaming

**For our application**: Use causal convolution with left-padding

## Receptive Field Calculation

For simple 1D conv with stride=1:
```
receptive_field = kernel_size
```

For stacked convolutions:
```
receptive_field = 1 + sum((kernel_size[i] - 1) for each layer i)
```

Example with 2 layers of kernel size 3:
```
receptive_field = 1 + (3-1) + (3-1) = 5 frames
```

## Dilated Convolution (Optional Enhancement)

Dilated convolution expands receptive field without increasing parameters:
```
receptive_field = 1 + (kernel_size - 1) * dilation_rate
```

Example: kernel=3, dilation=2
```
receptive_field = 1 + (3-1) * 2 = 5 frames
```

**Trade-off**: Larger receptive field but skips intermediate frames

For small buffers (4-8 frames), standard convolution is sufficient.

## Architecture Patterns

### Pattern 1: Single Conv Layer (Simplest)
```
Input [B, T, F] → Conv1D(kernel=3, filters=128) → Output [B, T, 128]
```
- Minimal latency
- Receptive field: 3 frames
- Good for velocity-based features

### Pattern 2: Stacked Conv Layers
```
Input [B, T, F] 
  → Conv1D(kernel=3, filters=64)
  → ReLU
  → Conv1D(kernel=3, filters=128)
  → Output [B, T, 128]
```
- Receptive field: 5 frames
- More expressive
- Still lightweight

### Pattern 3: Residual Connection
```
Input [B, T, F]
  → Conv1D(kernel=3, filters=128)
  → ReLU
  → Conv1D(kernel=3, filters=128)
  → Add(Input projected to 128)
  → Output [B, T, 128]
```
- Helps gradient flow
- Preserves input information
- Slightly more complex

## Recommended Configuration

For 4-8 frame buffer with frame-by-frame processing:

**Single lightweight temporal layer**:
- Kernel size: 3
- Filters: 128 (match D_MODEL)
- Padding: 'causal' (left-pad)
- Activation: ReLU or GELU
- No pooling (preserve temporal resolution)

**Rationale**:
- Kernel 3 captures immediate temporal context
- Causal padding ensures streaming compatibility
- 128 filters match audio model's D_MODEL
- Single layer minimizes latency for CPU inference

## Implementation Notes

**TensorFlow/Keras**:
```python
tf.keras.layers.Conv1D(
    filters=128,
    kernel_size=3,
    padding='causal',  # Left-pad for streaming
    activation='relu'
)
```

**Input preparation**: Stack buffer frames along time dimension
```python
# Buffer: list of 8 frames, each [85] features
buffer_tensor = tf.stack(buffer, axis=0)  # [8, 85]
buffer_tensor = tf.expand_dims(buffer_tensor, axis=0)  # [1, 8, 85]
```

**Output**: Take last timestep for current frame embedding
```python
temporal_features = conv1d(buffer_tensor)  # [1, 8, 128]
current_embedding = temporal_features[:, -1, :]  # [1, 128]
```

## References

- Temporal Convolutional Networks (TCNs) use stacked dilated causal convolutions
- Receptive field grows exponentially with dilation: R = (f-1)(2^K - 1) + 1
- For small sequences, simple convolution is more efficient than dilated
- Causal convolution essential for real-time streaming applications

*Content was rephrased for compliance with licensing restrictions*
