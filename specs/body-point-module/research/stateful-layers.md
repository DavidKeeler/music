# Stateful Keras Layers for Streaming

## Overview

For frame-by-frame streaming inference, the body point module needs to maintain internal state (history buffer) between calls. This document explores patterns for implementing stateful behavior in TensorFlow/Keras.

## Stateful vs Stateless

### Stateless (Default)
- Each call is independent
- No memory between invocations
- Caller manages state

### Stateful (Required for our module)
- Maintains internal state between calls
- Memory persists across invocations
- Module manages state internally

## Approaches for Stateful Behavior

### 1. Stateful RNN Layers (LSTM/GRU)

Keras provides built-in stateful RNN support:

```python
lstm = tf.keras.layers.LSTM(
    units=128,
    stateful=True,
    return_sequences=True,
    batch_input_shape=(1, 1, features)  # Fixed batch size required
)
```

**Key points**:
- `stateful=True` preserves hidden state between batches
- Requires fixed `batch_input_shape` (batch size must be specified)
- State persists until manually reset with `layer.reset_states()`
- Default initial state is zeros

**Usage pattern**:
```python
# Process frames one at a time
for frame in video_stream:
    output = model.predict(frame)  # State automatically maintained

# Reset when switching to new video
model.reset_states()
```

**Limitation**: Only works for RNN layers (LSTM, GRU), not for custom buffer logic.

### 2. Custom Layer with tf.Variable

Create custom layer that maintains state using `tf.Variable`:

```python
class StatefulBufferLayer(tf.keras.layers.Layer):
    def __init__(self, buffer_size, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.feature_dim = feature_dim
        
    def build(self, input_shape):
        # Create stateful buffer as trainable=False variable
        self.buffer = self.add_weight(
            name='buffer',
            shape=(self.buffer_size, self.feature_dim),
            initializer='zeros',
            trainable=False  # Not a model parameter
        )
        self.buffer_index = self.add_weight(
            name='buffer_index',
            shape=(),
            initializer='zeros',
            trainable=False,
            dtype=tf.int32
        )
        
    def call(self, inputs):
        # Update buffer with new frame
        # ... buffer update logic ...
        return processed_output
        
    def reset_state(self):
        self.buffer.assign(tf.zeros_like(self.buffer))
        self.buffer_index.assign(0)
```

**Pros**: 
- Full control over state management
- Works with any logic (not just RNN)
- State persists between calls

**Cons**:
- More complex to implement
- Need to handle state updates carefully
- State is part of model weights (saved/loaded with model)

### 3. Python Class with Internal State (Simplest)

Wrap Keras model in Python class that manages state:

```python
class StatefulBodyPointModule:
    def __init__(self, model, buffer_size):
        self.model = model  # Stateless Keras model
        self.buffer = []
        self.buffer_size = buffer_size
        
    def __call__(self, frame):
        # Update buffer
        self.buffer.append(frame)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        # Prepare input from buffer
        if len(self.buffer) < self.buffer_size:
            # Pad with zeros or repeat first frame
            padded_buffer = self._pad_buffer()
        else:
            padded_buffer = self.buffer
            
        # Run model
        buffer_tensor = tf.stack(padded_buffer, axis=0)
        output = self.model(buffer_tensor)
        return output
        
    def reset(self):
        self.buffer = []
```

**Pros**:
- Simple and intuitive
- Easy to debug
- State management is explicit
- Model itself remains stateless

**Cons**:
- State not saved with model
- Need to manage wrapper separately

## Recommended Approach

**For our body point module**: Use **Approach 3** (Python wrapper)

**Rationale**:
1. **Simplicity**: Buffer management is straightforward Python list operations
2. **Flexibility**: Easy to modify buffer logic (e.g., change size, add smoothing)
3. **Debugging**: State is visible and inspectable
4. **Separation of concerns**: Keras model handles computation, wrapper handles state
5. **Training compatibility**: Stateless model can be trained normally with batches

## Implementation Pattern

```python
class BodyPointModule:
    def __init__(self, pose_model, encoder_model, buffer_size=8):
        self.pose_model = pose_model  # MoveNet (frozen)
        self.encoder_model = encoder_model  # MLP + temporal conv (trainable)
        self.buffer_size = buffer_size
        self.buffer = []
        
    def process_frame(self, frame_image):
        # 1. Extract keypoints
        keypoints = self.pose_model(frame_image)
        
        # 2. Normalize and build features
        features = self._build_features(keypoints)
        
        # 3. Update buffer
        self.buffer.append(features)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        # 4. Prepare buffer tensor
        buffer_tensor = self._prepare_buffer_tensor()
        
        # 5. Encode to embedding
        embedding = self.encoder_model(buffer_tensor)
        
        return embedding
        
    def reset(self):
        """Reset buffer for new video stream"""
        self.buffer = []
        
    def _prepare_buffer_tensor(self):
        # Pad if buffer not full
        if len(self.buffer) < self.buffer_size:
            padding = [tf.zeros_like(self.buffer[0])] * (self.buffer_size - len(self.buffer))
            full_buffer = padding + self.buffer
        else:
            full_buffer = self.buffer
        return tf.stack(full_buffer, axis=0)
```

## State Management Considerations

### When to Reset State
- Switching to new video stream
- After long pause in streaming
- On explicit user request

### State Persistence
- State lives in Python object (not saved with model)
- To save state: pickle the wrapper object
- To restore: unpickle and restore buffer

### Thread Safety
- Python wrapper is not thread-safe by default
- For multi-threaded inference, use locks or separate instances per thread

## Training vs Inference

**Training**: Use stateless model with full sequences
```python
# Input: [batch, time, features]
# Model processes entire sequence at once
output = encoder_model(sequence_batch)
```

**Inference**: Use stateful wrapper with single frames
```python
# Input: single frame
# Wrapper maintains buffer internally
output = body_point_module.process_frame(frame)
```

## Summary

**Recommended**: Python wrapper class with internal buffer state
- Keras model remains stateless (easier to train)
- Wrapper manages buffer as Python list
- Simple, flexible, and debuggable
- State management is explicit and controllable

This pattern is widely used in production systems for streaming inference.

## References

- Keras stateful LSTM requires fixed batch size and manual state reset
- Custom layers with tf.Variable can maintain state but add complexity
- Python wrappers are standard pattern for streaming inference
- State should be reset between independent sequences

*Content was rephrased for compliance with licensing restrictions*
