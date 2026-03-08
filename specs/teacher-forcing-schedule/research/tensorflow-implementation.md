# TensorFlow/Keras Implementation Patterns

## Available TensorFlow APIs

### TensorFlow Addons (Recommended)

`tfa.seq2seq.ScheduledOutputTrainingSampler` provides built-in scheduled sampling support:

```python
import tensorflow_addons as tfa

sampler = tfa.seq2seq.ScheduledOutputTrainingSampler(
    sampling_probability=tf_ratio,  # Teacher forcing ratio (0-1)
    time_major=False,
    seed=None,
    next_inputs_fn=None  # Optional: custom function to process sampled outputs
)
```

**Key features:**
- Integrates with `tfa.seq2seq` decoder infrastructure
- Handles probability-based sampling automatically
- Supports custom input processing via `next_inputs_fn`

**Limitations:**
- Requires TensorFlow Addons (additional dependency)
- Designed for RNN-based seq2seq (may need adaptation for Transformers)
- TensorFlow Addons is in maintenance mode (limited future updates)

### Custom Implementation in Keras

For Transformer-based models or custom training loops, implement scheduled sampling manually:

**Approach 1: Custom Training Step**

Override `train_step` in Keras Model:

```python
class MelGeneratorWithScheduling(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tf_ratio = tf.Variable(1.0, trainable=False)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            # Autoregressive generation with scheduled sampling
            predictions = []
            current_input = x[:, 0:1, :]  # First frame
            
            for t in range(y.shape[1]):
                # Predict next frame
                pred = self.model(current_input, training=True)
                predictions.append(pred[:, -1:, :])
                
                # Scheduled sampling: use ground truth or prediction
                use_teacher = tf.random.uniform([]) < self.tf_ratio
                next_input = tf.cond(
                    use_teacher,
                    lambda: y[:, t:t+1, :],  # Ground truth
                    lambda: pred[:, -1:, :]   # Model prediction
                )
                current_input = tf.concat([current_input, next_input], axis=1)
            
            predictions = tf.concat(predictions, axis=1)
            loss = self.compute_loss(y=y, y_pred=predictions)
        
        # Update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": loss}
```

**Approach 2: Probability-Based Masking**

Use probabilistic masking to blend ground truth and predictions:

```python
def scheduled_sampling_step(ground_truth, predictions, tf_ratio):
    """
    Args:
        ground_truth: [batch, time, features]
        predictions: [batch, time, features]
        tf_ratio: scalar tensor (0-1)
    
    Returns:
        Mixed inputs for next step
    """
    batch_size = tf.shape(ground_truth)[0]
    time_steps = tf.shape(ground_truth)[1]
    
    # Sample random values for each timestep
    random_vals = tf.random.uniform([batch_size, time_steps, 1])
    
    # Create mask: 1 = use ground truth, 0 = use prediction
    mask = tf.cast(random_vals < tf_ratio, ground_truth.dtype)
    
    # Blend inputs
    mixed = mask * ground_truth + (1 - mask) * predictions
    
    return mixed
```

## Integration with Current Codebase

### Current Training Structure

From `src/music_generation/train.py`, the current implementation uses:
- Pure teacher forcing (ground truth always fed as input)
- Autoregressive training with shifted targets
- Custom training loop with `tf.GradientTape`

### Modification Points

1. **Add teacher forcing ratio tracking:**
   ```python
   self.tf_ratio = tf.Variable(initial_tf_ratio, trainable=False, dtype=tf.float32)
   ```

2. **Implement schedule update:**
   ```python
   def update_tf_ratio(self, step):
       new_ratio = self.schedule_fn(step)
       self.tf_ratio.assign(new_ratio)
   ```

3. **Modify training step:**
   - Add conditional logic to choose between ground truth and predictions
   - Use `tf.cond` or probabilistic masking
   - Ensure gradients flow through predictions when used

4. **Track as metric:**
   ```python
   self.tf_ratio_metric = tf.keras.metrics.Mean(name='tf_ratio')
   ```

## Performance Considerations

### Computational Overhead

**Minimal impact:**
- Random number generation: O(batch_size * time_steps)
- Conditional selection: O(batch_size * time_steps * features)
- Negligible compared to model forward pass

**Memory:**
- No additional memory for activations
- Small overhead for random values and masks

### Training Speed

- Pure teacher forcing: Fastest (parallel processing possible)
- Scheduled sampling: Slightly slower (sequential autoregressive generation)
- Trade-off: Better inference quality vs. training speed

## Best Practices

1. **Start with pure teacher forcing:**
   - Train for initial epochs with tf_ratio=1.0
   - Ensures stable initial learning

2. **Gradual decay:**
   - Use smooth schedules (exponential, inverse sigmoid)
   - Avoid abrupt changes that destabilize training

3. **Monitor metrics:**
   - Track tf_ratio alongside loss
   - Watch for training instability as ratio decreases

4. **Minimum ratio:**
   - Keep tf_ratio > 0 (e.g., 0.05-0.1)
   - Some teacher forcing helps maintain stability

5. **Validation:**
   - Evaluate with tf_ratio=0 (pure autoregressive)
   - Matches inference conditions

## References

[1] TensorFlow Addons ScheduledOutputTrainingSampler: https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq/ScheduledOutputTrainingSampler

[2] Stack Overflow: How to Implement Scheduled Sampling in Keras: https://stackoverflow.com/questions/56464316/how-to-implement-scheduled-sampling-in-keras

[3] TensorFlow Guide: Customizing what happens in fit(): https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
