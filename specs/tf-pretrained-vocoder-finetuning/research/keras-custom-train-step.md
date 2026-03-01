# Keras Custom train_step with Metrics

## Overview

Keras allows customizing the training loop by overriding the `train_step()` method of a `Model` subclass. This enables custom training logic while still using `fit()` with callbacks, distribution, etc.

## Basic Pattern

```python
class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        
        # Return dict of metric name -> current value
        return {m.name: m.result() for m in self.metrics}
```

## Adding Custom Metrics (e.g., Gradient Norm)

To add custom metrics like gradient norm:

```python
class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.grad_norm_tracker = keras.metrics.Mean(name="grad_norm")
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = keras.losses.mean_squared_error(y, y_pred)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Compute gradient norm
        grad_norm = tf.sqrt(sum([tf.reduce_sum(g**2) for g in gradients if g is not None]))
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update custom metrics
        self.loss_tracker.update_state(loss)
        self.grad_norm_tracker.update_state(grad_norm)
        
        return {
            "loss": self.loss_tracker.result(),
            "grad_norm": self.grad_norm_tracker.result()
        }
    
    @property
    def metrics(self):
        # List metrics so reset_states() is called automatically
        return [self.loss_tracker, self.grad_norm_tracker]
```

## Key Points

1. **Return dictionary:** `train_step()` must return a dict mapping metric names to current values
2. **Metric tracking:** Create `keras.metrics.Mean` instances to track values across batches
3. **Reset states:** List metrics in the `metrics` property so they reset between epochs
4. **Gradient computation:** Use `tf.GradientTape()` to compute gradients
5. **Gradient norm:** Compute L2 norm with `tf.sqrt(sum([tf.reduce_sum(g**2) for g in gradients]))`

## Benefits

- Still use `fit()` with all its features (callbacks, distribution, etc.)
- Custom training logic with full control
- Automatic metric tracking and logging
- Works with TensorBoard callbacks

## Reference

- TensorFlow Guide: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
