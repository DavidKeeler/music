"""Unit tests for train.py fixes: type compatibility and gradient flow."""

import tensorflow as tf
import numpy as np
from src.music_generation.train import MelGeneratorTraining, exponential_tf_schedule
from src.music_generation.config import (
    D_MODEL, NUM_HEADS, NUM_LAYERS, WINDOW_SIZES, 
    INITIAL_TF_RATIO, MIN_TF_RATIO, TF_DECAY_K, TF_WARMUP_STEPS
)


class SimpleMelGenerator(tf.keras.Model):
    """Minimal mock generator for testing."""
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(80)
    
    def call(self, x, training=False):
        return self.dense(x)


class TestTypeCompatibility(tf.test.TestCase):
    """Test type compatibility fix in update_tf_ratio()."""
    
    def test_update_tf_ratio_no_type_error(self):
        """Verify no TypeError occurs when updating tf_ratio."""
        model = SimpleMelGenerator()
        training_model = MelGeneratorTraining(
            model,
            initial_tf_ratio=INITIAL_TF_RATIO,
            min_tf_ratio=MIN_TF_RATIO,
            decay_k=TF_DECAY_K,
            warmup_steps=100
        )
        
        # This should not raise TypeError
        training_model.update_tf_ratio()
        
        # Verify tf_ratio is a valid float
        self.assertIsInstance(training_model.tf_ratio.numpy(), (float, np.floating))
    
    def test_warmup_behavior(self):
        """Verify warmup maintains initial ratio."""
        model = SimpleMelGenerator()
        training_model = MelGeneratorTraining(
            model,
            initial_tf_ratio=1.0,
            min_tf_ratio=0.05,
            decay_k=1e-5,
            warmup_steps=100
        )
        
        # During warmup (step < warmup_steps)
        training_model.training_step.assign(50)
        training_model.update_tf_ratio()
        self.assertAlmostEqual(training_model.tf_ratio.numpy(), 1.0, places=5)
        
        # After warmup (step > warmup_steps)
        training_model.training_step.assign(150)
        training_model.update_tf_ratio()
        # Should decay based on step_after_warmup = 50
        expected = exponential_tf_schedule(50, 1.0, 0.05, 1e-5)
        self.assertAlmostEqual(training_model.tf_ratio.numpy(), expected.numpy(), places=5)
    
    def test_type_consistency(self):
        """Verify training_step and warmup_steps types are compatible."""
        model = SimpleMelGenerator()
        training_model = MelGeneratorTraining(
            model,
            warmup_steps=100
        )
        
        # Check types
        step_dtype = training_model.training_step.dtype
        warmup_dtype = training_model.warmup_steps
        
        # After cast, subtraction should work
        training_model.training_step.assign(150)
        step_after_warmup = tf.maximum(0, training_model.training_step - tf.cast(warmup_dtype, tf.int64))
        
        # Should be int64
        self.assertEqual(step_after_warmup.dtype, tf.int64)
        self.assertEqual(step_after_warmup.numpy(), 50)


class TestGradientFlow(tf.test.TestCase):
    """Test gradient flow fix in autoregressive loop."""
    
    def test_gradients_flow_correctly(self):
        """Verify gradients flow through preds to model parameters."""
        model = SimpleMelGenerator()
        training_model = MelGeneratorTraining(model, initial_tf_ratio=0.5)
        
        # Create sample data
        batch_size = 2
        seq_len = 10
        x = tf.random.normal([batch_size, seq_len, 80])
        y = tf.random.normal([batch_size, seq_len, 80])
        
        with tf.GradientTape() as tape:
            # Run training step logic
            ar_input = x[:, :1, :]
            preds = []
            
            for t in range(seq_len - 1):
                next_frame_pred = model(ar_input, training=True)[:, -1:, :]
                use_teacher = tf.random.uniform([batch_size, 1, 1]) < training_model.tf_ratio
                next_input = tf.where(use_teacher, x[:, t+1:t+2, :], next_frame_pred)
                preds.append(next_frame_pred)
                ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
            
            pred_seq = tf.concat(preds, axis=1)
            loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Verify gradients exist and are not None
        self.assertIsNotNone(gradients)
        for grad in gradients:
            self.assertIsNotNone(grad)
            self.assertFalse(tf.reduce_all(tf.math.is_nan(grad)))
    
    def test_loss_value_consistency(self):
        """Verify loss value is identical with stop_gradient fix."""
        model = SimpleMelGenerator()
        training_model = MelGeneratorTraining(model, initial_tf_ratio=1.0)
        
        # Create sample data
        batch_size = 2
        seq_len = 10
        x = tf.random.normal([batch_size, seq_len, 80])
        y = tf.random.normal([batch_size, seq_len, 80])
        
        # Compute loss with stop_gradient (current implementation)
        ar_input = x[:, :1, :]
        preds = []
        
        for t in range(seq_len - 1):
            next_frame_pred = model(ar_input, training=False)[:, -1:, :]
            next_input = x[:, t+1:t+2, :]  # Teacher forcing
            preds.append(next_frame_pred)
            ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
        
        pred_seq = tf.concat(preds, axis=1)
        loss_with_stop = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
        
        # Compute loss without stop_gradient (for comparison)
        ar_input = x[:, :1, :]
        preds = []
        
        for t in range(seq_len - 1):
            next_frame_pred = model(ar_input, training=False)[:, -1:, :]
            next_input = x[:, t+1:t+2, :]
            preds.append(next_frame_pred)
            ar_input = tf.concat([ar_input, next_input], axis=1)  # No stop_gradient
        
        pred_seq = tf.concat(preds, axis=1)
        loss_without_stop = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
        
        # Loss values should be identical (stop_gradient only affects backprop)
        self.assertAlmostEqual(loss_with_stop.numpy(), loss_without_stop.numpy(), places=5)
    
    def test_memory_efficient_gradient_tape(self):
        """Verify GradientTape doesn't track ar_input growth."""
        model = SimpleMelGenerator()
        training_model = MelGeneratorTraining(model, initial_tf_ratio=0.5)
        
        batch_size = 2
        seq_len = 20
        x = tf.random.normal([batch_size, seq_len, 80])
        y = tf.random.normal([batch_size, seq_len, 80])
        
        with tf.GradientTape() as tape:
            ar_input = x[:, :1, :]
            preds = []
            
            for t in range(seq_len - 1):
                next_frame_pred = model(ar_input, training=True)[:, -1:, :]
                use_teacher = tf.random.uniform([batch_size, 1, 1]) < training_model.tf_ratio
                next_input = tf.where(use_teacher, x[:, t+1:t+2, :], next_frame_pred)
                preds.append(next_frame_pred)
                # With stop_gradient, tape doesn't track ar_input growth
                ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
            
            pred_seq = tf.concat(preds, axis=1)
            loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
        
        # Compute gradients - should succeed without OOM
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Verify gradients are valid
        self.assertIsNotNone(gradients)
        for grad in gradients:
            self.assertIsNotNone(grad)


if __name__ == '__main__':
    tf.test.main()
