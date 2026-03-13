"""Unit tests for train.py fixes: type compatibility and gradient flow."""

import tensorflow as tf
import numpy as np
from src.music_generation.train import MelGeneratorTraining, exponential_tf_schedule
from src.music_generation.config import (
    D_MODEL, NUM_HEADS, NUM_LAYERS, WINDOW_SIZES, 
    INITIAL_TF_RATIO, MIN_TF_RATIO, TF_DECAY_K, TF_WARMUP_STEPS,
    GROUPED_MEL_DIM
)


class SimpleMelGenerator(tf.keras.Model):
    """Minimal mock generator for testing with reduction factor support."""
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(GROUPED_MEL_DIM)
    
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
        x = tf.random.normal([batch_size, seq_len, GROUPED_MEL_DIM])
        y = tf.random.normal([batch_size, seq_len, GROUPED_MEL_DIM])
        
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
        x = tf.random.normal([batch_size, seq_len, GROUPED_MEL_DIM])
        y = tf.random.normal([batch_size, seq_len, GROUPED_MEL_DIM])
        
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
        x = tf.random.normal([batch_size, seq_len, GROUPED_MEL_DIM])
        y = tf.random.normal([batch_size, seq_len, GROUPED_MEL_DIM])
        
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


class TestIntegration(tf.test.TestCase):
    """Integration tests for full training with both fixes."""
    
    def test_full_training_100_steps(self):
        """Verify training runs for 100 steps without OOM or type errors."""
        # Force eager execution for this test
        tf.config.run_functions_eagerly(True)
        
        model = SimpleMelGenerator()
        training_model = MelGeneratorTraining(
            model,
            initial_tf_ratio=1.0,
            min_tf_ratio=0.05,
            decay_k=1e-5,
            warmup_steps=50
        )
        
        # Compile model
        training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
        
        # Create sample data
        batch_size = 2
        seq_len = 32
        x = tf.random.normal([batch_size, seq_len, GROUPED_MEL_DIM])
        y = tf.random.normal([batch_size, seq_len, GROUPED_MEL_DIM])
        
        # Train for 100 steps - should not raise OOM or TypeError
        for step in range(100):
            loss_dict = training_model.train_step((x, y))
            
            # Verify loss is valid
            self.assertIsNotNone(loss_dict)
            self.assertIn('loss', loss_dict)
            self.assertFalse(tf.math.is_nan(loss_dict['loss']))
            self.assertFalse(tf.math.is_inf(loss_dict['loss']))
        
        # Verify tf_ratio was updated
        self.assertLess(training_model.tf_ratio.numpy(), 1.0)
        self.assertGreaterEqual(training_model.tf_ratio.numpy(), 0.05)
        
        # Verify training_step was incremented
        self.assertEqual(training_model.training_step.numpy(), 100)
        
        tf.config.run_functions_eagerly(False)
    
    def test_checkpoint_save_and_load(self):
        """Verify checkpoints can be saved and loaded correctly."""
        import tempfile
        import os
        
        # Force eager execution for this test
        tf.config.run_functions_eagerly(True)
        
        # Create and train model
        model = SimpleMelGenerator()
        training_model = MelGeneratorTraining(
            model,
            initial_tf_ratio=1.0,
            min_tf_ratio=0.05,
            decay_k=1e-5,
            warmup_steps=50
        )
        training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
        
        # Train for a few steps to build the model
        batch_size = 2
        seq_len = 16
        x = tf.random.normal([batch_size, seq_len, GROUPED_MEL_DIM])
        y = tf.random.normal([batch_size, seq_len, GROUPED_MEL_DIM])
        
        for _ in range(10):
            training_model.train_step((x, y))
        
        # Get model weights before saving
        weights_before = [w.numpy() for w in training_model.base_model.trainable_variables]
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.weights.h5')
            
            # Build the model explicitly
            training_model.build(input_shape=(batch_size, seq_len, 80))
            training_model.save_weights(checkpoint_path)
            
            # Create new model and load checkpoint
            new_model = SimpleMelGenerator()
            new_training_model = MelGeneratorTraining(
                new_model,
                initial_tf_ratio=1.0,
                min_tf_ratio=0.05,
                decay_k=1e-5,
                warmup_steps=50
            )
            new_training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
            
            # Build model with same input shape and run one step to initialize weights
            new_training_model.build(input_shape=(batch_size, seq_len, 80))
            new_training_model.train_step((x, y))
            
            # Load checkpoint
            new_training_model.load_weights(checkpoint_path)
            
            # Verify model weights were restored
            weights_after = [w.numpy() for w in new_training_model.base_model.trainable_variables]
            self.assertEqual(len(weights_before), len(weights_after))
            for w_before, w_after in zip(weights_before, weights_after):
                self.assertTrue(np.allclose(w_before, w_after))
            
            # Verify training can resume without errors
            loss_after = new_training_model.train_step((x, y))
            self.assertIsNotNone(loss_after)
            self.assertIn('loss', loss_after)
            self.assertFalse(tf.math.is_nan(loss_after['loss']))
        
        tf.config.run_functions_eagerly(False)


if __name__ == '__main__':
    tf.test.main()
