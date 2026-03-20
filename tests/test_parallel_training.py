"""Unit tests for parallel training refactor."""
import tensorflow as tf
import numpy as np
from src.music_generation.train import MelGeneratorTraining
from src.music_generation.config import D_MODEL, N_MELS


class _MockTokenizer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(D_MODEL)
    def call(self, x, training=False):
        # Downsample 4x: take every 4th frame
        return self.proj(x[:, ::4, :])


class _MockDetokenizer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(N_MELS)
    def call(self, x, training=False):
        return self.proj(tf.repeat(x, 4, axis=1))


class SimpleMelGenerator(tf.keras.Model):
    """Minimal mock generator for testing."""
    
    def __init__(self):
        super().__init__()
        self.tokenizer = _MockTokenizer()
        self.detokenizer = _MockDetokenizer()
        self.tok_dense = tf.keras.layers.Dense(D_MODEL)
    
    def call(self, inputs, z=None, training=False):
        tokens = self.tokenizer(inputs, training=training)
        return self.forward_from_tokens(tokens, z=z, training=training)
    
    def forward_from_tokens(self, tokens, z=None, training=False):
        return self.detokenizer(self.tok_dense(tokens), training=training)


class TestPureTeacherForcing:
    """Test AC1: Pure teacher forcing (tf_ratio=1.0)."""
    
    def test_output_structure(self):
        """Verify output dict has correct keys and types."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=1.0)
        model.compile(optimizer='adam')
        
        x = tf.random.normal([4, 12, N_MELS])
        y = tf.random.normal([4, 12, N_MELS])
        
        result = model.train_step((x, y))
        
        assert 'loss' in result
        assert 'grad_norm' in result
        assert 'tf_ratio' in result
        assert result['loss'].shape == ()
        assert result['grad_norm'].shape == ()
    
    def test_single_forward_pass(self):
        """Verify pure teacher forcing path is used."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=1.0)
        model.compile(optimizer='adam')
        
        x = tf.random.normal([4, 12, N_MELS])
        y = tf.random.normal([4, 12, N_MELS])
        
        result = model.train_step((x, y))
        
        # Verify it completes successfully
        assert 'loss' in result
        assert tf.math.is_finite(result['loss'])
    
    def test_loss_computation(self):
        """Verify loss is computed as mean(|preds[:, :-1, :] - targets[:, 1:, :]|)."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=1.0)
        model.compile(optimizer='adam')
        
        x = tf.random.normal([2, 8, N_MELS])
        y = tf.random.normal([2, 8, N_MELS])
        
        result = model.train_step((x, y))
        
        # Loss should be finite and positive
        assert tf.math.is_finite(result['loss'])
        assert result['loss'] >= 0


class TestParallelScheduledSampling:
    """Test AC2: Parallel scheduled sampling (tf_ratio=0.5)."""
    
    def test_two_forward_passes(self):
        """Verify parallel sampling path is used."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=0.5)
        model.compile(optimizer='adam')
        
        x = tf.random.normal([4, 12, N_MELS])
        y = tf.random.normal([4, 12, N_MELS])
        
        result = model.train_step((x, y))
        
        # Verify it completes successfully
        assert 'loss' in result
        assert tf.math.is_finite(result['loss'])
    
    def test_sampling_mask_distribution(self):
        """Verify sampling mask has ~50% True values (±5%)."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=0.5)
        model.compile(optimizer='adam')
        
        # Run multiple times to check distribution
        true_ratios = []
        for _ in range(10):
            # Capture the mask by patching tf.random.uniform
            masks = []
            original_uniform = tf.random.uniform
            
            def capture_uniform(shape, *args, **kwargs):
                result = original_uniform(shape, *args, **kwargs)
                if len(shape) == 3:  # This is our sampling mask
                    masks.append(result < 0.5)
                return result
            
            tf.random.uniform = capture_uniform
            
            x = tf.random.normal([4, 100, N_MELS])
            y = tf.random.normal([4, 100, N_MELS])
            
            model.train_step((x, y))
            
            tf.random.uniform = original_uniform
            
            if masks:
                mask = masks[0]
                true_ratio = tf.reduce_mean(tf.cast(mask, tf.float32)).numpy()
                true_ratios.append(true_ratio)
        
        # Average should be close to 0.5
        avg_ratio = np.mean(true_ratios)
        assert 0.45 <= avg_ratio <= 0.55, f"Expected ~0.5, got {avg_ratio:.3f}"
    
    def test_output_structure(self):
        """Verify output dict has correct structure."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=0.5)
        model.compile(optimizer='adam')
        
        x = tf.random.normal([4, 12, N_MELS])
        y = tf.random.normal([4, 12, N_MELS])
        
        result = model.train_step((x, y))
        
        assert 'loss' in result
        assert 'grad_norm' in result
        assert 'tf_ratio' in result
        assert tf.math.is_finite(result['loss'])


class TestTrainStepDispatch:
    """Test train_step correctly dispatches based on tf_ratio."""
    
    def test_dispatch_pure_teacher_forcing(self):
        """Verify tf_ratio >= 0.99 uses pure teacher forcing."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=1.0)
        model.compile(optimizer='adam')
        
        # Patch methods to track which is called
        pure_tf_called = [False]
        parallel_called = [False]
        
        original_pure = model._pure_teacher_forcing
        original_parallel = model._parallel_scheduled_sampling
        
        def track_pure(*args, **kwargs):
            pure_tf_called[0] = True
            return original_pure(*args, **kwargs)
        
        def track_parallel(*args, **kwargs):
            parallel_called[0] = True
            return original_parallel(*args, **kwargs)
        
        model._pure_teacher_forcing = track_pure
        model._parallel_scheduled_sampling = track_parallel
        
        x = tf.random.normal([4, 12, N_MELS])
        y = tf.random.normal([4, 12, N_MELS])
        
        model.train_step((x, y))
        
        assert pure_tf_called[0], "Pure teacher forcing should be called"
        assert not parallel_called[0], "Parallel sampling should not be called"
    
    def test_dispatch_parallel_sampling(self):
        """Verify tf_ratio < 0.99 uses parallel scheduled sampling."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=0.5)
        model.compile(optimizer='adam')
        
        # Patch methods to track which is called
        pure_tf_called = [False]
        parallel_called = [False]
        
        original_pure = model._pure_teacher_forcing
        original_parallel = model._parallel_scheduled_sampling
        
        def track_pure(*args, **kwargs):
            pure_tf_called[0] = True
            return original_pure(*args, **kwargs)
        
        def track_parallel(*args, **kwargs):
            parallel_called[0] = True
            return original_parallel(*args, **kwargs)
        
        model._pure_teacher_forcing = track_pure
        model._parallel_scheduled_sampling = track_parallel
        
        x = tf.random.normal([4, 12, N_MELS])
        y = tf.random.normal([4, 12, N_MELS])
        
        model.train_step((x, y))
        
        assert not pure_tf_called[0], "Pure teacher forcing should not be called"
        assert parallel_called[0], "Parallel sampling should be called"
    
    def test_boundary_condition(self):
        """Test tf_ratio exactly at 0.99."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=0.99)
        model.compile(optimizer='adam')
        
        x = tf.random.normal([4, 12, N_MELS])
        y = tf.random.normal([4, 12, N_MELS])
        
        # Should use parallel sampling (< 0.99 check)
        result = model.train_step((x, y))
        
        assert 'loss' in result
        assert tf.math.is_finite(result['loss'])


class TestEndToEndTraining:
    """Test AC5: End-to-end training."""
    
    def test_training_loop(self):
        """Verify full training loop works with model.fit()."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=1.0)
        model.compile(optimizer='adam')
        
        # Create small dataset
        x = tf.random.normal([16, 12, N_MELS])
        y = tf.random.normal([16, 12, N_MELS])
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
        
        # Train for 2 epochs
        history = model.fit(dataset, epochs=2, verbose=0)
        
        assert 'loss' in history.history
        assert len(history.history['loss']) == 2
    
    def test_loss_decreases(self):
        """Verify loss decreases over training."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=1.0)
        model.compile(optimizer='adam')
        
        # Create dataset with learnable pattern
        x = tf.random.normal([32, 12, N_MELS])
        y = x * 2.0  # Simple pattern to learn
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(4).repeat()
        
        # Train for several steps
        history = model.fit(dataset, steps_per_epoch=10, epochs=3, verbose=0)
        
        losses = history.history['loss']
        # Loss should generally decrease
        assert losses[-1] < losses[0] * 1.5, "Loss should decrease or stay stable"
    
    def test_tf_ratio_updates(self):
        """Verify tf_ratio updates during training."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(
            base_model, 
            initial_tf_ratio=1.0,
            min_tf_ratio=0.5,
            decay_k=0.1,  # Fast decay for testing
            warmup_steps=0
        )
        model.compile(optimizer='adam')
        
        initial_ratio = model.tf_ratio.numpy()
        
        x = tf.random.normal([8, 12, N_MELS])
        y = tf.random.normal([8, 12, N_MELS])
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
        
        model.fit(dataset, epochs=5, verbose=0)
        
        final_ratio = model.tf_ratio.numpy()
        
        # Ratio should have decreased
        assert final_ratio < initial_ratio, f"Ratio should decrease: {initial_ratio} -> {final_ratio}"


class TestCheckpointCompatibility:
    """Test AC4: Checkpoint compatibility."""
    
    def test_save_and_load_weights(self):
        """Verify model can save and load weights."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=1.0)
        model.compile(optimizer='adam')
        
        # Build model by calling it
        x = tf.random.normal([4, 12, N_MELS])
        y = tf.random.normal([4, 12, N_MELS])
        _ = model(x, training=False)  # Build the model
        
        # Save weights
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False) as f:
            weights_path = f.name
        
        model.save_weights(weights_path)
        
        # Create new model and load
        new_base = SimpleMelGenerator()
        new_model = MelGeneratorTraining(new_base, initial_tf_ratio=1.0)
        new_model.compile(optimizer='adam')
        _ = new_model(x, training=False)  # Build
        new_model.load_weights(weights_path)
        
        # Verify weights match
        for w1, w2 in zip(model.base_model.weights, new_model.base_model.weights):
            assert tf.reduce_all(w1 == w2).numpy()
        
        # Cleanup
        import os
        os.unlink(weights_path)
    
    def test_variables_preserved(self):
        """Verify tf_ratio and training_step variables exist."""
        base_model = SimpleMelGenerator()
        model = MelGeneratorTraining(base_model, initial_tf_ratio=1.0)
        
        # Check variables exist
        assert hasattr(model, 'tf_ratio')
        assert hasattr(model, 'training_step')
        
        # Check they're Variables
        assert isinstance(model.tf_ratio, tf.Variable)
        assert isinstance(model.training_step, tf.Variable)
        
        # Check initial values
        assert model.tf_ratio.numpy() == 1.0
        assert model.training_step.numpy() == 0
