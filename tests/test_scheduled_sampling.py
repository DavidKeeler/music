"""Tests for teacher forcing schedule function."""
import tensorflow as tf
import pytest
from src.music_generation.train import exponential_tf_schedule
from src.music_generation import config


class TestExponentialTFSchedule:
    """Test exponential decay schedule function."""
    
    def test_initial_step_returns_initial_ratio(self):
        """At step 0, should return initial_ratio."""
        ratio = exponential_tf_schedule(step=0, initial_ratio=1.0, min_ratio=0.05, decay_k=1e-5)
        assert tf.abs(ratio - 1.0) < 1e-6
    
    def test_respects_min_ratio_floor(self):
        """Ratio should never fall below min_ratio."""
        # Very large step should hit the floor
        ratio = exponential_tf_schedule(step=1000000, initial_ratio=1.0, min_ratio=0.05, decay_k=1e-5)
        assert ratio >= 0.05
        assert tf.abs(ratio - 0.05) < 1e-6
    
    def test_exponential_decay(self):
        """Ratio should decrease exponentially."""
        ratio_0 = exponential_tf_schedule(step=0, initial_ratio=1.0, min_ratio=0.0, decay_k=1e-5)
        ratio_10k = exponential_tf_schedule(step=10000, initial_ratio=1.0, min_ratio=0.0, decay_k=1e-5)
        ratio_20k = exponential_tf_schedule(step=20000, initial_ratio=1.0, min_ratio=0.0, decay_k=1e-5)
        
        # Should decrease
        assert ratio_10k < ratio_0
        assert ratio_20k < ratio_10k
        
        # Should follow exponential decay (ratio at 2x step ≈ ratio² at x step)
        expected_ratio_20k = ratio_10k * tf.exp(-1e-5 * 10000)
        assert tf.abs(ratio_20k - expected_ratio_20k) < 1e-6
    
    def test_different_decay_rates(self):
        """Larger decay_k should result in faster decay."""
        ratio_slow = exponential_tf_schedule(step=10000, initial_ratio=1.0, min_ratio=0.0, decay_k=1e-6)
        ratio_fast = exponential_tf_schedule(step=10000, initial_ratio=1.0, min_ratio=0.0, decay_k=1e-4)
        
        assert ratio_fast < ratio_slow
    
    def test_accepts_tensor_input(self):
        """Should work with TensorFlow tensor inputs."""
        step_tensor = tf.constant(5000)
        ratio = exponential_tf_schedule(step=step_tensor, initial_ratio=1.0, min_ratio=0.05, decay_k=1e-5)
        
        assert isinstance(ratio, tf.Tensor)
        assert ratio.dtype == tf.float32
    
    def test_batch_computation(self):
        """Should work with batch of steps."""
        steps = tf.constant([0, 10000, 20000, 50000, 100000])
        ratios = tf.map_fn(
            lambda s: exponential_tf_schedule(s, initial_ratio=1.0, min_ratio=0.05, decay_k=1e-5),
            steps,
            dtype=tf.float32
        )
        
        # Should be monotonically decreasing
        for i in range(len(steps) - 1):
            assert ratios[i] >= ratios[i + 1]
        
        # First should be ~1.0
        assert tf.abs(ratios[0] - 1.0) < 1e-6
        
        # All should be >= min_ratio
        for ratio in ratios:
            assert ratio >= 0.05


class TestConfigParameters:
    """Test that config has required teacher forcing parameters."""
    
    def test_config_has_tf_parameters(self):
        """Config should define all teacher forcing schedule parameters."""
        assert hasattr(config, 'INITIAL_TF_RATIO')
        assert hasattr(config, 'MIN_TF_RATIO')
        assert hasattr(config, 'TF_DECAY_K')
        assert hasattr(config, 'TF_WARMUP_STEPS')
    
    def test_config_default_values(self):
        """Config parameters should have correct default values."""
        assert config.INITIAL_TF_RATIO == 1.0
        assert config.MIN_TF_RATIO == 0.05
        assert config.TF_DECAY_K == 5e-7
        assert config.TF_WARMUP_STEPS == 500000
    
    def test_config_values_are_valid(self):
        """Config values should be in valid ranges."""
        assert 0.0 <= config.INITIAL_TF_RATIO <= 1.0
        assert 0.0 <= config.MIN_TF_RATIO <= 1.0
        assert config.MIN_TF_RATIO <= config.INITIAL_TF_RATIO
        assert config.TF_DECAY_K > 0
        assert config.TF_WARMUP_STEPS >= 0



class _MockTokenizer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(256)
    def call(self, x, training=False):
        T = tf.shape(x)[1]
        T_tok = tf.maximum(T // 4, 1)
        indices = tf.cast(tf.linspace(0.0, tf.cast(T - 1, tf.float32), T_tok), tf.int32)
        return self.proj(tf.gather(x, indices, axis=1))


class _MockDetokenizer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(80)
    def call(self, x, training=False):
        return self.proj(tf.repeat(x, 4, axis=1))


class SimpleMelGenerator(tf.keras.Model):
    """Minimal mock generator for testing."""
    def __init__(self):
        super().__init__()
        self.tokenizer = _MockTokenizer()
        self.detokenizer = _MockDetokenizer()
        self.dense = tf.keras.layers.Dense(256)
        self.out = tf.keras.layers.Dense(80)
        self.tok_dense = tf.keras.layers.Dense(256)
    
    def call(self, x, z=None, training=False):
        return self.out(self.dense(x))
    
    def forward_from_tokens(self, tokens, z=None, training=False):
        return self.detokenizer(self.tok_dense(tokens), training=training)


class TestTFRatioTracking:
    """Test teacher forcing ratio tracking in trainer."""
    
    def test_initialization(self):
        """Trainer should initialize with correct TF ratio."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(base_model)
        
        assert trainer.tf_ratio.numpy() == config.INITIAL_TF_RATIO
        assert trainer.training_step.numpy() == 0
        assert trainer.initial_tf_ratio == config.INITIAL_TF_RATIO
        assert trainer.min_tf_ratio == config.MIN_TF_RATIO
        assert trainer.decay_k == config.TF_DECAY_K
        assert trainer.warmup_steps == config.TF_WARMUP_STEPS
    
    def test_custom_initialization(self):
        """Trainer should accept custom TF parameters."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(
            base_model,
            initial_tf_ratio=0.8,
            min_tf_ratio=0.1,
            decay_k=1e-4,
            warmup_steps=1000
        )
        
        assert tf.abs(trainer.tf_ratio - 0.8) < 1e-6
        assert trainer.initial_tf_ratio == 0.8
        assert trainer.min_tf_ratio == 0.1
        assert trainer.decay_k == 1e-4
        assert trainer.warmup_steps == 1000
    
    def test_update_tf_ratio(self):
        """update_tf_ratio should update ratio and step."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(base_model, initial_tf_ratio=1.0, decay_k=1e-3)
        
        initial_ratio = trainer.tf_ratio.numpy()
        trainer.update_tf_ratio()
        
        assert trainer.training_step.numpy() == 1
        # Ratio should decrease (or stay at min)
        assert trainer.tf_ratio.numpy() <= initial_ratio
    
    def test_warmup_period(self):
        """Ratio should stay constant during warmup."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(
            base_model,
            initial_tf_ratio=1.0,
            warmup_steps=10,
            decay_k=1e-3
        )
        
        # During warmup (steps 0-9), ratio should stay at initial
        for i in range(10):
            trainer.update_tf_ratio()
            # After update, training_step is i+1, but ratio was computed using step i
            assert tf.abs(trainer.tf_ratio - 1.0) < 1e-6, f"Failed at step {i}"
        
        # At step 10 (after 10 updates), we're at the boundary
        # The ratio was computed using step 9, so still 1.0
        assert trainer.training_step.numpy() == 10
        assert tf.abs(trainer.tf_ratio - 1.0) < 1e-6
        
        # After one more update, we compute using step 10 (step_after_warmup=0), still 1.0
        trainer.update_tf_ratio()
        assert trainer.training_step.numpy() == 11
        assert tf.abs(trainer.tf_ratio - 1.0) < 1e-6
        
        # After another update, we compute using step 11 (step_after_warmup=1), decay starts
        trainer.update_tf_ratio()
        assert trainer.training_step.numpy() == 12
        assert trainer.tf_ratio < 1.0
    
    def test_ratio_metric_tracking(self):
        """Trainer should track ratio as a metric."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(base_model)
        
        assert hasattr(trainer, 'tf_ratio_metric')
        assert isinstance(trainer.tf_ratio_metric, tf.keras.metrics.Mean)
        assert trainer.tf_ratio_metric.name == "tf_ratio"
    
    def test_metrics_property(self):
        """Trainer should expose metrics for auto-reset."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(base_model)
        
        metrics = trainer.metrics
        assert len(metrics) == 3
        assert metrics[0] is trainer.tf_ratio_metric
    
    def test_train_step_updates_ratio(self):
        """train_step should update and log TF ratio."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(base_model)
        trainer.compile(optimizer='adam')
        
        # Create dummy batch
        x = tf.random.normal([2, 12, 80])
        y = tf.random.normal([2, 12, 80])
        
        initial_step = trainer.training_step.numpy()
        result = trainer.train_step((x, y))
        
        # Should increment step
        assert trainer.training_step.numpy() == initial_step + 1
        
        # Should return tf_ratio in result
        assert 'tf_ratio' in result
        assert 0.0 <= result['tf_ratio'].numpy() <= 1.0
    
    def test_serialization(self):
        """Trainer should serialize/deserialize with TF params."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(
            base_model,
            initial_tf_ratio=0.9,
            min_tf_ratio=0.1,
            decay_k=1e-4,
            warmup_steps=500
        )
        
        config_dict = trainer.get_config()
        assert config_dict['initial_tf_ratio'] == 0.9
        assert config_dict['min_tf_ratio'] == 0.1
        assert config_dict['decay_k'] == 1e-4
        assert config_dict['warmup_steps'] == 500
        
        # Note: Full deserialization requires registered models
        # Just verify config structure is correct


class TestAutoregressiveTrainingStep:
    """Test autoregressive training step with scheduled sampling."""
    
    def test_pure_teacher_forcing(self):
        """Test with tf_ratio=1.0 (pure teacher forcing)."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(base_model, initial_tf_ratio=1.0)
        trainer.compile(optimizer='adam')
        
        x = tf.random.normal([2, 8, 80])
        y = tf.random.normal([2, 8, 80])
        
        result = trainer.train_step((x, y))
        
        assert 'loss' in result
        assert 'tf_ratio' in result
        assert tf.abs(result['tf_ratio'] - 1.0) < 1e-6
        assert not tf.math.is_nan(result['loss'])
    
    def test_pure_autoregressive(self):
        """Test with tf_ratio=0.0 (pure autoregressive)."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(base_model, initial_tf_ratio=0.0, min_tf_ratio=0.0)
        trainer.compile(optimizer='adam')
        
        x = tf.random.normal([2, 8, 80])
        y = tf.random.normal([2, 8, 80])
        
        result = trainer.train_step((x, y))
        
        assert 'loss' in result
        assert 'tf_ratio' in result
        assert tf.abs(result['tf_ratio'] - 0.0) < 1e-6
        assert not tf.math.is_nan(result['loss'])
    
    def test_mixed_sampling(self):
        """Test with tf_ratio=0.5 (mixed sampling)."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(base_model, initial_tf_ratio=0.5)
        trainer.compile(optimizer='adam')
        
        x = tf.random.normal([2, 8, 80])
        y = tf.random.normal([2, 8, 80])
        
        result = trainer.train_step((x, y))
        
        assert 'loss' in result
        assert 'tf_ratio' in result
        assert tf.abs(result['tf_ratio'] - 0.5) < 1e-6
        assert not tf.math.is_nan(result['loss'])
    
    def test_gradients_flow(self):
        """Test that gradients flow correctly."""
        from src.music_generation.train import MelGeneratorTraining
        
        base_model = SimpleMelGenerator()
        trainer = MelGeneratorTraining(base_model, initial_tf_ratio=0.0, min_tf_ratio=0.0)
        trainer.compile(optimizer='adam')
        
        x = tf.random.normal([2, 8, 80])
        y = tf.random.normal([2, 8, 80])
        
        # Build the model first
        _ = trainer.train_step((x, y))
        
        # Get initial weights
        initial_weights = [w.numpy().copy() for w in trainer.base_model.trainable_variables]
        
        # Run training step again
        result = trainer.train_step((x, y))
        
        # Check that weights changed
        final_weights = [w.numpy() for w in trainer.base_model.trainable_variables]
        
        import numpy as np
        weights_changed = any(
            not np.allclose(w1, w2, atol=1e-6)
            for w1, w2 in zip(initial_weights, final_weights)
        )
        
        assert weights_changed, "Weights should change after training step"
        assert result['grad_norm'] > 0, "Gradient norm should be positive"
