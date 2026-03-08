"""Tests for teacher forcing schedule function."""
import tensorflow as tf
import pytest
from src.music_generation.train import exponential_tf_schedule


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
