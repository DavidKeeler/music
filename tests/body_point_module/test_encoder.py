"""Unit tests for pose encoder."""

import tensorflow as tf
import pytest
from src.body_point_module.encoder import build_mlp_encoder, build_pose_encoder


class TestMLPEncoder:
    """Tests for MLP encoder."""
    
    def test_mlp_builds(self):
        """Test MLP encoder builds successfully."""
        mlp = build_mlp_encoder(input_dim=85, hidden_dim=256, output_dim=128)
        assert mlp is not None
        assert mlp.name == 'mlp_encoder'
    
    def test_mlp_output_shape(self):
        """Test MLP encoder output shape."""
        mlp = build_mlp_encoder(input_dim=85, hidden_dim=256, output_dim=128)
        features = tf.random.normal([1, 85])
        embedding = mlp(features)
        assert embedding.shape == (1, 128)
    
    def test_mlp_batch_processing(self):
        """Test MLP encoder handles batches."""
        mlp = build_mlp_encoder()
        features = tf.random.normal([4, 85])
        embedding = mlp(features)
        assert embedding.shape == (4, 128)
    
    def test_mlp_trainable(self):
        """Test MLP encoder has trainable parameters."""
        mlp = build_mlp_encoder()
        mlp.build(input_shape=(None, 85))
        trainable_params = sum([tf.size(w).numpy() for w in mlp.trainable_weights])
        assert trainable_params > 80000  # Approximately 88K
    
    def test_mlp_parameter_count(self):
        """Test MLP encoder parameter count is reasonable."""
        mlp = build_mlp_encoder(input_dim=85, hidden_dim=256, output_dim=128)
        mlp.build(input_shape=(None, 85))
        total_params = mlp.count_params()
        # Expected: (85*256 + 256) + (256*256 + 256) + (256*128 + 128) ≈ 120K
        assert 80000 < total_params < 150000


class TestPoseEncoder:
    """Tests for complete pose encoder."""
    
    def test_encoder_builds(self):
        """Test pose encoder builds successfully."""
        encoder = build_pose_encoder(buffer_size=8)
        assert encoder is not None
        assert encoder.name == 'pose_encoder'
    
    def test_encoder_output_shape(self):
        """Test pose encoder output shape."""
        encoder = build_pose_encoder(buffer_size=8)
        buffer_tensor = tf.random.normal([1, 8, 85])
        embedding = encoder(buffer_tensor)
        assert embedding.shape == (1, 128)
    
    def test_encoder_batch_processing(self):
        """Test pose encoder handles batches."""
        encoder = build_pose_encoder(buffer_size=8)
        buffer_tensor = tf.random.normal([4, 8, 85])
        embedding = encoder(buffer_tensor)
        assert embedding.shape == (4, 128)
    
    def test_encoder_trainable(self):
        """Test pose encoder has trainable parameters."""
        encoder = build_pose_encoder()
        trainable_params = sum([tf.size(w).numpy() for w in encoder.trainable_weights])
        assert trainable_params > 0
    
    def test_encoder_temporal_conv(self):
        """Test temporal convolution layer exists."""
        encoder = build_pose_encoder()
        # Check that Conv1D layer exists
        conv_layers = [layer for layer in encoder.layers if isinstance(layer, tf.keras.layers.Conv1D)]
        assert len(conv_layers) == 1
        assert conv_layers[0].kernel_size == (3,)
        assert conv_layers[0].padding == 'causal'
    
    def test_encoder_last_timestep(self):
        """Test encoder extracts last timestep."""
        encoder = build_pose_encoder(buffer_size=8)
        buffer_tensor = tf.random.normal([1, 8, 85])
        embedding = encoder(buffer_tensor)
        # Output should be [batch, embedding_dim], not [batch, time, embedding_dim]
        assert len(embedding.shape) == 2
    
    def test_encoder_custom_dimensions(self):
        """Test encoder with custom dimensions."""
        encoder = build_pose_encoder(
            input_dim=100,
            hidden_dim=128,
            output_dim=64,
            buffer_size=4
        )
        buffer_tensor = tf.random.normal([1, 4, 100])
        embedding = encoder(buffer_tensor)
        assert embedding.shape == (1, 64)
    
    def test_encoder_forward_pass(self):
        """Test encoder forward pass executes without errors."""
        encoder = build_pose_encoder()
        buffer_tensor = tf.random.normal([2, 8, 85])
        embedding = encoder(buffer_tensor, training=False)
        assert embedding.shape == (2, 128)
        assert not tf.reduce_any(tf.math.is_nan(embedding))
        assert not tf.reduce_any(tf.math.is_inf(embedding))


class TestEncoderIntegration:
    """Integration tests for encoder components."""
    
    def test_mlp_in_time_distributed(self):
        """Test MLP works with TimeDistributed wrapper."""
        mlp = build_mlp_encoder()
        buffer_tensor = tf.random.normal([1, 8, 85])
        
        # Apply MLP to each timestep
        time_dist = tf.keras.layers.TimeDistributed(mlp)
        output = time_dist(buffer_tensor)
        
        assert output.shape == (1, 8, 128)
    
    def test_encoder_gradient_flow(self):
        """Test gradients flow through encoder."""
        encoder = build_pose_encoder()
        buffer_tensor = tf.random.normal([1, 8, 85])
        
        with tf.GradientTape() as tape:
            embedding = encoder(buffer_tensor, training=True)
            loss = tf.reduce_mean(embedding ** 2)
        
        gradients = tape.gradient(loss, encoder.trainable_weights)
        
        # Check that gradients exist and are not None
        assert all(g is not None for g in gradients)
        # Check that gradients are not all zeros
        assert any(tf.reduce_sum(tf.abs(g)) > 0 for g in gradients)
