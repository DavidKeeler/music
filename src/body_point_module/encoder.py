"""Pose encoder model for embedding extraction."""

import tensorflow as tf


def build_mlp_encoder(input_dim=85, hidden_dim=256, output_dim=128):
    """
    Build MLP encoder for single frame.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        
    Returns:
        Keras Sequential model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_dim),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation('gelu'),
        tf.keras.layers.Dense(hidden_dim),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation('gelu'),
        tf.keras.layers.Dense(output_dim)
    ], name='mlp_encoder')


def build_temporal_pose_encoder(input_dim=85, hidden_dim=256, output_dim=128):
    """Build temporal pose encoder: [B, T, input_dim] -> [B, T, output_dim].

    TimeDistributed MLP per frame + causal Conv1D for temporal context.
    Supports variable-length sequences.
    """
    mlp = build_mlp_encoder(input_dim, hidden_dim, output_dim)
    inputs = tf.keras.Input(shape=(None, input_dim))
    x = tf.keras.layers.TimeDistributed(mlp)(inputs)
    x = tf.keras.layers.Conv1D(output_dim, kernel_size=3, padding='causal')(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name='temporal_pose_encoder')


def build_pose_encoder(input_dim=85, hidden_dim=256, output_dim=128, buffer_size=8):
    """
    Build complete pose encoder with temporal modeling.
    
    Args:
        input_dim: Input feature dimension per frame
        hidden_dim: MLP hidden dimension
        output_dim: Output embedding dimension
        buffer_size: Number of frames in buffer
        
    Returns:
        Keras Model
    """
    # Build per-frame MLP
    mlp = build_mlp_encoder(input_dim, hidden_dim, output_dim)
    
    # Full model
    inputs = tf.keras.Input(shape=(buffer_size, input_dim))
    
    # Apply MLP to each frame
    x = tf.keras.layers.TimeDistributed(mlp)(inputs)  # [buffer_size, output_dim]
    
    # Temporal convolution
    x = tf.keras.layers.Conv1D(
        filters=output_dim,
        kernel_size=3,
        padding='causal',
        activation='relu'
    )(x)  # [buffer_size, output_dim]
    
    # Extract last timestep
    outputs = x[:, -1, :]  # [output_dim]
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='pose_encoder')
    return model
