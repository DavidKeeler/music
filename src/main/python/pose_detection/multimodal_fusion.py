import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention

def linear_interpolate(x, target_length):
    """Linear interpolation for 1D temporal sequences."""
    x_len = tf.shape(x)[1]
    positions = tf.linspace(0.0, tf.cast(x_len - 1, tf.float32), target_length)
    indices = tf.cast(tf.floor(positions), tf.int32)
    next_indices = tf.minimum(indices + 1, x_len - 1)
    alpha = positions - tf.cast(indices, tf.float32)
    
    return ((1 - alpha[:, None]) * tf.gather(x, indices, axis=1, batch_dims=1) + 
            alpha[:, None] * tf.gather(x, next_indices, axis=1, batch_dims=1))

class MultimodalFusion(tf.keras.layers.Layer):
    """
    Fuses pose embeddings with audio tokens through cross-attention or addition.
    """
    
    def __init__(self, audio_dim=256, pose_dim=256, fusion_type='cross_attention', **kwargs):
        super().__init__(**kwargs)
        self.audio_dim = audio_dim
        self.pose_dim = pose_dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'add':
            # Project pose to audio dimension for addition
            self.pose_projection = Dense(audio_dim)
        elif fusion_type == 'concat':
            # Concatenate and project back to audio dimension
            self.fusion_projection = Dense(audio_dim)
        elif fusion_type == 'cross_attention':
            # Multi-head cross-attention
            self.cross_attention = MultiHeadAttention(
                num_heads=4, 
                key_dim=audio_dim // 4,
                dropout=0.1
            )
            
        self.layer_norm = LayerNormalization()
    
    def call(self, audio_tokens, pose_embeddings, training=False):
        """
        Args:
            audio_tokens: [batch, audio_time, audio_dim]
            pose_embeddings: [batch, pose_time, pose_dim]
        Returns:
            fused_tokens: [batch, audio_time, audio_dim]
        """
        # Align temporal dimensions using linear interpolation
        audio_time = tf.shape(audio_tokens)[1]
        pose_time = tf.shape(pose_embeddings)[1]
        
        if pose_time != audio_time:
            pose_embeddings = linear_interpolate(pose_embeddings, audio_time)
        
        if self.fusion_type == 'add':
            # Project pose to audio dimension and add
            pose_projected = self.pose_projection(pose_embeddings)
            fused = audio_tokens + pose_projected
            
        elif self.fusion_type == 'concat':
            # Concatenate along feature dimension
            concatenated = tf.concat([audio_tokens, pose_embeddings], axis=-1)
            fused = self.fusion_projection(concatenated)
            
        elif self.fusion_type == 'cross_attention':
            # Cross-attention: audio as query, pose as key/value
            attended = self.cross_attention(
                query=audio_tokens,
                value=pose_embeddings,
                key=pose_embeddings,
                training=training
            )
            fused = audio_tokens + attended
        
        # Layer normalization
        fused = self.layer_norm(fused)
        
        return fused

def sync_pose_to_audio(pose_embeddings, audio_length, pose_fps=30, audio_fps=86.13):
    """
    Synchronize pose embeddings to audio token timeline using linear interpolation.
    
    Args:
        pose_embeddings: [batch, pose_frames, pose_dim]
        audio_length: Number of audio tokens
        pose_fps: Pose detection frame rate
        audio_fps: Audio token rate (22050 / 256 ≈ 86.13 for STFT)
    
    Returns:
        synced_embeddings: [batch, audio_length, pose_dim]
    """
    # Calculate time alignment
    pose_frames = tf.shape(pose_embeddings)[1]
    pose_duration = tf.cast(pose_frames, tf.float32) / pose_fps
    audio_duration = tf.cast(audio_length, tf.float32) / audio_fps
    
    # Use the shorter duration to avoid extrapolation
    sync_duration = tf.minimum(pose_duration, audio_duration)
    sync_frames = tf.cast(sync_duration * audio_fps, tf.int32)
    
    # Linear interpolation to match audio timeline
    synced = linear_interpolate(pose_embeddings, sync_frames)
    
    # Pad or truncate to exact audio length
    if sync_frames < audio_length:
        padding = audio_length - sync_frames
        synced = tf.pad(synced, [[0, 0], [0, padding], [0, 0]])
    else:
        synced = synced[:, :audio_length, :]
    
    return synced
