"""Mel generator model."""

import tensorflow as tf
from src.music_generation.config import D_MODEL, NUM_LAYERS, NUM_HEADS, SEQ_LEN, N_MELS
from src.music_generation.layers import CausalConvBlock, TransformerBlock


class MelGenerator(tf.keras.Model):
    """Autoregressive mel spectrogram generator.
    
    Architecture (strictly causal):
    - Input projection: Dense(N_MELS -> D_MODEL)
    - 2 causal conv layers (kernel_size=3, with residual connections)
    - NUM_LAYERS transformer blocks (local window attention + FFN)
    - 1 causal conv head (kernel_size=3, with residual connection)
    - Output projection: Dense(D_MODEL -> N_MELS)
    
    All operations preserve causality: output at time t depends only on inputs ≤ t.
    Supports both parallel forward pass (training) and autoregressive generation (inference).
    """
    
    def __init__(self):
        super().__init__()
        
        # Input projection
        self.input_proj = tf.keras.layers.Dense(D_MODEL)
        
        # Causal conv layers
        self.conv1 = CausalConvBlock(D_MODEL, kernel_size=3, residual=True)
        self.conv2 = CausalConvBlock(D_MODEL, kernel_size=3, residual=True)
        
        # Transformer stack
        self.transformer_blocks = [
            TransformerBlock(D_MODEL, NUM_HEADS) for _ in range(NUM_LAYERS)
        ]
        
        # Conv head
        self.conv_head = CausalConvBlock(D_MODEL, kernel_size=3, residual=True)
        
        # Output projection
        self.output_proj = tf.keras.layers.Dense(N_MELS)
    
    def call(self, mel, training=False):
        """Forward pass (parallel processing for training).
        
        Args:
            mel: Input mel spectrogram [batch, seq_len, N_MELS]
            training: Whether in training mode
        
        Returns:
            Output mel spectrogram [batch, seq_len, N_MELS]
        """
        # Input projection: [B, T, N_MELS] -> [B, T, D]
        x = self.input_proj(mel)
        
        # Causal convs: [B, T, D] -> [B, T, D]
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        
        # Transformer stack: [B, T, D] -> [B, T, D]
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Conv head: [B, T, D] -> [B, T, D]
        x = self.conv_head(x, training=training)
        
        # Output projection: [B, T, D] -> [B, T, N_MELS]
        x = self.output_proj(x)
        
        return x
    
    def generate(
        self,
        seed_mel,
        num_frames,
        temperature=1.0,
        top_p=0.9
    ):
        """Autoregressive generation frame-by-frame.
        
        Generates mel spectrogram autoregressively by predicting one frame at a time.
        Maintains a sliding context window of SEQ_LEN frames for efficiency.
        
        Args:
            seed_mel: Seed mel spectrogram [seed_len, N_MELS]
                     If longer than SEQ_LEN, only last SEQ_LEN frames are used
            num_frames: Number of frames to generate
            temperature: Temperature scaling (higher = more random)
                        Adds Gaussian noise scaled by temperature * 0.1
            top_p: Nucleus sampling threshold (not used for continuous mel values)
        
        Returns:
            Generated mel spectrogram [num_frames, N_MELS]
        
        Note:
            For continuous mel values, temperature controls noise level rather than
            probability distribution sharpness (as in discrete token generation).
        """
        # Initialize context with seed (keep last SEQ_LEN frames)
        if seed_mel.shape[0] > SEQ_LEN:
            context = seed_mel[-SEQ_LEN:]
        else:
            context = seed_mel
        
        generated = []
        
        for _ in range(num_frames):
            # Keep last SEQ_LEN frames as context
            if context.shape[0] > SEQ_LEN:
                context = context[-SEQ_LEN:]
            
            # Forward pass: [1, T, N_MELS] -> [1, T, N_MELS]
            context_batch = tf.expand_dims(context, 0)  # [1, T, N_MELS]
            output = self(context_batch, training=False)  # [1, T, N_MELS]
            next_frame = output[0, -1, :]  # [N_MELS]
            
            # Temperature scaling: add noise proportional to temperature
            if temperature > 0.0:
                noise = tf.random.normal(tf.shape(next_frame)) * temperature * 0.1
                next_frame = next_frame + noise
            
            # Append to context and generated
            context = tf.concat([context, tf.expand_dims(next_frame, 0)], axis=0)
            generated.append(next_frame)
        
        return tf.stack(generated, axis=0)  # [num_frames, N_MELS]
