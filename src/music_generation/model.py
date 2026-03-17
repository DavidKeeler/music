"""Mel generator model."""

import tensorflow as tf
from src.music_generation.config import (
    D_MODEL, NUM_HEADS, SEQ_LEN, N_MELS, WINDOW_SIZES,
    REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN,
    CONV_DILATION_RATES, FRAME_STEP, SAMPLE_RATE,
    LATENT_DIM
)
from src.music_generation.layers import CausalConvBlock, TransformerBlock


class LatentEncoder(tf.keras.Model):
    """VAE-style encoder: mel sequence -> global latent vector z."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(128, 5, strides=2, padding="same", activation="gelu")
        self.conv2 = tf.keras.layers.Conv1D(256, 5, strides=2, padding="same", activation="gelu")
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(256, activation="gelu")
        self.z_mean_head = tf.keras.layers.Dense(latent_dim)
        self.z_logvar_head = tf.keras.layers.Dense(latent_dim)

    def call(self, mel, training=False):
        """Encode mel sequence to z, z_mean, z_logvar.

        Args:
            mel: [B, T, GROUPED_MEL_DIM]

        Returns:
            (z, z_mean, z_logvar) each [B, latent_dim]
        """
        x = self.conv1(mel)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dense(x)
        z_mean = self.z_mean_head(x)
        z_logvar = self.z_logvar_head(x)
        eps = tf.random.normal(tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_logvar) * eps
        return z, z_mean, z_logvar



class MelGenerator(tf.keras.Model):
    """Autoregressive mel spectrogram generator with reduction factor.
    
    Architecture (strictly causal):
    - Input projection: Dense(GROUPED_MEL_DIM -> D_MODEL)
    - N causal conv layers built from CONV_DILATION_RATES (kernel_size=3, residual)
    - 3 transformer blocks with increasing window sizes
    - Output projection: Dense(D_MODEL -> GROUPED_MEL_DIM)
    
    With R=4 reduction factor:
    - Input: [B, T/R, R*100] grouped frames
    - Output: [B, T/R, R*100] grouped frame predictions
    - Each position predicts R consecutive frames
    
    All operations preserve causality: output at time t depends only on inputs ≤ t.
    Supports both parallel forward pass (training) and autoregressive generation (inference).
    """
    
    def __init__(self):
        super().__init__()
        
        import logging
        
        # Calculate and log receptive field
        receptive_field_frames = 1 + 2 * sum(CONV_DILATION_RATES)
        frame_duration_ms = (FRAME_STEP / SAMPLE_RATE) * 1000
        receptive_field_ms = receptive_field_frames * frame_duration_ms
        
        logging.info(
            f"MelGenerator initialized with dilation rates {list(CONV_DILATION_RATES)}"
        )
        logging.info(
            f"Total receptive field: {receptive_field_frames} frames "
            f"(~{receptive_field_ms:.1f} ms)"
        )
        
        # Input projection: GROUPED_MEL_DIM -> D_MODEL
        self.input_proj = tf.keras.layers.Dense(D_MODEL)
        
        # Dynamic causal conv stack from config
        self.conv_layers = [
            CausalConvBlock(D_MODEL, kernel_size=3, dilation_rate=d, residual=True)
            for d in CONV_DILATION_RATES
        ]
        
        # Transformer stack with explicit layers and increasing window sizes
        self.transformer1 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[0])
        self.transformer2 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[1])
        self.transformer3 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[2])
        
        # Output projection: D_MODEL -> GROUPED_MEL_DIM
        self.output_proj = tf.keras.layers.Dense(GROUPED_MEL_DIM)
        
        # Latent conditioning projection
        self.z_proj = tf.keras.layers.Dense(D_MODEL)
    
    def call(self, mel, z=None, training=False):
        """Forward pass (parallel processing for training).
        
        Args:
            mel: Input mel spectrogram [batch, seq_len/R, GROUPED_MEL_DIM]
            z: Optional latent vector [batch, latent_dim]. If None, sampled from N(0,1).
            training: Whether in training mode
        
        Returns:
            Output mel spectrogram [batch, seq_len/R, GROUPED_MEL_DIM]
        """
        batch_size = tf.shape(mel)[0]
        seq_len = tf.shape(mel)[1]
        
        # Sample z from prior if not provided
        if z is None:
            z = tf.random.normal([batch_size, LATENT_DIM])
        
        # Input projection: [B, T/R, GROUPED_MEL_DIM] -> [B, T/R, D]
        x = self.input_proj(mel)
        
        # Inject latent conditioning: [B, latent_dim] -> [B, 1, D] broadcast to [B, T, D]
        z_embed = self.z_proj(z)  # [B, D]
        z_embed = tf.expand_dims(z_embed, axis=1)  # [B, 1, D]
        z_embed = tf.broadcast_to(z_embed, [batch_size, seq_len, D_MODEL])
        x = x + z_embed
        
        # Causal conv stack: [B, T/R, D] -> [B, T/R, D]
        for conv in self.conv_layers:
            x = conv(x, training=training)
        
        # Transformer stack: [B, T/R, D] -> [B, T/R, D]
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)
        
        # Output projection: [B, T/R, D] -> [B, T/R, GROUPED_MEL_DIM]
        x = self.output_proj(x)
        
        return x
    
    def generate(
        self,
        seed_mel,
        num_frames,
        temperature=1.0,
        top_p=0.9,
        z=None
    ):
        """Autoregressive generation with grouped frame prediction.
        
        Generates mel spectrogram autoregressively by predicting R frames at a time.
        Maintains a sliding context window of EFFECTIVE_SEQ_LEN grouped frames.
        
        Args:
            seed_mel: Seed mel spectrogram [seed_len, N_MELS]
                     Truncated to R-divisible length and reshaped to grouped frames
            num_frames: Number of individual frames to generate
            temperature: Temperature scaling (higher = more random)
                        Adds Gaussian noise scaled by temperature * 0.1
            top_p: Nucleus sampling threshold (not used for continuous mel values)
            z: Optional latent vector [1, latent_dim]. If None, sampled from N(0,1).
        
        Returns:
            Generated mel spectrogram [num_frames, N_MELS]
        
        Note:
            - Input/output use individual frames [T, N_MELS] for API compatibility
            - Internally operates on grouped frames [T/R, GROUPED_MEL_DIM]
            - Seed is truncated to R-divisible length before reshaping
        """
        # Sample z once for the entire generation if not provided
        if z is None:
            z = tf.random.normal([1, LATENT_DIM])
        # Truncate seed to R-divisible length
        seed_len = seed_mel.shape[0]
        truncated_len = (seed_len // REDUCTION_FACTOR) * REDUCTION_FACTOR
        if truncated_len > 0:
            seed_mel = seed_mel[:truncated_len]
        else:
            # If seed too short, pad to R frames
            pad_len = REDUCTION_FACTOR - seed_len
            seed_mel = tf.concat([
                tf.zeros([pad_len, N_MELS], dtype=seed_mel.dtype),
                seed_mel
            ], axis=0)
            truncated_len = REDUCTION_FACTOR
        
        # Reshape seed to grouped frames: [T, 80] -> [T/R, R*80]
        seed_grouped = tf.reshape(seed_mel, [-1, GROUPED_MEL_DIM])
        
        # Initialize context with seed (keep last EFFECTIVE_SEQ_LEN grouped frames)
        if seed_grouped.shape[0] > EFFECTIVE_SEQ_LEN:
            context = seed_grouped[-EFFECTIVE_SEQ_LEN:]
        else:
            context = seed_grouped
        
        generated = []
        
        # Calculate number of generation steps (each step produces R frames)
        num_steps = (num_frames + REDUCTION_FACTOR - 1) // REDUCTION_FACTOR
        
        for _ in range(num_steps):
            # Keep last EFFECTIVE_SEQ_LEN grouped frames as context
            if context.shape[0] > EFFECTIVE_SEQ_LEN:
                context = context[-EFFECTIVE_SEQ_LEN:]
            
            # Forward pass: [1, T/R, GROUPED_MEL_DIM] -> [1, T/R, GROUPED_MEL_DIM]
            context_batch = tf.expand_dims(context, 0)
            output = self(context_batch, z=z, training=False)
            next_grouped = output[0, -1, :]  # [GROUPED_MEL_DIM]
            
            # Temperature scaling: add noise proportional to temperature
            if temperature > 0.0:
                noise = tf.random.normal(tf.shape(next_grouped)) * 0.02 * temperature
                next_grouped = next_grouped + noise
            
            # Append to context and generated
            context = tf.concat([context, tf.expand_dims(next_grouped, 0)], axis=0)
            generated.append(next_grouped)
        
        # Stack and reshape to individual frames: [num_steps, R*80] -> [num_steps*R, 80]
        generated_grouped = tf.stack(generated, axis=0)  # [num_steps, GROUPED_MEL_DIM]
        generated_frames = tf.reshape(generated_grouped, [-1, N_MELS])  # [num_steps*R, N_MELS]
        
        # Truncate to exact num_frames requested
        return generated_frames[:num_frames]
