"""Mel generator model."""

import tensorflow as tf
from src.music_generation.config import (
    D_MODEL, NUM_HEADS, SEQ_LEN, N_MELS, WINDOW_SIZES,
    REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN,
    CONV_DILATION_RATES, FRAME_STEP, SAMPLE_RATE
)
from src.music_generation.layers import CausalConvBlock, TransformerBlock



class MelGenerator(tf.keras.Model):
    """Autoregressive mel spectrogram generator with reduction factor.
    
    Architecture (strictly causal):
    - Input projection: Dense(GROUPED_MEL_DIM -> D_MODEL)
    - 2 causal conv layers (kernel_size=3, with residual connections)
    - 3 transformer blocks with increasing window sizes (128, 256, 512)
    - 1 causal conv head (kernel_size=3, with residual connection)
    - Output projection: Dense(D_MODEL -> GROUPED_MEL_DIM)
    
    With R=4 reduction factor:
    - Input: [B, T/R, R*80] grouped frames
    - Output: [B, T/R, R*80] grouped frame predictions
    - Each position predicts R consecutive frames
    
    All operations preserve causality: output at time t depends only on inputs ≤ t.
    Supports both parallel forward pass (training) and autoregressive generation (inference).
    """
    
    def __init__(self):
        super().__init__()
        
        # Get dilation rates (use first 3, pad with 1 if insufficient)
        dilations = list(CONV_DILATION_RATES) + [1, 1, 1]
        d1, d2, d3 = dilations[0], dilations[1], dilations[2]
        
        # Warn if list length mismatch
        if len(CONV_DILATION_RATES) < 3:
            import logging
            logging.warning(
                f"CONV_DILATION_RATES has {len(CONV_DILATION_RATES)} values, "
                f"expected 3. Padding with 1s: {[d1, d2, d3]}"
            )
        elif len(CONV_DILATION_RATES) > 3:
            import logging
            logging.warning(
                f"CONV_DILATION_RATES has {len(CONV_DILATION_RATES)} values, "
                f"expected 3. Using first 3: {[d1, d2, d3]}"
            )
        
        # Calculate and log receptive field
        receptive_field_frames = 1 + 2 * (d1 + d2 + d3)
        frame_duration_ms = (FRAME_STEP / SAMPLE_RATE) * 1000
        receptive_field_ms = receptive_field_frames * frame_duration_ms
        
        import logging
        logging.info(
            f"MelGenerator initialized with dilation rates [{d1}, {d2}, {d3}]"
        )
        logging.info(
            f"Total receptive field: {receptive_field_frames} frames "
            f"(~{receptive_field_ms:.1f} ms)"
        )
        
        # Input projection: GROUPED_MEL_DIM -> D_MODEL
        self.input_proj = tf.keras.layers.Dense(D_MODEL)
        
        # Causal conv layers with progressive dilation
        self.conv1 = CausalConvBlock(D_MODEL, kernel_size=3, dilation_rate=d1, residual=True)
        self.conv2 = CausalConvBlock(D_MODEL, kernel_size=3, dilation_rate=d2, residual=True)
        
        # Transformer stack with explicit layers and increasing window sizes
        self.transformer1 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[0])
        self.transformer2 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[1])
        self.transformer3 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[2])
        
        # Conv head with largest dilation
        self.conv_head = CausalConvBlock(D_MODEL, kernel_size=3, dilation_rate=d3, residual=True)
        
        # Output projection: D_MODEL -> GROUPED_MEL_DIM
        self.output_proj = tf.keras.layers.Dense(GROUPED_MEL_DIM)
    
    def call(self, mel, training=False):
        """Forward pass (parallel processing for training).
        
        Args:
            mel: Input mel spectrogram [batch, seq_len/R, GROUPED_MEL_DIM]
            training: Whether in training mode
        
        Returns:
            Output mel spectrogram [batch, seq_len/R, GROUPED_MEL_DIM]
        """
        # Input projection: [B, T/R, GROUPED_MEL_DIM] -> [B, T/R, D]
        x = self.input_proj(mel)
        
        # Causal convs: [B, T/R, D] -> [B, T/R, D]
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        
        # Transformer stack: [B, T/R, D] -> [B, T/R, D]
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)
        
        # Conv head: [B, T/R, D] -> [B, T/R, D]
        x = self.conv_head(x, training=training)
        
        # Output projection: [B, T/R, D] -> [B, T/R, GROUPED_MEL_DIM]
        x = self.output_proj(x)
        
        return x
    
    def generate(
        self,
        seed_mel,
        num_frames,
        temperature=1.0,
        top_p=0.9
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
        
        Returns:
            Generated mel spectrogram [num_frames, N_MELS]
        
        Note:
            - Input/output use individual frames [T, N_MELS] for API compatibility
            - Internally operates on grouped frames [T/R, GROUPED_MEL_DIM]
            - Seed is truncated to R-divisible length before reshaping
        """
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
            output = self(context_batch, training=False)
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
