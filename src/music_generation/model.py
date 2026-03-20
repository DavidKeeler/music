"""Mel generator model."""

import tensorflow as tf
from src.music_generation.config import (
    D_MODEL, NUM_HEADS, N_MELS, WINDOW_SIZES,
    CONV_DILATION_RATES, FRAME_STEP, SAMPLE_RATE,
    LATENT_DIM, TOKEN_COMPRESSION_RATIO, TOKEN_NUM_CONV_LAYERS,
    TOKEN_SEQ_LEN
)
from src.music_generation.layers import CausalConvBlock, CausalConv1D, TransformerBlock


class MelTokenizer(tf.keras.layers.Layer):
    """Causal strided Conv1D encoder: [B, T, N_MELS] -> [B, T//C, D_MODEL].

    Uses TOKEN_NUM_CONV_LAYERS successive stride-2 causal convolutions for
    learned temporal compression. Strict causality via left-padding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n = TOKEN_NUM_CONV_LAYERS
        # Filter progression: N_MELS -> ... -> D_MODEL
        filters = [
            N_MELS + (D_MODEL - N_MELS) * (i + 1) // n
            for i in range(n)
        ]
        self.blocks = []
        for i in range(n):
            kernel_size = 3
            padding = kernel_size - 1  # causal left-pad for stride-2
            conv = tf.keras.layers.Conv1D(
                filters[i], kernel_size, strides=2, padding='valid'
            )
            norm = tf.keras.layers.LayerNormalization()
            self.blocks.append((padding, conv, norm))
        self.proj = tf.keras.layers.Dense(D_MODEL)

    def call(self, x, training=False):
        """Tokenize mel spectrogram.

        Args:
            x: [B, T, N_MELS] where T is divisible by TOKEN_COMPRESSION_RATIO
        Returns:
            [B, T // TOKEN_COMPRESSION_RATIO, D_MODEL]
        """
        for padding, conv, norm in self.blocks:
            x = tf.pad(x, [[0, 0], [padding, 0], [0, 0]])
            x = conv(x)
            x = norm(x)
            x = tf.nn.gelu(x)
        return self.proj(x)


class MelDetokenizer(tf.keras.layers.Layer):
    """Causal upsampling decoder: [B, T_tok, D_MODEL] -> [B, T_tok*C, N_MELS].

    Uses tf.repeat nearest-neighbor upsampling followed by causal Conv1D.
    Strict causality preserved throughout.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n = TOKEN_NUM_CONV_LAYERS
        # Filter progression: D_MODEL -> ... -> N_MELS (mirrored)
        filters = [
            D_MODEL - (D_MODEL - N_MELS) * (i + 1) // n
            for i in range(n)
        ]
        self.blocks = []
        for i in range(n):
            conv = CausalConv1D(filters[i], kernel_size=3)
            norm = tf.keras.layers.LayerNormalization()
            self.blocks.append((conv, norm))
        self.proj = tf.keras.layers.Dense(N_MELS)

    def call(self, x, training=False):
        """Detokenize tokens to mel spectrogram.

        Args:
            x: [B, T_tok, D_MODEL]
        Returns:
            [B, T_tok * TOKEN_COMPRESSION_RATIO, N_MELS]
        """
        for conv, norm in self.blocks:
            x = tf.repeat(x, repeats=2, axis=1)
            x = conv(x)
            x = norm(x)
            x = tf.nn.gelu(x)
        return self.proj(x)


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
            mel: [B, T, N_MELS] (raw mel spectrogram)

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
    """Autoregressive mel spectrogram generator with learned tokenizer/detokenizer.
    
    Architecture (strictly causal):
    - MelTokenizer: causal strided Conv1D encoder [B, T, N_MELS] -> [B, T/C, D_MODEL]
    - N causal conv layers built from CONV_DILATION_RATES (kernel_size=3, residual)
    - 3 transformer blocks with increasing window sizes
    - MelDetokenizer: causal upsampling decoder [B, T/C, D_MODEL] -> [B, T, N_MELS]
    
    All operations preserve causality: output at time t depends only on inputs ≤ t.
    Supports both parallel forward pass (training) and autoregressive generation (inference).
    """
    
    def __init__(self):
        super().__init__()
        
        import logging
        
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
        
        # Learned tokenizer/detokenizer replace input_proj/output_proj
        self.tokenizer = MelTokenizer()
        self.detokenizer = MelDetokenizer()
        
        # Dynamic causal conv stack from config
        self.conv_layers = [
            CausalConvBlock(D_MODEL, kernel_size=3, dilation_rate=d, residual=True)
            for d in CONV_DILATION_RATES
        ]
        
        # Transformer stack with explicit layers and increasing window sizes
        self.transformer1 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[0])
        self.transformer2 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[1])
        self.transformer3 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[2])
        
        # Latent conditioning projection
        self.z_proj = tf.keras.layers.Dense(D_MODEL)
        
        # Token-space projection head for forward_tokens()
        self.projection_head = tf.keras.layers.Dense(D_MODEL)
    
    def call(self, mel, z=None, training=False):
        """Forward pass (parallel processing for training).
        
        Args:
            mel: Input mel spectrogram [B, T, N_MELS]
            z: Optional latent vector [B, latent_dim]. If None, sampled from N(0,1).
            training: Whether in training mode
        
        Returns:
            Output mel spectrogram [B, T, N_MELS]
        """
        tokens = self.tokenizer(mel, training=training)
        return self.forward_from_tokens(tokens, z=z, training=training)
    
    def forward_tokens(self, tokens, z=None, training=False):
        """Token-space forward pass — stops before detokenizer.
        
        Args:
            tokens: [B, T_tok, D_MODEL]
            z: Optional latent vector [B, latent_dim]. If None, sampled from N(0,1).
            training: Whether in training mode
        
        Returns:
            [B, T_tok, D_MODEL] — projected token predictions
        """
        batch_size = tf.shape(tokens)[0]
        seq_len = tf.shape(tokens)[1]
        
        if z is None:
            z = tf.random.normal([batch_size, LATENT_DIM])
        
        z_embed = self.z_proj(z)
        z_embed = tf.expand_dims(z_embed, axis=1)
        z_embed = tf.broadcast_to(z_embed, [batch_size, seq_len, D_MODEL])
        x = tokens + z_embed
        
        for conv in self.conv_layers:
            x = conv(x, training=training)
        
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)
        
        return self.projection_head(x)
    
    def forward_from_tokens(self, tokens, z=None, training=False):
        """Forward pass from pre-tokenized input (for scheduled sampling).
        
        Args:
            tokens: [B, T_tok, D_MODEL] pre-tokenized input
            z: Optional latent vector [B, latent_dim]
            training: Whether in training mode
        
        Returns:
            Output mel spectrogram [B, T_tok * TOKEN_COMPRESSION_RATIO, N_MELS]
        """
        batch_size = tf.shape(tokens)[0]
        seq_len = tf.shape(tokens)[1]
        
        if z is None:
            z = tf.random.normal([batch_size, LATENT_DIM])
        
        # Latent conditioning in token space
        z_embed = self.z_proj(z)  # [B, D_MODEL]
        z_embed = tf.expand_dims(z_embed, axis=1)  # [B, 1, D_MODEL]
        z_embed = tf.broadcast_to(z_embed, [batch_size, seq_len, D_MODEL])
        x = tokens + z_embed
        
        # Causal conv stack
        for conv in self.conv_layers:
            x = conv(x, training=training)
        
        # Transformer stack
        x = self.transformer1(x)
        x = self.transformer2(x)
        x = self.transformer3(x)
        
        # Detokenize back to mel space
        return self.detokenizer(x, training=training)
    
    def generate(self, seed_mel, num_frames, temperature=1.0, top_p=0.9, z=None):
        """Autoregressive generation in token space.
        
        Args:
            seed_mel: Seed mel spectrogram [seed_len, N_MELS]
            num_frames: Number of individual frames to generate
            temperature: Temperature scaling for noise
            top_p: Nucleus sampling threshold (unused for continuous values)
            z: Optional latent vector [1, latent_dim]
        
        Returns:
            Generated mel spectrogram [num_frames, N_MELS]
        """
        C = TOKEN_COMPRESSION_RATIO
        
        if z is None:
            z = tf.random.normal([1, LATENT_DIM])
        
        # Truncate seed to C-divisible length
        seed_len = seed_mel.shape[0]
        truncated_len = (seed_len // C) * C
        if truncated_len > 0:
            seed_mel = seed_mel[:truncated_len]
        else:
            pad_len = C - seed_len
            seed_mel = tf.concat([
                tf.zeros([pad_len, N_MELS], dtype=seed_mel.dtype),
                seed_mel
            ], axis=0)
        
        # Tokenize seed: [1, seed_len, N_MELS] -> [1, seed_tokens, D_MODEL]
        seed_tokens = self.tokenizer(seed_mel[None], training=False)
        context = seed_tokens[0]  # [seed_tokens, D_MODEL]
        
        if context.shape[0] > TOKEN_SEQ_LEN:
            context = context[-TOKEN_SEQ_LEN:]
        
        generated_tokens = []
        num_steps = (num_frames + C - 1) // C
        
        for _ in range(num_steps):
            if context.shape[0] > TOKEN_SEQ_LEN:
                context = context[-TOKEN_SEQ_LEN:]
            
            # Forward pass in token space
            context_batch = context[None]  # [1, T_tok, D_MODEL]
            output_mel = self.forward_from_tokens(context_batch, z=z, training=False)
            # Re-tokenize output to get predicted next token
            output_tokens = self.tokenizer(output_mel, training=False)
            next_token = output_tokens[0, -1:]  # [1, D_MODEL]
            
            if temperature > 0.0:
                noise = tf.random.normal(tf.shape(next_token)) * 0.02 * temperature
                next_token = next_token + noise
            
            context = tf.concat([context, next_token], axis=0)
            generated_tokens.append(next_token)
        
        # Detokenize all generated tokens at once
        all_tokens = tf.concat(generated_tokens, axis=0)[None]  # [1, num_steps, D_MODEL]
        generated_mel = self.detokenizer(all_tokens, training=False)
        
        return generated_mel[0, :num_frames]
