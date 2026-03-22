"""Mel generator model."""

import tensorflow as tf
import keras
from src.music_generation.config import (
    D_MODEL, NUM_HEADS, N_MELS, WINDOW_SIZES,
    CONV_DILATION_RATES, FRAME_STEP, SAMPLE_RATE,
    LATENT_DIM, TOKEN_COMPRESSION_RATIO, TOKEN_NUM_CONV_LAYERS,
    TOKEN_SEQ_LEN, POSE_FEATURE_DIM, POSE_EMBEDDING_DIM
)
from src.music_generation.layers import CausalConvBlock, CausalConv1D, TransformerBlock
from src.body_point_module.encoder import build_temporal_pose_encoder


@keras.saving.register_keras_serializable()
class MelTokenizer(tf.keras.layers.Layer):
    """Causal strided Conv1D encoder: [B, T, N_MELS] -> [B, ceil(T/C), D_MODEL].

    Uses TOKEN_NUM_CONV_LAYERS successive stride-2 causal convolutions for
    learned temporal compression. Strict causality via left-padding.
    Pads input to next multiple of TOKEN_COMPRESSION_RATIO before encoding.
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
                filters[i], kernel_size, strides=2, padding='valid',
                name=f'conv_{i}'
            )
            norm = tf.keras.layers.LayerNormalization(name=f'norm_{i}')
            self.blocks.append((padding, conv, norm))
        self.proj = tf.keras.layers.Dense(D_MODEL, name='proj')

    def call(self, x, training=False):
        """Tokenize mel spectrogram.

        Args:
            x: [B, T, N_MELS] — T can be any length.
        Returns:
            [B, ceil(T / TOKEN_COMPRESSION_RATIO), D_MODEL]
        """
        # Pad to next multiple of C so stride-2 convs produce exact T_padded/C tokens
        C = TOKEN_COMPRESSION_RATIO
        pad_amount = (-tf.shape(x)[1]) % C
        x = tf.pad(x, [[0, 0], [0, pad_amount], [0, 0]])
        for padding, conv, norm in self.blocks:
            x = tf.pad(x, [[0, 0], [padding, 0], [0, 0]])
            x = conv(x)
            x = norm(x)
            x = tf.nn.gelu(x)
        return self.proj(x)


@keras.saving.register_keras_serializable()
class MelDetokenizer(tf.keras.layers.Layer):
    """Causal upsampling decoder: [B, T_tok, D_MODEL] -> [B, T_tok*C, N_MELS].

    Uses UpSampling1D + CausalConv1D for learned upsampling.
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
            up = tf.keras.layers.UpSampling1D(size=2, name=f'up_{i}')
            conv = CausalConv1D(filters[i], kernel_size=3, name=f'conv_{i}')
            norm = tf.keras.layers.LayerNormalization(name=f'norm_{i}')
            self.blocks.append((up, conv, norm))
        self.proj = tf.keras.layers.Dense(N_MELS, name='proj')

    def call(self, x, target_length=None, training=False):
        """Detokenize tokens to mel spectrogram.

        Args:
            x: [B, T_tok, D_MODEL]
            target_length: If provided, trim output to [:, :target_length, :]
        Returns:
            [B, T_tok * TOKEN_COMPRESSION_RATIO, N_MELS] or [B, target_length, N_MELS]
        """
        for up, conv, norm in self.blocks:
            x = up(x)
            x = conv(x)
            x = norm(x)
            x = tf.nn.gelu(x)
        x = self.proj(x)
        if target_length is not None:
            x = x[:, :target_length, :]
        return x


@keras.saving.register_keras_serializable()
class LatentEncoder(tf.keras.Model):
    """VAE-style encoder: mel sequence -> global latent vector z."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(128, 5, strides=2, padding="same", activation="gelu", name="conv1")
        self.conv2 = tf.keras.layers.Conv1D(256, 5, strides=2, padding="same", activation="gelu", name="conv2")
        self.pool = tf.keras.layers.GlobalAveragePooling1D(name="pool")
        self.dense = tf.keras.layers.Dense(256, activation="gelu", name="dense")
        self.z_mean_head = tf.keras.layers.Dense(latent_dim, name="z_mean")
        self.z_logvar_head = tf.keras.layers.Dense(latent_dim, name="z_logvar")

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



@keras.saving.register_keras_serializable()
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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
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
        self.tokenizer = MelTokenizer(name='mel_tokenizer')
        self.detokenizer = MelDetokenizer(name='mel_detokenizer')
        
        # Dynamic causal conv stack from config
        self.conv_layers = [
            CausalConvBlock(D_MODEL, kernel_size=3, dilation_rate=d, residual=True, name=f'conv_block_{i}')
            for i, d in enumerate(CONV_DILATION_RATES)
        ]
        
        # Transformer stack with explicit layers and increasing window sizes
        self.transformer1 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[0], name='transformer_0')
        self.transformer2 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[1], enable_cross_attn=True, name='transformer_1')
        self.transformer3 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=WINDOW_SIZES[2], enable_cross_attn=True, name='transformer_2')
        
        # Latent conditioning projection
        self.z_proj = tf.keras.layers.Dense(D_MODEL, name='z_proj')
        
        # Token-space projection head for forward_tokens()
        self.projection_head = tf.keras.layers.Dense(D_MODEL, name='projection_head')
        
        # Pose conditioning
        self.pose_encoder = build_temporal_pose_encoder(
            input_dim=POSE_FEATURE_DIM, output_dim=POSE_EMBEDDING_DIM
        )
        self.pose_proj = tf.keras.layers.Dense(D_MODEL, name='pose_proj')
        self.pose_stride_conv = tf.keras.layers.Conv1D(
            D_MODEL, kernel_size=TOKEN_COMPRESSION_RATIO,
            strides=TOKEN_COMPRESSION_RATIO, padding='same', name='pose_stride_conv'
        )
        self.pose_alpha = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='pose_alpha')
    
    def call(self, mel, z=None, pose=None, training=False):
        """Forward pass (parallel processing for training).
        
        Args:
            mel: Input mel spectrogram [B, T, N_MELS] — T can be any length
            z: Optional latent vector [B, latent_dim]. If None, sampled from N(0,1).
            pose: Optional pose features [B, T, POSE_FEATURE_DIM]. If None, no conditioning.
            training: Whether in training mode
        
        Returns:
            Output mel spectrogram [B, T, N_MELS] (same T as input)
        """
        original_T = tf.shape(mel)[1]
        tokens = self.tokenizer(mel, training=training)
        
        pose_embedded = None
        if pose is not None:
            pose_embedded = self.pose_encoder(pose, training=training)
            pose_embedded = self.pose_proj(pose_embedded)
            pose_embedded = self.pose_stride_conv(pose_embedded)
        
        token_preds = self.forward_tokens(tokens, z=z, pose_embedded=pose_embedded, training=training)
        return self.detokenizer(token_preds, target_length=original_T, training=training)
    
    def forward_tokens(self, tokens, z=None, pose_embedded=None, training=False):
        """Token-space forward pass — stops before detokenizer.
        
        Args:
            tokens: [B, T_tok, D_MODEL]
            z: Optional latent vector [B, latent_dim]. If None, sampled from N(0,1).
            pose_embedded: Optional pre-processed pose [B, T_tok, D_MODEL]. If None, no conditioning.
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
        x = self.transformer2(x, cross_attn_kv=pose_embedded, cross_attn_alpha=self.pose_alpha)
        x = self.transformer3(x, cross_attn_kv=pose_embedded, cross_attn_alpha=self.pose_alpha)
        
        return self.projection_head(x)
    
    def forward_from_tokens(self, tokens, z=None, pose_embedded=None, training=False):
        """Forward pass from pre-tokenized input (for scheduled sampling).
        
        Args:
            tokens: [B, T_tok, D_MODEL] pre-tokenized input
            z: Optional latent vector [B, latent_dim]
            pose_embedded: Optional pre-processed pose [B, T_tok, D_MODEL]
            training: Whether in training mode
        
        Returns:
            Output mel spectrogram [B, T_tok * TOKEN_COMPRESSION_RATIO, N_MELS]
        """
        token_preds = self.forward_tokens(tokens, z=z, pose_embedded=pose_embedded, training=training)
        return self.detokenizer(token_preds, training=training)
    
    def generate(self, seed_mel, num_frames, temperature=1.0, top_p=0.9, z=None, pose=None):
        """Autoregressive generation in token space.
        
        Tokenizes seed once, loops via forward_tokens() (no mel roundtrips),
        detokenizes all generated tokens once at the end.
        
        Args:
            seed_mel: Seed mel spectrogram [seed_len, N_MELS]
            num_frames: Number of individual frames to generate
            temperature: Temperature scaling for noise
            top_p: Nucleus sampling threshold (unused for continuous values)
            z: Optional latent vector [1, latent_dim]
            pose: Optional pose features [1, T, POSE_FEATURE_DIM]
        
        Returns:
            Generated mel spectrogram [num_frames, N_MELS]
        """
        C = TOKEN_COMPRESSION_RATIO
        
        if z is None:
            z = tf.random.normal([1, LATENT_DIM])
        
        # Pre-process pose if provided
        pose_embedded = None
        if pose is not None:
            pe = self.pose_encoder(pose, training=False)
            pe = self.pose_proj(pe)
            pose_embedded = self.pose_stride_conv(pe)
        
        # Tokenize seed once (only tokenizer call)
        seed_tokens = self.tokenizer(seed_mel[None], training=False)
        context = seed_tokens[0]  # [seed_tokens, D_MODEL]
        
        if context.shape[0] > TOKEN_SEQ_LEN:
            context = context[-TOKEN_SEQ_LEN:]
        
        generated_tokens = []
        num_steps = (num_frames + C - 1) // C
        
        for _ in range(num_steps):
            if context.shape[0] > TOKEN_SEQ_LEN:
                context = context[-TOKEN_SEQ_LEN:]
            
            # Forward pass entirely in token space — no mel roundtrip
            pred_tokens = self.forward_tokens(context[None], z=z, pose_embedded=pose_embedded, training=False)
            next_token = pred_tokens[0, -1:]  # [1, D_MODEL]
            
            if temperature > 0.0:
                noise = tf.random.normal(tf.shape(next_token)) * 0.02 * temperature
                next_token = next_token + noise
            
            context = tf.concat([context, next_token], axis=0)
            generated_tokens.append(next_token)
        
        # Detokenize all generated tokens once (only detokenizer call)
        all_tokens = tf.concat(generated_tokens, axis=0)[None]  # [1, num_steps, D_MODEL]
        generated_mel = self.detokenizer(all_tokens, training=False)
        
        return generated_mel[0, :num_frames]
