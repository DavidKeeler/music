"""Custom Keras layers for mel generation."""

import tensorflow as tf
import math


class CausalConv1D(tf.keras.layers.Layer):
    """Causal 1D convolution with left-only padding.
    
    Ensures strict causality: output at time t depends only on inputs <= t.
    """
    
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = (kernel_size - 1) * dilation_rate
        self.conv = tf.keras.layers.Conv1D(
            filters, kernel_size, dilation_rate=dilation_rate, padding='valid'
        )
    
    def call(self, x):
        """Apply causal convolution.
        
        Args:
            x: Input tensor [batch, time, channels]
        
        Returns:
            Output tensor [batch, time, channels]
        """
        x = tf.pad(x, [[0, 0], [self.padding, 0], [0, 0]])
        return self.conv(x)


class CausalConvBlock(tf.keras.layers.Layer):
    """Causal convolution block with normalization and activation."""
    
    def __init__(self, filters, kernel_size, dilation_rate=1, residual=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.residual = residual
        self.conv = CausalConv1D(filters, kernel_size, dilation_rate)
        self.norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.layers.Activation('gelu')
    
    def call(self, x):
        """Apply causal conv block.
        
        Args:
            x: Input tensor [batch, time, channels]
        
        Returns:
            Output tensor [batch, time, channels]
        """
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        
        if self.residual:
            out = out + x
        
        return out


class LocalWindowAttention(tf.keras.layers.Layer):
    """Local window causal multi-head attention with relative position bias."""
    
    def __init__(self, d_model, num_heads, window_size=128, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.qkv = tf.keras.layers.Dense(3 * d_model)
        self.out_proj = tf.keras.layers.Dense(d_model)
        
        # Learned relative position bias [num_heads, window_size]
        self.relative_position_bias = self.add_weight(
            name='relative_position_bias',
            shape=[num_heads, window_size],
            initializer=tf.random_normal_initializer(stddev=0.01),
            trainable=True
        )
    
    def call(self, x):
        """Apply local window causal attention with relative position bias.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, T, 3*D]
        qkv = tf.reshape(qkv, [B, T, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [3, B, H, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(self.head_dim)  # [B, H, T, T]
        
        # Add relative position bias
        positions = tf.range(T)
        rel_distances = positions[:, None] - positions[None, :]  # [T, T]
        rel_distances = tf.clip_by_value(rel_distances, 0, self.window_size - 1)
        bias = tf.gather(self.relative_position_bias, rel_distances, axis=1)  # [H, T, T]
        scores = scores + bias[None, :, :, :]  # [B, H, T, T]
        
        # Create causal mask
        causal_mask = tf.linalg.band_part(tf.ones([T, T]), -1, 0)
        causal_mask = 1.0 - causal_mask
        
        # Create window mask
        window_mask = tf.cast(positions[:, None] - positions[None, :] >= self.window_size, tf.float32)
        
        # Combine masks
        mask = tf.maximum(causal_mask, window_mask)
        scores = scores + (mask * -1e9)
        
        # Apply softmax and attention
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = tf.where(tf.math.is_nan(attn_weights), 0.0, attn_weights)
        
        out = tf.matmul(attn_weights, v)  # [B, H, T, head_dim]
        out = tf.transpose(out, [0, 2, 1, 3])  # [B, T, H, head_dim]
        out = tf.reshape(out, [B, T, self.d_model])
        
        return self.out_proj(out)


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with local window attention and feed-forward network."""
    
    def __init__(self, d_model, num_heads, window_size=128, ffn_hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        if ffn_hidden_dim is None:
            ffn_hidden_dim = 4 * d_model
        
        self.attn = LocalWindowAttention(d_model, num_heads, window_size)
        self.norm1 = tf.keras.layers.LayerNormalization()
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn_hidden_dim, activation='gelu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.norm2 = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        """Apply transformer block.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Attention with residual
        x = x + self.attn(self.norm1(x))
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x
