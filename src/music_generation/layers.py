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
    """Local window causal multi-head attention with relative position bias.
    
    Uses sliding windows to avoid allocating [T, T] attention matrices.
    Memory scales as O(B × H × T × window_size) instead of O(B × H × T²).
    """
    
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
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, H, T, head_dim]
        
        # Pad K and V to ensure all positions have a full window
        # Pad left with window_size - 1 zeros for causal attention
        pad_width = self.window_size - 1
        k_padded = tf.pad(k, [[0, 0], [0, 0], [pad_width, 0], [0, 0]])  # [B, H, T+pad, head_dim]
        v_padded = tf.pad(v, [[0, 0], [0, 0], [pad_width, 0], [0, 0]])
        
        # Compute windowed attention using vectorized gather (avoid tf.signal.frame Metal bug)
        # Create indices for each position's window: [T, window_size]
        T_static = tf.shape(k_padded)[2] - pad_width  # Get T as tensor
        indices = tf.range(T_static)[:, None] + tf.range(self.window_size)[None, :]  # [T, window_size]
        
        # Gather windows: [B, H, T, window_size, head_dim]
        k_windows = tf.gather(k_padded, indices, axis=2, batch_dims=0)
        v_windows = tf.gather(v_padded, indices, axis=2, batch_dims=0)
        
        # Compute attention scores using einsum: [B, H, T, head_dim] x [B, H, T, window_size, head_dim]
        scores = tf.einsum('bhti,bhtji->bhtj', q, k_windows) / math.sqrt(self.head_dim)  # [B, H, T, window_size]
        
        # Add relative position bias: [num_heads, window_size] -> [1, H, 1, window_size]
        bias = tf.reshape(self.relative_position_bias, [1, self.num_heads, 1, self.window_size])
        scores = scores + bias
        
        # Apply softmax
        attn_weights = tf.nn.softmax(scores, axis=-1)  # [B, H, T, window_size]
        
        # Apply attention to values using einsum
        out = tf.einsum('bhtj,bhtji->bhti', attn_weights, v_windows)  # [B, H, T, head_dim]
        
        # Reshape to [B, T, D]
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
