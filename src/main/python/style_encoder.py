import tensorflow as tf
from audio_transformer_freq import ComplexDense, ComplexTransformerBlock

class StyleEncoder(tf.keras.Model):
    def __init__(self, d_model=128, num_blocks=2, style_dim=64, pos_period=512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.style_dim = style_dim
        self.pos_period = pos_period
        
        # Input projection for STFT features
        self.input_proj = ComplexDense(d_model)
        
        # Transformer blocks (reused from audio model)
        self.transformer_blocks = [
            ComplexTransformerBlock(d_model) 
            for _ in range(num_blocks)
        ]
        
        # Global pooling and style projection
        self.style_proj = tf.keras.layers.Dense(style_dim)
        
    def get_repeating_positional_encoding(self, seq_len, d_model):
        # Create positions that repeat every pos_period steps
        pos = tf.cast(tf.range(seq_len), tf.float32) % self.pos_period
        pos = pos[:, None]
        
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))
        
        pe_sin = tf.sin(pos * div_term)
        pe_cos = tf.cos(pos * div_term)
        pe = tf.concat([pe_sin, pe_cos], axis=1)[:, :d_model]
        
        return pe
    
    def call(self, stft_input, training=False):
        # stft_input shape: [batch, time, freq_bins] - complex STFT
        seq_len = tf.shape(stft_input)[1]
        
        # Project to model dimension
        x = self.input_proj(stft_input)
        
        # Add repeating positional encoding
        pos_emb = self.get_repeating_positional_encoding(seq_len, self.d_model)
        x = x + tf.cast(pos_emb, tf.complex64)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # Global average pooling over time dimension
        x_magnitude = tf.abs(x)  # Convert to real for pooling
        style_embedding = tf.reduce_mean(x_magnitude, axis=1)  # [batch, d_model]
        
        # Project to style dimension
        style_embedding = self.style_proj(style_embedding)
        
        return style_embedding
