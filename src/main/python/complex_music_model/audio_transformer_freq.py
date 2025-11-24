import tensorflow as tf
from tensorflow.keras.layers import Layer
import warnings

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- Complex Layers (Dense, Conv, ConvTranspose, ModReLU, LayerNorm) --- #

def complex_modrelu(z, bias):
    mag = tf.abs(z)
    b = tf.cast(bias, tf.float32)
    activated = tf.nn.relu(mag + b)
    eps = 1e-7  # Slightly larger for stability
    scale = activated / (mag + eps)
    return tf.cast(scale, tf.complex64) * z

class ComplexDense(Layer):
    def __init__(self, units, use_bias=True, spectral_norm=False, activation=True, **kwargs):
        super().__init__(**kwargs)
        self.units, self.use_bias, self.spectral_norm, self.activation = units, use_bias, spectral_norm, activation

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        # Store real and imaginary parts separately (combine in call)
        self.w_real = self.add_weight((in_dim, self.units), initializer='glorot_uniform', trainable=True)
        self.w_imag = self.add_weight((in_dim, self.units), initializer='glorot_uniform', trainable=True)
        
        if self.activation:
            self.relu_bias = self.add_weight((self.units,), initializer='zeros', trainable=True)
        if self.use_bias:
            self.b = self.add_weight((self.units,), initializer='zeros', trainable=True)

    def call(self, x):
        # Combine weights in call() to avoid graph scope issues
        w = tf.complex(self.w_real, self.w_imag)
        z = tf.matmul(x, w)
        
        if self.use_bias:
            z += tf.cast(self.b, tf.complex64)
        return complex_modrelu(z, self.relu_bias) if self.activation else z

class ComplexConv1D(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.filters, self.kernel_size, self.strides, self.padding = filters, kernel_size, strides, padding

    def build(self, input_shape):
        self.W_real = self.add_weight((self.kernel_size, input_shape[-1], self.filters), initializer='glorot_uniform')
        self.W_imag = self.add_weight((self.kernel_size, input_shape[-1], self.filters), initializer='glorot_uniform')

    def call(self, x):
        xr, xi = tf.math.real(x), tf.math.imag(x)
        r = tf.nn.conv1d(xr, self.W_real, stride=self.strides, padding=self.padding.upper()) - \
            tf.nn.conv1d(xi, self.W_imag, stride=self.strides, padding=self.padding.upper())
        i = tf.nn.conv1d(xr, self.W_imag, stride=self.strides, padding=self.padding.upper()) + \
            tf.nn.conv1d(xi, self.W_real, stride=self.strides, padding=self.padding.upper())
        return tf.complex(r, i)

class ComplexConv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.filters, self.kernel_size, self.strides, self.padding = filters, kernel_size, strides, padding

    def build(self, input_shape):
        self.W_real = self.add_weight((self.kernel_size, self.filters, input_shape[-1]), initializer='glorot_uniform')
        self.W_imag = self.add_weight((self.kernel_size, self.filters, input_shape[-1]), initializer='glorot_uniform')

    def call(self, x):
        xr, xi = tf.math.real(x), tf.math.imag(x)
        batch_size, length = tf.shape(x)[0], tf.shape(x)[1]
        out_len = length * self.strides
        out_shape = [batch_size, out_len, self.filters]
        r = tf.nn.conv1d_transpose(xr, self.W_real, output_shape=out_shape, strides=self.strides, padding=self.padding.upper()) - \
            tf.nn.conv1d_transpose(xi, self.W_imag, output_shape=out_shape, strides=self.strides, padding=self.padding.upper())
        i = tf.nn.conv1d_transpose(xr, self.W_imag, output_shape=out_shape, strides=self.strides, padding=self.padding.upper()) + \
            tf.nn.conv1d_transpose(xi, self.W_real, output_shape=out_shape, strides=self.strides, padding=self.padding.upper())
        return tf.complex(r, i)

class ComplexLayerNorm(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight((input_shape[-1],), initializer='ones', trainable=True)
        self.beta = self.add_weight((input_shape[-1],), initializer='zeros', trainable=True)

    def call(self, x):
        xr, xi = tf.math.real(x), tf.math.imag(x)
        mean_r, mean_i = tf.reduce_mean(xr, -1, keepdims=True), tf.reduce_mean(xi, -1, keepdims=True)
        var_r, var_i = tf.reduce_mean((xr - mean_r)**2, -1, keepdims=True), tf.reduce_mean((xi - mean_i)**2, -1, keepdims=True)
        denom = tf.sqrt(var_r + var_i + self.epsilon)
        norm_r, norm_i = (xr - mean_r) / denom, (xi - mean_i) / denom
        return tf.complex(norm_r * self.gamma + self.beta, norm_i * self.gamma + self.beta)

# --- Transformer Blocks --- #
class ComplexMultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = ComplexDense(d_model, spectral_norm=True)
        self.wk = ComplexDense(d_model, spectral_norm=True)
        self.wv = ComplexDense(d_model, spectral_norm=True)
        self.dense = ComplexDense(d_model, spectral_norm=True)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, [0, 2, 1, 3])  # (B, H, T, D)

    def call(self, v, k, q, mask=None):
        # project Q, K, V
        q = self.wq(q)  # complex
        k = self.wk(k)  # complex
        v = self.wv(v)  # complex

        q = self.split_heads(q)  # (B,H,T,D)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # --- fully complex attention ---
        # Hermitian dot-product: Q @ K^H
        scores = tf.matmul(q, tf.linalg.adjoint(k))  # (B,H,T,T), complex64
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scores = scores / tf.cast(tf.sqrt(dk), tf.complex64)

        # softmax must operate on real numbers, use real part
        logits = tf.math.real(scores)

        # apply mask if provided
        if mask is not None:
            mask = tf.cast(mask, tf.float32)[tf.newaxis, tf.newaxis, :, :]
            neg_inf = tf.constant(-1e9, dtype=logits.dtype)
            logits = logits * mask + (1.0 - mask) * neg_inf

        attn_weights = tf.nn.softmax(logits, axis=-1)  # (B,H,T,T)
        attn_weights = tf.cast(attn_weights, tf.complex64)  # back to complex

        # apply attention
        output = tf.matmul(attn_weights, v)  # (B,H,T,D)
        output = tf.transpose(output, [0, 2, 1, 3])
        batch_size = tf.shape(output)[0]
        output = tf.reshape(output, [batch_size, -1, self.num_heads * self.depth])

        # final complex linear projection
        return self.dense(output)

#
# class ComplexMultiHeadAttention(Layer):
#     def __init__(self, d_model, num_heads, **kwargs):
#         super().__init__(**kwargs)
#         assert d_model % num_heads == 0
#         self.d_model, self.num_heads, self.depth = d_model, num_heads, d_model // num_heads
#         self.wq = ComplexDense(d_model)
#         self.wk = ComplexDense(d_model)
#         self.wv = ComplexDense(d_model)
#         self.dense = ComplexDense(d_model)
#
#     def split_heads(self, x):
#         batch_size = tf.shape(x)[0]
#         x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
#         return tf.transpose(x, [0,2,1,3])
#
#     def call(self, v, k, q, mask=None):
#         q, k, v = self.wq(q), self.wk(k), self.wv(v)
#         q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
#         qr, qi = tf.math.real(q), tf.math.imag(q)
#         kr, ki = tf.math.real(k), tf.math.imag(k)
#         q_cat = tf.concat([qr, qi], axis=-1)
#         k_cat = tf.concat([kr, -ki], axis=-1)
#         dk = tf.cast(tf.shape(k_cat)[-1], tf.float32)
#         logits = tf.matmul(q_cat, k_cat, transpose_b=True) / tf.sqrt(dk)
#         if mask is not None:
#             mask = tf.cast(mask, tf.float32)[tf.newaxis, tf.newaxis, :, :]
#             logits = logits * mask + (1.0 - mask) * -1e9
#         attn_weights = tf.cast(tf.nn.softmax(logits, axis=-1), tf.complex64)
#         output = tf.matmul(attn_weights, v)
#         output = tf.transpose(output, [0,2,1,3])
#         batch_size = tf.shape(output)[0]
#         output = tf.reshape(output, [batch_size, -1, self.num_heads*self.depth])
#         return self.dense(output)

class ComplexTransformerBlock(Layer):
    def __init__(self, d_model, num_heads=4, dff=512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.attention = ComplexMultiHeadAttention(d_model, num_heads)
        self.ffn = ComplexDense(d_model, spectral_norm=True)
        self.layernorm1 = ComplexLayerNorm()
        self.layernorm2 = ComplexLayerNorm()

    def call(self, x, mask=None, training=False):
        # --- Self-attention ---
        attn_output = self.attention(x, x, x, mask=mask)  # complex64
        x = x + attn_output
        x = self.layernorm1(x)  # complex64 normalization

        # --- Feed-forward ---
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.layernorm2(x)  # complex64 normalization

        return x

# class ComplexTransformerBlock(Layer):
#     def __init__(self, d_model, num_heads=4, dff=256, **kwargs):
#         super().__init__(**kwargs)
#         self.attention = ComplexMultiHeadAttention(d_model, num_heads)
#         self.ffn = ComplexDense(d_model)
#         self.layernorm1 = ComplexLayerNorm()
#         self.layernorm2 = ComplexLayerNorm()
#
#     def call(self, x, mask=None, training=False):
#         x = x + self.attention(x,x,x, mask=mask)
#         x = self.layernorm1(x)
#         x = x + self.ffn(x)
#         x = self.layernorm2(x)
#         return x

# --- STFT Consistency --- #
class STFTConsistencyLayer(Layer):
    def __init__(self, frame_length=1024, frame_step=256, **kwargs):
        super().__init__(**kwargs)
        self.frame_length, self.frame_step = frame_length, frame_step

    def call(self, pred_complex):
        # Proper windowing for inverse STFT with correct scaling
        recon = tf.signal.inverse_stft(
            pred_complex, 
            frame_length=self.frame_length, 
            frame_step=self.frame_step,
            window_fn=tf.signal.hann_window
        )
        return tf.signal.stft(
            recon, 
            frame_length=self.frame_length, 
            frame_step=self.frame_step,
            window_fn=tf.signal.hann_window
        )

# --- Audio Transformer Model --- #
class AudioTransformerFreq(tf.keras.Model):
    def __init__(self, d_model=256, num_heads=4, num_layers=1, dff=512, compression_rate=2):
        super().__init__()
        self.d_model = d_model

        # Encoder: Compress 513 STFT bins → 128 (divisible by num_heads=4)
        self.input_proj = ComplexDense(128, activation=False)  # Fix 513→128
        self.encoder = ComplexDense(128, activation=False)
        self.pre_dense = ComplexDense(d_model)

        # Down/up sampling (keep complex)
        self.down = ComplexConv1D(filters=d_model, kernel_size=3, strides=2)
        self.up = ComplexConv1DTranspose(filters=d_model, kernel_size=3, strides=2)

        # Transformer stack (complex)
        self.transformer_blocks = [
            ComplexTransformerBlock(d_model, num_heads, dff)
            for _ in range(num_layers)
        ]

        # Compressed-space residual prediction
        self.final_dense = ComplexDense(128)
        # Expand back to 128, then to full STFT bins
        self.pre_decoder = ComplexDense(128, activation=False)
        self.decoder = ComplexDense(513, activation=False)

        # STFT consistency (complex)
        self.consistency_layer = STFTConsistencyLayer(frame_length=1024, frame_step=256)

    def call(self, stft_input, training=False):
        # Project 513 → 512 (divisible by num_heads)
        x = self.input_proj(stft_input)
        
        # Encode → complex features
        compressed = self.encoder(x)
        x = self.pre_dense(compressed)

        # Positional encoding (real → complex)
        seq_len = tf.shape(x)[1]
        pos_encoding = self.get_positional_encoding(seq_len, tf.shape(x)[-1])
        pos_encoding = tf.cast(pos_encoding, tf.complex64)
        x = x + pos_encoding[tf.newaxis, :, :]

        # Downsampling (complex)
        x = self.down(x)

        # Causal mask
        seq_len = tf.shape(x)[1]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = tf.cast(mask, tf.bool)

        # Transformer stack (fully complex)
        for block in self.transformer_blocks:
            x = block(x, mask=mask, training=training)

        # Upsampling (complex)
        x = self.up(x)

        # Compressed-space residual
        out = self.final_dense(x)
        min_len = tf.minimum(tf.shape(out)[1], tf.shape(compressed)[1])
        out = out[:, :min_len, :] + compressed[:, :min_len, :]

        # Decode → 128 → full STFT bins
        out_128 = self.pre_decoder(out)
        full_stft = self.decoder(out_128)

        # STFT consistency
        out_consistent = self.consistency_layer(full_stft)
        return out_consistent

    def get_positional_encoding(self, seq_len, d_model, max_len=2048):
        pos = tf.cast(tf.range(max_len), tf.float32)[:, None]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32))
        )
        pe_sin = tf.sin(pos * div_term)
        pe_cos = tf.cos(pos * div_term)
        pe = tf.concat([pe_sin, pe_cos], axis=1)[:, :d_model]
        return pe[:seq_len]
    
    def generate(self, seed_stft, num_steps, temperature=1.0):
        """
        Efficient autoregressive generation with preallocation.
        
        Args:
            seed_stft: [1, seed_len, 513] initial STFT frames
            num_steps: number of new frames to generate
            temperature: sampling temperature (1.0 = no change)
        
        Returns:
            generated_stft: [1, seed_len + num_steps, 513]
        """
        batch_size = tf.shape(seed_stft)[0]
        seed_len = tf.shape(seed_stft)[1]
        
        # Preallocate output tensor for efficiency
        total_len = seed_len + num_steps
        output_shape = [batch_size, total_len, 513]
        generated = tf.Variable(tf.zeros(output_shape, dtype=tf.complex64))
        
        # Initialize with seed
        generated[:, :seed_len, :].assign(seed_stft)
        
        # Generate step by step
        for step in range(num_steps):
            current_len = seed_len + step
            current_seq = generated[:, :current_len, :]
            
            # Forward pass
            next_frame = self(current_seq, training=False)
            
            # Take only the last predicted frame
            next_frame = next_frame[:, -1:, :]
            
            # Optional temperature scaling
            if temperature != 1.0:
                # Scale magnitude by temperature
                mag = tf.abs(next_frame)
                phase = tf.math.angle(next_frame)
                mag = mag ** (1.0 / temperature)
                next_frame = mag * tf.exp(1j * tf.cast(phase, tf.complex64))
            
            # Assign to preallocated tensor
            generated[:, current_len:current_len+1, :].assign(next_frame)
        
        return generated.value()

