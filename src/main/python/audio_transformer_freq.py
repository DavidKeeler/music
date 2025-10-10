import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization
import warnings

# Suppress TensorFlow complex casting warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')


@tf.function
def modrelu(z, bias):
    abs_z = tf.abs(z)
    scale = tf.nn.relu(abs_z + bias) / (abs_z + 1e-6)
    scale = tf.cast(scale, tf.complex64)
    return scale * z


class ComplexDense(Layer):
    def __init__(self, units, use_bias=True):
        super().__init__()
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.w_real = self.add_weight(shape=(in_dim, self.units), initializer="glorot_uniform", trainable=True)
        self.w_imag = self.add_weight(shape=(in_dim, self.units), initializer="glorot_uniform", trainable=True)
        if self.use_bias:
            self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def call(self, x):
        xr, xi = tf.math.real(x), tf.math.imag(x)
        real = tf.matmul(xr, self.w_real) - tf.matmul(xi, self.w_imag)
        imag = tf.matmul(xr, self.w_imag) + tf.matmul(xi, self.w_real)
        z = tf.complex(real, imag)
        if self.use_bias:
            z = z + tf.cast(self.b, tf.complex64)
            bias_val = tf.cast(self.b, tf.float32)
        else:
            bias_val = 0.0
        return modrelu(z, bias_val)


class ComplexTransformerBlock(Layer):
    def __init__(self, d_model, num_heads=4, dff=512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = ComplexDense(dff)
        self.proj = ComplexDense(d_model)
        self.ln1 = LayerNormalization()

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        # Extract real and imaginary parts
        x_real = tf.math.real(x)
        x_imag = tf.math.imag(x)

        # Attention on real part only with causal mask
        attn_out = self.attn(x_real, x_real, attention_mask=mask)
        x_real_norm = self.ln1(x_real + attn_out)  # Re-enabled LayerNorm

        # Reconstruct complex tensor
        x_complex = tf.complex(x_real_norm, x_imag)

        # Complex FFN
        ffn_out = self.proj(self.ffn(x_complex))

        return x_complex + ffn_out


@tf.keras.utils.register_keras_serializable()
class AudioTransformerFreq(tf.keras.Model):
    def __init__(self, d_model=256, num_blocks=4, freq_bins=513, seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.freq_bins = freq_bins
        self.seq_len = seq_len

        self.input_proj = ComplexDense(d_model)
        self.transformer_blocks = [ComplexTransformerBlock(d_model) for _ in range(num_blocks)]
        self.output_proj = ComplexDense(freq_bins)

    def call(self, stft_input, training=True):
        batch_size = tf.shape(stft_input)[0]
        seq_len = tf.shape(stft_input)[1]

        # Create causal mask
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = tf.where(mask == 0, -1e9, 0.0)

        x = self.input_proj(stft_input)

        for block in self.transformer_blocks:
            x = block(x, mask=mask)

        stft_output = self.output_proj(x)
        return stft_output

    def generate(self, initial_frames, num_steps):
        """Autoregressive generation"""
        batch_size = tf.shape(initial_frames)[0]
        generated = initial_frames

        for _ in range(num_steps):
            # Take last seq_len frames
            input_seq = generated[:, -self.seq_len:]

            # Predict next frame
            next_frame = self.call(input_seq, training=False)[:, -1:, :]

            # Append to generated sequence
            generated = tf.concat([generated, next_frame], axis=1)

        return generated
