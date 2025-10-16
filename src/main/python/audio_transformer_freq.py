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
    def __init__(self, units, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias
        })
        return config

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.w_real = self.add_weight(shape=(in_dim, self.units), initializer="glorot_uniform", trainable=True)
        self.w_imag = self.add_weight(shape=(in_dim, self.units), initializer="glorot_uniform", trainable=True)
        if self.use_bias:
            self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)

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
    def __init__(self, d_model, num_heads=4, dff=512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.layers.Dense(dff, activation='relu')
        self.proj = tf.keras.layers.Dense(d_model)
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.mag_norm = LayerNormalization()

    def call(self, x, mask=None):
        magnitude = tf.abs(x)
        phase = tf.math.angle(x)
        magnitude = self.mag_norm(magnitude)
        magphase = tf.concat([magnitude, phase], axis=-1)

        attn_out = self.attn(self.ln1(magphase), self.ln1(magphase), attention_mask=mask)
        magphase = magphase + attn_out

        ffn_out = self.proj(self.ffn(self.ln2(magphase)))
        magphase = magphase + ffn_out

        mag_out = magphase[..., :self.d_model // 2]
        phase_out = magphase[..., self.d_model // 2:]
        mag_out = tf.nn.softplus(mag_out)

        return tf.complex(mag_out * tf.cos(phase_out), mag_out * tf.sin(phase_out))


class ComplexConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.W_real = self.add_weight(
            shape=(self.kernel_size, input_shape[-1], self.filters),
            initializer='glorot_uniform',
            name='W_real'
        )
        self.W_imag = self.add_weight(
            shape=(self.kernel_size, input_shape[-1], self.filters),
            initializer='glorot_uniform',
            name='W_imag'
        )

    def call(self, inputs):
        real_part = tf.math.real(inputs)
        imag_part = tf.math.imag(inputs)
        real_conv_real = tf.nn.conv1d(real_part, self.W_real, stride=self.strides, padding=self.padding.upper())
        real_conv_imag = tf.nn.conv1d(real_part, self.W_imag, stride=self.strides, padding=self.padding.upper())
        imag_conv_real = tf.nn.conv1d(imag_part, self.W_real, stride=self.strides, padding=self.padding.upper())
        imag_conv_imag = tf.nn.conv1d(imag_part, self.W_imag, stride=self.strides, padding=self.padding.upper())
        output_real = real_conv_real - imag_conv_imag
        output_imag = real_conv_imag + imag_conv_real
        return tf.complex(output_real, output_imag)


class ComplexConv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.W_real = self.add_weight(
            shape=(self.kernel_size, self.filters, input_shape[-1]),
            initializer='glorot_uniform',
            name='W_real'
        )
        self.W_imag = self.add_weight(
            shape=(self.kernel_size, self.filters, input_shape[-1]),
            initializer='glorot_uniform',
            name='W_imag'
        )

    def call(self, inputs):
        real_part = tf.math.real(inputs)
        imag_part = tf.math.imag(inputs)
        batch_size = tf.shape(inputs)[0]
        out_len = tf.shape(inputs)[1] * self.strides
        out_shape = [batch_size, out_len, self.filters]
        real_conv_real = tf.nn.conv1d_transpose(real_part, self.W_real, output_shape=out_shape,
                                               strides=self.strides, padding=self.padding.upper())
        real_conv_imag = tf.nn.conv1d_transpose(real_part, self.W_imag, output_shape=out_shape,
                                               strides=self.strides, padding=self.padding.upper())
        imag_conv_real = tf.nn.conv1d_transpose(imag_part, self.W_real, output_shape=out_shape,
                                               strides=self.strides, padding=self.padding.upper())
        imag_conv_imag = tf.nn.conv1d_transpose(imag_part, self.W_imag, output_shape=out_shape,
                                               strides=self.strides, padding=self.padding.upper())
        output_real = real_conv_real - imag_conv_imag
        output_imag = real_conv_imag + imag_conv_real
        return tf.complex(output_real, output_imag)


@tf.keras.utils.register_keras_serializable()
class AudioTransformerFreq(tf.keras.Model):
    def __init__(self, d_model=256, num_blocks=1, freq_bins=513, seq_len=64, compression_factor=2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.freq_bins = freq_bins
        self.seq_len = seq_len
        self.compression_factor = compression_factor

        self.input_proj = ComplexDense(d_model)
        self.skip_conv = ComplexConv1D(filters=d_model, kernel_size=compression_factor,
                                       strides=compression_factor, padding='same')
        self.transformer_blocks = [ComplexTransformerBlock(d_model * 2) for _ in range(num_blocks)]
        self.inverse_conv = ComplexConv1DTranspose(filters=d_model, kernel_size=compression_factor,
                                                   strides=compression_factor, padding='same')
        self.output_proj = ComplexDense(freq_bins)

    def get_positional_encoding(self, seq_len, d_model):
        pos = tf.cast(tf.range(seq_len), tf.float32)[:, None]
        div_term = tf.exp(tf.cast(tf.range(0, d_model, 2), tf.float32) * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))
        pe_sin = tf.sin(pos * div_term)
        pe_cos = tf.cos(pos * div_term)
        pe = tf.concat([pe_sin, pe_cos], axis=1)[:, :d_model]
        return pe

    def call(self, stft_input, training=True):
        seq_len = tf.shape(stft_input)[1]
        x = self.input_proj(stft_input)
        pos_emb = self.get_positional_encoding(seq_len, self.d_model)
        x = x + tf.cast(pos_emb, tf.complex64)
        x_compressed = self.skip_conv(x)
        mask = tf.linalg.band_part(tf.ones((tf.shape(x_compressed)[1], tf.shape(x_compressed)[1])), -1, 0)
        mask = tf.where(mask == 0, -1e9, 0.0)
        for block in self.transformer_blocks:
            x_compressed = block(x_compressed, mask=mask)
        x_expanded = self.inverse_conv(x_compressed)
        expanded_len = tf.shape(x_expanded)[1]
        x_expanded = tf.cond(
            expanded_len < seq_len,
            lambda: tf.pad(x_expanded, [[0, 0], [0, seq_len - expanded_len], [0, 0]]),
            lambda: x_expanded[:, :seq_len, :]
        )
        return self.output_proj(x_expanded)

    def generate(self, initial_frames, num_steps):
        generated = initial_frames
        for _ in range(num_steps):
            input_seq = generated[:, -self.seq_len:]
            next_frame = self.call(input_seq, training=False)[:, -1:, :]
            generated = tf.concat([generated, next_frame], axis=1)
        return generated
