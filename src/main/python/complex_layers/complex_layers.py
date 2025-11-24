import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization
import warnings

# Suppress TensorFlow complex casting warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

def spectral_norm_complex(w, u=None, num_iters=1, eps=1e-12):
    """
    Complex spectral normalization (power iteration).
    w: complex tensor with shape [..., out_features] or (..., in, out)
    returns normalized w and updated u (complex).
    """
    w_shape = tf.shape(w)
    # flatten to matrix W_mat of shape [out, in]
    W_mat = tf.reshape(w, [-1, w_shape[-1]])  # (..., out) flattened by rows; careful if w is (in, out)
    # better explicit for dense weights shaped (in, out) -> want [out, in]
    W_mat = tf.transpose(tf.reshape(w, [w_shape[0], -1]))  # [out, in]

    out, inp = tf.shape(W_mat)[0], tf.shape(W_mat)[1]

    if u is None:
        # u shape [out, 1]
        u = tf.complex(tf.random.normal([out, 1]), tf.random.normal([out, 1]))
        u = tf.nn.l2_normalize(u, axis=0)

    # power iterations: v = normalize(W^H u), u = normalize(W v)
    for _ in range(num_iters):
        v = tf.matmul(tf.linalg.adjoint(W_mat), u)   # shape [in,1]
        v = tf.nn.l2_normalize(v, axis=0)
        u = tf.matmul(W_mat, v)                      # shape [out,1]
        u = tf.nn.l2_normalize(u, axis=0)

    # sigma = u^H W v  -> scalar (complex), take abs(real) since sigma should be real >= 0
    sigma = tf.matmul(tf.matmul(tf.linalg.adjoint(u), W_mat), v)  # shape [1,1] complex
    sigma = tf.cast(tf.abs(sigma), tf.float32)
    sigma = tf.maximum(sigma, eps)

    W_norm = w / tf.cast(sigma, w.dtype)
    return W_norm, u

def complex_modrelu(z, bias):
    # z: complex tensor, bias: real tensor broadcastable to channels
    mag = tf.abs(z)
    phase = tf.math.angle(z)
    # bias must be float32 and broadcast
    b = tf.cast(bias, tf.float32)
    activated = tf.nn.relu(mag + b)  # modReLU: relu(mag + b)
    # avoid divide-by-zero: use z / (mag + eps) * activated
    eps = 1e-8
    scale = activated / (mag + eps)
    return tf.cast(scale, tf.complex64) * z

class ComplexDense(Layer):
    def __init__(self, units, use_bias=True, spectral_norm=False, activation=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.spectral_norm = spectral_norm
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "spectral_norm": self.spectral_norm,
            "activation": self.activation,
        })
        return config

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.w_real = self.add_weight(shape=(in_dim, self.units), initializer="glorot_uniform", trainable=True)
        self.w_imag = self.add_weight(shape=(in_dim, self.units), initializer="glorot_uniform", trainable=True)
        if self.activation:
            self.relu_bias = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        if self.use_bias:
            self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        if self.spectral_norm:
            self.u_real = self.add_weight(shape=(self.units, 1), initializer="random_normal", trainable=False)
            self.u_imag = self.add_weight(shape=(self.units, 1), initializer="zeros", trainable=False)

    def call(self, x):
        # Construct complex weight matrix
        w = tf.complex(self.w_real, self.w_imag)
        
        # Apply spectral normalization if enabled
        if self.spectral_norm:
            u = tf.complex(self.u_real, self.u_imag)
            w, u_new = spectral_norm_complex(w, u)
            self.u_real.assign(tf.math.real(u_new))
            self.u_imag.assign(tf.math.imag(u_new))
        
        xr, xi = tf.math.real(x), tf.math.imag(x)
        wr, wi = tf.math.real(w), tf.math.imag(w)
        real = tf.matmul(xr, wr) - tf.matmul(xi, wi)
        imag = tf.matmul(xr, wi) + tf.matmul(xi, wr)
        z = tf.complex(real, imag)
        if self.use_bias:
            z = z + tf.cast(self.b, tf.complex64)
        
        if self.activation:
            return complex_modrelu(z, self.relu_bias)
        else:
            return z

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
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Complex attention
        qr, qi = tf.math.real(q), tf.math.imag(q)
        kr, ki = tf.math.real(k), tf.math.imag(k)
        vr, vi = tf.math.real(v), tf.math.imag(v)
        
        # Attention scores (real part only for stability)
        scores_real = tf.matmul(qr, kr, transpose_b=True) + tf.matmul(qi, ki, transpose_b=True)
        scores_real = scores_real / tf.sqrt(tf.cast(self.depth, tf.float32))
        
        if mask is not None:
            scores_real += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scores_real, axis=-1)
        
        # Apply attention to complex values
        output_real = tf.matmul(attention_weights, vr)
        output_imag = tf.matmul(attention_weights, vi)
        output = tf.complex(output_real, output_imag)
        
        # Concatenate heads
        output = tf.transpose(output, [0, 2, 1, 3])
        batch_size = tf.shape(output)[0]
        output = tf.reshape(output, [batch_size, -1, self.d_model])
        
        return self.dense(output)

class ComplexLayerNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # gamma and beta are real scalars for scaling magnitude and shifting phase
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zeros', trainable=True)

    def call(self, x):
        # x: complex tensor (B,T,C)
        xr, xi = tf.math.real(x), tf.math.imag(x)

        # mean and variance of complex magnitude
        mean_r = tf.reduce_mean(xr, axis=-1, keepdims=True)
        mean_i = tf.reduce_mean(xi, axis=-1, keepdims=True)

        var_r = tf.reduce_mean(tf.square(xr - mean_r), axis=-1, keepdims=True)
        var_i = tf.reduce_mean(tf.square(xi - mean_i), axis=-1, keepdims=True)

        denom = tf.sqrt(var_r + var_i + self.epsilon)

        norm_r = (xr - mean_r) / denom
        norm_i = (xi - mean_i) / denom

        # scale & shift (gamma is real scalar per channel)
        out = tf.complex(norm_r * self.gamma + self.beta,
                         norm_i * self.gamma + self.beta)
        return out

class ComplexTransformerBlock(Layer):
    def __init__(self, d_model, num_heads=4, dff=512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        
        self.attention = ComplexMultiHeadAttention(d_model, num_heads)
        self.ffn = ComplexDense(d_model, spectral_norm=True)
        self.layernorm1 = ComplexLayerNorm()
        self.layernorm2 = ComplexLayerNorm()

    def call(self, x, mask=None, training=False):
        # Self-attention
        attn_output = self.attention(x, x, x, mask=mask)
        x = x + attn_output
        x = self.layernorm1(x)
        
        # Feed forward
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.layernorm2(x)
        
        return x

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
        
        real_output = real_conv_real - imag_conv_imag
        imag_output = real_conv_imag + imag_conv_real
        
        return tf.complex(real_output, imag_output)

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
        
        real_conv_real = tf.nn.conv1d_transpose(real_part, self.W_real, out_shape, self.strides, padding=self.padding.upper())
        real_conv_imag = tf.nn.conv1d_transpose(real_part, self.W_imag, out_shape, self.strides, padding=self.padding.upper())
        imag_conv_real = tf.nn.conv1d_transpose(imag_part, self.W_real, out_shape, self.strides, padding=self.padding.upper())
        imag_conv_imag = tf.nn.conv1d_transpose(imag_part, self.W_imag, out_shape, self.strides, padding=self.padding.upper())
        
        real_output = real_conv_real - imag_conv_imag
        imag_output = real_conv_imag + imag_conv_real
        
        return tf.complex(real_output, imag_output)
