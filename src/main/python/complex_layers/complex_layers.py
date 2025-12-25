import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization
import warnings

# Suppress TensorFlow complex casting warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

def spectral_norm_complex(w, u=None, num_iters=1, eps=1e-12):
    """
    Complex spectral normalization (power iteration).
    w: complex tensor with shape (in, out)
    returns normalized w and updated u (complex).
    """
    w_shape = tf.shape(w)
    # Reshape to matrix (in, out) then transpose to (out, in)
    W_mat = tf.reshape(w, [w_shape[0], w_shape[1]])  # (in, out)
    W_mat = tf.transpose(W_mat)  # (out, in)

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

    # Normalize the matrix and reshape back
    W_norm_mat = W_mat / tf.cast(sigma, W_mat.dtype)
    W_norm_mat = tf.transpose(W_norm_mat)  # back to (in, out)
    W_norm = tf.reshape(W_norm_mat, tf.shape(w))
    return W_norm, u

def complex_modrelu(z, bias):
    # z: complex tensor, bias: real tensor broadcastable to channels
    mag = tf.abs(z)
    phase = tf.math.angle(z)
    # bias must match z's real dtype
    b = tf.cast(bias, z.dtype.real_dtype)
    activated = tf.nn.relu(mag + b)  # modReLU: relu(mag + b)
    # avoid divide-by-zero: use z / (mag + eps) * activated
    eps = 1e-8
    scale = activated / (mag + eps)
    return tf.complex(scale, tf.zeros_like(scale)) * z

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

class ComplexMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Linear projections for complex Q, K, V
        self.wq = ComplexDense(d_model, spectral_norm=True)
        self.wk = ComplexDense(d_model, spectral_norm=True)
        self.wv = ComplexDense(d_model, spectral_norm=True)

        # Output projection
        self.dense = ComplexDense(d_model, spectral_norm=True)

    def split_heads(self, x):
        # x: (B, T, D) -> (B, H, T, D/H)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, [0, 2, 1, 3])  # (B, H, T, D/H)

    def call(self, v, k, q, mask=None, window_size=64):
        # Apply linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Complex Hermitian dot product
        scores = tf.matmul(q, tf.linalg.adjoint(k))  # (B,H,T,T), complex
        scores /= tf.cast(tf.sqrt(tf.cast(self.depth, tf.float32)), tf.complex64)

        # Split magnitude and phase
        mag = tf.abs(scores)
        phase = tf.math.angle(scores)

        # Apply local window mask
        seq_len = tf.shape(mag)[-1]
        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]
        local_mask = tf.cast((i >= j) & ((i - j) < window_size), tf.float32)
        mag += (1.0 - local_mask)[None, None, :, :] * -1e9

        # Optional user-provided mask
        if mask is not None:
            if len(mask.shape) == 2:  # (T,T)
                mask = mask[None, :, :]
            mag += mask[:, None, :, :] * -1e9

        # Softmax on magnitude
        att_weights = tf.nn.softmax(mag, axis=-1)

        # Reapply phase: att = |softmax| * exp(i*phase)
        att_weights = tf.complex(att_weights * tf.cos(phase), att_weights * tf.sin(phase))

        # Multiply by V
        output = tf.matmul(att_weights, v)  # (B,H,T,D/H)

        # Merge heads
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [tf.shape(output)[0], -1, self.d_model])

        # Output projection
        return self.dense(output)


class ComplexLayerNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zeros', trainable=True)

    def call(self, x):
        # Proper complex layer norm using covariance matrix
        # Following Trabelsi et al. (Deep Complex Networks)
        
        # Center the data
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        x_centered = x - mean
        
        # Compute covariance components
        # Vrr = E[x_r * x_r], Vii = E[x_i * x_i], Vri = E[x_r * x_i]
        xr = tf.math.real(x_centered)
        xi = tf.math.imag(x_centered)
        
        Vrr = tf.reduce_mean(xr * xr, axis=-1, keepdims=True)
        Vii = tf.reduce_mean(xi * xi, axis=-1, keepdims=True)
        Vri = tf.reduce_mean(xr * xi, axis=-1, keepdims=True)
        
        # Compute normalization using 2x2 covariance matrix inverse square root
        # det = Vrr * Vii - Vri^2
        det = Vrr * Vii - Vri * Vri
        det = tf.maximum(det, self.epsilon)
        
        # Inverse square root of covariance matrix
        s = tf.sqrt(det)
        t = tf.sqrt(Vrr + Vii + 2 * s)
        inverse_st = 1.0 / (t + self.epsilon)
        
        # Whitening transformation
        Wrr = (Vii + s) * inverse_st
        Wii = (Vrr + s) * inverse_st
        Wri = -Vri * inverse_st
        
        # Apply whitening
        norm_r = Wrr * xr + Wri * xi
        norm_i = Wri * xr + Wii * xi
        
        # Scale and shift
        out = tf.complex(norm_r * self.gamma + self.beta,
                         norm_i * self.gamma)
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


import tensorflow as tf


class ComplexGRU(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        d = input_shape[-1]
        init = 'glorot_uniform'

        def w(name):
            return self.add_weight(
                name=name, shape=(d, self.units),
                initializer=init, trainable=True
            )

        def u(name):
            return self.add_weight(
                name=name, shape=(self.units, self.units),
                initializer=init, trainable=True
            )

        def b(name):
            return self.add_weight(
                name=name, shape=(self.units,),
                initializer='zeros', trainable=True
            )

        # Update gate
        self.Wz_r, self.Wz_i = w("Wz_r"), w("Wz_i")
        self.Uz_r, self.Uz_i = u("Uz_r"), u("Uz_i")
        self.bz_r = b("bz_r")
        self.bz_i = b("bz_i")

        # Reset gate
        self.Wr_r, self.Wr_i = w("Wr_r"), w("Wr_i")
        self.Ur_r, self.Ur_i = u("Ur_r"), u("Ur_i")
        self.br_r = b("br_r")
        self.br_i = b("br_i")

        # Candidate
        self.Wh_r, self.Wh_i = w("Wh_r"), w("Wh_i")
        self.Uh_r, self.Uh_i = u("Uh_r"), u("Uh_i")
        self.bh_r = b("bh_r")
        self.bh_i = b("bh_i")

    def call(self, inputs):
        xr = tf.math.real(inputs)
        xi = tf.math.imag(inputs)

        batch = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        hr = tf.zeros((batch, self.units), dtype=tf.float32)
        hi = tf.zeros((batch, self.units), dtype=tf.float32)

        ta = tf.TensorArray(tf.complex64, size=seq_len)

        def step(t, hr, hi, ta):
            xr_t = xr[:, t, :]
            xi_t = xi[:, t, :]

            # --- Update gate z ---
            z_r = xr_t @ self.Wz_r - xi_t @ self.Wz_i + hr @ self.Uz_r - hi @ self.Uz_i + self.bz_r
            z_i = xr_t @ self.Wz_i + xi_t @ self.Wz_r + hr @ self.Uz_i + hi @ self.Uz_r + self.bz_i

            # Magnitude-based sigmoid: σ(|z|) * z / |z|
            z_mag = tf.sqrt(z_r**2 + z_i**2 + 1e-8)
            z_gate = tf.nn.sigmoid(z_mag)
            z_r = z_gate * z_r / z_mag
            z_i = z_gate * z_i / z_mag

            # --- Reset gate r ---
            r_r = xr_t @ self.Wr_r - xi_t @ self.Wr_i + hr @ self.Ur_r - hi @ self.Ur_i + self.br_r
            r_i = xr_t @ self.Wr_i + xi_t @ self.Wr_r + hr @ self.Ur_i + hi @ self.Ur_r + self.br_i

            r_mag = tf.sqrt(r_r**2 + r_i**2 + 1e-8)
            r_gate = tf.nn.sigmoid(r_mag)
            r_r = r_gate * r_r / r_mag
            r_i = r_gate * r_i / r_mag

            # Apply reset gate (complex multiplication)
            rh_r = r_r * hr - r_i * hi
            rh_i = r_r * hi + r_i * hr

            # --- Candidate ---
            h_r = xr_t @ self.Wh_r - xi_t @ self.Wh_i + rh_r @ self.Uh_r - rh_i @ self.Uh_i + self.bh_r
            h_i = xr_t @ self.Wh_i + xi_t @ self.Wh_r + rh_r @ self.Uh_i + rh_i @ self.Uh_r + self.bh_i

            # Magnitude-based tanh
            h_mag = tf.sqrt(h_r**2 + h_i**2 + 1e-8)
            h_act = tf.tanh(h_mag)
            h_r = h_act * h_r / h_mag
            h_i = h_act * h_i / h_mag

            # --- Final update (complex multiplication) ---
            # (1-z) * h_old + z * h_new
            hr_new = (1 - z_r) * hr - (-z_i) * hi + z_r * h_r - z_i * h_i
            hi_new = (1 - z_r) * hi + (-z_i) * hr + z_r * h_i + z_i * h_r

            ta = ta.write(t, tf.complex(hr_new, hi_new))
            return t + 1, hr_new, hi_new, ta

        _, _, _, ta = tf.while_loop(
            cond=lambda t, *_: t < seq_len,
            body=step,
            loop_vars=[0, hr, hi, ta]
        )

        out = ta.stack()
        return tf.transpose(out, [1, 0, 2])

