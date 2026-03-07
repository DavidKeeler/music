# -*- coding: utf-8 -*-
"""Helper layers for HiFi-GAN (extracted from TensorFlowTTS)."""

import tensorflow as tf
from tensorflow.python.keras import initializers


def get_initializer(initializer_seed=42):
    """Creates a `tf.initializers.glorot_normal` with the given seed."""
    return tf.keras.initializers.GlorotNormal(seed=initializer_seed)


class TFReflectionPad1d(tf.keras.layers.Layer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size, padding_type="REFLECT", **kwargs):
        super().__init__(**kwargs)
        self.padding_size = padding_size
        self.padding_type = padding_type

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, C).
        """
        return tf.pad(
            x,
            [[0, 0], [self.padding_size, self.padding_size], [0, 0]],
            self.padding_type,
        )


class TFConvTranspose1d(tf.keras.layers.Layer):
    """Tensorflow ConvTranspose1d module."""

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        is_weight_norm,
        initializer_seed,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding="same",
            kernel_initializer=get_initializer(initializer_seed),
        )
        if is_weight_norm:
            self.conv1d_transpose = WeightNormalization(self.conv1d_transpose)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T', C').
        """
        x = tf.expand_dims(x, 2)
        x = self.conv1d_transpose(x)
        x = tf.squeeze(x, 2)
        return x


class WeightNormalization(tf.keras.layers.Wrapper):
    """Layer wrapper to decouple magnitude and direction of the layer's weights.
    
    Extracted from TensorFlowTTS/tensorflow_tts/utils/weight_norm.py
    """

    def __init__(self, layer, data_init=True, **kwargs):
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                "Please initialize `WeightNorm` layer with a `tf.keras.layers.Layer` instance."
            )

        super().__init__(layer, **kwargs)
        self.data_init = data_init
        self._track_trackable(layer, name="layer")
        
        layer_type = type(layer).__name__
        self.filter_axis = -2 if layer_type == "Conv2DTranspose" else -1

    def _compute_weights(self):
        """Generate weights with normalization."""
        new_axis = -self.filter_axis - 3
        self.layer.kernel = tf.nn.l2_normalize(
            self.v, axis=self.kernel_norm_axes
        ) * tf.expand_dims(self.g, new_axis)

    def _init_norm(self):
        """Set the norm of the weight vector."""
        kernel_norm = tf.sqrt(
            tf.reduce_sum(tf.square(self.v), axis=self.kernel_norm_axes)
        )
        self.g.assign(kernel_norm)

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""
        self._compute_weights()
        activation = self.layer.activation
        self.layer.activation = None

        use_bias = self.layer.bias is not None
        if use_bias:
            bias = self.layer.bias
            self.layer.bias = tf.zeros_like(bias)

        x_init = self.layer(inputs)
        norm_axes_out = list(range(x_init.shape.rank - 1))
        m_init, v_init = tf.nn.moments(x_init, norm_axes_out)
        scale_init = 1.0 / tf.sqrt(v_init + 1e-10)

        self.g.assign(self.g * scale_init)
        if use_bias:
            self.layer.bias = bias
            self.layer.bias.assign(-m_init * scale_init)
        self.layer.activation = activation

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, "kernel"):
                raise ValueError(
                    "`WeightNorm` must wrap a layer that contains a `kernel` for weights"
                )

            self.kernel_norm_axes = list(range(self.layer.kernel.shape.ndims))
            self.kernel_norm_axes.pop(self.filter_axis)

            self.v = self.layer.kernel
            self.layer.kernel = None
            
            self.g = self.add_weight(
                name="g",
                shape=(int(self.v.shape[self.filter_axis]),),
                initializer="ones",
                dtype=self.v.dtype,
                trainable=True,
            )
            self.initialized = self.add_weight(
                name="initialized", dtype=tf.bool, trainable=False
            )
            self.initialized.assign(False)

        super().build()

    def call(self, inputs):
        if not self.initialized.numpy():
            if self.data_init:
                self._data_dep_init(inputs)
            else:
                self._init_norm()
            self.initialized.assign(True)

        self._compute_weights()
        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = self.layer.compute_output_shape(input_shape)
        if isinstance(output_shape, tf.TensorShape):
            return output_shape
        return tf.TensorShape(output_shape)


# Simplified GroupConv1D - just use regular Conv1D with groups parameter
# TensorFlow 2.13+ supports groups natively
GroupConv1D = tf.keras.layers.Conv1D
