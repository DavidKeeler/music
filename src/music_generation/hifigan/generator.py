# -*- coding: utf-8 -*-
"""HiFi-GAN Generator (extracted from TensorFlowTTS)."""

import tensorflow as tf
from .layers import TFReflectionPad1d, TFConvTranspose1d, WeightNormalization


class TFHifiResBlock(tf.keras.layers.Layer):
    """Tensorflow Hifigan resblock module."""

    def __init__(
        self,
        kernel_size,
        filters,
        dilation_rate,
        use_bias,
        nonlinear_activation,
        nonlinear_activation_params,
        is_weight_norm,
        initializer_seed,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.blocks_1 = []
        self.blocks_2 = []

        for i in range(len(dilation_rate)):
            self.blocks_1.append(
                [
                    TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate[i]),
                    tf.keras.layers.Conv1D(
                        filters=filters,
                        kernel_size=kernel_size,
                        dilation_rate=dilation_rate[i],
                        use_bias=use_bias,
                    ),
                ]
            )
            self.blocks_2.append(
                [
                    TFReflectionPad1d((kernel_size - 1) // 2 * 1),
                    tf.keras.layers.Conv1D(
                        filters=filters,
                        kernel_size=kernel_size,
                        dilation_rate=1,
                        use_bias=use_bias,
                    ),
                ]
            )

        self.activation = getattr(tf.keras.layers, nonlinear_activation)(
            **nonlinear_activation_params
        )

        if is_weight_norm:
            self._apply_weightnorm(self.blocks_1)
            self._apply_weightnorm(self.blocks_2)

    def call(self, x, training=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T, C).
        """
        for c1, c2 in zip(self.blocks_1, self.blocks_2):
            xt = self.activation(x)
            for c in c1:
                xt = c(xt)
            xt = self.activation(xt)
            for c in c2:
                xt = c(xt)
            x = xt + x
        return x

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            for j in range(len(list_layers[i])):
                try:
                    layer = list_layers[i][j]
                    layer_name = layer.name.lower()
                    if "conv1d" in layer_name or "dense" in layer_name:
                        list_layers[i][j] = WeightNormalization(layer)
                except Exception:
                    pass


class TFMultiHifiResBlock(tf.keras.layers.Layer):
    """Tensorflow Multi Hifigan resblock module."""

    def __init__(self, list_resblock, **kwargs):
        super().__init__(**kwargs)
        self.list_resblock = list_resblock

    def call(self, x, training=False):
        xs = None
        for resblock in self.list_resblock:
            if xs is None:
                xs = resblock(x, training=training)
            else:
                xs += resblock(x, training=training)
        return xs / len(self.list_resblock)


class TFHifiGANGenerator(tf.keras.Model):
    """HiFi-GAN Generator."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        # Check hyper parameter is valid
        assert (
            config.stacks
            == len(config.stack_kernel_size)
            == len(config.stack_dilation_rate)
        )

        # Build layers
        layers = []
        layers += [
            TFReflectionPad1d(
                (config.kernel_size - 1) // 2,
                padding_type=config.padding_type,
                name="first_reflect_padding",
            ),
            tf.keras.layers.Conv1D(
                filters=config.filters,
                kernel_size=config.kernel_size,
                use_bias=config.use_bias,
            ),
        ]

        for i, upsample_scale in enumerate(config.upsample_scales):
            # Add upsampling layer
            layers += [
                getattr(tf.keras.layers, config.nonlinear_activation)(
                    **config.nonlinear_activation_params
                ),
                TFConvTranspose1d(
                    filters=config.filters // (2 ** (i + 1)),
                    kernel_size=upsample_scale * 2,
                    strides=upsample_scale,
                    padding="same",
                    is_weight_norm=config.is_weight_norm,
                    initializer_seed=config.initializer_seed,
                    name="conv_transpose_._{}".format(i),
                ),
            ]

            # Add residual stack layer
            layers += [
                TFMultiHifiResBlock(
                    list_resblock=[
                        TFHifiResBlock(
                            kernel_size=config.stack_kernel_size[j],
                            filters=config.filters // (2 ** (i + 1)),
                            dilation_rate=config.stack_dilation_rate[j],
                            use_bias=config.use_bias,
                            nonlinear_activation=config.nonlinear_activation,
                            nonlinear_activation_params=config.nonlinear_activation_params,
                            is_weight_norm=config.is_weight_norm,
                            initializer_seed=config.initializer_seed,
                            name="hifigan_resblock_._{}".format(j),
                        )
                        for j in range(config.stacks)
                    ],
                    name="multi_hifigan_resblock_._{}".format(i),
                )
            ]
        
        # Add final layer
        layers += [
            getattr(tf.keras.layers, config.nonlinear_activation)(
                **config.nonlinear_activation_params
            ),
            TFReflectionPad1d(
                (config.kernel_size - 1) // 2,
                padding_type=config.padding_type,
                name="last_reflect_padding",
            ),
            tf.keras.layers.Conv1D(
                filters=config.out_channels,
                kernel_size=config.kernel_size,
                use_bias=config.use_bias,
                dtype=tf.float32,
            ),
        ]
        
        if config.use_final_nolinear_activation:
            layers += [tf.keras.layers.Activation("tanh", dtype=tf.float32)]

        if config.is_weight_norm is True:
            self._apply_weightnorm(layers)

        self.hifigan = tf.keras.models.Sequential(layers)

    def call(self, mels, **kwargs):
        """Calculate forward propagation.
        Args:
            mels (Tensor): Input tensor (B, T, 80)
        Returns:
            Tensor: Output tensor (B, T * prod(upsample_scales), 1)
        """
        return self.inference(mels)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, 80], dtype=tf.float32, name="mels")
        ]
    )
    def inference(self, mels):
        return self.hifigan(mels)

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass

    def _build(self):
        """Build model by passing fake input."""
        fake_mels = tf.random.uniform(shape=[1, 100, 80], dtype=tf.float32)
        self(fake_mels)
