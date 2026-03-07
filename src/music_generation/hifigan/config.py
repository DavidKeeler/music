# -*- coding: utf-8 -*-
"""HiFi-GAN configuration."""


class HiFiGANConfig:
    """Configuration for HiFi-GAN vocoder."""

    def __init__(self, **kwargs):
        # Generator architecture
        self.filters = kwargs.get("filters", 512)
        self.kernel_size = kwargs.get("kernel_size", 7)
        self.upsample_scales = kwargs.get("upsample_scales", [8, 8, 2, 2])
        self.stacks = kwargs.get("stacks", 3)
        self.stack_kernel_size = kwargs.get("stack_kernel_size", [3, 7, 11])
        self.stack_dilation_rate = kwargs.get(
            "stack_dilation_rate", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        )
        
        # Output
        self.out_channels = kwargs.get("out_channels", 1)
        self.use_final_nolinear_activation = kwargs.get(
            "use_final_nolinear_activation", True
        )
        
        # Activation
        self.nonlinear_activation = kwargs.get("nonlinear_activation", "LeakyReLU")
        self.nonlinear_activation_params = kwargs.get(
            "nonlinear_activation_params", {"alpha": 0.2}
        )
        
        # Normalization
        self.is_weight_norm = kwargs.get("is_weight_norm", True)
        self.use_bias = kwargs.get("use_bias", True)
        self.padding_type = kwargs.get("padding_type", "REFLECT")
        
        # Initialization
        self.initializer_seed = kwargs.get("initializer_seed", 42)


def get_default_config():
    """Get default HiFi-GAN configuration for 22kHz audio."""
    return HiFiGANConfig(
        filters=512,
        kernel_size=7,
        upsample_scales=[8, 8, 2, 2],  # Total upsampling: 256x
        stacks=3,
        stack_kernel_size=[3, 7, 11],
        stack_dilation_rate=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        out_channels=1,
        use_final_nolinear_activation=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"alpha": 0.2},
        is_weight_norm=False,  # Disabled for now - causes issues in TF 2.13+
        use_bias=True,
        padding_type="REFLECT",
        initializer_seed=42,
    )
