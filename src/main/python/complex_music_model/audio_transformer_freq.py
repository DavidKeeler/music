import tensorflow as tf
from tensorflow.keras.layers import Layer
import warnings
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from complex_layers import (
    ComplexDense,
    ComplexConv1D,
    ComplexConv1DTranspose,
    ComplexLayerNorm,
    ComplexMultiHeadAttention,
    ComplexTransformerBlock,
    complex_modrelu
)

tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

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
    def __init__(self, d_model=256, num_heads=4, num_layers=2, dff=512, n_mels=128):
        super().__init__()
        self.d_model = d_model
        self.n_mels = n_mels

        # Mel encoder (real → d_model real)
        self.mel_proj = tf.keras.layers.Dense(d_model, activation='relu')
        self.mel_conv1 = tf.keras.layers.Conv1D(d_model, 3, padding='same', activation='relu')
        self.mel_norm1 = tf.keras.layers.LayerNormalization()
        self.mel_conv2 = tf.keras.layers.Conv1D(d_model, 3, padding='same', activation='relu')

        # STFT encoder
        self.stft_proj = ComplexDense(d_model, activation=False)

        # Gated fusion
        self.gate_proj = tf.keras.layers.Dense(d_model, activation='sigmoid')

        # Lift to complex
        self.imag_proj = tf.keras.layers.Dense(d_model)
        self.complex_input_proj = ComplexDense(d_model, activation=False)

        # Transformer stack (complex)
        self.transformer_blocks = [
            ComplexTransformerBlock(d_model, num_heads, dff)
            for _ in range(num_layers)
        ]

        # Decoder head (complex → 513 bins)
        self.decoder = ComplexDense(513, activation=False)

        # STFT consistency
        self.consistency_layer = STFTConsistencyLayer(frame_length=1024, frame_step=256)
        
        # Learnable positional encoding scale
        self.pos_scale = self.add_weight(name="pos_scale", shape=(), initializer="ones")

    def call(self, inputs, training=False):
        mel_input, stft_input = inputs
        # Mel encoder: [B, T, n_mels] → [B, T, d_model] real
        x = self.mel_proj(mel_input)
        x = self.mel_conv1(x)
        x = self.mel_norm1(x)
        x = self.mel_conv2(x)

        # STFT encoder: [B, T, 513] → [B, T, d_model] complex
        stft_features = self.stft_proj(stft_input)

        # Lift mel to complex: real → complex(real, imag)
        zr = x
        zi = self.imag_proj(x)
        mel_complex = tf.complex(zr, zi)
        mel_complex = self.complex_input_proj(mel_complex)

        # Gated fusion: α = sigmoid(Dense(mel_features)), z = stft + α * mel
        alpha = self.gate_proj(x)
        alpha = tf.cast(alpha, tf.complex64)
        z = stft_features + alpha * mel_complex

        # Positional encoding
        seq_len = tf.shape(z)[1]
        pos_encoding = self.get_positional_encoding(seq_len, self.d_model)
        z = z + self.pos_scale * pos_encoding[None, :, :]

        # Causal mask
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = 1.0 - mask

        # Transformer stack
        for block in self.transformer_blocks:
            z = block(z, mask=mask, training=training)

        # Decode: complex → residual
        residual = self.decoder(z)
        
        # Constrain residual magnitude
        residual = residual * 0.1
        
        # Add residual to input STFT
        pred_complex = stft_input + residual
        
        # Apply STFT consistency
        pred_complex = self.consistency_layer(pred_complex)

        return pred_complex

    def get_positional_encoding(self, seq_len, d_model, max_len=2048):
        pos = tf.cast(tf.range(max_len), tf.float32)[:, None]
        div_term = tf.exp(
            tf.range(0, d_model, dtype=tf.float32) * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32))
        )
        # Complex positional encoding: e^(i * pos * div_term)
        angles = pos * div_term
        pe_real = tf.cos(angles)
        pe_imag = tf.sin(angles)
        pe = tf.complex(pe_real, pe_imag)
        return pe[:seq_len]

