import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
sys.path.append('/Users/davidkeeler/code/conducting/src/main/python')
from complex_layers.complex_layers import ComplexDense, ComplexTransformerBlock

N_MELS = 128
FRAME_LENGTH = 2048
FRAME_STEP = 512
SAMPLE_RATE = 22050
N_STFT_BINS = FRAME_LENGTH // 2 + 1

class AudioTransformerMel(keras.Model):
    def __init__(self, d_model=256, num_heads=4, num_layers=2, dff=512):
        super().__init__()
        self.d_model = d_model
        
        self.mel_proj = layers.Dense(d_model, activation='relu')
        self.mel_encoder = keras.Sequential([
            layers.Conv1D(d_model, 3, padding='same', activation='relu'),
            layers.LayerNormalization(),
            layers.Conv1D(d_model, 3, padding='same', activation='relu'),
        ])
        
        self.imag_proj = layers.Dense(d_model)
        
        self.complex_input_proj = ComplexDense(d_model, activation=False)
        self.transformer_blocks = [ComplexTransformerBlock(d_model, num_heads, dff) for _ in range(num_layers)]
        
        self.dec_real = layers.Dense(N_STFT_BINS)
        self.dec_imag = layers.Dense(N_STFT_BINS)
        
    def call(self, mel_input, training=False):
        x = self.mel_proj(mel_input)
        x = self.mel_encoder(x)
        
        zr = x
        zi = self.imag_proj(x)
        z = tf.complex(zr, zi)
        
        z = self.complex_input_proj(z)
        
        seq_len = tf.shape(z)[1]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = 1.0 - mask
        
        for block in self.transformer_blocks:
            z = block(z, mask=mask, training=training)
        
        zr, zi = tf.math.real(z), tf.math.imag(z)
        out_real = self.dec_real(zr)
        out_imag = self.dec_imag(zi)
        pred_complex = tf.complex(out_real, out_imag)
        
        return pred_complex

def audio_to_mel(waveform):
    stft = tf.signal.stft(waveform, FRAME_LENGTH, FRAME_STEP, pad_end=True)
    magnitude = tf.abs(stft)
    
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=N_STFT_BINS,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=8000.0
    )
    
    mel = tf.matmul(magnitude, mel_matrix)
    mel = tf.math.log(mel + 1e-6)
    return mel

def stft_to_audio(stft_frames):
    return tf.signal.inverse_stft(stft_frames, FRAME_LENGTH, FRAME_STEP)
