import tensorflow as tf
import numpy as np
import soundfile as sf
import os
import time
from datetime import datetime
from tqdm import tqdm
from audio_transformer_freq import ComplexDense, ComplexConv1D, ComplexConv1DTranspose, ComplexTransformerBlock, AudioTransformerFreq

import tensorflow as tf

# STFT Configuration - Fixed for smooth phase evolution
FRAME_LENGTH = 1024
FRAME_STEP = 256  # 75% overlap instead of 50%
MEL_BINS = 128
SAMPLE_RATE = 22050

import tensorflow as tf
import numpy as np

def complex_stft_loss(y_true, y_pred, eps=1e-8):
    """
    Lightweight complex STFT loss with stable phase smoothness.

    Args:
        y_true, y_pred: [batch, time, freq_bins], tf.complex64
        eps: small value to avoid division by zero
    Returns:
        total_loss, mag_loss, phase_loss, continuity_loss, smooth_loss
    """
    # -------------------------------
    # 1. Magnitude loss
    # -------------------------------
    mag_true = tf.abs(y_true)
    mag_pred = tf.abs(y_pred)
    mag_loss = tf.reduce_mean(tf.square(tf.math.log1p(mag_true) - tf.math.log1p(mag_pred)))

    # -------------------------------
    # 2. Phase loss (global)
    # -------------------------------
    conj_prod = y_true * tf.math.conj(y_pred)
    phase_loss = 1.0 - tf.reduce_mean(tf.math.cos(tf.math.angle(conj_prod)))

    # -------------------------------
    # 3. Spectral centroid continuity
    # -------------------------------
    freq_bins = tf.range(tf.shape(y_true)[-1], dtype=tf.float32)
    centroid_true = tf.reduce_sum(mag_true * freq_bins[None, None, :], axis=-1) / (
                tf.reduce_sum(mag_true, axis=-1) + eps)
    centroid_pred = tf.reduce_sum(mag_pred * freq_bins[None, None, :], axis=-1) / (
                tf.reduce_sum(mag_pred, axis=-1) + eps)
    continuity_loss = tf.reduce_mean(tf.square((centroid_true[:, 1:] - centroid_true[:, :-1]) -
                                               (centroid_pred[:, 1:] - centroid_pred[:, :-1])))

    # -------------------------------
    # 4. Phase smoothness (stable)
    # -------------------------------
    # Use angle(a * conj(b)) for frame-to-frame difference
    delta_true = tf.math.angle(y_true[:, 1:, :] * tf.math.conj(y_true[:, :-1, :]))
    delta_pred = tf.math.angle(y_pred[:, 1:, :] * tf.math.conj(y_pred[:, :-1, :]))

    # Use magnitude-weighted smoothness instead of hard masking
    mag_weights = tf.minimum(mag_true[:, 1:, :], mag_true[:, :-1, :])
    mag_weights = mag_weights / (tf.reduce_max(mag_weights, axis=-1, keepdims=True) + eps)
    
    # Compute weighted phase difference
    phase_diff = tf.square(delta_true - delta_pred)
    smooth_loss = tf.reduce_sum(mag_weights * phase_diff) / (tf.reduce_sum(mag_weights) + eps)

    # Normalize phase difference by pi to keep loss scale reasonable
    smooth_loss = smooth_loss / (np.pi * np.pi)

    # -------------------------------
    # 5. Weighted sum
    # -------------------------------
    total_loss = mag_loss + 1.0 * phase_loss + 1.0 * continuity_loss + 1.0 * smooth_loss
    total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, 1.0)

    return total_loss, mag_loss, phase_loss, continuity_loss, smooth_loss


def multi_resolution_stft_loss(
    y_true,
    y_pred,
    fft_sizes=(256, 1024, 4096),
    hop_sizes=(64, 256, 1024),
    win_lengths=(256, 1024, 4096),
    phase_weight=0.05,
    env_weight=0.3,
    eps=1e-7
):
    """Multi-resolution perceptual STFT loss for music generation.
       Returns: total_loss, mag_loss, env_loss, phase_loss
    """

    mag_losses = []
    env_losses = []
    phase_losses = []

    for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):

        # Compute STFTs
        S_true = tf.signal.stft(y_true, win, hop, fft, pad_end=True)
        S_pred = tf.signal.stft(y_pred, win, hop, fft, pad_end=True)

        mag_true = tf.abs(S_true) + eps
        mag_pred = tf.abs(S_pred) + eps

        # ----------------------------------------------------
        # 1. Log-magnitude loss (core perceptual term)
        # ----------------------------------------------------
        mag_loss = tf.reduce_mean(
            tf.square(tf.math.log1p(mag_true) - tf.math.log1p(mag_pred))
        )
        mag_losses.append(mag_loss)

        # ----------------------------------------------------
        # 2. Spectral envelope (smooth mel-like structure)
        # ----------------------------------------------------
        # Soft frequency weighting to emphasize low/mid bands
        freqs = tf.linspace(0.0, 1.0, mag_true.shape[-1])
        weight = tf.exp(-3.0 * freqs)[None, None, :]

        env_true = tf.reduce_sum(mag_true * weight, axis=-1)
        env_pred = tf.reduce_sum(mag_pred * weight, axis=-1)

        env_loss = tf.reduce_mean(tf.square(env_true - env_pred))
        env_losses.append(env_loss)

        # ----------------------------------------------------
        # 3. Phase-difference loss (instantaneous frequency)
        # Only applied on small FFT sizes
        # ----------------------------------------------------
        if fft <= 1024 and phase_weight > 0:
            phase_true = tf.math.angle(S_true)
            phase_pred = tf.math.angle(S_pred)

            # frame-to-frame difference: instantaneous frequency proxy
            dphi_true = tf.sin(phase_true[:, 1:, :] - phase_true[:, :-1, :])
            dphi_pred = tf.sin(phase_pred[:, 1:, :] - phase_pred[:, :-1, :])

            # Onset-aware masking: ignore high spectral flux frames
            flux = tf.reduce_mean(tf.abs(mag_true[:, 1:, :] - mag_true[:, :-1, :]), axis=-1)
            mask = 1.0 / (1.0 + 10.0 * flux)   # suppress phase loss on transients
            mask = mask[:, :, None]

            phase_loss = tf.reduce_mean(mask * tf.square(dphi_true - dphi_pred))
            phase_losses.append(phase_loss)

    # Sum over all resolutions
    mag_loss = tf.add_n(mag_losses)
    env_loss = env_weight * tf.add_n(env_losses)
    phase_loss = phase_weight * (tf.add_n(phase_losses) if len(phase_losses) else 0.0)

    total = mag_loss + env_loss + phase_loss
    total = tf.where(tf.math.is_finite(total), total, 1.0)

    return total, mag_loss, env_loss, phase_loss

def load_full_audio(file_path, sr=22050):
    audio, file_sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if file_sr != sr:
        ratio = sr / file_sr
        new_length = int(len(audio) * ratio)
        audio = np.interp(np.linspace(0, len(audio) - 1, new_length),
                          np.arange(len(audio)), audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio.astype(np.float32)

def audio_to_stft(audio):
    return tf.signal.stft(
        audio, 
        frame_length=FRAME_LENGTH, 
        frame_step=FRAME_STEP,
        window_fn=tf.signal.hann_window
    )

def stft_to_audio(stft):
    return tf.math.real(tf.signal.inverse_stft(
        stft, 
        frame_length=FRAME_LENGTH, 
        frame_step=FRAME_STEP,
        window_fn=tf.signal.hann_window
    ))



def make_batches(stft_data, seq_len=64, batch_size=8):
    # Keep original STFT data - model will handle mel conversion internally
    x_data = []
    y_data = []
    for i in range(len(stft_data) - seq_len - 1):
        x_data.append(stft_data[i:i + seq_len])
        y_data.append(stft_data[i + 1:i + seq_len + 1])
    
    x_data = tf.stack(x_data)
    y_data = tf.stack(y_data)
    
    ds = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    return ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

class SaveBaseModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
    
    def on_epoch_end(self, epoch, logs=None):
        self.model.base_model.save_weights(self.filepath)
        print(f"Epoch {epoch + 1}: saved base model weights to {self.filepath}")

class TrainingModel(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
    
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            pred = self.base_model(x, training=True)
            total_loss, mag_loss, phase_loss, continuity_loss, smooth_loss = complex_stft_loss(y, pred)
            
            # Diagnostics for smooth loss
            delta_true = tf.math.angle(y[:, 1:, :] * tf.math.conj(y[:, :-1, :]))
            delta_pred = tf.math.angle(pred[:, 1:, :] * tf.math.conj(pred[:, :-1, :]))
            
            mean_delta_true = tf.reduce_mean(tf.abs(delta_true))
            mean_delta_pred = tf.reduce_mean(tf.abs(delta_pred))
        
        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        
        # Check for NaN gradients and clip more aggressively
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else g for g in grads]
        grads = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in grads]
        
        # Calculate gradient norm for monitoring
        grad_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(g)) for g in grads if g is not None]))
        
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {
            "0_loss": total_loss,
            "1_mag": mag_loss,
            "2_phase": phase_loss,
            "3_continuity": continuity_loss,
            "4_smooth": smooth_loss,
            "5_lr": self.optimizer.learning_rate,
            "6_grad_norm": grad_norm,
            "7_delta_true": mean_delta_true,
            "8_delta_pred": mean_delta_pred
        }

class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=0.0):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # --- Warmup: linear from 0 → base_lr ---
        warmup_lr = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))

        # --- Cosine decay after warmup ---
        progress = (step - self.warmup_steps) / tf.maximum(
            1.0, self.total_steps - self.warmup_steps
        )
        progress = tf.clip_by_value(progress, 0.0, 1.0)

        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + tf.cos(np.pi * progress))

        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }

def train_curriculum():
    input_file = "/Users/davidkeeler/data/music/musicnet/test_data/2416.wav"
    checkpoint_dir = "/Users/davidkeeler/models/conducting/checkpoints/"
    output_dir = "/Users/davidkeeler/data/music/model_out"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Training Started: {datetime.now()} ===")

    audio = load_full_audio(input_file)
    stft_data = audio_to_stft(audio)
    print(f"STFT shape: {stft_data.shape}")

    base_model = AudioTransformerFreq()
    model = TrainingModel(base_model)

    teacher_forcing = 1.0

    stages = [(128, 30)] + [(128, 10)] * 10
    
    # Calculate total training steps
    steps_per_epoch = (len(stft_data) - 32 - 1) // 8
    total_steps = steps_per_epoch * sum(epochs for _, epochs in stages)

    lr_schedule = WarmupCosineSchedule(
        base_lr=3e-5,
        min_lr=1e-5,
        warmup_steps= steps_per_epoch * 1,
        total_steps=total_steps,
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer)
    checkpoint_path = os.path.join(checkpoint_dir, 'autoreg_weights.weights.h5')

    # Load existing weights if available
    if os.path.exists(checkpoint_path):
        try:
            dummy = tf.zeros((1, 64, 513), dtype=tf.complex64)
            model.base_model(dummy)
            model.base_model.load_weights(checkpoint_path)
            print(f"Loaded weights from: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print("No existing checkpoint found")

    # Build and show model summary
    sample = stft_data[:32][tf.newaxis, ...]
    _ = model.base_model(sample)
    model.base_model.summary()
    for stage, (seq_len, epochs) in enumerate(stages):
        print(f"\n--- Stage {stage+1} | seq_len={seq_len} | epochs={epochs} | teacher_forcing={teacher_forcing:.2f} ---")
        
        # Set teacher forcing probability
        model.teacher_forcing_prob = teacher_forcing
        
        dataset = make_batches(stft_data, seq_len, batch_size=8)
        
        # Create custom checkpoint callback for base model
        checkpoint_callback = SaveBaseModelCallback(checkpoint_path)
        
        # Train using Keras fit
        history = model.fit(dataset, epochs=epochs, verbose=1, callbacks=[checkpoint_callback])

        teacher_forcing *= 0.9

    print("=== Training Complete ===")

    # Generate output
    init = stft_data[:64][tf.newaxis, ...]
    seq = [init]
    
    for _ in range(200):
        inp = tf.concat(seq, axis=1)
        output = model(inp, training=False)
        next_frame = output[:, -1:, :]
        seq.append(next_frame)
    
    gen = tf.concat(seq, axis=1)[0]
    out_audio = stft_to_audio(gen).numpy()
    sf.write(os.path.join(output_dir, "autoregressive_generated.wav"), out_audio, 22050)
    print("Saved generated audio output.")

if __name__ == "__main__":
    train_curriculum()
