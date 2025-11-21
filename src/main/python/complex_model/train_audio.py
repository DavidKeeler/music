import tensorflow as tf
import numpy as np
import soundfile as sf
import os
import time
from datetime import datetime
from tqdm import tqdm
from audio_transformer_freq import ComplexDense, ComplexConv1D, ComplexConv1DTranspose, ComplexTransformerBlock, AudioTransformerFreq

import tensorflow as tf

# STFT Configuration
FRAME_LENGTH = 1024
FRAME_STEP = 512
MEL_BINS = 128
SAMPLE_RATE = 22050

def stable_complex_stft_loss(y_true, y_pred, eps=1e-5):
    # ----- Magnitude -----
    mag_true = tf.abs(y_true)
    mag_pred = tf.abs(y_pred)

    # Clamp to avoid log NaNs
    mag_true = tf.clip_by_value(mag_true, 0.0, 1e6)
    mag_pred = tf.clip_by_value(mag_pred, 0.0, 1e6)

    # Mel perceptual weights
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=80,
        num_spectrogram_bins=513,
        sample_rate=22050,
        lower_edge_hertz=0.0,
        upper_edge_hertz=11025.0
    )
    mel_weights = tf.reduce_sum(mel_weights, axis=1)
    mel_weights = mel_weights / (tf.reduce_max(mel_weights) + eps)  # [513]

    # Magnitude loss (perceptual log)
    mag_loss = tf.square(tf.math.log1p(mag_true) - tf.math.log1p(mag_pred))
    mag_loss = tf.reduce_mean(mel_weights * mag_loss)

    # ----- Phase (robust cos–sin) -----
    phase_true = tf.math.angle(y_true)
    phase_pred = tf.math.angle(y_pred)

    true_vec = tf.stack([tf.cos(phase_true), tf.sin(phase_true)], axis=-1)
    pred_vec = tf.stack([tf.cos(phase_pred), tf.sin(phase_pred)], axis=-1)

    phase_error = tf.reduce_sum(tf.square(true_vec - pred_vec), axis=-1)

    # Magnitude weighting (stable per-sample)
    mean_mag = tf.reduce_mean(mag_true, axis=[-1], keepdims=True)
    weights = mag_true / (mean_mag + eps)
    weights = tf.clip_by_value(weights, 0.3, 5.0)

    phase_loss = tf.reduce_mean(weights * phase_error)

    # ----- Phase smoothness -----
    delta_true = tf.sin(phase_true[:, 1:, :] - phase_true[:, :-1, :])
    delta_pred = tf.sin(phase_pred[:, 1:, :] - phase_pred[:, :-1, :])
    smooth_loss = tf.reduce_mean(tf.square(delta_true - delta_pred))

    # Only apply small weight
    smooth_loss *= 0.1

    # ----- Total -----
    total = mag_loss + phase_loss + smooth_loss

    # Guard
    total = tf.where(tf.math.is_finite(total), total, 1.0)
    mag_loss = tf.where(tf.math.is_finite(mag_loss), mag_loss, 0.0)
    phase_loss = tf.where(tf.math.is_finite(phase_loss), phase_loss, 0.0)
    smooth_loss = tf.where(tf.math.is_finite(smooth_loss), smooth_loss, 0.0)
    
    return total, mag_loss, phase_loss, smooth_loss

def robust_weighted_wrapped_phase_loss(y_true, y_pred, eps=1e-8, min_weight=0.3, max_weight=5.0):
    """
    Magnitude-weighted phase loss using [cos, sin] representation.

    - Converts phase to unit vectors in 2D (cos, sin)
    - Computes squared Euclidean distance
    - Magnitude-weighted
    """
    phase_true = tf.math.angle(y_true)
    phase_pred = tf.math.angle(y_pred)
    mag_true = tf.abs(y_true)

    # Convert phases to 2D unit vectors
    true_vec = tf.stack([tf.cos(phase_true), tf.sin(phase_true)], axis=-1)
    pred_vec = tf.stack([tf.cos(phase_pred), tf.sin(phase_pred)], axis=-1)

    # Squared Euclidean distance in 2D plane
    phase_error = tf.reduce_sum(tf.square(true_vec - pred_vec), axis=-1)

    # Magnitude-based weights
    weights = mag_true / (tf.reduce_mean(mag_true) + eps)
    weights = tf.clip_by_value(weights, min_weight, max_weight)

    # Weighted loss
    loss = tf.reduce_mean(weights * phase_error)

    # Guard against NaN/Inf
    loss = tf.where(tf.math.is_finite(loss), loss, 0.0)
    return loss


def phase_loss_complex_recon(y_true, y_pred):
    """
    Complex reconstruction loss: measures error in the complex plane.
    """
    mag_true = tf.abs(y_true)
    phase_true = tf.math.angle(y_true)
    phase_pred = tf.math.angle(y_pred)

    pred_complex = tf.cast(mag_true, tf.complex64) * tf.exp(1j * tf.cast(phase_pred, tf.complex64))
    true_complex = tf.cast(mag_true, tf.complex64) * tf.exp(1j * tf.cast(phase_true, tf.complex64))

    return tf.reduce_mean(tf.square(tf.abs(pred_complex - true_complex)))

def phase_smoothness_loss(y_true, y_pred):
    """Phase smoothness loss - encourages smooth phase transitions"""
    phase_true = tf.math.angle(y_true)
    phase_pred = tf.math.angle(y_pred)
    
    delta_true = tf.sin(phase_true[:, 1:, :] - phase_true[:, :-1, :])
    delta_pred = tf.sin(phase_pred[:, 1:, :] - phase_pred[:, :-1, :])
    phase_smooth = tf.reduce_mean(tf.square(delta_true - delta_pred))
    
    return phase_smooth

def complex_music_loss(y_true, y_pred, eps=1e-8):
    # Both are now STFT (513 bins)
    
    # Create mel-scale perceptual weights
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=80,
        num_spectrogram_bins=513,
        sample_rate=22050,
        lower_edge_hertz=0.0,
        upper_edge_hertz=11025.0
    )
    # Sum mel weights per frequency bin to get perceptual importance
    perceptual_weights = tf.reduce_sum(mel_weights, axis=1)  # [513] - sum across mel bins
    perceptual_weights = perceptual_weights / tf.reduce_max(perceptual_weights)  # Normalize
    
    mag_true = tf.abs(y_true)
    mag_pred = tf.abs(y_pred)
    
    # Perceptually weighted magnitude loss
    mag_error = tf.square(tf.math.log1p(mag_true) - tf.math.log1p(mag_pred))
    mag_loss = tf.reduce_mean(perceptual_weights * mag_error)

    # Perceptually weighted phase loss
    wrapped_phase_loss = robust_weighted_wrapped_phase_loss(y_true, y_pred)
    smooth_loss = phase_smoothness_loss(y_true, y_pred)
    complex_loss = phase_loss_complex_recon(y_true, y_pred) * 0.05  # Scale down

    # Guard against NaN/Inf in all losses
    mag_loss = tf.where(tf.math.is_finite(mag_loss), mag_loss, 0.0)
    wrapped_phase_loss = tf.where(tf.math.is_finite(wrapped_phase_loss), wrapped_phase_loss, 0.0)
    smooth_loss = tf.where(tf.math.is_finite(smooth_loss), smooth_loss, 0.0)
    complex_loss = tf.where(tf.math.is_finite(complex_loss), complex_loss, 0.0)

    total_loss = (
        1.0 * mag_loss +
        1.0 * wrapped_phase_loss +
        1.0 * smooth_loss
    )

    total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, 1.0)
    return total_loss, mag_loss, wrapped_phase_loss, smooth_loss, complex_loss

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
    return tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)

def stft_to_audio(stft):
    return tf.math.real(tf.signal.inverse_stft(stft, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP))



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
            total_loss, mag_loss, phase_loss, smooth_loss = stable_complex_stft_loss(y, pred)
        
        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        
        # Check for NaN gradients and clip more aggressively
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) if g is not None else g for g in grads]
        grads = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in grads]
        
        # Calculate gradient norm for monitoring
        grad_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in grads if g is not None]))
        
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {
            "1_loss": total_loss,
            "2_mag": mag_loss,
            "3_phase": phase_loss,
            "4_smooth": smooth_loss,
            "5_lr": self.optimizer.learning_rate,
            "6_grad_norm": grad_norm
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

    teacher_forcing = 0.9

    stages = [(32, 50)] * 20
    
    # Calculate total training steps
    steps_per_epoch = (len(stft_data) - 32 - 1) // 8
    total_steps = steps_per_epoch * sum(epochs for _, epochs in stages)

    lr_schedule = WarmupCosineSchedule(
        base_lr=2e-5,
        min_lr=1e-7,
        warmup_steps= steps_per_epoch * 2,
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
