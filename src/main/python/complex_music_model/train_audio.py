import tensorflow as tf
import numpy as np
import soundfile as sf
import os
from datetime import datetime
from audio_transformer_freq import AudioTransformerFreq

FRAME_LENGTH = 1024
FRAME_STEP = 256
N_MELS = 128
SAMPLE_RATE = 22050

def audio_to_mel(waveform):
    stft = tf.signal.stft(waveform, FRAME_LENGTH, FRAME_STEP, pad_end=True)
    magnitude = tf.abs(stft)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=FRAME_LENGTH // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20.0,
        upper_edge_hertz=8000.0
    )
    mel = tf.matmul(magnitude, mel_matrix)
    return tf.math.log(mel + 1e-6)

def audio_to_stft(audio):
    return tf.signal.stft(audio, FRAME_LENGTH, FRAME_STEP, window_fn=tf.signal.hann_window)

def stft_to_audio(stft):
    return tf.math.real(tf.signal.inverse_stft(stft, FRAME_LENGTH, FRAME_STEP, window_fn=tf.signal.hann_window))

def complex_stft_loss(y_true, y_pred, eps=1e-8):
    """
    Fully stabilized complex STFT loss.
    Removes all NaN/Inf sources:
      - angle() on near-zero magnitudes
      - correlation denominator collapse
      - IF weight normalization
    """

    # -----------------------------------------------------------
    # 0. Safe complex jitter to avoid undefined angle(0+0j)
    # -----------------------------------------------------------
    jitter = eps * 1e2   # tiny, but enough to avoid zero magnitude
    noise_real = tf.random.normal(tf.shape(y_pred), stddev=jitter)
    noise_imag = tf.random.normal(tf.shape(y_pred), stddev=jitter)

    y_true_safe = y_true + tf.complex(noise_real, noise_imag)
    y_pred_safe = y_pred + tf.complex(noise_real, noise_imag)  # same noise OK

    mag_true = tf.abs(y_true_safe)
    mag_pred = tf.abs(y_pred_safe)

    # -----------------------------------------------------------
    # 1. Magnitude (log-L2)
    # -----------------------------------------------------------
    mag_loss = tf.reduce_mean(
        tf.square(tf.math.log1p(mag_true) - tf.math.log1p(mag_pred))
    )

    # -----------------------------------------------------------
    # 2a. Stable global cosine phase loss
    # -----------------------------------------------------------
    conj_prod = y_true_safe * tf.math.conj(y_pred_safe)
    phase_loss_cos = 1.0 - tf.reduce_mean(
        tf.math.cos(tf.math.angle(conj_prod))
    )

    # -----------------------------------------------------------
    # 2b. Wrapped local phase difference
    # -----------------------------------------------------------
    phase_true = tf.math.angle(y_true_safe)
    phase_pred = tf.math.angle(y_pred_safe)

    phase_diff = tf.atan2(
        tf.sin(phase_true - phase_pred),
        tf.cos(phase_true - phase_pred)
    )

    phase_loss_wrapped = tf.reduce_mean(tf.square(phase_diff))

    phase_loss = 0.5 * phase_loss_cos + 0.5 * phase_loss_wrapped

    # -----------------------------------------------------------
    # 3. IF loss (wrapped + stable weighting)
    # -----------------------------------------------------------
    d_true = tf.math.angle(
        y_true_safe[:, 1:, :] * tf.math.conj(y_true_safe[:, :-1, :])
    )
    d_pred = tf.math.angle(
        y_pred_safe[:, 1:, :] * tf.math.conj(y_pred_safe[:, :-1, :])
    )

    d_diff = tf.atan2(tf.sin(d_true - d_pred), tf.cos(d_true - d_pred))

    w = tf.minimum(mag_true[:, 1:, :], mag_true[:, :-1, :])

    # Safe normalization: only normalize if max(w) > threshold
    max_w = tf.reduce_max(w, axis=[1, 2], keepdims=True)
    safe_norm = tf.where(max_w > 1e-6, w / max_w, tf.zeros_like(w))

    if_loss = (
        tf.reduce_sum(safe_norm * tf.square(d_diff)) /
        (tf.reduce_sum(safe_norm) + eps)
    )

    # -----------------------------------------------------------
    # 4. Merged temporal + correlation loss
    # -----------------------------------------------------------
    m1 = mag_pred[:, 1:, :]
    m2 = mag_pred[:, :-1, :]

    delta_mag = m1 - m2

    # centered
    m1m = m1 - tf.reduce_mean(m1, axis=-1, keepdims=True)
    m2m = m2 - tf.reduce_mean(m2, axis=-1, keepdims=True)

    num = tf.reduce_sum(m1m * m2m, axis=-1)

    # Safe denominator: avoid 0 via floor
    den = tf.sqrt(
        tf.reduce_sum(m1m ** 2, axis=-1) *
        tf.reduce_sum(m2m ** 2, axis=-1)
    )
    den = tf.maximum(den, 1e-4)  # prevents blow-up

    corr = num / den  # [-1,1]

    temp_corr_loss = tf.reduce_mean(
        tf.square(delta_mag) * (1.0 - corr[:, :, tf.newaxis])
    )

    # -----------------------------------------------------------
    # Weighted total (your original weights)
    # -----------------------------------------------------------
    total_loss = (
        2.0 * mag_loss +
        30.0 * phase_loss +
        2.0 * if_loss +
        1.0 * temp_corr_loss
    )

    # Absolute safety check (should never trigger now)
    total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, 1.0)

    return total_loss, mag_loss, phase_loss, if_loss, temp_corr_loss

def multi_resolution_stft_loss(
    y_true,
    y_pred,
    fft_sizes=(2048, 1024, 512),
    hop_sizes=(256, 128, 64),
    win_lengths=None,
    mag_loss_power=1,
    eps=1e-7,
    window_fn=tf.signal.hann_window,
    res_weights=None,
):
    """
    Correct MR-STFT loss (HiFi-GAN style) that always compares waveforms
    across multiple STFT resolutions.

    Args:
      y_true, y_pred: either waveforms [B, T] (float) OR complex STFTs [B, F, T?] (complex).
                      If complex STFTs are passed, they are ISTFT'ed to waveforms first.
      fft_sizes, hop_sizes, win_lengths: iterables of ints (same length).
      mag_loss_power: 1 -> L1 on log1p(mag), 2 -> squared
      res_weights: optional list/tuple of same length as fft_sizes for weighting each resolution.
      eps: numeric stability
      window_fn: callable to create window for stft/istft
    Returns:
      total_loss: scalar tensor
      details: dict with per-resolution 'sc_losses' and 'mag_losses' and their means
    """
    if win_lengths is None:
        win_lengths = fft_sizes
    if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths)):
        raise ValueError("fft_sizes, hop_sizes, win_lengths must have same length")

    def _ensure_waveform(x):
        if x.dtype.is_complex:
            frame_length = win_lengths[0]
            frame_step = hop_sizes[0]
            return tf.math.real(tf.signal.inverse_stft(x, frame_length=frame_length,
                                                       frame_step=frame_step, window_fn=window_fn))
        else:
            return x

    y_true_w = _ensure_waveform(y_true)
    y_pred_w = _ensure_waveform(y_pred)

    sc_losses = []
    mag_losses = []
    per_res = []

    if res_weights is None:
        res_weights = [1.0] * len(fft_sizes)
    else:
        if len(res_weights) != len(fft_sizes):
            raise ValueError("res_weights must match number of resolutions")

    for i, (fft, hop, win) in enumerate(zip(fft_sizes, hop_sizes, win_lengths)):
        S_true = tf.signal.stft(y_true_w, frame_length=win, frame_step=hop,
                                fft_length=fft, window_fn=window_fn, pad_end=True)
        S_pred = tf.signal.stft(y_pred_w, frame_length=win, frame_step=hop,
                                fft_length=fft, window_fn=window_fn, pad_end=True)

        mag_true = tf.abs(S_true) + eps
        mag_pred = tf.abs(S_pred) + eps

        num = tf.sqrt(tf.reduce_sum((mag_true - mag_pred) ** 2, axis=[-2, -1]))
        den = tf.sqrt(tf.reduce_sum((mag_true) ** 2, axis=[-2, -1])) + eps
        sc_per_example = num / den
        sc_res = tf.reduce_mean(sc_per_example)

        log_true = tf.math.log1p(mag_true)
        log_pred = tf.math.log1p(mag_pred)
        diff = tf.abs(log_true - log_pred)
        if mag_loss_power == 1:
            mag_res = tf.reduce_mean(diff)
        else:
            mag_res = tf.reduce_mean(diff ** 2)

        w = float(res_weights[i])
        sc_losses.append(sc_res * w)
        mag_losses.append(mag_res * w)
        per_res.append((sc_res, mag_res))

    sc_loss = tf.add_n(sc_losses) / tf.cast(len(sc_losses), tf.float32)
    mag_loss = tf.add_n(mag_losses) / tf.cast(len(mag_losses), tf.float32)

    total_loss = sc_loss + mag_loss

    details = {
        "sc_loss": sc_loss,
        "mag_loss": mag_loss,
        "per_res": per_res
    }
    return total_loss, details


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



def make_batches(mel_data, stft_data, seq_len=64, batch_size=8):
    min_len = min(len(mel_data), len(stft_data))
    
    def generator():
        for i in range(min_len - seq_len - 1):
            yield mel_data[i:i + seq_len], stft_data[i + 1:i + seq_len + 1]
    
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, mel_data.shape[1]), dtype=tf.float32),
            tf.TensorSpec(shape=(seq_len, stft_data.shape[1]), dtype=tf.complex64)
        )
    )
    return ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

class SaveBaseModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
    
    def on_epoch_end(self, epoch, logs=None):
        self.model.base_model.save_weights(self.filepath)
        print(f"Epoch {epoch + 1}: saved base model weights to {self.filepath}")

class TrainingModel(tf.keras.Model):
    def __init__(self, base_model, loss_type="complex_stft", combined_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.loss_type = loss_type
        self.combined_weight = combined_weight
    
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            pred = self.base_model([x, y], training=True)
            
            if self.loss_type == "complex_stft":
                total_loss, mag_loss, phase_loss, if_loss, temp_corr_loss = complex_stft_loss(y, pred)
            elif self.loss_type == "mr_stft":
                total_loss, details = multi_resolution_stft_loss(y, pred)
                mag_loss = details["mag_loss"]
                phase_loss = tf.constant(0.0)
                if_loss = details["sc_loss"]
                temp_corr_loss = tf.constant(0.0)
            else:
                loss1, mag_loss, phase_loss, if_loss, temp_corr_loss = complex_stft_loss(y, pred)
                loss2, details = multi_resolution_stft_loss(y, pred)
                total_loss = self.combined_weight * loss1 + (1 - self.combined_weight) * loss2
        
        grads = tape.gradient(total_loss, self.base_model.trainable_variables)
        grads = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {
            "0-loss": total_loss,
            "mag": mag_loss,
            "phase": phase_loss,
            "if": if_loss,
            "temp": temp_corr_loss,
            "lr": self.optimizer.learning_rate,
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
    LOSS_TYPE = "complex_stft"
    COMBINED_WEIGHT = 0.5
    
    input_file = "/Users/davidkeeler/data/music/musicnet/test_data/2416.wav"
    checkpoint_dir = "/Users/davidkeeler/models/conducting/checkpoints/"
    output_dir = "/Users/davidkeeler/data/music/model_out"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Training Started: {datetime.now()} ===")

    audio, _ = sf.read(input_file)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    audio = audio.astype(np.float32)
    
    mel_data = audio_to_mel(audio)
    stft_data = audio_to_stft(audio)
    print(f"Mel shape: {mel_data.shape}, STFT shape: {stft_data.shape}")

    base_model = AudioTransformerFreq()
    model = TrainingModel(base_model, loss_type=LOSS_TYPE, combined_weight=COMBINED_WEIGHT)
    print(f"Using loss type: {LOSS_TYPE}")

    stages = [(16, 200)]
    
    checkpoint_path = os.path.join(checkpoint_dir, 'autoreg_weights.weights.h5')

    if os.path.exists(checkpoint_path):
        try:
            dummy = tf.zeros((1, 64, N_MELS))
            model.base_model(dummy)
            model.base_model.load_weights(checkpoint_path)
            print(f"Loaded weights from: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load weights: {e}")

    sample = [mel_data[:32][tf.newaxis, ...], stft_data[:32][tf.newaxis, ...]]
    _ = model.base_model(sample)
    print("Model built successfully")
    model.base_model.summary()
    
    for stage, (seq_len, epochs) in enumerate(stages):
        print(f"\n--- Stage {stage+1} | seq_len={seq_len} | epochs={epochs} ---")

        steps_per_epoch = (len(mel_data) - seq_len - 1) // 8
        total_steps = steps_per_epoch * epochs
        print(f"Steps per epoch: {steps_per_epoch}")

        lr_schedule = WarmupCosineSchedule(
            base_lr=1e-5,
            min_lr=5e-6,
            warmup_steps=0, #steps_per_epoch,
            total_steps=total_steps,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer)
        print("Model compiled")
        
        print("Creating dataset...")
        dataset = make_batches(mel_data, stft_data, seq_len, batch_size=8)
        print("Dataset created")
        checkpoint_callback = SaveBaseModelCallback(checkpoint_path)
        print("Starting fit...")
        model.fit(dataset, epochs=epochs, verbose=1, callbacks=[checkpoint_callback])

    print("=== Training Complete ===")
    print("Saved generated audio output.")

if __name__ == "__main__":
    train_curriculum()
