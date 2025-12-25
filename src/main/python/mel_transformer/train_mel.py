import tensorflow as tf
import numpy as np
import soundfile as sf
import os
from datetime import datetime
from audio_transformer_mel import MelTransformer, audio_to_mel, N_MELS, FRAME_STEP, SAMPLE_RATE

def load_audio(file_path, sr=SAMPLE_RATE):
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

def make_batches(audio, seq_len=64, batch_size=8):
    mel = audio_to_mel(audio)
    
    # Create waveform chunks aligned with mel frames
    num_frames = tf.shape(mel)[0]
    waveform_chunks = []
    for i in range(num_frames):
        start = i * FRAME_STEP
        chunk = audio[start:start + FRAME_STEP]
        if len(chunk) < FRAME_STEP:
            chunk = tf.pad(chunk, [[0, FRAME_STEP - len(chunk)]])
        waveform_chunks.append(chunk)
    waveform_chunks = tf.stack(waveform_chunks)
    
    x_data = []
    y_data = []
    for i in range(len(mel) - seq_len - 1):
        x_data.append(mel[i:i + seq_len])
        y_data.append(waveform_chunks[i + 1:i + seq_len + 1])
    
    x_data = tf.stack(x_data)
    y_data = tf.stack(y_data)
    
    ds = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    return ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def multi_resolution_stft_loss(y_true, y_pred, fft_sizes=(2048, 1024, 512), 
                               hop_sizes=(256, 128, 64), eps=1e-7):
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    
    sc_losses = []
    mag_losses = []
    
    for fft, hop in zip(fft_sizes, hop_sizes):
        S_true = tf.signal.stft(y_true_flat, frame_length=fft, frame_step=hop,
                                fft_length=fft, window_fn=tf.signal.hann_window, pad_end=True)
        S_pred = tf.signal.stft(y_pred_flat, frame_length=fft, frame_step=hop,
                                fft_length=fft, window_fn=tf.signal.hann_window, pad_end=True)
        
        mag_true = tf.abs(S_true) + eps
        mag_pred = tf.abs(S_pred) + eps
        
        num = tf.sqrt(tf.reduce_sum((mag_true - mag_pred) ** 2, axis=[-2, -1]))
        den = tf.sqrt(tf.reduce_sum(mag_true ** 2, axis=[-2, -1])) + eps
        sc_losses.append(tf.reduce_mean(num / den))
        
        log_diff = tf.abs(tf.math.log1p(mag_true) - tf.math.log1p(mag_pred))
        mag_losses.append(tf.reduce_mean(log_diff))
    
    sc_loss = tf.add_n(sc_losses) / len(sc_losses)
    mag_loss = tf.add_n(mag_losses) / len(mag_losses)
    
    return sc_loss + mag_loss, sc_loss, mag_loss

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
            loss, sc_loss, mag_loss = multi_resolution_stft_loss(y, pred)
        
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        grad_norm = tf.sqrt(tf.add_n([tf.reduce_sum(tf.square(g)) for g in grads if g is not None]))
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {
            "0_loss": loss,
            "1_sc": sc_loss,
            "2_mag": mag_loss,
            "3_lr": self.optimizer.learning_rate,
            "4_grad_norm": grad_norm
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
        warmup_lr = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))
        progress = (step - self.warmup_steps) / tf.maximum(1.0, self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

def train():
    input_file = "/Users/davidkeeler/data/music/musicnet/test_data/2416.wav"
    checkpoint_dir = "/Users/davidkeeler/models/conducting/mel_checkpoints/"
    output_dir = "/Users/davidkeeler/data/music/model_out"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Training Started: {datetime.now()} ===")

    audio = load_audio(input_file)
    print(f"Audio shape: {audio.shape}")

    base_model = MelTransformer()
    model = TrainingModel(base_model)

    stages = [(64, 20), (128, 20), (256, 20)]
    checkpoint_path = os.path.join(checkpoint_dir, 'mel_weights.weights.h5')

    if os.path.exists(checkpoint_path):
        try:
            dummy = tf.zeros((1, 64, N_MELS))
            model.base_model(dummy)
            model.base_model.load_weights(checkpoint_path)
            print(f"Loaded weights from: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load weights: {e}")

    for stage, (seq_len, epochs) in enumerate(stages):
        print(f"\n--- Stage {stage+1} | seq_len={seq_len} | epochs={epochs} ---")

        dataset = make_batches(audio, seq_len, batch_size=8)
        steps_per_epoch = len(list(dataset))
        total_steps = steps_per_epoch * epochs

        lr_schedule = WarmupCosineSchedule(
            base_lr=1e-4,
            min_lr=1e-6,
            warmup_steps=steps_per_epoch,
            total_steps=total_steps,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer)
        
        model.fit(dataset, epochs=epochs, verbose=1)
        model.base_model.save_weights(checkpoint_path)
        print(f"Saved weights to {checkpoint_path}")

    print("=== Training Complete ===")

    # Generate output
    mel = audio_to_mel(audio)
    init = mel[:64][tf.newaxis, ...]
    generated_chunks = []
    
    for i in range(200):
        pred = model(init, training=False)
        generated_chunks.append(pred[0, -1, :])
        
        next_mel = mel[64 + i:64 + i + 1]
        if len(next_mel) > 0:
            init = tf.concat([init[:, 1:, :], next_mel[tf.newaxis, ...]], axis=1)
    
    out_audio = tf.concat(generated_chunks, axis=0).numpy().flatten()
    sf.write(os.path.join(output_dir, "mel_generated.wav"), out_audio, SAMPLE_RATE)
    print("Saved generated audio output.")

if __name__ == "__main__":
    train()
