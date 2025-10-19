import tensorflow as tf
import numpy as np
import soundfile as sf
import os
import time
from audio_transformer_freq import AudioTransformerFreq
from datetime import datetime
from tqdm import tqdm


def get_latest_checkpoint(checkpoint_dir):
    """Find the most recently created checkpoint file"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.weights.h5')]
    if not checkpoint_files:
        return None
    
    # Get full paths and sort by modification time
    full_paths = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
    latest_checkpoint = max(full_paths, key=os.path.getmtime)
    return latest_checkpoint


def magnitude_spectral_loss(y_true, y_pred):
    # Use MSE on magnitudes instead of complex difference
    mag_true = tf.abs(y_true)
    mag_pred = tf.abs(y_pred)
    return tf.reduce_mean(tf.square(mag_true - mag_pred))


def phase_loss(y_true, y_pred):
    """Phase consistency loss using cosine distance"""
    # Get magnitudes and add epsilon for stability
    mag_true = tf.abs(y_true) + 1e-8
    mag_pred = tf.abs(y_pred) + 1e-8
    
    # Normalize to unit circle
    y_true_norm = y_true / tf.cast(mag_true, tf.complex64)
    y_pred_norm = y_pred / tf.cast(mag_pred, tf.complex64)
    
    # Phase difference using normalized complex multiplication
    phase_diff = y_true_norm * tf.math.conj(y_pred_norm)
    
    # Use real part (cosine of phase difference)
    cos_phase_diff = tf.math.real(phase_diff)
    
    # Clamp for stability
    cos_phase_diff = tf.clip_by_value(cos_phase_diff, -0.999, 0.999)
    
    # Phase loss: 1 - mean cosine similarity
    return 1.0 - tf.reduce_mean(cos_phase_diff)


def combined_loss(y_true, y_pred, magnitude_weight=1.0, phase_weight=0.0):
    """Combined magnitude and phase loss - temporarily disable phase loss"""
    mag_loss = magnitude_spectral_loss(y_true, y_pred)
    if phase_weight > 0:
        ph_loss = phase_loss(y_true, y_pred)
        return magnitude_weight * mag_loss + phase_weight * ph_loss
    return magnitude_weight * mag_loss


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


def audio_to_stft(audio, frame_length=1024, frame_step=256):
    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    # Replace any NaN or infinite values in real and imaginary parts
    real_part = tf.math.real(stft)
    imag_part = tf.math.imag(stft)
    real_part = tf.where(tf.math.is_finite(real_part), real_part, 0.0)
    imag_part = tf.where(tf.math.is_finite(imag_part), imag_part, 0.0)
    return tf.complex(real_part, imag_part)


def stft_to_audio(stft, frame_length=1024, frame_step=256):
    audio = tf.signal.inverse_stft(stft, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    return tf.math.real(audio)


def create_teacher_forcing_dataset(stft_data, seq_len=64, batch_size=8):
    def generator():
        time_steps = stft_data.shape[0]
        for i in range(time_steps - seq_len):
            input_seq = stft_data[i:i + seq_len]
            target_seq = stft_data[i + 1:i + seq_len + 1]
            yield input_seq, target_seq
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, stft_data.shape[1]), dtype=tf.complex64),
            tf.TensorSpec(shape=(seq_len, stft_data.shape[1]), dtype=tf.complex64)
        )
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def safe_transfer_weights(old_model, new_model):
    old_map = {w.name: w.numpy() for w in old_model.weights}
    assigned = 0
    for w in new_model.weights:
        name = w.name
        if name in old_map and old_map[name].shape == tuple(w.shape):
            w.assign(old_map[name])
            assigned += 1
    print(f"Transferred {assigned}/{len(new_model.weights)} tensors (matched by name & shape).")


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = magnitude_spectral_loss(y, pred)
        
        # Check for NaN in predictions
        if tf.reduce_any(tf.math.is_nan(tf.abs(pred))):
            tf.print("NaN detected in model output!")
            
    grads = tape.gradient(loss, model.trainable_variables)
    
    # Clip gradients to prevent explosion
    grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
    
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_curriculum():
    input_file = "/Users/davidkeeler/data/music/musicnet/test_data/2416.wav"
    checkpoint_dir = "/Users/davidkeeler/models/conducting/checkpoints/"
    output_dir = "/Users/davidkeeler/data/music/model_out"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Training Started: {datetime.now()} ===")
    audio_segment = load_full_audio(input_file)
    stft_segment = audio_to_stft(audio_segment)
    print(f"Audio length: {len(audio_segment)} samples")
    print(f"STFT shape: {stft_segment.shape}")
    
    real_nan = tf.reduce_sum(tf.cast(~tf.math.is_finite(tf.math.real(stft_segment)), tf.int32))
    imag_nan = tf.reduce_sum(tf.cast(~tf.math.is_finite(tf.math.imag(stft_segment)), tf.int32))
    print(f"NaN/Inf values in STFT real: {real_nan.numpy()}, imag: {imag_nan.numpy()}")
    if real_nan > 0 or imag_nan > 0:
        print("WARNING: Input data contains NaN/Inf values!")

    stages = [
        (32, 30, 1e-3),
        (64, 30, 1e-3),
        (128, 30, 1e-3),
        (256, 30, 1e-3),
        (512, 30, 1e-3),
        (1024, 30, 1e-3),
    ]

    model = None
    checkpoint_path = os.path.join(checkpoint_dir, 'curriculum_weights.weights.h5')

    for idx, (seq_len, epochs, lr) in enumerate(stages):
        print(f"\n--- Stage {idx+1}: seq_len={seq_len}, epochs={epochs} ---")

        new_model = AudioTransformerFreq(d_model=128, num_blocks=1, freq_bins=513, seq_len=seq_len)
        dummy_input = tf.zeros((1, seq_len, 513), dtype=tf.complex64)
        new_model(dummy_input)

        if model is not None:
            # safe_transfer_weights(model, new_model)  # Disable for debugging
            pass
        else:
            # Load most recent checkpoint
            latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                try:
                    new_model.load_weights(latest_checkpoint)
                    print(f"Loaded weights from: {os.path.basename(latest_checkpoint)}")
                except Exception as e:
                    print(f"Failed to load weights: {e}")
                    print("Starting with fresh weights")
            else:
                print("No existing checkpoints found.")

        model = new_model
        optimizer = tf.keras.optimizers.Adam(lr, clipnorm=1.0)
        dataset = create_teacher_forcing_dataset(stft_segment, seq_len, batch_size=8)

        for epoch in range(epochs):
            epoch_losses = []
            dataset_iter = iter(dataset)
            
            # Count total steps
            total_steps = sum(1 for _ in dataset)
            dataset = create_teacher_forcing_dataset(stft_segment, seq_len, batch_size=8)
            
            epoch_start = time.time()
            with tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for step, (x, y) in enumerate(dataset):
                    step_start = time.time()
                    loss = train_step(model, optimizer, x, y)
                    step_time = time.time() - step_start
                    
                    epoch_losses.append(loss.numpy())
                    
                    # Moving average of last 10 steps
                    window_size = min(10, len(epoch_losses))
                    moving_avg = np.mean(epoch_losses[-window_size:])
                    pbar.set_postfix({
                        'loss': f'{moving_avg:.6f}',
                        'ms/step': f'{step_time*1000:.1f}'
                    })
                    pbar.update(1)
            
            epoch_time = time.time() - epoch_start
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f} - {epoch_time:.1f}s - {epoch_time/total_steps*1000:.1f}ms/step")
            model.save_weights(checkpoint_path)

    print("=== Curriculum Training Complete ===")
    model.summary()

    final_seq_len = stages[-1][0]
    init = stft_segment[:final_seq_len][tf.newaxis, ...]
    gen = model.generate(init, num_steps=min(500, stft_segment.shape[0] - final_seq_len))
    out_audio = stft_to_audio(gen[0]).numpy()
    sf.write(os.path.join(output_dir, "curriculum_generated.wav"), out_audio, 22050)
    print("Saved final audio output.")


if __name__ == "__main__":
    train_curriculum()
