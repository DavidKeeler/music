import tensorflow as tf
import numpy as np
import soundfile as sf
import os
from audio_transformer_freq import AudioTransformerFreq
import glob

def magnitude_spectral_loss(y_true, y_pred):
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Magnitude loss - preserve STFT magnitude
    true_mag = tf.abs(y_true)
    pred_mag = tf.abs(y_pred)
    mag_loss = tf.reduce_mean(tf.square(true_mag - pred_mag))
    
    # Spectral loss - compare frequency bins directly (already in frequency domain)
    spectral_loss = tf.reduce_mean(tf.square(tf.abs(y_true) - tf.abs(y_pred)))
    
    # Energy preservation loss
    true_energy = tf.reduce_mean(tf.square(tf.abs(y_true)), axis=-1, keepdims=True)
    pred_energy = tf.reduce_mean(tf.square(tf.abs(y_pred)), axis=-1, keepdims=True)
    energy_loss = tf.reduce_mean(tf.square(true_energy - pred_energy))

    total_loss = mae_loss + 1.5 * mag_loss + 0.4 * energy_loss
    return total_loss


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
    return stft


def stft_to_audio(stft, frame_length=1024, frame_step=256):
    audio = tf.signal.inverse_stft(stft, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    return tf.math.real(audio)


def create_teacher_forcing_dataset(stft_data, seq_len=64, batch_size=8):
    """Create input/target pairs for teacher forcing"""

    def generator():
        time_steps = stft_data.shape[0]
        while True:
            for i in range(time_steps - seq_len):
                input_seq = stft_data[i:i + seq_len]  # Input sequence
                target_seq = stft_data[i + 1:i + seq_len + 1]  # Target shifted by 1
                yield input_seq, target_seq

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, stft_data.shape[1]), dtype=tf.complex64),
            tf.TensorSpec(shape=(seq_len, stft_data.shape[1]), dtype=tf.complex64)
        )
    )
    return dataset.batch(batch_size).take(500)


def train_curriculum():
    from datetime import datetime

    input_file = "/Users/davidkeeler/data/music/musicnet/test_data/2416.wav"
    output_dir = "/Users/davidkeeler/data/music/model_out"
    checkpoint_dir = "/Users/davidkeeler/models/conducting/checkpoints/"

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"=== Curriculum Training Started: {datetime.now()} ===")

    audio_segment = load_full_audio(input_file)
    stft_segment = audio_to_stft(audio_segment)
    print(f"Audio length: {len(audio_segment)} samples ({len(audio_segment) / 22050:.2f} seconds)")
    print(f"STFT shape: {stft_segment.shape}")

    # Curriculum stages: seq_len, epochs, and learning_rate for each stage
    stages = [
        (32, 10, 1e-3),   # 0.37 seconds
        (64, 10, 8e-4),   # 0.74 seconds
        (128, 10, 5e-4),  # 1.5 seconds
        (256, 5, 3e-4),   # 3 seconds
        (512, 5, 3e-4),   # 6 seconds
        (1024, 1, 3e-4)   # 12 seconds
    ]

    model = None
    
    for stage_idx, (seq_len, epochs, lr) in enumerate(stages):
        print(f"\n--- Stage {stage_idx + 1}: seq_len={seq_len}, epochs={epochs}, lr={lr} ---")

        if model is None:
            model = AudioTransformerFreq(d_model=128, num_blocks=2, freq_bins=513, seq_len=seq_len)
        else:
            # Transfer weights to new model with different seq_len
            old_weights = model.get_weights()
            model = AudioTransformerFreq(d_model=128, num_blocks=2, freq_bins=513, seq_len=seq_len)
            
            # Build new model to initialize weights
            dummy_input = tf.zeros((1, seq_len, 513), dtype=tf.complex64)
            model(dummy_input)
            
            try:
                model.set_weights(old_weights)
                print("Transferred weights from previous stage")
            except Exception as e:
                print(f"Could not transfer weights: {e}, starting fresh for this stage")

        dataset = create_teacher_forcing_dataset(stft_segment, seq_len=seq_len, batch_size=8)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=magnitude_spectral_loss,
            metrics=['mae']
        )
        
        # Build model with actual training data for proper summary
        for batch in dataset.take(1):
            model(batch[0])
            break
        model.summary()

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, f'stage_{stage_idx + 1}_epoch_{{epoch:04d}}.keras'),
            save_freq=5,
            verbose=0
        )

        class StageProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, stage_name):
                self.stage_name = stage_name
                
            def on_epoch_end(self, epoch, logs=None):
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] {self.stage_name} Epoch {epoch + 1}/{epochs} - loss: {logs['loss']:.6f} - mae: {logs['mae']:.6f}")
                
            def on_batch_end(self, batch, logs=None):
                if batch % 50 == 0:  # Print every 50 batches
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"[{current_time}] {self.stage_name} Batch {batch} - loss: {logs['loss']:.6f}")

        try:
            model.fit(dataset, epochs=epochs, verbose=1,
                      callbacks=[checkpoint_callback, StageProgressCallback(f"Stage {stage_idx + 1}")])
        except KeyboardInterrupt:
            print(f"\nTraining interrupted at stage {stage_idx + 1}")
            break

    print(f"\n=== Curriculum Training Complete: {datetime.now()} ===")

    # Generate final audio with longest sequence model
    final_seq_len = stages[-1][0] if stages else 64
    initial_frames = stft_segment[:final_seq_len][tf.newaxis, ...]
    generated_stft = model.generate(initial_frames, num_steps=min(500, stft_segment.shape[0] - final_seq_len))
    generated_audio = stft_to_audio(generated_stft[0]).numpy()

    sf.write(os.path.join(output_dir, "curriculum_original.wav"), audio_segment, 22050)
    sf.write(os.path.join(output_dir, "curriculum_generated.wav"), generated_audio, 22050)
    
    model.save(os.path.join(checkpoint_dir, 'curriculum_final.keras'))
    print("Final curriculum model saved")


if __name__ == "__main__":
    train_curriculum()
