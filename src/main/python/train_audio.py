import tensorflow as tf
import numpy as np
import soundfile as sf
import os
from audio_transformer_freq import AudioTransformerFreq
from datetime import datetime


def magnitude_spectral_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.abs(y_true - y_pred)))


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
    return tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)


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
    old_weights = old_model.get_weights()
    new_weights = new_model.get_weights()
    if len(old_weights) != len(new_weights):
        print(f"Warning: mismatched weight count ({len(old_weights)} vs {len(new_weights)}), copying overlapping")
    shared_len = min(len(old_weights), len(new_weights))
    new_weights[:shared_len] = old_weights[:shared_len]
    new_model.set_weights(new_weights)
    print(f"Transferred {shared_len}/{len(new_weights)} weight tensors.")


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

    stages = [
        (16, 2, 5e-6),
        (32, 2, 5e-6),
        (64, 2, 5e-6),
        (128, 2, 5e-6),
        (256, 2, 5e-6),
    ]

    model = None
    checkpoint_path = os.path.join(checkpoint_dir, 'curriculum_weights.weights.h5')

    for idx, (seq_len, epochs, lr) in enumerate(stages):
        print(f"\n--- Stage {idx+1}: seq_len={seq_len}, epochs={epochs} ---")

        new_model = AudioTransformerFreq(d_model=128, num_blocks=1, freq_bins=513, seq_len=seq_len)
        dummy_input = tf.zeros((1, seq_len, 513), dtype=tf.complex64)
        new_model(dummy_input)

        if model is not None:
            safe_transfer_weights(model, new_model)
        elif os.path.exists(checkpoint_path):
            try:
                new_model.load_weights(checkpoint_path)
                print("Loaded existing weights.")
            except Exception as e:
                print(f"Failed to load weights: {e}")

        model = new_model
        optimizer = tf.keras.optimizers.Adam(lr)
        dataset = create_teacher_forcing_dataset(stft_segment, seq_len, batch_size=8)

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                loss = magnitude_spectral_loss(y, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        for epoch in range(epochs):
            avg_loss = 0
            for step, (x, y) in enumerate(dataset):
                loss = train_step(x, y)
                avg_loss += loss
                if step % 50 == 0:
                    print(f"  Step {step}: loss={loss:.6f}")
            avg_loss /= tf.cast(step + 1, tf.float32)
            print(f"Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.6f}")
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
