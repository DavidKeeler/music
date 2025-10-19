import tensorflow as tf
import numpy as np
import soundfile as sf
import os
import time
from audio_transformer_freq import AudioTransformerFreq
from datetime import datetime
from tqdm import tqdm


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
    grads = tape.gradient(loss, model.trainable_variables)
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

    stages = [
        (16, 20, 5e-5),
        (32, 20, 5e-5),
        (64, 20, 5e-5),
        (128, 20, 5e-5),
        (256, 20, 5e-5),
        (512, 20, 5e-5),
        (1024, 20, 5e-5),
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
