import tensorflow as tf
import numpy as np
import soundfile as sf
import os
from datetime import datetime
from audio_transformer_freq import AudioTransformerFreq

def audio_to_stft(audio, frame_length=1024, frame_step=256):
    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    return stft

def stft_to_audio(stft, frame_length=1024, frame_step=256):
    audio = tf.signal.inverse_stft(stft, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    return tf.math.real(audio)

def load_seed_audio(file_path, duration_sec=2, sr=22050):
    """Load seed audio for generation"""
    audio, file_sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    if file_sr != sr:
        ratio = sr / file_sr
        new_length = int(len(audio) * ratio)
        audio = np.interp(np.linspace(0, len(audio) - 1, new_length),
                          np.arange(len(audio)), audio)
    
    # Take seed from middle of audio
    start_sample = len(audio) // 2
    end_sample = start_sample + int(duration_sec * sr)
    seed = audio[start_sample:end_sample]
    
    seed = seed / (np.max(np.abs(seed)) + 1e-8)
    return seed.astype(np.float32)

def generate_music_30sec():
    print(f"=== Music Generation Started: {datetime.now()} ===")

    checkpoint_dir = "/Users/davidkeeler/models/conducting/checkpoints"
    seed_file = "/Users/davidkeeler/data/music/musicnet/test_data/2416.wav"
    output_dir = "/Users/davidkeeler/data/music/model_out"
    
    # Create model with current training parameters
    print("Creating model...")
    model = AudioTransformerFreq(d_model=128, num_blocks=1, freq_bins=513, seq_len=16)
    
    # Build model by running a dummy forward pass
    dummy_real = tf.random.normal((1, 16, 513))
    dummy_imag = tf.random.normal((1, 16, 513))
    dummy_input = tf.complex(dummy_real, dummy_imag)
    _ = model(dummy_input)
    
    # Load weights from current checkpoint
    print("Loading weights...")
    try:
        model.load_weights(os.path.join(checkpoint_dir, "curriculum_weights.weights.h5"))
        print("Loaded weights from curriculum_weights.weights.h5")
    except Exception as e:
        print(f"Could not load weights: {e}")
        raise Exception("No suitable checkpoint found")
    
    # Load seed audio
    print("Loading seed audio...")
    seed_audio = load_seed_audio(seed_file, duration_sec=2)
    seed_stft = audio_to_stft(seed_audio)
    print(f"Seed STFT shape: {seed_stft.shape}")
    
    # Generate 30 seconds of audio
    target_duration = 30  # seconds
    sr = 22050
    target_samples = target_duration * sr
    frame_step = 256
    target_frames = target_samples // frame_step
    
    print(f"Generating {target_duration} seconds ({target_frames} frames)...")
    
    # Use seed frames as initial context
    seq_len = 256  # Final stage sequence length
    initial_frames = seed_stft[:seq_len][tf.newaxis, ...]
    
    # Generate remaining frames
    frames_to_generate = target_frames - seq_len
    print(f"Using {seq_len} seed frames, generating {frames_to_generate} new frames")
    
    # Autoregressive generation with stability measures
    generated_frames = []
    current_context = initial_frames[0]  # Remove batch dimension
    
    for i in range(frames_to_generate):
        if i % 100 == 0:
            print(f"Generated {i}/{frames_to_generate} frames ({i/frames_to_generate*100:.1f}%)")
        
        # Predict next frame
        context_batch = current_context[tf.newaxis, ...]  # Add batch dimension
        logits = model(context_batch, training=False)[0, -1:, :]  # Get last frame
        
        # Apply temperature scaling
        temperature = 0.3
        magnitude = tf.abs(logits)
        phase = tf.math.angle(logits)
        
        # Scale magnitude with temperature
        log_magnitude = tf.math.log(magnitude + 1e-8)
        scaled_log_magnitude = log_magnitude / temperature
        scaled_magnitude = tf.exp(scaled_log_magnitude)
        
        # Clip magnitude to prevent explosion
        clipped_magnitude = tf.clip_by_value(scaled_magnitude, 0.0, 20.0)
        next_frame = tf.complex(clipped_magnitude * tf.cos(phase), 
                               clipped_magnitude * tf.sin(phase))
        
        # Append to generated frames
        generated_frames.append(next_frame[0])
        
        # Update context (sliding window)
        current_context = tf.concat([current_context[1:], next_frame], axis=0)
    
    # Combine seed and generated frames
    all_frames = tf.concat([seed_stft, tf.stack(generated_frames)], axis=0)
    
    # Convert back to audio
    print("Converting to audio...")
    generated_audio = stft_to_audio(all_frames).numpy()
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_output = os.path.join(output_dir, f"seed_{timestamp}.wav")
    generated_output = os.path.join(output_dir, f"generated_30sec_{timestamp}.wav")
    
    sf.write(seed_output, seed_audio, sr)
    sf.write(generated_output, generated_audio, sr)
    
    print(f"=== Generation Complete: {datetime.now()} ===")
    print(f"Seed audio saved: {seed_output}")
    print(f"Generated audio saved: {generated_output}")
    print(f"Generated audio duration: {len(generated_audio)/sr:.2f} seconds")

if __name__ == "__main__":
    generate_music_30sec()
