import numpy as np
import soundfile as sf
import os
from scipy import signal

def analyze_audio_quality(file_path):
    """Check if audio is noise by analyzing spectral and temporal properties"""
    audio, sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # 1. Spectral flatness (noise has flat spectrum)
    f, psd = signal.welch(audio, sr, nperseg=1024)
    spectral_flatness = np.exp(np.mean(np.log(psd + 1e-10))) / np.mean(psd)
    
    # 2. Zero crossing rate (noise has high ZCR)
    zero_crossings = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
    
    # 3. Spectral centroid stability (music has stable centroid)
    stft = np.abs(signal.stft(audio, sr, nperseg=1024)[2])
    freqs = np.fft.fftfreq(1024, 1/sr)[:513]
    centroids = np.sum(stft * freqs[:, None], axis=0) / (np.sum(stft, axis=0) + 1e-10)
    centroid_std = np.std(centroids)
    
    # 4. Amplitude distribution (noise is more gaussian)
    hist, _ = np.histogram(audio, bins=50, density=True)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    
    print(f"Audio: {os.path.basename(file_path)}")
    print(f"Spectral flatness: {spectral_flatness:.4f} (noise > 0.5)")
    print(f"Zero crossing rate: {zero_crossings:.4f} (noise > 0.1)")
    print(f"Centroid std: {centroid_std:.1f} (noise > 2000)")
    print(f"Amplitude entropy: {entropy:.2f} (noise > 3.5)")
    
    # Noise detection thresholds
    is_noise = (spectral_flatness > 0.5 or 
                zero_crossings > 0.1 or 
                centroid_std > 2000 or 
                entropy > 3.5)
    
    print(f"Result: {'NOISE' if is_noise else 'MUSIC'}")
    print("-" * 40)
    
    return is_noise

def check_latest_generation():
    """Check the most recent generated audio and compare with training song"""
    output_dir = "/Users/davidkeeler/data/music/model_out"
    training_file = "/Users/davidkeeler/data/music/musicnet/test_data/2416.wav"
    
    # Check generated audio
    files = [f for f in os.listdir(output_dir) if f.startswith("generated_30sec_")]
    if not files:
        print("No generated files found")
        return
    
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
    generated_path = os.path.join(output_dir, latest_file)
    
    print("=== COMPARISON ===")
    analyze_audio_quality(training_file)
    analyze_audio_quality(generated_path)

if __name__ == "__main__":
    check_latest_generation()
