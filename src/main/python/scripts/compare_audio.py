import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import librosa

def compare_audio(original_path, generated_path):
    # Load audio files
    orig, sr1 = sf.read(original_path)
    gen, sr2 = sf.read(generated_path)
    
    print(f"Original: {orig.shape}, SR: {sr1}")
    print(f"Generated: {gen.shape}, SR: {sr2}")
    
    # Ensure same length
    min_len = min(len(orig), len(gen))
    orig = orig[:min_len]
    gen = gen[:min_len]
    
    # Basic metrics
    mse = np.mean((orig - gen) ** 2)
    mae = np.mean(np.abs(orig - gen))
    correlation = np.corrcoef(orig, gen)[0, 1]
    
    print(f"\nMetrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Correlation: {correlation:.6f}")
    
    # Spectral comparison
    f1, t1, S1 = signal.spectrogram(orig, sr1, nperseg=1024)
    f2, t2, S2 = signal.spectrogram(gen, sr2, nperseg=1024)
    
    spectral_mse = np.mean((S1 - S2) ** 2)
    print(f"Spectral MSE: {spectral_mse:.6f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Waveforms
    plt.subplot(2, 2, 1)
    plt.plot(orig[:1000], label='Original', alpha=0.7)
    plt.plot(gen[:1000], label='Generated', alpha=0.7)
    plt.title('Waveform Comparison (first 1000 samples)')
    plt.legend()
    
    # Spectrograms
    plt.subplot(2, 2, 2)
    plt.imshow(10*np.log10(S1 + 1e-10), aspect='auto', origin='lower')
    plt.title('Original Spectrogram')
    
    plt.subplot(2, 2, 3)
    plt.imshow(10*np.log10(S2 + 1e-10), aspect='auto', origin='lower')
    plt.title('Generated Spectrogram')
    
    # Difference
    plt.subplot(2, 2, 4)
    plt.plot(orig - gen)
    plt.title('Difference (Original - Generated)')
    
    plt.tight_layout()
    plt.savefig('/Users/davidkeeler/data/music/model_out/comparison.png')
    plt.show()

if __name__ == "__main__":
    original = "/Users/davidkeeler/data/music/model_out/overfit_original.wav"
    generated = "/Users/davidkeeler/data/music/model_out/overfit_generated.wav"
    compare_audio(original, generated)
