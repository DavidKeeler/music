"""Compare two audio files and report similarity metrics.

Usage:
    python -m scripts.compare_audio --original ~/data/music/musicnet/train_data/1727.wav --reconstructed output_vocoder_roundtrip.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare two audio files")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--sr", type=int, default=24000, help="Sample rate for comparison")
    args = parser.parse_args(argv)

    for p in (args.original, args.reconstructed):
        if not p.exists():
            print(f"File not found: {p}")
            sys.exit(1)

    # Load both files at same sample rate
    orig, _ = librosa.load(str(args.original), sr=args.sr, mono=True)
    recon, _ = librosa.load(str(args.reconstructed), sr=args.sr, mono=True)

    # Trim to same length
    min_len = min(len(orig), len(recon))
    orig = orig[:min_len]
    recon = recon[:min_len]

    print(f"Original:      {args.original} ({len(orig)/args.sr:.1f}s)")
    print(f"Reconstructed: {args.reconstructed} ({len(recon)/args.sr:.1f}s)")
    print(f"Compared length: {min_len/args.sr:.1f}s ({min_len} samples)")
    print()

    # Waveform metrics
    mse = np.mean((orig - recon) ** 2)
    rmse = np.sqrt(mse)
    max_abs = np.max(np.abs(orig - recon))
    correlation = np.corrcoef(orig, recon)[0, 1]

    print("=== Waveform ===")
    print(f"  MSE:         {mse:.6f}")
    print(f"  RMSE:        {rmse:.6f}")
    print(f"  Max error:   {max_abs:.6f}")
    print(f"  Correlation: {correlation:.4f}")
    print()

    # SNR
    signal_power = np.mean(orig ** 2)
    noise_power = mse
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    print(f"  SNR:         {snr_db:.1f} dB")
    print()

    # Mel spectrogram comparison
    orig_mel = librosa.feature.melspectrogram(y=orig, sr=args.sr, n_mels=100, n_fft=1024, hop_length=256)
    recon_mel = librosa.feature.melspectrogram(y=recon, sr=args.sr, n_mels=100, n_fft=1024, hop_length=256)

    orig_log = np.log(orig_mel + 1e-8)
    recon_log = np.log(recon_mel + 1e-8)

    mel_mse = np.mean((orig_log - recon_log) ** 2)
    mel_corr = np.corrcoef(orig_log.flatten(), recon_log.flatten())[0, 1]

    print("=== Mel Spectrogram (log) ===")
    print(f"  MSE:         {mel_mse:.4f}")
    print(f"  Correlation: {mel_corr:.4f}")
    print()

    # Verdict
    print("=== Verdict ===")
    if mel_corr > 0.95:
        print("  Roundtrip looks solid — mel pipeline is working correctly.")
    elif mel_corr > 0.8:
        print("  Reasonable reconstruction. Some loss expected from Griffin-Lim.")
    elif mel_corr > 0.5:
        print("  Mediocre reconstruction. Consider using HiFi-GAN or Vocos.")
    else:
        print("  Poor reconstruction. Something is likely wrong in the mel pipeline.")


if __name__ == "__main__":
    main()
