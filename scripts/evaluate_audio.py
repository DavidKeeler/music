"""Evaluate whether generated audio resembles music or noise.

Computes spectral metrics and prints a summary with a music/noise verdict.

Metrics:
  - Spectral Flatness: low = tonal (music-like), high = noise-like
  - Harmonic-to-Noise Ratio (HNR): higher = more harmonic content
  - Onset Rate: musical audio has structured onsets; noise has very few or too many
  - Tempo Confidence: whether a stable tempo can be detected
  - Dynamic Range: music has variation; flat noise does not
  - ZCR Variance: noise has constant zero-crossing rate; music varies
  - Spectral Centroid Variance: music's frequency content moves; noise stays put

Usage:
    python -m scripts.evaluate_audio output_generated.wav
    python -m scripts.evaluate_audio --reference real_music.wav output_generated.wav
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class AudioMetrics:
    spectral_flatness_mean: float
    hnr_db: float
    onset_rate: float  # onsets per second
    tempo_confidence: float  # strength of detected tempo
    dynamic_range_db: float
    zcr_variance: float
    spectral_centroid_variance: float
    duration_s: float
    voiced_ratio: float
    pitch_std: float
    spectral_bandwidth_mean: float
    spectral_contrast_mean: float
    clip_ratio: float
    energy: float


def compute_metrics(path: str, sr=24000) -> AudioMetrics:
    y, sr = librosa.load(path, sr=sr)
    duration = len(y) / sr

    # Spectral flatness (geometric mean / arithmetic mean of power spectrum)
    flatness = librosa.feature.spectral_flatness(y=y)
    flatness_mean = float(np.mean(flatness))

    # Harmonic-to-noise ratio via harmonic/percussive separation
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.sum(y_harm ** 2) + 1e-10
    noise_energy = np.sum(y_perc ** 2) + 1e-10
    hnr_db = float(10 * np.log10(harm_energy / noise_energy))

    # Onset rate
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    onset_rate = len(onsets) / duration if duration > 0 else 0.0

    # Tempo confidence (strength of the dominant beat)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    ac = librosa.autocorrelate(onset_env, max_size=sr // 512 * 4)
    if len(ac) > 1 and ac[0] > 0:
        tempo_confidence = float(np.max(ac[1:]) / ac[0])
    else:
        tempo_confidence = 0.0

    # Dynamic range (dB difference between loud and quiet frames)
    rms = librosa.feature.rms(y=y)[0]
    rms = rms[rms > 0]
    if len(rms) > 1:
        rms_db = librosa.amplitude_to_db(rms)
        dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
    else:
        dynamic_range = 0.0

    # ZCR variance (noise has constant ZCR; music varies)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_variance = float(np.var(zcr))

    # Spectral centroid variance (music moves; noise stays put)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid_variance = float(np.var(centroid))

    # Voiced ratio and pitch std via pyin
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=2000, sr=sr)
    voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_std = float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0

    # Spectral bandwidth and contrast
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_bandwidth_mean = float(np.mean(bandwidth))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = float(np.mean(contrast))

    # Clipping ratio and energy
    clip_ratio = float(np.mean(np.abs(y) > 0.99))
    energy = float(np.sum(y ** 2) / len(y))

    return AudioMetrics(
        spectral_flatness_mean=flatness_mean,
        hnr_db=hnr_db,
        onset_rate=onset_rate,
        tempo_confidence=tempo_confidence,
        dynamic_range_db=dynamic_range,
        zcr_variance=zcr_variance,
        spectral_centroid_variance=spectral_centroid_variance,
        duration_s=duration,
        voiced_ratio=voiced_ratio,
        pitch_std=pitch_std,
        spectral_bandwidth_mean=spectral_bandwidth_mean,
        spectral_contrast_mean=spectral_contrast_mean,
        clip_ratio=clip_ratio,
        energy=energy,
    )


def verdict(m: AudioMetrics) -> tuple[str, list[str]]:
    """Return (label, reasons) — 'music', 'maybe music', or 'noise'."""
    score = 0
    reasons = []

    if m.spectral_flatness_mean < 0.1:
        score += 2
        reasons.append(f"tonal (flatness={m.spectral_flatness_mean:.4f})")
    elif m.spectral_flatness_mean < 0.3:
        score += 1
        reasons.append(f"somewhat tonal (flatness={m.spectral_flatness_mean:.4f})")
    else:
        score -= 2
        reasons.append(f"noise-like spectrum (flatness={m.spectral_flatness_mean:.4f})")

    if m.hnr_db > 5:
        score += 2
        reasons.append(f"strong harmonic content (HNR={m.hnr_db:.1f} dB)")
    elif m.hnr_db > 0:
        score += 1
        reasons.append(f"some harmonic content (HNR={m.hnr_db:.1f} dB)")
    else:
        score -= 1
        reasons.append(f"weak harmonics (HNR={m.hnr_db:.1f} dB)")

    if 1.0 < m.onset_rate < 15.0:
        score += 1
        reasons.append(f"reasonable onset rate ({m.onset_rate:.1f}/s)")
    else:
        reasons.append(f"unusual onset rate ({m.onset_rate:.1f}/s)")

    if m.tempo_confidence > 0.3:
        score += 1
        reasons.append(f"detectable tempo (confidence={m.tempo_confidence:.2f})")
    else:
        reasons.append(f"no clear tempo (confidence={m.tempo_confidence:.2f})")

    if m.dynamic_range_db > 10:
        score += 1
        reasons.append(f"good dynamic range ({m.dynamic_range_db:.1f} dB)")
    elif m.dynamic_range_db > 3:
        reasons.append(f"limited dynamic range ({m.dynamic_range_db:.1f} dB)")
    else:
        score -= 1
        reasons.append(f"flat dynamics ({m.dynamic_range_db:.1f} dB)")

    if m.zcr_variance > 0.002:
        score += 1
        reasons.append(f"varied ZCR ({m.zcr_variance:.4f})")
    else:
        score -= 1
        reasons.append(f"constant ZCR ({m.zcr_variance:.4f})")

    if m.spectral_centroid_variance > 1e5:
        score += 1
        reasons.append(f"moving spectral centroid (var={m.spectral_centroid_variance:.0f})")
    else:
        score -= 1
        reasons.append(f"static spectral centroid (var={m.spectral_centroid_variance:.0f})")

    if score >= 5:
        label = "music"
    elif score >= 2:
        label = "maybe music"
    else:
        label = "noise"

    return label, reasons


def print_report(name: str, m: AudioMetrics):
    label, reasons = verdict(m)
    print(f"\n{'=' * 60}")
    print(f"  {name}  ({m.duration_s:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Spectral Flatness : {m.spectral_flatness_mean:.4f}")
    print(f"  HNR               : {m.hnr_db:.1f} dB")
    print(f"  Onset Rate        : {m.onset_rate:.1f} /s")
    print(f"  Tempo Confidence  : {m.tempo_confidence:.2f}")
    print(f"  Dynamic Range     : {m.dynamic_range_db:.1f} dB")
    print(f"  ZCR Variance      : {m.zcr_variance:.4f}")
    print(f"  Centroid Variance : {m.spectral_centroid_variance:.0f}")
    print(f"  ────────────────────────────────────")
    print(f"  Verdict: {label.upper()}")
    for r in reasons:
        print(f"    • {r}")
    print(f"{'=' * 60}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate if audio is music or noise")
    parser.add_argument("input", type=Path, help="Audio file to evaluate")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Optional reference music file for comparison")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    if args.reference:
        if not args.reference.exists():
            print(f"Error: {args.reference} not found", file=sys.stderr)
            sys.exit(1)
        ref_metrics = compute_metrics(str(args.reference))
        print_report(f"Reference: {args.reference.name}", ref_metrics)

    metrics = compute_metrics(str(args.input))
    print_report(f"Evaluated: {args.input.name}", metrics)


if __name__ == "__main__":
    main()
