"""Vocoder roundtrip sanity check: audio → mel → vocoder → wav.

Skips the model entirely. If the output sounds bad, the problem is in
the mel extraction or vocoder, not the model.

Usage:
    python -m scripts.vocoder_roundtrip --input ~/data/music/musicnet/train_data/some_file.wav
    python -m scripts.vocoder_roundtrip --input seed.wav --vocoder hifigan
"""

import argparse
import logging
import sys
from pathlib import Path

import soundfile as sf
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _audio_to_mel_magnitude(waveform):
    """Legacy magnitude-based mel (for comparison only)."""
    from src.music_generation.config import FRAME_LENGTH, FRAME_STEP, N_MELS, SAMPLE_RATE as SR, F_MIN, F_MAX
    stft = tf.signal.stft(waveform, FRAME_LENGTH, FRAME_STEP, pad_end=True)
    magnitude = tf.abs(stft)
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS, num_spectrogram_bins=FRAME_LENGTH // 2 + 1,
        sample_rate=SR, lower_edge_hertz=F_MIN, upper_edge_hertz=F_MAX
    )
    return tf.math.log(tf.matmul(magnitude, mel_matrix) + 1e-8)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vocoder roundtrip: audio → mel → vocoder → wav")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input .wav file")
    parser.add_argument("--output", type=Path, default=Path("output_vocoder_roundtrip.wav"),
                        help="Output .wav file path")
    parser.add_argument("--vocoder", type=str, default="griffin-lim",
                        choices=["hifigan", "vocos", "griffin-lim"],
                        help="Vocoder backend")
    parser.add_argument("--magnitude", action="store_true",
                        help="Use magnitude spectrogram instead of power (legacy, not recommended)")
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error("Input audio not found: %s", args.input)
        sys.exit(1)

    from src.music_generation.audio_utils import load_audio, audio_to_mel, normalize_mel, denormalize_mel
    from src.music_generation.config import SAMPLE_RATE

    # Audio → mel
    waveform = load_audio(str(args.input))
    logger.info("Loaded audio: %d samples (%.1fs)", len(waveform), len(waveform) / SAMPLE_RATE)

    if args.magnitude:
        logger.info("Using MAGNITUDE spectrogram (legacy mode)")
        mel = _audio_to_mel_magnitude(waveform)
    else:
        logger.info("Using power spectrogram (audio_to_mel)")
        mel = audio_to_mel(waveform)  # [T, N_MELS]

    # Load normalization stats if available (matches training pipeline)
    import json
    stats_path = Path.home() / "data" / "music" / "musicnet" / "cache" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        mel_mean, mel_std = stats["mean"], stats["std"]
        logger.info("Normalizing mel (mean=%.4f, std=%.4f) then denormalizing for vocoder", mel_mean, mel_std)
        # Normalize → denormalize is identity, but this tests the full pipeline path
        # For the roundtrip we skip normalization since we're bypassing the model
        logger.info("(Skipping normalize/denormalize — no model in the loop)")
    else:
        logger.info("No stats.json found — using raw mel (no normalization)")
    logger.info("Mel shape: %s  min=%.2f  max=%.2f  mean=%.2f",
                mel.shape, tf.reduce_min(mel).numpy(),
                tf.reduce_max(mel).numpy(), tf.reduce_mean(mel).numpy())

    # Mel → vocoder → audio
    from src.music_generation.vocoder import load_pretrained_vocoder
    vocoder = load_pretrained_vocoder(backend=args.vocoder, enable_fallback=True)

    mel_batched = tf.expand_dims(mel, 0)  # [1, T, N_MELS]
    audio_out = tf.squeeze(vocoder(mel_batched)).numpy()

    duration = len(audio_out) / SAMPLE_RATE
    sf.write(str(args.output), audio_out, SAMPLE_RATE)
    logger.info("Saved %.1fs of audio to %s", duration, args.output)


if __name__ == "__main__":
    main()
