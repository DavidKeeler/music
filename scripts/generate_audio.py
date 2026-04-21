"""Generate audio from a trained mel generator checkpoint.

Usage:
    python -m scripts.generate_audio
    python -m scripts.generate_audio --checkpoint checkpoints/mel_generator.keras
    python -m scripts.generate_audio --num_frames 1024 --temperature 0.8
    python -m scripts.generate_audio --vocoder griffin-lim
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate audio from mel generator checkpoint")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path.home() / "models" / "conducting" / "mel_checkpoints_24khz",
                        help="Path to checkpoint directory or .keras file")
    parser.add_argument("--output", type=Path, default=Path("output_generated.wav"),
                        help="Output .wav file path")
    parser.add_argument("--num_frames", type=int, default=512,
                        help="Number of mel frames to generate (~5.5s at 512)")
    parser.add_argument("--seed_frames", type=int, default=64,
                        help="Number of random seed mel frames")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (lower = more conservative)")
    parser.add_argument("--vocoder", type=str, default="griffin-lim",
                        choices=["hifigan", "vocos", "griffin-lim"],
                        help="Vocoder backend for mel-to-audio")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args(argv)

    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    # Load checkpoint
    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    from src.music_generation.model import MelGenerator
    from src.music_generation.config import N_MELS, SEQ_LEN

    if args.checkpoint.is_dir():
        # Directory with .weights.h5 files — build model then load weights
        weights_path = args.checkpoint / "base_model.weights.h5"
        if not weights_path.exists():
            logger.error("No base_model.weights.h5 found in %s", args.checkpoint)
            sys.exit(1)
        logger.info("Building MelGenerator and loading weights from %s", weights_path)
        mel_generator = MelGenerator()
        # Build by running a dummy forward pass
        dummy = tf.zeros([1, SEQ_LEN, N_MELS])
        mel_generator(dummy, training=False)
        mel_generator.load_weights(str(weights_path))
    elif str(args.checkpoint).endswith('.keras'):
        logger.info("Loading mel generator from %s", args.checkpoint)
        mel_generator = tf.keras.models.load_model(str(args.checkpoint))
    else:
        # Assume .h5 weights file directly
        logger.info("Building MelGenerator and loading weights from %s", args.checkpoint)
        mel_generator = MelGenerator()
        dummy = tf.zeros([1, SEQ_LEN, N_MELS])
        mel_generator(dummy, training=False)
        mel_generator.load_weights(str(args.checkpoint))

    logger.info("Model loaded successfully")

    # Create a random seed mel spectrogram
    from src.music_generation.config import SAMPLE_RATE, FRAME_STEP
    seed_mel = tf.random.normal([args.seed_frames, N_MELS], mean=-5.0, stddev=2.0)

    logger.info("Generating %d mel frames (temperature=%.2f)...", args.num_frames, args.temperature)
    generated_mel = mel_generator.generate(
        seed_mel, args.num_frames, temperature=args.temperature
    )
    logger.info("Generated mel shape: %s", generated_mel.shape)

    # Denormalize model output (model outputs normalized mels)
    import json
    from src.music_generation.audio_utils import denormalize_mel
    stats_path = Path.home() / "data" / "music" / "musicnet" / "cache" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        logger.info("Denormalizing mel (mean=%.4f, std=%.4f)", stats["mean"], stats["std"])
        generated_mel = denormalize_mel(generated_mel, stats["mean"], stats["std"])
    else:
        logger.warning("No stats.json found — skipping denormalization (output may sound wrong)")

    # Load vocoder and convert mel to audio
    from src.music_generation.vocoder import load_pretrained_vocoder
    logger.info("Loading vocoder: %s", args.vocoder)
    vocoder = load_pretrained_vocoder(backend=args.vocoder, enable_fallback=True)

    mel_for_vocoder = tf.expand_dims(generated_mel, 0)  # [1, T, N_MELS]
    audio = vocoder(mel_for_vocoder)
    audio = tf.squeeze(audio).numpy()

    # Write output
    duration = len(audio) / SAMPLE_RATE
    sf.write(str(args.output), audio, SAMPLE_RATE)
    logger.info("Saved %.1fs of audio to %s", duration, args.output)


if __name__ == "__main__":
    main()
