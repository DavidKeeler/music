"""Generate audio using teacher forcing (non-autoregressive single forward pass).

Feeds ground-truth mel to the model and vocodes the output, bypassing
the autoregressive loop. Useful for diagnosing whether bad output comes
from the model itself or from autoregressive error accumulation.

Usage:
    python -m scripts.generate_audio_teacher_forced --input seed.wav
    python -m scripts.generate_audio_teacher_forced --input seed.wav --checkpoint ./checkpoints
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
    parser = argparse.ArgumentParser(description="Generate audio with teacher forcing (single forward pass)")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input .wav file (ground truth audio)")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path.home() / "models" / "conducting" / "mel_checkpoints_24khz",
                        help="Path to checkpoint directory or .keras/.h5 file")
    parser.add_argument("--output", type=Path, default=Path("output_teacher_forced.wav"),
                        help="Output .wav file path")
    parser.add_argument("--vocoder", type=str, default="griffin-lim",
                        choices=["hifigan", "vocos", "griffin-lim"],
                        help="Vocoder backend for mel-to-audio")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args(argv)

    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    if not args.input.exists():
        logger.error("Input audio not found: %s", args.input)
        sys.exit(1)
    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    from src.music_generation.audio_utils import load_audio, audio_to_mel
    from src.music_generation.model import MelGenerator
    from src.music_generation.config import N_MELS, SEQ_LEN, SAMPLE_RATE

    # Load model
    if args.checkpoint.is_dir():
        weights_path = args.checkpoint / "base_model.weights.h5"
        if not weights_path.exists():
            logger.error("No base_model.weights.h5 found in %s", args.checkpoint)
            sys.exit(1)
        mel_generator = MelGenerator()
        mel_generator(tf.zeros([1, SEQ_LEN, N_MELS]), training=False)
        mel_generator.load_weights(str(weights_path))
    elif str(args.checkpoint).endswith('.keras'):
        mel_generator = tf.keras.models.load_model(str(args.checkpoint))
    else:
        mel_generator = MelGenerator()
        mel_generator(tf.zeros([1, SEQ_LEN, N_MELS]), training=False)
        mel_generator.load_weights(str(args.checkpoint))

    logger.info("Model loaded from %s", args.checkpoint)

    # Load input audio and convert to mel
    waveform = load_audio(str(args.input))
    input_mel = audio_to_mel(waveform)  # [T, N_MELS]
    logger.info("Input mel shape: %s", input_mel.shape)

    # Single forward pass (teacher forcing — model sees ground truth at every step)
    output_mel = mel_generator(input_mel[None], training=False)  # [1, T, N_MELS]
    logger.info("Output mel shape: %s", output_mel.shape)

    # Denormalize model output before vocoding
    import json
    from src.music_generation.audio_utils import denormalize_mel
    stats_path = Path.home() / "data" / "music" / "musicnet" / "cache" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        logger.info("Denormalizing mel (mean=%.4f, std=%.4f)", stats["mean"], stats["std"])
        output_mel = denormalize_mel(output_mel, stats["mean"], stats["std"])
    else:
        logger.warning("No stats.json found — skipping denormalization")

    # Vocode
    from src.music_generation.vocoder import load_pretrained_vocoder
    vocoder = load_pretrained_vocoder(backend=args.vocoder, enable_fallback=True)
    audio = tf.squeeze(vocoder(output_mel)).numpy()

    duration = len(audio) / SAMPLE_RATE
    sf.write(str(args.output), audio, SAMPLE_RATE)
    logger.info("Saved %.1fs of audio to %s", duration, args.output)


if __name__ == "__main__":
    main()
