"""CLI entry point for generating beat-signature conducting gesture datasets.

Pairs mel spectrograms with procedurally generated conducting keypoint
sequences aligned to detected beats, outputting TFRecords.

Usage:
    python -m src.music_generation.generate_dataset \
        --input_dir ~/data/music/wav_files \
        --output_dir ./beat_dataset \
        --fps 30 --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from .audio_utils import load_audio, audio_to_mel
from .beat_analyzer import BeatAnalyzer, BeatInfo
from .conducting_patterns import PatternGenerator
from .config import SAMPLE_RATE, FRAME_STEP
from .dataset_writer import write_sample
from .skeleton_builder import SkeletonBuilder

logger = logging.getLogger(__name__)


def _slice_beat_info(beat_info: BeatInfo, start: float, end: float) -> BeatInfo:
    """Slice BeatInfo to a time window [start, end)."""
    mask = (beat_info.beat_times >= start) & (beat_info.beat_times < end)
    db_mask = (beat_info.downbeat_times >= start) & (beat_info.downbeat_times < end)
    return BeatInfo(
        beat_times=beat_info.beat_times[mask] - start,
        downbeat_times=beat_info.downbeat_times[db_mask] - start,
        tempo=beat_info.tempo,
        time_signature=beat_info.time_signature,
    )


def process_file(
    audio_path: Path,
    output_dir: Path,
    analyzer: BeatAnalyzer,
    fps: int,
    sample_duration: float,
    variations_per_file: int,
    base_seed: int,
    file_idx: int,
    ts_override: str | None,
) -> int:
    """Process a single audio file, return number of samples written."""
    # Load and compute mel
    audio = load_audio(str(audio_path))
    mel = audio_to_mel(audio).numpy()
    audio_duration = len(audio.numpy()) / SAMPLE_RATE

    if audio_duration < sample_duration:
        logger.warning("Skipping %s: too short (%.1fs < %.1fs)", audio_path.name, audio_duration, sample_duration)
        return 0

    # Beat analysis
    beat_info = analyzer.analyze(str(audio_path), time_signature_override=ts_override)

    mel_fps = SAMPLE_RATE / FRAME_STEP
    pattern_gen = PatternGenerator()
    skeleton = SkeletonBuilder()
    samples_written = 0

    # Number of non-overlapping windows
    num_windows = int(audio_duration // sample_duration)

    tfrecord_path = output_dir / f"{audio_path.stem}.tfrecord"
    writer = tf.io.TFRecordWriter(str(tfrecord_path))

    try:
        for var_idx in range(variations_per_file):
            seed = base_seed + file_idx * variations_per_file + var_idx

            for win_idx in range(num_windows):
                start_t = win_idx * sample_duration
                end_t = start_t + sample_duration

                # Slice mel
                mel_start = int(round(start_t * mel_fps))
                mel_end = int(round(end_t * mel_fps))
                mel_slice = mel[mel_start:mel_end]

                # Slice beats and generate keypoints
                window_beats = _slice_beat_info(beat_info, start_t, end_t)
                if len(window_beats.beat_times) < 2:
                    continue

                wrist = pattern_gen.generate_wrist_trajectory(
                    window_beats, fps=fps, duration=sample_duration, seed=seed + win_idx,
                )
                keypoints = skeleton.build_sequence(
                    wrist, window_beats.downbeat_times, fps,
                )

                write_sample(
                    writer, mel_slice, keypoints,
                    beat_info.time_signature, beat_info.tempo,
                    fps, audio_path.name,
                )
                samples_written += 1
    finally:
        writer.close()

    return samples_written


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Generate beat-signature conducting gesture dataset")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of .wav files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for TFRecords")
    parser.add_argument("--fps", type=int, default=30, help="Keypoint frame rate (default: 30)")
    parser.add_argument("--time_signature", type=str, default=None, help="Override time signature, e.g. 3/4")
    parser.add_argument("--sample_duration", type=float, default=10.0, help="Duration per sample in seconds (default: 10)")
    parser.add_argument("--variations_per_file", type=int, default=3, help="Variations per file (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    wav_files = sorted(args.input_dir.glob("*.wav"))
    if not wav_files:
        logger.error("No .wav files found in %s", args.input_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = BeatAnalyzer()
    total_samples = 0

    for file_idx, wav_path in enumerate(wav_files):
        logger.info("[%d/%d] Processing %s", file_idx + 1, len(wav_files), wav_path.name)
        try:
            n = process_file(
                wav_path, args.output_dir, analyzer,
                args.fps, args.sample_duration, args.variations_per_file,
                args.seed,
                file_idx, args.time_signature,
            )
            total_samples += n
            logger.info("  → %d samples written", n)
        except Exception as e:
            logger.warning("Skipping %s: %s", wav_path.name, e)

    logger.info("Done. %d samples from %d files → %s", total_samples, len(wav_files), args.output_dir)


if __name__ == "__main__":
    main()
