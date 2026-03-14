"""Beat analysis using BeatNet for beat/downbeat/meter detection."""

import logging
import sys
import types
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def _patch_beatnet_deps():
    """Patch numpy deprecations and mock pyaudio for BeatNet compatibility."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        np.int = np.int64
        np.float = np.float64
        np.bool = np.bool_
        np.complex = np.complex128
        np.object = np.object_
        np.str = np.str_
    if "pyaudio" not in sys.modules:
        sys.modules["pyaudio"] = types.ModuleType("pyaudio")


@dataclass
class BeatInfo:
    """Metric structure extracted from audio."""
    beat_times: np.ndarray       # seconds, all beat positions
    downbeat_times: np.ndarray   # seconds, beat-1 positions only
    tempo: float                 # BPM
    time_signature: tuple        # e.g. (4, 4)


def _parse_time_signature(ts_str: str) -> tuple:
    """Parse '3/4' -> (3, 4)."""
    num, denom = ts_str.strip().split("/")
    return (int(num), int(denom))


def _infer_time_signature(output: np.ndarray) -> tuple:
    """Infer time signature from BeatNet output beat numbering.

    BeatNet outputs [beat_time, beat_number] where beat_number resets
    at each downbeat (beat_number=1). The max beat number before reset
    gives beats per measure.
    """
    beat_numbers = output[:, 1].astype(int)
    max_beat = int(np.max(beat_numbers))
    if max_beat < 2:
        max_beat = 4  # fallback
    return (max_beat, 4)


def _estimate_tempo(beat_times: np.ndarray) -> float:
    """Estimate BPM from median inter-beat interval."""
    if len(beat_times) < 2:
        return 120.0
    intervals = np.diff(beat_times)
    median_ibi = float(np.median(intervals))
    if median_ibi <= 0:
        return 120.0
    return 60.0 / median_ibi


class BeatAnalyzer:
    """Wraps BeatNet for offline beat/downbeat/meter detection."""

    def __init__(self, model: int = 1, device: str = "cpu"):
        _patch_beatnet_deps()
        from BeatNet.BeatNet import BeatNet
        self._beatnet = BeatNet(
            model, mode="offline", inference_model="DBN", device=device
        )

    def analyze(
        self, audio_path: str, time_signature_override: str = None
    ) -> BeatInfo:
        """Analyze audio file for beats, downbeats, tempo, and meter.

        Args:
            audio_path: Path to .wav file.
            time_signature_override: e.g. '3/4' to override detected meter.

        Returns:
            BeatInfo with detected metric structure.

        Raises:
            RuntimeError: If beat detection fails completely.
        """
        output = self._beatnet.process(audio_path)
        if output is None or len(output) == 0:
            raise RuntimeError(f"Beat detection failed for {audio_path}")

        beat_times = output[:, 0].astype(np.float64)
        beat_numbers = output[:, 1].astype(int)
        downbeat_times = beat_times[beat_numbers == 1]
        tempo = _estimate_tempo(beat_times)

        if time_signature_override:
            time_sig = _parse_time_signature(time_signature_override)
        else:
            time_sig = _infer_time_signature(output)

        logger.info(
            "Detected: tempo=%.1f BPM, meter=%d/%d, %d beats, %d downbeats",
            tempo, time_sig[0], time_sig[1], len(beat_times), len(downbeat_times),
        )
        return BeatInfo(
            beat_times=beat_times,
            downbeat_times=downbeat_times,
            tempo=tempo,
            time_signature=time_sig,
        )
