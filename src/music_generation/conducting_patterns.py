"""Canonical conducting patterns, spline interpolation, and variation."""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .beat_analyzer import BeatInfo

logger = logging.getLogger(__name__)

# Canonical ictus waypoints: (x, y) in shoulder-normalized coordinates.
# y-positive = down (image coords). x-positive = right.
PATTERNS = {
    (2, 4): [(0.0, 1.0), (0.0, -0.5)],
    (3, 4): [(0.0, 1.0), (0.5, 0.3), (0.0, -0.5)],
    (4, 4): [(0.0, 1.0), (-0.5, 0.3), (0.5, 0.3), (0.0, -0.5)],
    # Stretch patterns
    (6, 8): [
        (0.0, 1.0), (0.0, 0.7), (-0.4, 0.3),
        (0.4, 0.3), (0.4, 0.0), (0.0, -0.5),
    ],
    (5, 4): [(0.0, 1.0), (-0.5, 0.3), (0.5, 0.3), (0.0, 0.5), (0.0, -0.5)],
    (7, 8): [
        (0.0, 1.0), (-0.5, 0.3), (0.5, 0.3), (0.0, -0.5),
        (0.0, 0.8), (0.3, 0.3), (0.0, -0.5),
    ],
}


@dataclass
class VariationParams:
    """Per-sample variation parameters."""
    amplitude_scale: float = 1.0   # multiplier on gesture size
    timing_jitter: float = 0.0     # fraction of beat duration (±)
    noise_std: float = 0.0         # Gaussian noise std on trajectory
    style: str = "legato"          # "legato" or "staccato"


def _sample_variation(rng: np.random.Generator) -> VariationParams:
    """Sample random variation parameters."""
    return VariationParams(
        amplitude_scale=rng.uniform(0.7, 1.3),
        timing_jitter=rng.uniform(0.0, 0.08),
        noise_std=rng.uniform(0.005, 0.02),
        style=rng.choice(["legato", "staccato"]),
    )


def interpolate_trajectory(
    waypoints: list[tuple[float, float]],
    beat_times: np.ndarray,
    fps: int,
    duration: float,
    variation: VariationParams | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Interpolate smooth wrist trajectory from ictus waypoints.

    Args:
        waypoints: List of (x, y) ictus positions for one measure.
        beat_times: Beat timestamps in seconds (may span multiple measures).
        fps: Target frame rate.
        duration: Total duration in seconds.
        variation: Variation parameters (amplitude, jitter, noise, style).
        rng: Random generator for jitter/noise.

    Returns:
        [num_frames, 2] array of (x, y) wrist positions.
    """
    if variation is None:
        variation = VariationParams()
    if rng is None:
        rng = np.random.default_rng()

    num_frames = int(duration * fps)
    if num_frames == 0 or len(beat_times) < 2:
        return np.zeros((max(num_frames, 1), 2), dtype=np.float32)

    beats_per_measure = len(waypoints)

    # Build time→waypoint mapping across all beats
    t_knots = []
    xy_knots = []

    for i, bt in enumerate(beat_times):
        if bt > duration:
            break
        wp_idx = i % beats_per_measure
        wx, wy = waypoints[wp_idx]

        # Apply amplitude scaling
        wx *= variation.amplitude_scale
        wy *= variation.amplitude_scale

        # Apply timing jitter
        jittered_t = bt
        if i > 0 and variation.timing_jitter > 0:
            ibi = beat_times[min(i, len(beat_times) - 1)] - beat_times[max(i - 1, 0)]
            jittered_t = bt + rng.uniform(-1, 1) * variation.timing_jitter * ibi

        t_knots.append(jittered_t)
        xy_knots.append([wx, wy])

    if len(t_knots) < 2:
        return np.zeros((num_frames, 2), dtype=np.float32)

    t_knots = np.array(t_knots)
    xy_knots = np.array(xy_knots)

    # Sort by time (jitter may reorder slightly)
    order = np.argsort(t_knots)
    t_knots = t_knots[order]
    xy_knots = xy_knots[order]

    # Remove duplicate times
    mask = np.diff(t_knots, prepend=-1.0) > 1e-6
    t_knots = t_knots[mask]
    xy_knots = xy_knots[mask]

    if len(t_knots) < 2:
        return np.zeros((num_frames, 2), dtype=np.float32)

    # Cubic spline interpolation
    bc = "clamped" if variation.style == "staccato" else "not-a-knot"
    cs = CubicSpline(t_knots, xy_knots, bc_type=bc)

    t_frames = np.linspace(0, duration, num_frames, endpoint=False)
    # Clamp to spline domain
    t_clamped = np.clip(t_frames, t_knots[0], t_knots[-1])
    trajectory = cs(t_clamped).astype(np.float32)

    # Add Gaussian noise
    if variation.noise_std > 0:
        trajectory += rng.normal(0, variation.noise_std, trajectory.shape).astype(
            np.float32
        )

    return trajectory


class PatternGenerator:
    """Generate conducting keypoint sequences from beat structure.

    This produces wrist trajectories only ([num_frames, 2]).
    Use SkeletonBuilder to expand to full 17-joint skeleton.
    """

    def generate_wrist_trajectory(
        self,
        beat_info: BeatInfo,
        fps: int,
        duration: float,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate right-wrist (x, y) trajectory.

        Args:
            beat_info: Beat analysis results.
            fps: Target frame rate.
            duration: Duration in seconds.
            seed: Random seed for reproducibility.

        Returns:
            [num_frames, 2] array of (x, y) wrist positions.

        Raises:
            ValueError: If time signature is not supported.
        """
        ts = tuple(beat_info.time_signature)
        if ts not in PATTERNS:
            raise ValueError(
                f"Unsupported time signature {ts}. "
                f"Supported: {list(PATTERNS.keys())}"
            )

        rng = np.random.default_rng(seed)
        variation = _sample_variation(rng)
        waypoints = PATTERNS[ts]

        logger.info(
            "Generating %s pattern: amp=%.2f, jitter=%.3f, style=%s",
            f"{ts[0]}/{ts[1]}",
            variation.amplitude_scale,
            variation.timing_jitter,
            variation.style,
        )

        return interpolate_trajectory(
            waypoints=waypoints,
            beat_times=beat_info.beat_times,
            fps=fps,
            duration=duration,
            variation=variation,
            rng=rng,
        )
