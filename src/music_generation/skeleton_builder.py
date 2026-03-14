"""Expand wrist trajectory to full 17-joint MoveNet skeleton."""

import numpy as np

# MoveNet Thunder joint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Rest pose in shoulder-normalized (x, y) coordinates.
# Origin = shoulder midpoint, unit = shoulder width.
REST_POSE_XY = {
    NOSE: (0.0, -0.8),
    LEFT_EYE: (-0.1, -0.85),
    RIGHT_EYE: (0.1, -0.85),
    LEFT_EAR: (-0.2, -0.8),
    RIGHT_EAR: (0.2, -0.8),
    LEFT_SHOULDER: (-0.5, 0.0),
    RIGHT_SHOULDER: (0.5, 0.0),
    LEFT_HIP: (-0.3, 1.2),
    RIGHT_HIP: (0.3, 1.2),
    LEFT_KNEE: (-0.3, 2.2),
    RIGHT_KNEE: (0.3, 2.2),
    LEFT_ANKLE: (-0.3, 3.2),
    RIGHT_ANKLE: (0.3, 3.2),
}

UPPER_ARM_LEN = 0.6  # shoulder widths


def _solve_elbow(shoulder_xy: np.ndarray, wrist_xy: np.ndarray) -> np.ndarray:
    """Simple 2-link IK: place elbow between shoulder and wrist.

    Uses midpoint biased toward shoulder, offset perpendicular to the arm
    to create a natural bend.
    """
    mid = 0.4 * shoulder_xy + 0.6 * wrist_xy
    arm_vec = wrist_xy - shoulder_xy
    arm_len = np.linalg.norm(arm_vec)
    if arm_len < 1e-6:
        return mid
    # Perpendicular offset (outward bend)
    perp = np.array([-arm_vec[1], arm_vec[0]])
    bend = 0.15 * min(1.0, UPPER_ARM_LEN / max(arm_len, 1e-6))
    return mid + perp * bend


def _downbeat_mask(downbeat_times: np.ndarray, fps: int, num_frames: int,
                   radius: int = 4) -> np.ndarray:
    """Return [num_frames] float mask: 1.0 at downbeat, fading over ±radius frames."""
    mask = np.zeros(num_frames, dtype=np.float32)
    for dt in downbeat_times:
        center = int(round(dt * fps))
        for offset in range(-radius, radius + 1):
            idx = center + offset
            if 0 <= idx < num_frames:
                weight = 1.0 - abs(offset) / (radius + 1)
                mask[idx] = max(mask[idx], weight)
    return mask


class SkeletonBuilder:
    """Expand wrist trajectory to 17-joint MoveNet skeleton."""

    def __init__(self, nod_amount: float = 0.06):
        self.nod_amount = nod_amount

    def build_sequence(
        self,
        right_wrist_xy: np.ndarray,
        downbeat_times: np.ndarray,
        fps: int,
    ) -> np.ndarray:
        """Build full skeleton sequence from right-wrist trajectory.

        Args:
            right_wrist_xy: [num_frames, 2] (x, y) in shoulder-normalized coords.
            downbeat_times: Downbeat timestamps in seconds.
            fps: Frame rate.

        Returns:
            [num_frames, 17, 3] in MoveNet (y, x, confidence) format.
        """
        num_frames = len(right_wrist_xy)
        out = np.zeros((num_frames, 17, 3), dtype=np.float32)

        # Downbeat nod mask
        nod = _downbeat_mask(downbeat_times, fps, num_frames) * self.nod_amount

        r_shoulder = np.array(REST_POSE_XY[RIGHT_SHOULDER])
        l_shoulder = np.array(REST_POSE_XY[LEFT_SHOULDER])

        for f in range(num_frames):
            rw = right_wrist_xy[f]  # (x, y)
            # Mirror for left wrist: negate x
            lw = np.array([-rw[0], rw[1]])

            # Elbows via IK
            r_elbow = _solve_elbow(r_shoulder, rw)
            l_elbow = _solve_elbow(l_shoulder, lw)

            # Shoulder vertical shift proportional to wrist height
            r_shift = -0.05 * rw[1]
            l_shift = -0.05 * lw[1]

            # Head nod on downbeat
            nod_y = nod[f]

            frame = {}
            # Head group
            for jid, (bx, by) in [(NOSE, REST_POSE_XY[NOSE]),
                                   (LEFT_EYE, REST_POSE_XY[LEFT_EYE]),
                                   (RIGHT_EYE, REST_POSE_XY[RIGHT_EYE]),
                                   (LEFT_EAR, REST_POSE_XY[LEFT_EAR]),
                                   (RIGHT_EAR, REST_POSE_XY[RIGHT_EAR])]:
                frame[jid] = (bx, by + nod_y)

            # Shoulders with shift
            frame[LEFT_SHOULDER] = (l_shoulder[0], l_shoulder[1] + l_shift)
            frame[RIGHT_SHOULDER] = (r_shoulder[0], r_shoulder[1] + r_shift)

            # Arms
            frame[LEFT_ELBOW] = tuple(l_elbow)
            frame[RIGHT_ELBOW] = tuple(r_elbow)
            frame[LEFT_WRIST] = tuple(lw)
            frame[RIGHT_WRIST] = tuple(rw)

            # Static lower body
            for jid in (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
                        LEFT_ANKLE, RIGHT_ANKLE):
                frame[jid] = REST_POSE_XY[jid]

            # Write as (y, x, confidence)
            for jid, (x, y) in frame.items():
                out[f, jid] = [y, x, 1.0]

        return out
