# Research: MoveNet Keypoint Mapping for Conducting

## MoveNet Thunder 17 Keypoints

From the body-point-module design (already in this project):

```
Index  Joint
0      nose
1      left_eye
2      right_eye
3      left_ear
4      right_ear
5      left_shoulder
6      right_shoulder
7      left_elbow
8      right_elbow
9      left_wrist
10     right_wrist
11     left_hip
12     right_hip
13     left_knee
14     right_knee
15     left_ankle
16     right_ankle
```

Output format: `[1, 1, 17, 3]` where each keypoint is `(y, x, confidence)`.
Note: MoveNet uses **(y, x)** order, not (x, y).

## Joints Relevant to Conducting

**Primary (drive the pattern):**
- 9: left_wrist, 10: right_wrist — the "baton tips", define the beat pattern trajectory
- 7: left_elbow, 8: right_elbow — follow wrists with damping

**Secondary (support motion):**
- 5: left_shoulder, 6: right_shoulder — slight raise/lower
- 0: nose — subtle nod on downbeat
- 11: left_hip, 12: right_hip — anchor points, mostly static

**Static (lower body):**
- 13-16: knees and ankles — conductors stand still, these stay fixed
- 1-4: eyes and ears — follow head, minimal independent motion

## Synthetic Generation Strategy

1. **Define wrist trajectories** from canonical beat patterns (see conducting-patterns.md)
2. **Derive elbow positions** via simple 2-link inverse kinematics (upper arm + forearm lengths)
3. **Derive shoulder positions** with slight vertical displacement proportional to arm raise
4. **Head/nose**: small nod (downward Y displacement) on beat 1
5. **Hips, knees, ankles**: static with very small noise
6. **Eyes, ears**: offset from nose position (fixed relative positions + tiny noise)
7. **Confidence**: set to 1.0 for all synthetic keypoints (or vary slightly for realism)

## Normalization Compatibility

The body-point-module normalizes keypoints:
- Center at shoulder midpoint
- Scale by shoulder width
- Convert to body-relative coordinates

The synthetic generator should output in the **same normalized space** so the data is directly compatible. This means we define conducting patterns in normalized coordinates and don't need to worry about absolute pixel positions.

## Sources
- Body-point-module design: specs/body-point-module/design.md (this project)
- MoveNet documentation: https://www.tensorflow.org/hub/tutorials/movenet
