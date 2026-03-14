# Research: Canonical Conducting Patterns

## Pattern Geometry

All conducting patterns share common structure (source: "Music in Motion" by Lesley Mann, CC-BY-4.0):
- Every pattern starts with a **downbeat** (downward arm movement)
- Every pattern ends with an **upbeat** (upward arm movement)
- Additional beats add horizontal movements between down and up
- Each beat has three components: **prep** (approach), **ictus** (change of direction = the beat point), **rebound** (follow-through)

### 2-Beat Pattern (2/4, 2/2)
```
Beat 1 (down): Start high → move down to lowest point (ictus)
Beat 2 (up):   Rebound → sweep up back to start
```
Trajectory: vertical line — down, up. Simple pendulum motion.

### 3-Beat Pattern (3/4, 3/8)
```
Beat 1 (down):  Start high → move down to lowest point (ictus)
Beat 2 (right): Sweep out to the right (ictus at rightmost point)
Beat 3 (up):    Sweep back up to start
```
Trajectory: triangle — down, right, up.

### 4-Beat Pattern (4/4, 4/8)
```
Beat 1 (down): Start high → move down to lowest point (ictus)
Beat 2 (in):   Sweep inward/left (ictus at leftmost point)
Beat 3 (out):  Sweep outward/right past center (ictus at rightmost point)
Beat 4 (up):   Sweep back up to start
```
Trajectory: cross pattern — down, left, right, up.

### 6-Beat Pattern (6/8)
```
Beat 1 (down): Down to lowest point
Beat 2:        Small rebound down-left
Beat 3 (left): Sweep left (ictus)
Beat 4 (right): Sweep right (ictus)
Beat 5:        Small rebound right
Beat 6 (up):   Sweep up to start
```
Compound meter — groups of 3 subdivisions. Can also be conducted in 2 at fast tempos.

### 5-Beat Pattern (5/4)
Typically conducted as 3+2 or 2+3:
- **3+2**: down, left, right (like 3-pattern), then in, up (like 2-pattern)
- **2+3**: down, up (like 2-pattern), then down, right, up (like 3-pattern)

### 7-Beat Pattern (7/8)
Typically 4+3 or 3+4 or 2+2+3, combining sub-patterns.

## Modeling as Keypoint Trajectories

For synthetic generation, each pattern can be modeled as:
1. Define **ictus waypoints** in 2D space (the beat points) for the right wrist
2. Interpolate smooth curves between waypoints using splines or bezier curves
3. Scale amplitude based on dynamics (louder = bigger gestures)
4. Map right wrist trajectory to full skeleton using inverse kinematics or simple joint chain

### Coordinate System (body-relative, matching body-point-module normalization)
- Origin: shoulder midpoint
- X: left-right (positive = right)
- Y: up-down (positive = down, matching image coordinates)
- Scale: normalized by shoulder width

### Ictus Waypoints (approximate, right hand, normalized coordinates)

**4/4 pattern:**
| Beat | X    | Y    | Description |
|------|------|------|-------------|
| 1    | 0.0  | 1.0  | Down center |
| 2    | -0.5 | 0.3  | Left        |
| 3    | 0.5  | 0.3  | Right       |
| 4    | 0.0  | -0.5 | Up          |

**3/4 pattern:**
| Beat | X    | Y    | Description |
|------|------|------|-------------|
| 1    | 0.0  | 1.0  | Down        |
| 2    | 0.5  | 0.3  | Right       |
| 3    | 0.0  | -0.5 | Up          |

**2/4 pattern:**
| Beat | X    | Y    | Description |
|------|------|------|-------------|
| 1    | 0.0  | 1.0  | Down        |
| 2    | 0.0  | -0.5 | Up          |

## Full Upper Body Motion

Beyond the dominant (right) hand:
- **Left hand**: mirrors right hand pattern (symmetrically — hands move in/out, not both left/right)
- **Elbows**: follow wrist trajectory with damping (elbow hinge is primary for basic timekeeping)
- **Shoulders**: slight raise/lower with arm movement
- **Torso**: subtle lean toward beat direction
- **Head**: slight nod on downbeat

## Adding Variation

Realistic variation sources:
- **Amplitude**: scale gesture size (±20-40%)
- **Timing**: slight anticipation/delay of ictus (±5-10% of beat duration)
- **Trajectory noise**: Perlin noise or Gaussian perturbation on waypoints
- **Style**: legato (rounded ictus) vs staccato (sharp ictus) — affects interpolation curve shape
- **Dynamics**: forte = large gestures, piano = small gestures

## Sources
- "Music in Motion: A Conductor's Guide" — Lesley Mann (CC-BY-4.0): https://pressbooks.pub/musicinmotion/chapter/basic_beat_patterns/
- Content was rephrased for compliance with licensing restrictions
