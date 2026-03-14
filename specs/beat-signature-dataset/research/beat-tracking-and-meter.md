# Research: Beat Tracking & Time Signature Detection

## Beat Tracking Libraries

### BeatNet (Recommended)
- **Repo**: https://github.com/mjhydri/BeatNet
- **Paper**: ISMIR 2021 — CRNN + particle filtering
- Joint beat, downbeat, tempo, and **meter tracking** in a single system
- Outputs: numpy array of (beat_time, beat_number) — beat_number=1 indicates downbeat
- Meter detection is built-in (tagged as `meter-detection` and `online-time-signature-detection`)
- Modes: streaming, realtime, online, offline
- Offline mode uses madmom's DBN inference with BeatNet's neural network
- Install: `pip install BeatNet` (requires librosa and madmom)
- **Key advantage**: Directly outputs meter/time signature, not just beat positions

### madmom
- **Repo**: https://github.com/CPJKU/madmom
- State-of-the-art MIR library with onset detection, beat/downbeat/meter tracking, tempo estimation
- `DBNDownBeatTrackingProcessor` can output beat positions with beat numbers (1=downbeat)
- Meter must be inferred from the pattern of downbeats and beat counts
- More manual work to get time signature vs. BeatNet's direct output
- Well-established, widely used in research

### librosa.beat
- `librosa.beat.beat_track()` — simple beat tracking
- No built-in downbeat or meter detection
- Would need additional logic to infer time signature from beat intervals
- Simplest API but least capable for our needs

## Recommendation

**BeatNet in offline mode** is the best fit:
1. Directly provides meter detection (time signature)
2. Joint beat + downbeat tracking gives us the full metric structure
3. Offline mode is fine since we're generating datasets, not running real-time
4. Falls back to madmom's DBN for robust offline inference

**Fallback**: madmom's `DBNDownBeatTrackingProcessor` with manual meter inference from downbeat spacing.

## Meter Detection Reliability

Time signature auto-detection is imperfect. Common failure modes:
- Confusing 4/4 with 2/2 (same beat pattern, different grouping)
- Confusing 3/4 with 6/8 (both have 3-feel)
- Irregular meters (5/4, 7/8) are hardest to detect
- Tempo changes or rubato can confuse beat trackers

**Mitigation**: Allow CLI override of detected time signature. Log confidence/detected meter so user can filter or correct.

## Sources
- BeatNet: https://github.com/mjhydri/BeatNet (CC-BY-4.0)
- madmom: https://github.com/CPJKU/madmom
- Content was rephrased for compliance with licensing restrictions
