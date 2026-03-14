# Dataset Download Ralph Job

Set up a Ralph job to download datasets for the conducting project (music → conducting motion / pose → beat structure).

## Datasets by Priority

### Tier 1 — True Conducting Datasets
- **PHENICX Conduct Dataset** (MTG, UPF): RGB video, Kinect depth, 3D skeleton, audio, scores, beat/loudness annotations. 25 conductors, Beethoven symphonies, ~1.5 hours. Academic access required.
- **Edinburgh Orchestral Conducting MoCap**: Optical mocap, 3D joint trajectories, baton tracking. 6 professional conductors, C3D files. University of Edinburgh data repository.

### Tier 2 — Musical Performance Motion
- **MOSA Dataset**: 3D mocap + audio + musical annotations (beats, phrases, articulation, dynamics). ~30 hours, ~742 performances. Zenodo or project repo.

### Tier 3 — Dance and Rhythm Motion
- **DeepDance MoCap**: Full body mocap, dance, music-synced. 30 dancers, ~540 sequences, 240 Hz. Zenodo.

### Tier 4 — Music Performance Video
- **URMP Dataset** (U of Rochester): Multi-instrument ensemble recordings, isolated tracks, synced video. 44 pieces.

### Tier 5 — Generic Human Motion (Pretraining)
- **Human3.6M**: Large-scale mocap, 11 actors, ~3.6M frames. License agreement required.
- **MPII Human Pose**: Annotated images with body keypoints.

### Tier 6 — Synthetic (Generate)
- **Synthetic Beat Trajectory Dataset**: Parametric baton trajectories for 2/4, 3/4, 4/4 patterns. 1M+ sequences.

### Tier 7 — Collected from YouTube
- **Orchestra Conducting Video Dataset**: YouTube orchestral recordings → pose estimation → beat alignment. 1000+ hours potential.

## Unified Output Format
- Pose: 17–33 joint skeleton
- Features: hand velocity, acceleration, trajectory
- Music: tempo, beat positions, downbeats, loudness envelope
