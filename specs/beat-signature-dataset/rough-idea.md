# Rough Idea: Beat Signature Dataset

## Overview

Create a synthetic dataset of music aligned with body point annotations for conducting beat signatures. Conductors use different motions for different time signatures (2/4, 3/4, 4/4, etc.), and we need training data that pairs music with the corresponding conducting gestures.

## Goal

Build a dataset pipeline that produces paired (audio, body keypoint sequence) samples where:
- Audio is annotated with beat/downbeat positions and time signature
- Body keypoint sequences represent canonical conducting patterns for each time signature
- The data can be used to train models that map audio → conducting motion

## Key Aspects

- **Time signatures**: 2/4, 3/4, 4/4, and potentially 6/8, 5/4, etc.
- **Beat alignment**: Body motion must be precisely synchronized with musical beats
- **Synthetic generation**: Conducting patterns can be procedurally generated from canonical beat patterns rather than requiring real motion capture
- **Music sources**: Existing music datasets (MusicNet, etc.) with beat/meter annotations

## Prior Research (from another agent — partially vetted)

Several existing audio-to-motion frameworks may inform the approach:

- **EMAGE** — Full-body audio gesture generation using Masked Audio-Gesture Transformer. Code and dataset available. Not conducting-specific but relevant architecture.
- **DuetGen** — Music-to-dance motion generation (SIGGRAPH 2025). Hierarchical generative model mapping music features → motion tokens.
- **music2dance** — GAN-based audio-driven dance motion generation.
- **Awesome Gesture Generation** — Community-curated list of audio-to-gesture/motion projects.

### Suggested approach from research:
1. Extract music features (tempo, beats, spectral flux, chroma) via Librosa
2. Model skeletal pose as keypoint vectors (17–33 joints)
3. Use audio encoder + motion decoder or transformer-based sequence mapping

**Note**: The research above is partially unvetted. Some links/claims may be outdated or inaccurate. Needs verification during the research phase.

## Context

This integrates with the existing TensorFlow music generation system and the `body-point-module` spec (which handles video → pose extraction → embeddings for cross-attention). This dataset would provide training data for that pipeline.
