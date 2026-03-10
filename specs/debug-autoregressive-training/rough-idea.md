# Rough Idea

## Problem Statement

After implementing autoregressive training for the mel generator, the training command fails to run successfully:

```bash
python3 -m src.music_generation.train
```

## Context

- Recently added autoregressive capability to mel training via a spec
- The training script was working before the autoregressive changes
- Need to identify and fix the bug to restore training functionality

## Goal

Make `python3 -m src.music_generation.train` run successfully with the new autoregressive implementation.
