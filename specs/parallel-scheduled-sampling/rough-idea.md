# Rough Idea: Parallel Scheduled Sampling for Mel Generator Training

## Problem

Current training implementation uses slow autoregressive generation:
- 128 forward passes per training step (one per sequence position)
- O(seq_len) complexity per batch
- Very slow training times

## Solution

Implement parallel scheduled sampling:
- Reduce to 2 forward passes per training step
- Maintain scheduled sampling behavior (model sees its own mistakes)
- Keep training fully parallelizable
- Expected 50-100× speedup

## Key Technique

Instead of:
```
predict → append → predict → append → predict (128 times)
```

Do:
```
predict whole sequence once
mix prediction + ground truth based on tf_ratio
compute loss
```

## Context

The technique is described in the provided context and is standard in modern fast sequence models. It maintains the benefits of scheduled sampling (exposure to model's own errors) while avoiding the sequential bottleneck.

## Current Implementation Location

`src/music_generation/train.py` - `MelGeneratorTraining.train_step()` method contains the autoregressive loop that needs optimization.
