# PROMPT: Debug Training Pipeline

## Objective

Fix runtime issues in the TensorFlow music generation training pipeline to ensure it runs without crashes for at least one complete epoch.

## Key Requirements

1. Fix TensorFlow graph execution error in train_step (replace Python control flow with tf.cond)
2. Add shape validation for dataset batches before training
3. Verify model builds correctly on first batch
4. Ensure training completes one full epoch without crashes
5. Validate checkpoint saving works correctly

## Acceptance Criteria

```gherkin
Given the training script is invoked with valid arguments
When training runs for one complete epoch
Then no runtime errors should occur
And loss values should be finite numbers
And at least one checkpoint should be saved
And training metrics should be displayed

Given a TensorFlow graph execution error occurs in train_step
When the error is related to symbolic tensor usage in control flow
Then the code should use tf.cond instead of Python if statements
And training should proceed without the error

Given the model processes a batch
When the batch contains valid mel spectrogram data
Then predictions should have the same shape as inputs
And loss should be computed successfully
And gradients should be applied to model weights
```

## Context

- Training code: `src/music_generation/train.py`
- Known issue: Line ~87 uses `if tf.math.is_nan(loss)` which fails in graph mode
- Model: MelGeneratorTraining wrapper around MelGenerator
- Dataset: tf.data pipeline loading mel spectrograms

## Test Command

```bash
python3 -m src.music_generation.train \
    --data_dir ~/data/music/musicnet/train_data \
    --checkpoint_dir ~/models/conducting/mel_checkpoints \
    --epochs 1 \
    --batch_size 4 \
    --lr 0.0001
```

## Reference

See detailed design and implementation plan in `specs/debug-training/`
