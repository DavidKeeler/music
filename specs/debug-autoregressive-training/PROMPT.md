# Fix Autoregressive Training Bug

## Objective

Fix `OperatorNotAllowedInGraphError` and `TypeError` in `src/music_generation/train.py` to make training run successfully. Replace Python `if` statement with TensorFlow's graph-compatible `tf.cond` and fix NaN/Inf detection type mismatch.

## Key Requirements

- Fix line 179 in `MelGeneratorTraining.train_step()`: `if self.tf_ratio >= 1.0 - 1e-6:`
- Extract pure teacher forcing logic (lines 180-195) into nested function
- Extract autoregressive logic (lines 197-246) into nested function
- Replace Python `if` with `tf.cond(self.tf_ratio >= 1.0 - 1e-6, pure_teacher_forcing, autoregressive_training)`
- **Fix NaN/Inf detection in both functions:** Replace problematic nested `tf.cond` with simple `tf.print`
  - Current: `tf.cond(..., lambda: tf.print(...), lambda: tf.constant(0))` (type mismatch)
  - Fixed: `tf.print("Loss:", loss, "IsNaN:", tf.math.is_nan(loss), "IsInf:", tf.math.is_inf(loss))`
- Both functions must return identical structure: `{"loss": ..., "grad_norm": ..., "tf_ratio": ...}`
- Minimal changes only - preserve all existing logic

## Acceptance Criteria

```gherkin
Given the fixed training script
When I run "python3 -m src.music_generation.train --data_dir ~/data/music/musicnet/train_data --checkpoint_dir ~/models/conducting/mel_checkpoints --epochs 1 --batch_size 4 --lr 0.00005"
Then the command should complete without errors
And training should complete at least one epoch
And metrics should be logged (loss, grad_norm, tf_ratio)
And no TypeError about mismatched return types should occur

Given tf_ratio is 1.0
When train_step executes
Then pure teacher forcing path should run

Given tf_ratio is 0.5
When train_step executes
Then autoregressive path should run
```

## Reference

See detailed design and plan in `specs/debug-autoregressive-training/`
