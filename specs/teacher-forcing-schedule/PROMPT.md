# PROMPT.md

## Objective

Implement exponential decay scheduled sampling for the mel spectrogram generator to reduce exposure bias and improve autoregressive inference quality.

## Problem

Current training uses pure teacher forcing (always feeding ground truth previous frames). During inference, the model uses its own predictions, causing train-test mismatch that leads to error accumulation and poor generation quality.

## Solution

Gradually transition from teacher forcing to model predictions during training using exponential decay: `ε(step) = max(ε_min, ε_initial * exp(-k * step))`

## Key Requirements

1. **Exponential decay schedule function**
   - Formula: `ε(step) = max(ε_min, ε_initial * exp(-k * step))`
   - Per-step updates (not per-epoch)
   - Support warmup period

2. **Configuration parameters** (add to `src/music_generation/config.py`)
   - `initial_tf_ratio: float = 1.0`
   - `min_tf_ratio: float = 0.05`
   - `tf_decay_k: float = 1e-5`
   - `tf_warmup_steps: int = 0`

3. **Memory-efficient training step** (modify `src/music_generation/train.py`)
   - Use TensorArray for prediction collection (NOT Python lists)
   - Use sliding window for context (NOT growing concatenation)
   - Short-circuit to single forward pass when `tf_ratio >= 1.0`
   - Probabilistic sampling: use ground truth with probability ε, else use model prediction

4. **Trainer modifications**
   - Add `tf_ratio` as non-trainable `tf.Variable`
   - Add `update_tf_ratio(step)` method
   - Track ratio as metric, log alongside loss

5. **Validation with pure autoregressive**
   - Evaluate with `tf_ratio = 0.0` (matches inference conditions)

6. **Checkpoint save/load**
   - Save/restore `tf_ratio` and `training_step`
   - Backward compatible with existing checkpoints

## Acceptance Criteria

### Given: Training starts with scheduled sampling enabled
**When:** First training step executes
**Then:** 
- Teacher forcing ratio initializes to `initial_tf_ratio`
- Training proceeds without errors
- Ratio is logged

### Given: Training in progress
**When:** Each step completes
**Then:**
- Ratio decreases according to exponential schedule
- Ratio never falls below `min_tf_ratio`
- Ratio is logged alongside loss

### Given: Training step with tf_ratio = 0.5
**When:** Autoregressive generation occurs
**Then:**
- ~50% of timesteps use ground truth
- ~50% of timesteps use model predictions
- Gradients flow through both paths

### Given: Training with tf_ratio = 0.0
**When:** Training step executes
**Then:**
- All timesteps use model predictions (pure autoregressive)
- Training completes without errors

### Given: Checkpoint saved during training
**When:** Training resumes from checkpoint
**Then:**
- Teacher forcing ratio restored
- Schedule continues from correct step

### Given: Validation evaluation
**When:** Validation runs
**Then:**
- Uses pure autoregressive (tf_ratio = 0)
- Reports validation loss

### Given: Warmup period configured
**When:** Step < warmup_steps
**Then:**
- Ratio remains at `initial_tf_ratio`
- Decay starts after warmup

## Critical Memory Patterns

**AVOID:**
- Growing concatenation: `tf.concat([seq, frame], axis=1)` in loop
- Python list accumulation: `predictions.append(pred)`
- Multiple `tf.cond` branches inside `GradientTape`

**USE:**
- TensorArray with `dynamic_size=False`
- Sliding window: `current_input[:, -max_context:, :]`
- Short-circuit when `tf_ratio >= 1.0`

## Reference

Full specification: `specs/teacher-forcing-schedule/`
- `design.md` - Detailed design with memory optimizations
- `plan.md` - 8-step implementation plan
- `research/` - Background and implementation patterns

## Implementation Notes

- Modify existing `src/music_generation/train.py`
- Add parameters to `src/music_generation/config.py`
- Follow TDD: write tests for each component
- Each step should result in working, integrated code
- Estimated effort: 12-18 hours
