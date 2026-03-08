# Implementation Plan

## Checklist

- [ ] Step 1: Add schedule function
- [ ] Step 2: Update configuration
- [ ] Step 3: Add teacher forcing ratio tracking to trainer
- [ ] Step 4: Implement memory-efficient training step
- [ ] Step 5: Update training loop
- [ ] Step 6: Add validation with pure autoregressive
- [ ] Step 7: Update checkpoint save/load
- [ ] Step 8: Add tests

---

## Step 1: Add Schedule Function

**Objective:** Create exponential decay schedule function for teacher forcing ratio.

**Implementation:**
- Add `exponential_tf_schedule()` function to `src/music_generation/train.py`
- Implement formula: `ε(step) = max(ε_min, ε_initial * exp(-k * step))`
- Use TensorFlow operations for compatibility with training loop
- Handle edge cases (step=0, large steps)

**Tests:**
- Unit test: verify exponential decay formula
- Unit test: verify min_ratio floor is respected
- Unit test: test with step=0 returns initial_ratio
- Unit test: test with large step approaches min_ratio

**Integration:**
- Function is pure (no side effects)
- Returns scalar float32 tensor
- Can be called per training step

**Demo:**
```python
# Verify schedule behavior
import matplotlib.pyplot as plt
steps = range(0, 100000, 1000)
ratios = [exponential_tf_schedule(s, 1.0, 0.05, 1e-5) for s in steps]
plt.plot(steps, ratios)
plt.xlabel('Step')
plt.ylabel('TF Ratio')
plt.title('Exponential Decay Schedule')
plt.savefig('tf_schedule.png')
```

---

## Step 2: Update Configuration

**Objective:** Add teacher forcing schedule parameters to config.

**Implementation:**
- Add four new fields to `TrainingConfig` in `src/music_generation/config.py`:
  - `initial_tf_ratio: float = 1.0`
  - `min_tf_ratio: float = 0.05`
  - `tf_decay_k: float = 1e-5`
  - `tf_warmup_steps: int = 0`
- Add validation in config `__post_init__` if present
- Add docstrings explaining each parameter

**Tests:**
- Unit test: verify default values
- Unit test: verify custom values are accepted
- Unit test: verify invalid values raise errors (ratios outside [0,1])

**Integration:**
- Config changes are backward compatible (new fields have defaults)
- Existing configs continue to work

**Demo:**
```python
# Verify config loading
from src.music_generation.config import TrainingConfig
config = TrainingConfig(
    initial_tf_ratio=1.0,
    min_tf_ratio=0.05,
    tf_decay_k=1e-5,
    tf_warmup_steps=5000
)
print(f"TF Schedule: {config.initial_tf_ratio} -> {config.min_tf_ratio}, k={config.tf_decay_k}")
```

---

## Step 3: Add Teacher Forcing Ratio Tracking to Trainer

**Objective:** Add tf_ratio variable and update logic to trainer class.

**Implementation:**
- In `Trainer.__init__()`:
  - Create `self.tf_ratio` as non-trainable `tf.Variable`
  - Store schedule parameters from config
  - Create `self.tf_ratio_metric` for logging
- Add `update_tf_ratio(step)` method:
  - Check if step < warmup_steps
  - Call `exponential_tf_schedule()` with effective step
  - Assign new ratio to `self.tf_ratio`
  - Update metric
- Call `update_tf_ratio()` at start of each training step

**Tests:**
- Unit test: verify tf_ratio initializes correctly
- Unit test: verify warmup period keeps ratio at initial value
- Unit test: verify ratio decreases after warmup
- Integration test: verify ratio updates during training loop

**Integration:**
- Trainer class maintains backward compatibility
- Ratio is tracked as a metric (logged with loss)

**Demo:**
```python
# Verify ratio tracking
trainer = Trainer(config)
print(f"Initial ratio: {trainer.tf_ratio.numpy()}")
for step in range(0, 10000, 1000):
    trainer.update_tf_ratio(step)
    print(f"Step {step}: ratio = {trainer.tf_ratio.numpy():.4f}")
```

---

## Step 4: Implement Memory-Efficient Training Step

**Objective:** Modify training step to use scheduled sampling with memory optimizations.

**Implementation:**
- Create new `train_step_with_scheduled_sampling()` method
- Use TensorArray for prediction collection (not Python list)
- Use sliding window for context (fixed memory)
- Short-circuit to single forward pass when `tf_ratio >= 1.0`
- Use `tf.cond` for per-step sampling decision
- Ensure gradients flow through model predictions

**Key memory optimizations:**
- TensorArray with `dynamic_size=False` for fixed allocation
- Sliding window: `current_input[:, -max_context:, :]`
- Avoid growing concatenation in loop
- Pre-allocate when possible

**Tests:**
- Unit test: verify tf_ratio=1.0 uses pure teacher forcing
- Unit test: verify tf_ratio=0.0 uses pure autoregressive
- Unit test: verify gradients flow correctly
- Integration test: verify loss computation matches expected
- Memory test: verify no memory growth over multiple steps

**Integration:**
- Replace existing `train_step()` or add as alternative
- Works with existing loss functions and optimizer

**Demo:**
```python
# Verify training step works
model = MelGenerator(config)
optimizer = tf.keras.optimizers.Adam()
x_batch, y_batch = next(iter(train_dataset))

# Test with different ratios
for ratio in [1.0, 0.5, 0.0]:
    trainer.tf_ratio.assign(ratio)
    loss = train_step_with_scheduled_sampling(
        model, x_batch, y_batch, trainer.tf_ratio, optimizer, loss_fn
    )
    print(f"TF ratio {ratio}: loss = {loss.numpy():.4f}")
```

---

## Step 5: Update Training Loop

**Objective:** Integrate scheduled sampling into main training loop.

**Implementation:**
- In main training loop (likely `train()` function):
  - Call `trainer.update_tf_ratio(step)` at start of each step
  - Use `train_step_with_scheduled_sampling()` instead of old training step
  - Log `tf_ratio` alongside loss metrics
  - Add to TensorBoard/logging output
- Ensure step counter is tracked correctly for schedule

**Tests:**
- Integration test: run training for multiple steps
- Integration test: verify ratio decreases over time
- Integration test: verify metrics are logged
- Integration test: verify training converges

**Integration:**
- Training loop maintains existing structure
- Logging includes new tf_ratio metric
- Compatible with existing callbacks and monitoring

**Demo:**
```bash
# Run training with scheduled sampling
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --checkpoint_dir ./checkpoints_scheduled \
  --epochs 10 \
  --batch_size 16 \
  --initial_tf_ratio 1.0 \
  --min_tf_ratio 0.05 \
  --tf_decay_k 1e-5

# Check logs show decreasing tf_ratio
tail -f logs/training.log | grep "tf_ratio"
```

---

## Step 6: Add Validation with Pure Autoregressive

**Objective:** Evaluate model with pure autoregressive generation during validation.

**Implementation:**
- Create validation step that uses `tf_ratio=0.0`
- Run validation at end of each epoch
- Log validation loss separately from training loss
- Compare validation loss with/without teacher forcing

**Tests:**
- Integration test: verify validation runs without errors
- Integration test: verify validation uses tf_ratio=0.0
- Integration test: verify validation loss is computed correctly

**Integration:**
- Validation step mirrors inference conditions
- Helps monitor exposure bias reduction
- Can be used for early stopping

**Demo:**
```python
# Run validation
val_loss_autoregressive = validate(model, val_dataset, tf_ratio=0.0)
val_loss_teacher_forcing = validate(model, val_dataset, tf_ratio=1.0)
print(f"Validation loss (autoregressive): {val_loss_autoregressive:.4f}")
print(f"Validation loss (teacher forcing): {val_loss_teacher_forcing:.4f}")
print(f"Gap: {val_loss_teacher_forcing - val_loss_autoregressive:.4f}")
```

---

## Step 7: Update Checkpoint Save/Load

**Objective:** Save and restore teacher forcing ratio and schedule state.

**Implementation:**
- In checkpoint save:
  - Add `tf_ratio` to checkpoint dict
  - Add `training_step` for schedule continuation
  - Add schedule config parameters
- In checkpoint load:
  - Restore `tf_ratio` if present
  - Restore `training_step` if present
  - Handle missing fields for backward compatibility
- Update checkpoint metadata

**Tests:**
- Integration test: save checkpoint mid-training
- Integration test: load checkpoint and verify ratio restored
- Integration test: continue training and verify schedule continues
- Integration test: load old checkpoint without tf_ratio (backward compat)

**Integration:**
- Checkpoint format remains backward compatible
- Old checkpoints load with default tf_ratio=1.0
- New checkpoints include schedule state

**Demo:**
```python
# Save and restore checkpoint
trainer.save_checkpoint('checkpoint_step_10000.h5', step=10000)
print(f"Saved with tf_ratio: {trainer.tf_ratio.numpy()}")

# Load in new trainer
new_trainer = Trainer(config)
new_trainer.load_checkpoint('checkpoint_step_10000.h5')
print(f"Restored tf_ratio: {new_trainer.tf_ratio.numpy()}")
```

---

## Step 8: Add Tests

**Objective:** Comprehensive test coverage for scheduled sampling.

**Implementation:**

**Unit tests** (`tests/test_scheduled_sampling.py`):
- Test `exponential_tf_schedule()` function
- Test config validation
- Test ratio update logic
- Test warmup period behavior

**Integration tests** (`tests/test_training_scheduled.py`):
- Test training step with different ratios
- Test full training loop
- Test checkpoint save/load
- Test validation step

**Memory tests** (`tests/test_memory_efficiency.py`):
- Test no memory growth over multiple steps
- Test TensorArray usage
- Test sliding window behavior

**End-to-end test**:
- Train small model for few steps
- Verify ratio decreases
- Verify loss converges
- Generate sample and verify quality

**Tests:**
- All tests pass
- Coverage > 90% for new code
- No memory leaks detected

**Integration:**
- Tests run with `pytest tests/`
- CI/CD pipeline includes new tests

**Demo:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_scheduled_sampling.py -v

# Run with coverage
pytest tests/ --cov=src.music_generation --cov-report=html
```

---

## Notes

**Dependencies between steps:**
- Step 1 → Step 3 (schedule function used by trainer)
- Step 2 → Step 3 (config used by trainer)
- Steps 1-3 → Step 4 (training step uses schedule and config)
- Step 4 → Step 5 (training loop uses new training step)
- Step 5 → Step 6 (validation uses training infrastructure)
- Steps 1-6 → Step 7 (checkpoint includes all state)
- Steps 1-7 → Step 8 (tests cover all functionality)

**Each step is independently testable and results in working, integrated code.**

**Estimated effort:**
- Steps 1-2: 1-2 hours (simple additions)
- Step 3: 1-2 hours (trainer modifications)
- Step 4: 3-4 hours (core implementation, memory optimization)
- Step 5: 1-2 hours (integration)
- Step 6: 1-2 hours (validation)
- Step 7: 2-3 hours (checkpoint handling)
- Step 8: 3-4 hours (comprehensive tests)
- **Total: 12-18 hours**
