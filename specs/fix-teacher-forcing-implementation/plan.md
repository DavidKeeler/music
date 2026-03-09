# Implementation Plan

## Checklist

- [ ] Step 1: Add MAX_CONTEXT_FRAMES to config.py
- [ ] Step 2: Update train.py to use configurable context
- [ ] Step 3: Add unit tests for sliding window logic
- [ ] Step 4: Add memory profiling test
- [ ] Step 5: Add integration test with full training step
- [ ] Step 6: Run tests and validate
- [ ] Step 7: Update documentation

---

## Step 1: Add MAX_CONTEXT_FRAMES to config.py

**Objective**: Add configuration parameter for context window size.

**Implementation**:
- Add `MAX_CONTEXT_FRAMES = SEQ_LEN` after `TF_WARMUP_STEPS` definition
- Add comment explaining purpose and default value
- Add assertion to validate positive integer value

**Tests**:
- Import config and verify `MAX_CONTEXT_FRAMES` exists
- Verify default value equals `SEQ_LEN`

**Integration**: Configuration is now available for use in training code.

**Demo**: Print config value, show it equals 512.

---

## Step 2: Update train.py to use configurable context

**Objective**: Replace hardcoded `max_context = 64` with configurable value.

**Implementation**:
- Import `MAX_CONTEXT_FRAMES` from config at top of file
- In `train_step` method (line ~201), replace `max_context = 64` with `max_context = MAX_CONTEXT_FRAMES`
- No other changes needed - sliding window logic is correct

**Tests**:
- Verify import works
- Check training step executes without errors
- Validate context window uses correct size

**Integration**: Training now uses full context by default.

**Demo**: Run single training step, verify it completes successfully.

---

## Step 3: Add unit tests for sliding window logic

**Objective**: Test sliding window update maintains constant size and correct behavior.

**Implementation**:
Create `tests/test_teacher_forcing.py`:

```python
def test_sliding_window_constant_size():
    """Verify sliding window maintains constant size."""
    # Create initial window and new frame
    # Simulate update: window[:, 1:] + new_frame
    # Assert output shape equals input shape

def test_sliding_window_drops_oldest():
    """Verify oldest frame is dropped."""
    # Create window with known values
    # Add new frame
    # Assert first frame is gone, last frame is new

def test_sliding_window_edge_cases():
    """Test edge cases: window_size=1, empty initial context."""
    # Test minimal window size
    # Test growing from small initial context
```

**Tests**: Run pytest on new test file.

**Integration**: Validates core sliding window logic in isolation.

**Demo**: Show all tests passing with clear assertions.

---

## Step 4: Add memory profiling test

**Objective**: Verify memory usage remains constant during autoregressive loop.

**Implementation**:
Add to `tests/test_teacher_forcing.py`:

```python
def test_memory_constant_during_training():
    """Profile memory usage during training step."""
    # Create small model and data
    # Run training step with memory profiler
    # Assert memory doesn't grow beyond threshold
    # Compare 64 vs 512 context sizes

def test_no_memory_leak():
    """Run multiple training steps, check for leaks."""
    # Run 10 training steps
    # Track memory after each step
    # Assert memory stabilizes (no continuous growth)
```

Use `tracemalloc` or `memory_profiler` for tracking.

**Tests**: Run memory tests, verify no leaks.

**Integration**: Ensures implementation is memory-safe.

**Demo**: Show memory usage graph - flat line after initial allocation.

---

## Step 5: Add integration test with full training step

**Objective**: Test complete training step with different MAX_CONTEXT_FRAMES values.

**Implementation**:
Add to `tests/test_teacher_forcing.py`:

```python
def test_training_step_with_different_context_sizes():
    """Test training step with various MAX_CONTEXT_FRAMES."""
    # Test with context sizes: 32, 64, 128, 512
    # Create trainer, run train_step
    # Verify loss is computed, gradients applied
    # Check no errors or shape mismatches

def test_training_step_scheduled_sampling():
    """Verify scheduled sampling works with configurable context."""
    # Set tf_ratio to 0.5
    # Run training step
    # Verify both teacher forcing and predictions used
```

**Tests**: Run integration tests with different configurations.

**Integration**: Validates end-to-end functionality.

**Demo**: Run training step with context=512, show successful completion.

---

## Step 6: Run tests and validate

**Objective**: Execute full test suite and verify all tests pass.

**Implementation**:
```bash
# Run all tests
pytest tests/test_teacher_forcing.py -v

# Run with coverage
pytest tests/test_teacher_forcing.py --cov=src.music_generation.train

# Run memory profiling tests separately if needed
pytest tests/test_teacher_forcing.py::test_memory_constant_during_training -v
```

**Tests**: All tests pass with >90% coverage of modified code.

**Integration**: Confirms implementation is correct and complete.

**Demo**: Show pytest output with all green checks.

---

## Step 7: Update documentation

**Objective**: Document the new configuration parameter and its usage.

**Implementation**:
Update `README.md`:
- Add `MAX_CONTEXT_FRAMES` to configuration section
- Explain default value and when to modify
- Add example of memory-constrained usage

Update docstrings in `config.py`:
- Document parameter purpose
- Explain trade-offs (memory vs model capacity)

**Tests**: Review documentation for clarity and completeness.

**Integration**: Users can understand and configure the parameter.

**Demo**: Show updated README with clear explanation.

---

## Summary

**Total steps**: 7
**Estimated effort**: 4-6 hours
**Core changes**: 2 lines of code (config + train.py)
**Test coverage**: Unit, integration, memory profiling
**Risk**: Low - minimal changes, comprehensive testing

**Key principle**: Each step builds working, tested functionality. No orphaned code.
