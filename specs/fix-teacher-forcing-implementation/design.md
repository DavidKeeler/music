# Design: Fix Teacher Forcing Implementation

## Overview

Make the autoregressive context window size configurable to allow full sequence context during training, improving model's ability to learn long-range dependencies. Add comprehensive tests to validate memory usage and correctness.

## Detailed Requirements

### Functional Requirements

1. **Configurable Context Window**
   - Add `MAX_CONTEXT_FRAMES` parameter to `config.py`
   - Default value: `SEQ_LEN` (512) for full context
   - Allow override for memory-constrained environments
   - Use in autoregressive training loop

2. **Memory-Safe Implementation**
   - Maintain fixed-size sliding window (no growing concatenation)
   - Handle case where initial context < max_context
   - Ensure constant memory footprint during training

3. **Testing**
   - Unit tests for sliding window logic
   - Memory usage validation
   - Correctness tests (predictions match expected behavior)
   - Integration test with full training step

### Non-Functional Requirements

1. **Performance**: No significant slowdown compared to current implementation
2. **Backward Compatibility**: Existing checkpoints continue to work
3. **Maintainability**: Clear configuration, well-documented behavior

## Architecture Overview

```mermaid
graph TD
    A[config.py] -->|MAX_CONTEXT_FRAMES| B[train.py]
    B --> C[train_step method]
    C --> D{tf_ratio >= 1.0?}
    D -->|Yes| E[Parallel Training]
    D -->|No| F[Autoregressive Loop]
    F --> G[Initialize: x[:, -MAX_CONTEXT_FRAMES:, :]]
    G --> H[For each timestep]
    H --> I[Predict next frame]
    I --> J[Scheduled sampling]
    J --> K[Sliding window update]
    K --> H
    H --> L[Compute loss]
    L --> M[Update weights]
```

## Components and Interfaces

### 1. Configuration (config.py)

**Add parameter**:
```python
# Maximum context frames for autoregressive training
# Default to SEQ_LEN for full context. Reduce for memory-constrained environments.
MAX_CONTEXT_FRAMES = SEQ_LEN  # 512
```

**Location**: After `TF_WARMUP_STEPS` definition

### 2. Training Loop Modification (train.py)

**Current implementation** (lines 199-237):
- Fixed `max_context = 64`
- Sliding window approach (correct)

**Required changes**:
```python
# Import from config
from .config import MAX_CONTEXT_FRAMES

# In train_step method, replace hardcoded value:
max_context = MAX_CONTEXT_FRAMES  # Instead of 64
```

**No other changes needed** - the sliding window logic is already correct.

### 3. Test Suite (tests/test_teacher_forcing.py)

**New test file** with:

1. **Test sliding window logic**
   - Verify window size stays constant
   - Check oldest frame is dropped, newest appended
   - Validate shape invariants

2. **Test memory usage**
   - Profile memory during training step
   - Ensure no growth over iterations
   - Compare different MAX_CONTEXT_FRAMES values

3. **Test correctness**
   - Verify predictions use correct context
   - Check scheduled sampling behavior
   - Validate loss computation

4. **Test edge cases**
   - Initial context < MAX_CONTEXT_FRAMES
   - MAX_CONTEXT_FRAMES = 1 (minimal context)
   - MAX_CONTEXT_FRAMES > SEQ_LEN (should work, uses available)

## Data Models

No new data models. Existing tensors:
- `current_input`: [batch, max_context, mel_dim] - sliding window buffer
- `predictions`: TensorArray of size seq_len
- `x`, `y`: [batch, seq_len, mel_dim] - input/target sequences

## Error Handling

**Configuration validation**:
```python
assert MAX_CONTEXT_FRAMES > 0, "MAX_CONTEXT_FRAMES must be positive"
assert isinstance(MAX_CONTEXT_FRAMES, int), "MAX_CONTEXT_FRAMES must be integer"
```

**Runtime checks** (already present):
- Shape assertions for tensor dimensions
- NaN/Inf detection in loss

## Acceptance Criteria

### Given: MAX_CONTEXT_FRAMES configured to 512
**When:** Autoregressive training step executes
**Then:** 
- Initial context uses last 512 frames from input
- Sliding window maintains 512 frames throughout
- Memory usage remains constant

### Given: MAX_CONTEXT_FRAMES configured to 64
**When:** Training step executes
**Then:**
- Behavior matches current implementation
- Memory usage is lower than with 512
- Training completes without errors

### Given: Initial context has 100 frames, MAX_CONTEXT_FRAMES = 512
**When:** Autoregressive loop starts
**Then:**
- Uses all 100 available frames
- Window grows to 512 as predictions are generated
- No errors or shape mismatches

### Given: Training with different MAX_CONTEXT_FRAMES values
**When:** Memory profiler runs
**Then:**
- Memory usage scales linearly with MAX_CONTEXT_FRAMES
- No memory leaks detected
- Peak memory is predictable

### Given: Unit tests for sliding window
**When:** Tests execute
**Then:**
- All edge cases pass
- Shape invariants maintained
- Correctness validated

## Testing Strategy

### Unit Tests
- Sliding window update logic (isolated)
- Configuration validation
- Edge case handling

### Integration Tests
- Full training step with different MAX_CONTEXT_FRAMES
- Checkpoint save/load compatibility
- End-to-end training for 10 steps

### Performance Tests
- Memory profiling with different context sizes
- Training speed comparison (64 vs 512)
- Benchmark against baseline

### Test Data
- Synthetic mel spectrograms (controlled shapes)
- Small batch size (2) for faster tests
- Short sequences (64 frames) for unit tests

## Appendices

### A. Technology Choices

**Why not change the loop implementation?**
- Research shows TensorArray + for-loop is efficient
- tf.scan is often slower (2x in benchmarks)
- Current approach is simple and debuggable

**Why default to SEQ_LEN?**
- Maximizes model's learning capacity
- Matches inference conditions (full context available)
- Users can reduce if memory-constrained

### B. Research Findings

See `research/loop-performance.md` for detailed analysis:
- tf.scan vs TensorArray performance comparison
- Memory characteristics of different approaches
- Best practices for autoregressive loops in TensorFlow

### C. Alternative Approaches Considered

1. **Dynamic context sizing**: Grow context during training
   - Rejected: Adds complexity, unclear benefit
   
2. **Multiple context sizes**: Train with varying sizes
   - Rejected: Out of scope, can be future enhancement
   
3. **Rewrite with tf.while_loop**: More "native" TensorFlow
   - Rejected: No performance benefit, harder to debug

### D. Implementation Notes

**Minimal change principle**: Only modify what's necessary
- Single line change in train.py (max_context value)
- One new parameter in config.py
- New test file for validation

**No breaking changes**:
- Existing code continues to work
- Default behavior improves (full context)
- Users can opt-in to reduced context if needed
