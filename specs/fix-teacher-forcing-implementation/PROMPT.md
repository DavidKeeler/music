# PROMPT.md

## Objective

Make the autoregressive context window size configurable in the teacher forcing implementation to enable full sequence context during training.

## Problem

The current implementation hardcodes `max_context = 64` in the autoregressive training loop. With `SEQ_LEN = 512`, this limits the model to only 64 frames of context, reducing its ability to learn long-range dependencies.

## Solution

Add `MAX_CONTEXT_FRAMES` configuration parameter (default: `SEQ_LEN`) and update the training loop to use it. Maintain the existing sliding window approach for memory efficiency.

## Key Requirements

1. **Add configuration parameter** (`src/music_generation/config.py`)
   - `MAX_CONTEXT_FRAMES = SEQ_LEN` (default: 512)
   - Add comment explaining purpose and trade-offs
   - Place after `TF_WARMUP_STEPS` definition

2. **Update training loop** (`src/music_generation/train.py`)
   - Import `MAX_CONTEXT_FRAMES` from config
   - Replace `max_context = 64` with `max_context = MAX_CONTEXT_FRAMES`
   - No other changes needed - sliding window logic is correct

3. **Add comprehensive tests** (`tests/test_teacher_forcing.py`)
   - Unit tests for sliding window logic
   - Memory profiling tests (verify constant memory usage)
   - Integration tests with different context sizes
   - Edge case tests (small context, initial context < max)

4. **Update documentation** (`README.md`)
   - Document `MAX_CONTEXT_FRAMES` parameter
   - Explain default value and when to modify
   - Add memory trade-off guidance

## Acceptance Criteria

### Given: MAX_CONTEXT_FRAMES configured to 512
**When:** Autoregressive training step executes
**Then:** 
- Initial context uses last 512 frames from input
- Sliding window maintains 512 frames throughout
- Memory usage remains constant

### Given: Unit tests for sliding window
**When:** Tests execute
**Then:**
- Window maintains constant size
- Oldest frame dropped, newest appended
- Edge cases pass (window_size=1, growing from small initial)

### Given: Memory profiling test
**When:** Training step runs for multiple iterations
**Then:**
- Memory usage is constant (no growth)
- Memory scales linearly with MAX_CONTEXT_FRAMES
- No memory leaks detected

### Given: Integration test with different context sizes
**When:** Training step runs with context sizes [32, 64, 128, 512]
**Then:**
- All complete without errors
- Loss computed correctly
- Gradients applied successfully

### Given: Documentation updated
**When:** User reads README
**Then:**
- MAX_CONTEXT_FRAMES parameter is documented
- Default value and purpose are clear
- Memory trade-offs are explained

## Critical Implementation Notes

**Minimal changes**:
- Only 2 lines of code change (config + train.py)
- Existing sliding window logic is correct - don't modify
- Keep TensorArray + for-loop approach (research shows it's efficient)

**Testing priority**:
- Memory profiling is critical - verify no leaks
- Test with different context sizes (32, 64, 128, 512)
- Use `tracemalloc` or `memory_profiler` for memory tests

**Don't change**:
- Loop implementation (TensorArray + for-loop is efficient)
- Sliding window update logic (already correct)
- Scheduled sampling behavior

## Reference

Full specification: `specs/fix-teacher-forcing-implementation/`
- `design.md` - Detailed design with architecture
- `plan.md` - 7-step implementation plan
- `research/loop-performance.md` - Performance analysis

## Implementation Estimate

4-6 hours total:
- Core changes: 1 hour
- Testing: 2-3 hours
- Documentation: 1 hour
- Validation: 1-2 hours
