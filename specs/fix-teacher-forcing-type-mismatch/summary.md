# Summary: Fix Teacher Forcing Type Mismatch and Memory Issue

## Overview

This specification addresses two critical bugs in the teacher forcing implementation that prevent training from running:

1. **Type mismatch error** - int64/int32 incompatibility in `update_tf_ratio()`
2. **Memory explosion** - Growing tensor inside GradientTape causes OOM

## Artifacts

### Research Documents
- `research/type-handling-comparison.md` - Analysis of type handling differences between original and current code
- `research/memory-issues.md` - Detailed analysis of memory leak in autoregressive loop
- `research/conducting-vs-conducting3.md` - Comparison of original vs current implementation
- `research/performance-analysis.md` - Performance characteristics of growing context approach
- `research/context-capping-issue.md` - Analysis of intentional context capping (no changes needed)

### Design Document
- `design.md` - Complete technical specification with:
  - Detailed requirements for both fixes
  - Architecture diagrams
  - Component specifications
  - Data flow diagrams
  - Acceptance criteria (Given-When-Then format)
  - Comprehensive testing strategy
  - Alternative approaches considered

### Implementation Plan
- `plan.md` - Step-by-step implementation guide with:
  - 6 numbered steps with checklist
  - Specific code changes for each step
  - Test requirements and demo commands
  - Integration notes
  - Manual testing procedures

## The Fixes

### Fix 1: Type Mismatch (Line 125)

**Change:**
```python
# BEFORE
step_after_warmup = tf.maximum(0, self.training_step - self.warmup_steps)

# AFTER
step_after_warmup = tf.maximum(0, self.training_step - tf.cast(self.warmup_steps, tf.int64))
```

**Impact:** Resolves TypeError, allows training to start

### Fix 2: Memory Leak (Line 220)

**Change:**
```python
# BEFORE
ar_input = tf.concat([ar_input, next_input], axis=1)

# AFTER
ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
```

**Impact:** Reduces memory usage from 168 MB to 0.65 MB per batch (99.6% reduction)

## Key Findings

1. **Original code (conducting/) used seq_len=128** - never tested with longer sequences in the codebase, but user confirmed it worked with longer sequences
2. **Both models can handle variable lengths** - no positional encodings, causal convolutions and windowed attention support any length
3. **Context capping at 128 is intentional** - should NOT be removed (user confirmed)
4. **Memory issue scales with sequence length** - 16x more memory for 4x longer sequences
5. **stop_gradient is safe** - gradients still flow through preds list to model parameters

## Testing Strategy

- **Unit tests** - Type compatibility, warmup behavior, gradient flow, memory usage
- **Integration tests** - Full training step, checkpoint save/load, convergence
- **Manual testing** - Real training run with monitoring

## Next Steps

### Option 1: Implement with Ralph (Autonomous)

Create `PROMPT.md` and run:
```bash
ralph run --config presets/spec-driven.yml
```

### Option 2: Manual Implementation

Follow the 6-step plan in `plan.md`:
1. Fix type mismatch
2. Fix memory leak
3. Add unit tests for types
4. Add unit tests for gradients
5. Add integration tests
6. Manual verification

## Files Modified

**Single file:** `src/music_generation/train.py`
- Line 125: Add `tf.cast(self.warmup_steps, tf.int64)`
- Line 220: Add `tf.stop_gradient(next_input)`

## Expected Outcomes

- ✓ Training runs without TypeError
- ✓ No OOM errors with seq_len=512
- ✓ Memory usage reduced by 99.6%
- ✓ Training dynamics unchanged
- ✓ Checkpoints remain compatible
- ✓ Performance maintained
