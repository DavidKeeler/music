# Summary: Parallel Autoregressive Training Refactoring

## Project Overview

Refactor the mel spectrogram generator training pipeline to eliminate the inefficient O(T) autoregressive loop and replace it with parallel training using causal masking. This achieves ~100x speedup while preserving exact autoregressive semantics.

## Artifacts Created

### Planning Documents

1. **rough-idea.md** - Original problem statement
2. **requirements.md** - Q&A record (empty - skipped to research)
3. **research/** - Four comprehensive research documents:
   - `01-current-implementation.md` - Analysis of existing O(T) loop bottleneck
   - `02-causal-masking-parallel-training.md` - Causal masking patterns and PSS algorithm
   - `03-teacher-forcing-integration.md` - Teacher forcing and scheduled sampling
   - `04-performance-implications.md` - Performance analysis and projections

### Design Document

4. **design.md** - Complete design specification including:
   - Detailed requirements (functional and non-functional)
   - Architecture diagrams (current vs proposed)
   - Component specifications with algorithms
   - Data models and tensor shapes
   - Error handling strategies
   - 8 acceptance criteria (Given-When-Then format)
   - Comprehensive testing strategy
   - Appendices (technology choices, research summary, alternatives, migration path)

### Implementation Plan

5. **plan.md** - 7-step incremental implementation plan:
   - Step 1: Extract pure teacher forcing method
   - Step 2: Implement parallel scheduled sampling
   - Step 3: Update train_step dispatch logic
   - Step 4: Add unit tests
   - Step 5: Add integration tests
   - Step 6: Benchmark and validate
   - Step 7: Update documentation

## Key Findings

### Current Problem

- Training uses O(T) autoregressive loop with 255 forward passes per batch
- Extremely slow: ~10 seconds per batch on CPU
- High memory usage: ~2GB peak per batch
- Limits batch size to 4-8 on 16GB RAM

### Solution

- Replace loop with 2-pass parallel scheduled sampling
- Pure teacher forcing: 1 forward pass (already optimal)
- Scheduled sampling: 2 forward passes (vs 255 currently)
- Model architecture already causal - no changes needed

### Expected Improvements

| Metric | Current | After Refactoring | Improvement |
|--------|---------|-------------------|-------------|
| Forward passes/batch | 255 | 2 | 127x fewer |
| Training time (100 epochs) | ~28 hours | ~17 minutes | ~100x faster |
| Peak memory/batch | ~2 GB | ~10 MB | ~200x less |
| Max batch size (16GB RAM) | 4 | 64 | 16x larger |
| Time per batch (CPU) | ~10 sec | ~0.1 sec | 100x faster |

## Technical Approach

### Two Training Paths

**Path 1: Pure Teacher Forcing (tf_ratio ≥ 0.99)**
```
Input: ground_truth
Forward: preds = model(ground_truth)
Loss: L1(preds[:, :-1, :], ground_truth[:, 1:, :])
Complexity: O(1) - single forward pass
```

**Path 2: Parallel Scheduled Sampling (tf_ratio < 0.99)**
```
Pass 1: preds = model(ground_truth, training=False)  # Inference
Sample: use_teacher ~ Bernoulli(tf_ratio)
Mix: input = where(use_teacher, ground_truth, shifted_preds)
Pass 2: preds = model(input, training=True)  # Training
Loss: L1(preds[:, :-1, :], ground_truth[:, 1:, :])
Complexity: O(2) - two forward passes
```

### Why This Works

- Model architecture already uses causal layers (CausalConv1D, LocalWindowAttention)
- Causal masking ensures position t only sees positions < t
- Parallel training is mathematically equivalent to sequential autoregressive training
- Two-pass PSS preserves scheduled sampling semantics

## Implementation Highlights

### Minimal Code Changes

Only `src/music_generation/train.py` needs modification:

1. Extract `_pure_teacher_forcing(x, y)` method (~15 lines)
2. Add `_parallel_scheduled_sampling(x, y)` method (~20 lines)
3. Simplify `train_step(data)` to dispatch (~10 lines)
4. Remove old nested functions (~100 lines deleted)

**Net change: ~55 lines added, ~100 lines removed**

### Backward Compatibility

- No model architecture changes
- Checkpoints remain compatible
- Same hyperparameters and configuration
- Same loss function and optimizer

### Testing Strategy

- 5 unit tests (shapes, loss, forward passes, sampling, dispatch)
- 3 integration tests (end-to-end, checkpoints, schedule)
- 1 benchmark script (performance validation)
- All tests run in < 1 minute

## Next Steps

### Option 1: Implement with Ralph (Autonomous)

Create `PROMPT.md` for Ralph to implement autonomously:

```bash
ralph run --config presets/spec-driven.yml
```

Ralph will:
- Read design.md and plan.md
- Implement all 7 steps
- Run tests and validate
- Generate final report

### Option 2: Manual Implementation

Follow the 7-step plan in `plan.md`:
1. Refactor training methods (2-3 hours)
2. Add tests (2-3 hours)
3. Benchmark and document (2 hours)

**Total effort: 6-8 hours**

### Option 3: Hybrid Approach

Use Ralph for initial implementation, then manually review and refine.

## Risk Assessment

### Low Risk

- ✅ Model architecture unchanged (checkpoints compatible)
- ✅ Mathematically equivalent (same loss function)
- ✅ Well-researched approach (ICLR 2020 paper)
- ✅ Incremental plan (testable at each step)

### Mitigation Strategies

- Validate on small dataset first
- Compare loss curves with baseline
- Benchmark before full training
- Keep old implementation as fallback

## Success Criteria

**Must achieve:**
- ✅ 50x+ speedup in training time
- ✅ All tests pass
- ✅ Checkpoint compatibility maintained
- ✅ Similar or better loss curves

**Nice to have:**
- ✅ 100x+ speedup
- ✅ Improved convergence
- ✅ Higher quality generations

## Files Reference

All artifacts located in: `specs/parallel-autoregressive-training/`

```
specs/parallel-autoregressive-training/
├── rough-idea.md           # Original problem statement
├── requirements.md         # Q&A record (empty)
├── research/               # Research findings
│   ├── 01-current-implementation.md
│   ├── 02-causal-masking-parallel-training.md
│   ├── 03-teacher-forcing-integration.md
│   └── 04-performance-implications.md
├── design.md               # Complete design specification
├── plan.md                 # 7-step implementation plan
└── summary.md              # This file
```

## Conclusion

This refactoring addresses a critical performance bottleneck with a well-researched, low-risk solution. The parallel training approach is standard in modern sequence models (GPT, etc.) and will bring the codebase in line with best practices while achieving massive speedup.

**Estimated impact:**
- Training time: 28 hours → 17 minutes (~100x faster)
- Development velocity: Much faster iteration cycles
- Resource efficiency: Can train on CPU, larger batches
- Code quality: Simpler, more maintainable training loop

**Ready for implementation.**
