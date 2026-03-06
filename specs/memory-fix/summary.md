# Memory Optimization Summary

## Project Overview

Comprehensive plan to diagnose and fix memory issues in the TensorFlow music generation training pipeline, enabling training on resource-constrained hardware (MacBook Air with limited RAM).

## Problem Identified

Four critical memory issues causing OOM errors:
1. **Autoregressive loop** - O(n²) memory growth from concatenating growing sequences
2. **Dataset statistics** - Loads entire dataset into memory during initialization
3. **No memory limits** - TensorFlow allocates all available memory upfront
4. **Large batch size** - Default batch_size=16 too large for MacBook Air

## Solution Approach

Five-step incremental optimization plan:
1. Configure TensorFlow memory growth
2. Implement streaming statistics computation
3. Replace growing concatenation with fixed-size sliding window
4. Add batch size warnings for resource-constrained systems
5. Add memory profiling tools for validation

## Artifacts Created

```
specs/memory-fix/
├── rough-idea.md          # Problem statement and context
├── requirements.md        # (Empty - no additional requirements gathering needed)
├── research/              # (Empty - no external research needed)
├── design.md              # Detailed technical design with 3 solution options
├── plan.md                # 5-step implementation plan with code examples
└── summary.md             # This file
```

## Key Design Decisions

**Streaming Statistics:** Chunked sum/sum-of-squares approach
- Simpler than Welford's algorithm
- Sufficient numerical stability for mel spectrograms
- Constant memory usage regardless of dataset size

**Autoregressive Loop:** Fixed-size sliding window (64 frames)
- Simplest implementation
- Most effective memory reduction (O(n²) → O(n))
- Minimal impact on model quality (Transformer has positional encoding)
- Alternative options documented: Truncated BPTT, TensorArray pre-allocation

**Batch Size:** Warning message instead of auto-adjustment
- Respects user intent
- Provides actionable guidance
- Non-breaking change

## Expected Impact

- **Memory reduction:** 50-70% peak memory usage
- **Training speed:** <10% slowdown (acceptable tradeoff)
- **Model quality:** No degradation (equivalent convergence)
- **Compatibility:** Fully backward compatible with existing checkpoints

## Next Steps

### Option 1: Manual Implementation
Follow the 5-step plan in `specs/memory-fix/plan.md`:
1. Add memory configuration (5 min)
2. Implement streaming statistics (15 min)
3. Optimize autoregressive loop (20 min)
4. Add batch size warning (10 min)
5. Add profiling tools (15 min)

**Total estimated time:** ~1 hour

### Option 2: Ralph Autonomous Implementation
Use Ralph's spec-driven preset to implement automatically:

```bash
# Create PROMPT.md first (see below)
ralph run --config presets/spec-driven.yml
```

## Ralph Integration

To enable autonomous implementation, create `specs/memory-fix/PROMPT.md`:

```markdown
# Objective
Fix memory issues in TensorFlow music generation training to enable training on MacBook Air with limited RAM.

# Key Requirements
1. Configure TensorFlow memory growth to prevent upfront allocation
2. Implement streaming statistics computation (constant memory)
3. Replace O(n²) autoregressive loop with O(n) fixed-window approach
4. Add batch size warnings for resource-constrained systems
5. Add memory profiling tools

# Acceptance Criteria
**Given** training runs on MacBook Air with 8GB RAM  
**When** using recommended batch size (2-4)  
**Then** training completes without OOM errors

**Given** dataset statistics are computed  
**When** processing 1000+ audio files  
**Then** memory usage remains constant

**Given** autoregressive training loop executes  
**When** processing 128-frame sequences  
**Then** memory growth is O(n) not O(n²)

# Reference
See detailed design and plan in `specs/memory-fix/`
```

Then run:
```bash
ralph run --config presets/spec-driven.yml
```

## Testing Validation

After implementation, validate with:

```bash
# Profile memory usage
python scripts/profile_memory.py \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --batch_size 4 \
  --epochs 2

# Run full training
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints \
  --epochs 100 \
  --batch_size 4  # Reduced from 16
```

Monitor with Activity Monitor (macOS) or htop (Linux) to verify:
- Peak memory stays under 6GB
- Memory usage is stable during training
- No memory growth over epochs

## References

- Original issue: Training OOM on MacBook Air
- Design document: `specs/memory-fix/design.md`
- Implementation plan: `specs/memory-fix/plan.md`
- Related code: `src/music_generation/train.py`, `src/music_generation/dataset.py`
