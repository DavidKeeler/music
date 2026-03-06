# Objective
Fix memory issues in TensorFlow music generation training to enable training on MacBook Air with limited RAM.

# Context
Training script experiences OOM errors due to:
- O(n²) memory growth in autoregressive loop (concatenating growing sequences)
- Loading entire dataset during statistics computation
- No TensorFlow memory growth configuration
- Batch size too large for resource-constrained hardware

# Key Requirements

## 1. Memory Configuration
- Add `configure_memory()` function in `src/music_generation/train.py`
- Enable TensorFlow memory growth for all GPUs
- Call before model creation in `main()`

## 2. Streaming Statistics
- Modify `MusicNetDataset._compute_statistics()` in `src/music_generation/dataset.py`
- Replace batch concatenation with chunked sum/sum-of-squares
- Compute mean/std from running totals (constant memory)

## 3. Fixed-Window Autoregressive Loop
- Modify `MelGeneratorTraining.train_step()` in `src/music_generation/train.py`
- Add `context_len=64` parameter to `__init__`
- Truncate `ar_input` to last 64 frames before each prediction
- Reduces O(n²) to O(n) memory growth

## 4. Batch Size Warning
- Add memory check in `train()` function
- Warn if batch_size > 8 and system RAM < 16GB
- Suggest appropriate batch size (2 for <8GB, 4 for <16GB)
- Requires `psutil` package

## 5. Memory Profiling Tools
- Create `scripts/profile_memory.py` for measuring peak memory
- Use `tracemalloc` to track memory usage during training
- Document expected memory usage in README

# Acceptance Criteria

**Given** training runs on MacBook Air with 8GB RAM  
**When** using batch_size=4  
**Then** training completes without OOM errors

**Given** dataset statistics are computed  
**When** processing 1000+ audio files  
**Then** memory usage remains constant (not growing with dataset size)

**Given** autoregressive training loop executes  
**When** processing 128-frame sequences  
**Then** memory growth is O(n) not O(n²), peak memory is constant per batch

**Given** user specifies batch_size=16 on system with 8GB RAM  
**When** training starts  
**Then** warning message suggests batch_size=2

**Given** memory profiling script runs  
**When** training for 2 epochs  
**Then** peak memory is reported and reduced by 50%+ vs original implementation

# Implementation Notes

- All changes maintain backward compatibility
- Existing checkpoints will load correctly
- No changes to model architecture or config.py defaults
- Teacher forcing behavior unchanged
- See `specs/memory-fix/design.md` for detailed technical design
- See `specs/memory-fix/plan.md` for step-by-step implementation guide

# Files to Modify

1. `src/music_generation/train.py` - Add memory config, fixed-window loop, batch size warning
2. `src/music_generation/dataset.py` - Streaming statistics computation
3. `scripts/profile_memory.py` - Create new profiling script
4. `requirements.txt` - Add psutil if not present

# Testing

- Verify training completes on small dataset (10 files, 5 epochs)
- Profile memory usage (should be constant during training)
- Compare loss curves to original (should be equivalent)
- Test batch size warning displays correctly
- Verify checkpoints save/load correctly

# Reference
Detailed design and plan in `specs/memory-fix/`
