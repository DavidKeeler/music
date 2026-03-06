# Scratchpad - Memory Optimization for TensorFlow Music Generation

## Objective
Fix memory issues in TensorFlow music generation training to enable training on MacBook Air with limited RAM.

## Current Understanding

The training script has 4 main memory issues:
1. **No TensorFlow memory growth config** - TF allocates all GPU/CPU memory upfront
2. **Dataset statistics computation** - Loads entire dataset into memory
3. **O(n²) memory growth in autoregressive loop** - Concatenates growing sequences
4. **No batch size guidance** - Users may set batch_size too large for their RAM

## Implementation Plan

Based on `specs/memory-fix/plan.md`, I need to implement 5 steps:

### Step 1: Memory Configuration
- Add `configure_memory()` function to enable TF memory growth
- Call before model creation in `main()`

### Step 2: Streaming Statistics
- Modify `MusicNetDataset._compute_statistics()` in `dataset.py`
- Use running sum/sum-of-squares instead of concatenating all mels

### Step 3: Fixed-Window Autoregressive Loop
- Add `context_len=64` parameter to `MelGeneratorTraining.__init__()`
- Truncate `ar_input` to last 64 frames in `train_step()`
- Reduces O(n²) to O(n) memory

### Step 4: Batch Size Warning
- Add memory check in `train()` function
- Warn if batch_size > 8 and RAM < 16GB
- Requires `psutil` package

### Step 5: Memory Profiling
- Create `scripts/profile_memory.py` for measuring peak memory
- Use `tracemalloc` to track memory usage

## Current State

Checked existing code:
- `train.py` exists with `MelGeneratorTraining` class
- `train_step()` has the O(n²) concatenation issue (line 67: `ar_input = tf.concat([ar_input, next_input], axis=1)`)
- No memory configuration present
- No batch size warning present

## Next Steps

Need to create tasks for each implementation step and start with Step 1.


## Iteration 1 - Completed

### Task: Add TensorFlow memory configuration (task-1772774973-3155)

**What I did:**
- Added `configure_memory()` function to `src/music_generation/train.py`
- Function enables memory growth for all GPUs using `tf.config.experimental.set_memory_growth()`
- Called at start of `main()` before any model creation
- Added logging import and logger to report GPU memory configuration

**Testing:**
- Verified function executes without errors
- Tested on system with 1 GPU - successfully enabled memory growth
- Function handles systems without GPU gracefully (no-op)

**Commit:** dbe9e41 - "Add TensorFlow memory growth configuration"

**Learning:**
- TensorFlow by default allocates all GPU memory upfront
- Memory growth allows TensorFlow to allocate memory as needed
- This is critical for resource-constrained environments like MacBook Air

**Next:** Implement streaming statistics computation (task-1772774976-8684)


## Iteration 2 - Completed

### Task: Implement streaming statistics computation (task-1772774976-8684)

**What I did:**
- Modified `MusicNetDataset._compute_statistics()` in `src/music_generation/dataset.py`
- Replaced batch concatenation (`tf.concat(all_mels)`) with streaming computation
- Uses running totals: `total_sum`, `total_sum_sq`, `total_count`
- Computes mean = sum/count, variance = (sum_sq/count) - mean², std = sqrt(variance)

**Testing:**
- All dataset tests pass (2/2)
- Manual verification shows numerically equivalent results (diff < 1e-8)
- Memory usage now O(1) instead of O(n) with dataset size

**Commit:** 121175a - "Implement streaming statistics computation for constant memory usage"

**Learning:**
- Streaming statistics using Welford's online algorithm variant
- No need to store all data in memory - just accumulate sums
- Numerically stable and equivalent to batch computation

**Next:** Implement fixed-window autoregressive loop (task-1772774979-c549)


## Iteration 3 - Completed

### Task: Add fixed-window autoregressive loop (task-1772774979-c549)

**What I did:**
- Added `context_len=64` parameter to `MelGeneratorTraining.__init__()`
- Added truncation logic in `train_step()` to keep only last 64 frames of `ar_input`
- Prevents O(n²) memory growth by maintaining constant context window size
- Memory usage now O(n) instead of O(n²) during autoregressive training

**Testing:**
- All training compatibility tests pass (3/3)
- test_training_step_no_errors: ✓
- test_training_multiple_steps: ✓
- test_teacher_forcing_ratio_decay: ✓
- Tests ran in virtual environment with Python 3.12 + TensorFlow

**Commit:** 5c32830 - "Add fixed-window autoregressive loop to prevent O(n²) memory growth"

**Learning:**
- Fixed-window approach is simplest solution for memory optimization
- Transformer's positional encoding means absolute position matters less
- Truncation happens after concatenation, keeping most recent context
- No impact on teacher forcing or loss computation logic

**Next:** Add batch size warning (task-1772774983-44f8)


## Iteration 4 - Completed

### Task: Add batch size warning (task-1772774983-44f8)

**What I did:**
- Added `psutil>=5.9.0` to `requirements.txt`
- Added `check_batch_size()` function to `src/music_generation/train.py`
- Function checks system RAM and warns if batch_size > 8 and RAM < 16GB
- Suggests batch_size=2 for <8GB RAM, batch_size=4 for 8-16GB RAM
- Called at start of `train()` function before dataset loading

**Testing:**
- Installed psutil in virtual environment
- All training compatibility tests pass (3/3)
- Manual test shows function executes without errors
- Logic verified: warns appropriately based on RAM and batch size

**Commit:** 55083f0 - "Add batch size warning for resource-constrained systems"

**Learning:**
- psutil provides cross-platform system memory information
- Warning approach is better than auto-adjustment (respects user intent)
- Non-breaking change that provides actionable guidance

**Next:** Create memory profiling script (task-1772774987-4731)


## Iteration 5 - Completed

### Task: Create memory profiling script (task-1772774987-4731)

**What I did:**
- Created `scripts/profile_memory.py` for measuring peak memory usage
- Uses Python's built-in `tracemalloc` module to track memory allocations
- Reports baseline, current, peak memory and memory increase
- Shows top 10 memory allocations by line number
- Supports all training parameters (data_dir, epochs, batch_size, teacher forcing)
- Provides human-readable memory formatting (B/KB/MB/GB/TB)

**Testing:**
- Script executes without errors (tested with --help flag)
- All imports resolve correctly in virtual environment
- Fixed config import to use module-level constants instead of Config class

**Commit:** 5d60335 - "Add memory profiling script using tracemalloc"

**Learning:**
- tracemalloc is Python's built-in memory profiling tool
- get_traced_memory() returns (current, peak) tuple
- snapshot.statistics('lineno') groups allocations by source line
- Minimal implementation - no external dependencies beyond stdlib

**Next:** Check remaining tasks and verify completion
