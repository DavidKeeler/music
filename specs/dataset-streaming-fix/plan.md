# Implementation Plan: Dataset Streaming Fix

## BEHAVIORAL RULES - READ FIRST

**ASK BEFORE CHANGING ANYTHING**

1. **Ask before changing code** - Never modify files without explicit permission, even if you think it's the right fix
2. **Describe, don't implement** - When asked "how to fix X", explain the solution, don't apply it
3. **Planning means planning** - When writing specs/plans, describe what needs to happen, not implement it
4. **Check before creating files** - Ask if a file should be created before writing it
5. **Numbers are choices, not commands** - When given a number (like "2"), that's selecting an option, not a command to implement
6. **Wait for confirmation** - After explaining something, wait for "do it" or explicit permission before taking action
7. **Never revert without asking** - If you make a mistake, don't automatically undo it - ask first

**When in doubt: Stop and ask what to do next.**

---

## Implementation Steps

### Step 1: Remove sequence index pre-building in dataset.py

**What to change:**
In `src/music_generation/dataset.py`, class `MusicNetDataset.__init__()`:
- Remove the block starting with `self.sequence_indices = []`
- Remove the for loop that calls `_get_mel_length()` for each file
- Replace with: `self.file_indices = list(range(len(self.audio_files)))`

**Why:** This eliminates the startup delay caused by scanning all files.

### Step 2: Update _generator() to create sequences on-demand

**What to change:**
In `src/music_generation/dataset.py`, method `_generator()`:
- Change loop from `for file_idx, start_frame in self.sequence_indices:` to `for file_idx in self.file_indices:`
- After loading mel with `_load_or_compute_mel()`, check if `mel.shape[0] <= SEQ_LEN` and skip if too short
- Add inner loop: `stride = SEQ_LEN // 2; for start_frame in range(0, mel.shape[0] - SEQ_LEN, stride):`
- Generate input/target sequences inside the inner loop

**Why:** Moves sequence generation from initialization to iteration time.

### Step 3: Update __len__() method

**What to change:**
In `src/music_generation/dataset.py`, method `__len__()`:
- Change from `return len(self.sequence_indices)` to `return len(self.file_indices) * 10`
- Add comment: "# Rough estimate for progress bars"

**Why:** Exact count isn't available without scanning files, but approximate is sufficient.

### Step 4: Test the changes

**What to verify:**
- Dataset initialization completes in <5 seconds
- Training starts immediately after "Loading dataset" message
- Batches have correct shapes: `[batch_size, SEQ_LEN, N_MELS]`
- Short files are skipped without errors

**Test command:**
```bash
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --checkpoint_dir ~/models/conducting/mel_checkpoints \
  --epochs 1 \
  --batch_size 4
```
