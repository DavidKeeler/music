# Fix Dataset Streaming Performance

## Objective

Modify the MusicNet dataset loader to enable true streaming without pre-scanning all audio files during initialization.

## Problem

Current implementation in `src/music_generation/dataset.py` calls `_get_mel_length()` for every audio file during `__init__`, which loads/processes all files before training starts. This defeats the purpose of streaming and causes long startup delays.

## Required Changes

### File: `src/music_generation/dataset.py`

**1. Replace sequence index pre-building (lines ~60-67):**

Remove:
```python
self.sequence_indices = []
for file_idx, audio_file in enumerate(self.audio_files):
    mel_length = self._get_mel_length(audio_file)
    if mel_length > SEQ_LEN:
        stride = SEQ_LEN // 2
        for start in range(0, mel_length - SEQ_LEN, stride):
            self.sequence_indices.append((file_idx, start))
```

Replace with:
```python
self.file_indices = list(range(len(self.audio_files)))
```

**2. Update `_generator()` method:**

Replace:
```python
def _generator(self):
    for file_idx, start_frame in self.sequence_indices:
        audio_file = self.audio_files[file_idx]
        mel = self._load_or_compute_mel(audio_file)
        input_mel = mel[start_frame:start_frame + SEQ_LEN]
        target_mel = mel[start_frame + 1:start_frame + SEQ_LEN + 1]
        yield input_mel.numpy(), target_mel.numpy()
```

With:
```python
def _generator(self):
    for file_idx in self.file_indices:
        audio_file = self.audio_files[file_idx]
        mel = self._load_or_compute_mel(audio_file)
        
        if mel.shape[0] <= SEQ_LEN:
            continue
        
        stride = SEQ_LEN // 2
        for start_frame in range(0, mel.shape[0] - SEQ_LEN, stride):
            input_mel = mel[start_frame:start_frame + SEQ_LEN]
            target_mel = mel[start_frame + 1:start_frame + SEQ_LEN + 1]
            yield input_mel.numpy(), target_mel.numpy()
```

**3. Update `__len__()` method:**

Replace:
```python
def __len__(self) -> int:
    return len(self.sequence_indices)
```

With:
```python
def __len__(self) -> int:
    return len(self.file_indices) * 10  # Rough estimate
```

## Acceptance Criteria

- Dataset initialization completes in <5 seconds regardless of number of audio files
- Training starts immediately after dataset creation
- Files are loaded on-demand during iteration only
- Short files (length <= SEQ_LEN) are skipped during iteration, not initialization
- All existing functionality preserved (caching, normalization, batching)

## Testing

Run training and verify:
```bash
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --checkpoint_dir ~/models/conducting/mel_checkpoints \
  --epochs 1 \
  --batch_size 4
```

Expected: "Loading dataset" message followed immediately by training progress within seconds.
