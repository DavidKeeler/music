# Memory Issue Diagnosis and Fix for Music Generation Training

## Problem Statement

The training script (`python -m src.music_generation.train`) is experiencing memory issues on a MacBook Air when running with:
- batch_size: 16
- epochs: 100
- Dataset: MusicNet audio files

## Current Command
```bash
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints \
  --epochs 100 \
  --batch_size 16
```

## Identified Memory Problems

### 1. Autoregressive Loop in train_step (Critical)
**Location:** `src/music_generation/train.py` - `MelGeneratorTraining.train_step()`

The autoregressive loop builds the full sequence in memory with gradient tracking:
```python
for t in range(1, seq_len):
    pred = self.base_model(ar_input, training=True)[:, -1:, :]
    preds.append(pred)
    # ... concatenates growing ar_input each iteration
    ar_input = tf.concat([ar_input, next_input], axis=1)
```

**Issues:**
- Gradient tape tracks all intermediate predictions
- `ar_input` grows by concatenation each step (O(seq_len²) memory)
- All predictions stored in `preds` list before concatenation
- For SEQ_LEN=128, this creates 128 forward passes with growing tensors

### 2. Dataset Statistics Computation (High Impact)
**Location:** `src/music_generation/dataset.py` - `MusicNetDataset._compute_statistics()`

Loads ALL mel spectrograms into memory at once:
```python
all_mels = []
for audio_file in self.audio_files:
    mel = audio_to_mel(waveform)
    all_mels.append(mel)
all_mels_tensor = tf.concat(all_mels, axis=0)  # Concatenates everything!
```

**Issues:**
- Loads entire dataset into memory
- No streaming statistics computation
- Only happens on first run, but blocks training start

### 3. No Memory Growth Limit
**Location:** `src/music_generation/train.py`

No TensorFlow memory growth configuration for GPU/CPU.

### 4. Batch Size Too Large for MacBook Air
Default batch_size=16 may be too large for limited RAM on MacBook Air.

## Desired Outcome

Training script should:
1. Run successfully on MacBook Air with limited memory
2. Use efficient autoregressive training without O(n²) memory growth
3. Compute dataset statistics in streaming fashion
4. Configure appropriate memory limits
5. Provide sensible default batch size for resource-constrained environments
