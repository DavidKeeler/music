# Implementation Plan

## Checklist

- [ ] Step 1: Add reduction factor configuration
- [ ] Step 2: Update dataset preprocessing
- [ ] Step 3: Modify model input/output projections
- [ ] Step 4: Update model generation method
- [ ] Step 5: Modify training loss computation
- [ ] Step 6: Add unit tests
- [ ] Step 7: Verify end-to-end training and inference

---

## Step 1: Add Reduction Factor Configuration

**Objective:** Add `REDUCTION_FACTOR` and derived constants to `config.py`

**Implementation:**
- Add `REDUCTION_FACTOR = 4` constant
- Add `EFFECTIVE_SEQ_LEN = SEQ_LEN // REDUCTION_FACTOR` (computed value)
- Add `GROUPED_MEL_DIM = N_MELS * REDUCTION_FACTOR` (computed value)
- Place these after the existing `SEQ_LEN` and `N_MELS` definitions

**Tests:**
- Verify `REDUCTION_FACTOR` is accessible from config
- Verify `GROUPED_MEL_DIM = 320` when `N_MELS=80` and `REDUCTION_FACTOR=4`
- Verify `EFFECTIVE_SEQ_LEN = 128` when `SEQ_LEN=512` and `REDUCTION_FACTOR=4`

**Integration:**
- Import these constants in subsequent steps
- No changes to existing code yet

**Demo:**
```python
from src.music_generation.config import REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN
print(f"R={REDUCTION_FACTOR}, Grouped Dim={GROUPED_MEL_DIM}, Effective Len={EFFECTIVE_SEQ_LEN}")
# Output: R=4, Grouped Dim=320, Effective Len=128
```

---

## Step 2: Update Dataset Preprocessing

**Objective:** Modify dataset pipeline to truncate and reshape mel sequences into grouped frames

**Implementation:**
- In `dataset.py`, update the sequence preparation logic:
  - Truncate mel to be divisible by R: `mel = mel[:len(mel) - (len(mel) % REDUCTION_FACTOR)]`
  - Reshape to grouped format: `mel_grouped = tf.reshape(mel, [-1, GROUPED_MEL_DIM])`
  - Create input/target pairs with 1-step offset on grouped sequences
- Update sequence length checks to use `EFFECTIVE_SEQ_LEN` instead of `SEQ_LEN`
- Ensure normalization is applied before reshaping

**Tests:**
- Test truncation: input [515, 80] → output [512, 80] for R=4
- Test reshaping: input [512, 80] → output [128, 320] for R=4
- Test input/target pairs: verify target is input shifted by 1 grouped frame
- Test values preserved: flatten and compare before/after reshape

**Integration:**
- Dataset now returns `[B, T/R, R*80]` tensors
- Compatible with updated model (Step 3)

**Demo:**
```python
dataset = create_dataset(data_dir, cache_dir, batch_size=2)
for batch in dataset.take(1):
    input_mel, target_mel = batch
    print(f"Input shape: {input_mel.shape}")  # [2, 128, 320]
    print(f"Target shape: {target_mel.shape}")  # [2, 128, 320]
```

---

## Step 3: Modify Model Input/Output Projections

**Objective:** Update `MelGenerator` to accept and produce grouped mel frames

**Implementation:**
- In `model.py`, import `GROUPED_MEL_DIM` from config
- Change input projection: `self.input_proj = tf.keras.layers.Dense(D_MODEL)` (input dim now R*80)
- Change output projection: `self.output_proj = tf.keras.layers.Dense(GROUPED_MEL_DIM)`
- Add comment explaining the dimension change
- No changes to transformer/conv layers (they operate on D_MODEL)

**Tests:**
- Test input projection: input [4, 128, 320] → output [4, 128, D_MODEL]
- Test output projection: input [4, 128, D_MODEL] → output [4, 128, 320]
- Test full forward pass: input [4, 128, 320] → output [4, 128, 320]
- Verify no shape errors during call

**Integration:**
- Model now processes grouped frames
- Compatible with updated dataset (Step 2)
- Generation method needs update (Step 4)

**Demo:**
```python
model = MelGenerator()
dummy_input = tf.random.normal([2, 128, 320])
output = model(dummy_input, training=False)
print(f"Output shape: {output.shape}")  # [2, 128, 320]
```

---

## Step 4: Update Model Generation Method

**Objective:** Modify `generate()` to handle grouped frame prediction and context management

**Implementation:**
- In `model.py`, update `MelGenerator.generate()`:
  - Truncate seed to be divisible by R
  - Reshape seed to grouped format: `[seed_len, 80] → [seed_len/R, R*80]`
  - Calculate `num_steps = (num_frames + R - 1) // R`
  - Update context window check to use `EFFECTIVE_SEQ_LEN`
  - After prediction, append grouped frame to context
  - After all steps, reshape output: `[num_steps, R*80] → [num_steps*R, 80]`
  - Truncate to exact `num_frames`
- Add inline comments explaining reshaping logic

**Tests:**
- Test seed truncation: input [100, 80] → truncated to [96, 80] for R=4
- Test seed reshaping: [96, 80] → [24, 320]
- Test generation steps: num_frames=500 → 125 steps
- Test output shape: generate(seed, 500) → [500, 80]
- Test context update: verify 1 grouped frame added per step

**Integration:**
- Generation now produces R frames per step
- Compatible with updated model architecture (Step 3)
- Inference API unchanged (accepts/returns individual frames)

**Demo:**
```python
model = MelGenerator()
seed = tf.random.normal([96, 80])
generated = model.generate(seed, num_frames=500)
print(f"Generated shape: {generated.shape}")  # [500, 80]
```

---

## Step 5: Modify Training Loss Computation

**Objective:** Update `train_step()` to compute per-frame loss and average across R frames

**Implementation:**
- In `train.py`, update `MelGeneratorTraining.train_step()`:
  - After forward pass, reshape predictions: `[B, T/R, R*80] → [B, T/R, R, 80]`
  - Reshape targets similarly: `[B, T/R, R*80] → [B, T/R, R, 80]`
  - Compute MSE per frame: `tf.reduce_mean(tf.square(pred - target), axis=-1)` → `[B, T/R, R]`
  - Average across all dimensions: `tf.reduce_mean(frame_losses)`
- Add inline comment explaining per-frame loss computation
- No changes to teacher forcing schedule logic

**Tests:**
- Test reshaping: [2, 128, 320] → [2, 128, 4, 80]
- Test loss shape: verify scalar output
- Test loss computation: compare with manual calculation
- Test gradient flow: verify gradients are computed correctly

**Integration:**
- Training now optimizes per-frame predictions
- Compatible with updated model and dataset
- Teacher forcing schedule unchanged

**Demo:**
```python
training_model = MelGeneratorTraining(base_model)
dataset = create_dataset(data_dir, cache_dir, batch_size=2)
for batch in dataset.take(1):
    result = training_model.train_step(batch)
    print(f"Loss: {result['loss']:.4f}")
    print(f"TF Ratio: {result['tf_ratio']:.4f}")
```

---

## Step 6: Add Unit Tests

**Objective:** Create unit tests for all shape transformations and layer dimensions

**Implementation:**
- Create `tests/test_reduction_factor.py` with tests:
  - `test_config_values()`: Verify REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN
  - `test_dataset_truncation()`: Verify truncation to R-divisible length
  - `test_dataset_reshaping()`: Verify [T, 80] → [T/R, R*80]
  - `test_model_input_projection()`: Verify input shape handling
  - `test_model_output_projection()`: Verify output shape
  - `test_model_forward_pass()`: Verify end-to-end shape
  - `test_loss_computation()`: Verify per-frame loss averaging
  - `test_generation_seed_truncation()`: Verify seed preprocessing
  - `test_generation_output_shape()`: Verify final output shape
  - `test_generation_context_update()`: Verify context management

**Tests:**
- Run `pytest tests/test_reduction_factor.py`
- All tests should pass
- Coverage should include config, dataset, model, training

**Integration:**
- Tests validate all components work correctly
- Provides regression protection for future changes

**Demo:**
```bash
pytest tests/test_reduction_factor.py -v
# Output: 10 passed in X.XXs
```

---

## Step 7: Verify End-to-End Training and Inference

**Objective:** Run a short training session and generate audio to verify the complete pipeline

**Implementation:**
- Run training for 10 steps with small batch:
  ```bash
  python -m src.music_generation.train \
    --data_dir ~/data/music/musicnet/train_data \
    --cache_dir ./cache \
    --checkpoint_dir ./checkpoints_r4 \
    --epochs 1 \
    --batch_size 2
  ```
- Verify training logs show correct shapes and loss values
- Run inference to generate audio:
  ```python
  from src.music_generation.inference import MusicGenerationModel
  model = MusicGenerationModel.from_checkpoints(
      mel_checkpoint='./checkpoints_r4/mel_generator.h5',
      normalizer=normalizer
  )
  audio = model.generate(seed_mel, num_frames=500)
  ```
- Verify generated audio shape and save to file
- Listen to audio to verify quality (subjective check)

**Tests:**
- Verify no runtime errors during training
- Verify checkpoint is saved correctly
- Verify inference loads checkpoint and generates audio
- Verify generated audio has correct sample rate and length

**Integration:**
- Complete pipeline works end-to-end
- Ready for full-scale training experiments

**Demo:**
```bash
# Training
python -m src.music_generation.train --epochs 1 --batch_size 2

# Inference
python -c "
from src.music_generation.inference import MusicGenerationModel
from src.music_generation.audio_utils import load_audio, audio_to_mel, MelNormalizer
import soundfile as sf

normalizer = MelNormalizer.from_dataset('./cache')
model = MusicGenerationModel.from_checkpoints(
    mel_checkpoint='./checkpoints_r4/mel_generator.h5',
    normalizer=normalizer
)
seed_audio = load_audio('seed.wav')
seed_mel = audio_to_mel(seed_audio)[:96]  # Use 96 frames (divisible by 4)
audio = model.generate(seed_mel, num_frames=500)
sf.write('generated_r4.wav', audio.numpy(), 22050)
print('Generated audio saved to generated_r4.wav')
"
```

---

## Notes

- Each step builds on the previous step and produces working, testable code
- Steps 1-2 focus on data pipeline (config + dataset)
- Steps 3-4 focus on model architecture (forward pass + generation)
- Step 5 focuses on training (loss computation)
- Steps 6-7 focus on validation (tests + end-to-end verification)
- Core end-to-end functionality is available after Step 5
- All steps maintain backward compatibility with existing APIs (external interfaces unchanged)
