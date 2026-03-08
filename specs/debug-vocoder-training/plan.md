# Debug Vocoder Training - Implementation Plan

## Checklist
- [ ] Step 1: Fix missing huggingface_hub dependency
- [ ] Step 2: Verify pretrained model loading
- [ ] Step 3: Test dataset loading and preprocessing
- [ ] Step 4: Run single training step
- [ ] Step 5: Monitor full training execution

---

## Step 1: Fix missing huggingface_hub dependency

**Objective:** Install missing dependency and update requirements.txt

**Implementation:**
- Install `huggingface_hub` in venv: `pip install huggingface_hub`
- Check if `requirements.txt` includes it, add if missing
- Verify import works: `python -c "from huggingface_hub import hf_hub_download"`

**Tests:**
- Import statement succeeds without ModuleNotFoundError

**Integration:**
- Dependency available for vocoder.py to use

**Demo:**
- Show successful import and updated requirements.txt

---

## Step 2: Verify pretrained model loading

**Objective:** Ensure HiFiGAN pretrained model downloads and loads correctly

**Implementation:**
- Run vocoder loading in isolation to catch any download/loading errors
- Check if model files are cached locally
- Verify model architecture loads on Metal GPU
- Handle any TensorFlow/Keras compatibility warnings

**Tests:**
- Model loads without errors
- Model can accept dummy mel spectrogram input
- Output shape is correct

**Integration:**
- Loaded model ready for training script

**Demo:**
- Show model summary and successful forward pass

---

## Step 3: Test dataset loading and preprocessing

**Objective:** Verify data pipeline works with provided data_dir

**Implementation:**
- Check if `~/data/music/musicnet` contains expected audio files
- Test audio loading and mel spectrogram conversion
- Verify batch creation with batch_size=16
- Check data shapes match model expectations (80 mel bins)

**Tests:**
- Dataset yields batches without errors
- Mel spectrograms have correct shape [batch, time, 80]
- Audio reconstruction target shapes are correct

**Integration:**
- Data pipeline feeds into training loop

**Demo:**
- Show sample batch shapes and statistics

---

## Step 4: Run single training step

**Objective:** Execute one training iteration to catch runtime errors

**Implementation:**
- Modify train_vocoder.py to run 1 step only (or add --debug flag)
- Execute forward pass, loss calculation, backward pass
- Check for Metal GPU memory issues
- Verify gradient updates work
- Log loss values

**Tests:**
- Single step completes without crashes
- Loss is finite (not NaN/Inf)
- Model weights update

**Integration:**
- Training loop ready for full execution

**Demo:**
- Show loss value and successful weight update

---

## Step 5: Monitor full training execution

**Objective:** Run full training with monitoring and error handling

**Implementation:**
- Start training with original parameters (50 epochs, batch_size=16)
- Add checkpointing every N steps
- Log training metrics (loss, learning rate)
- Monitor Metal GPU memory usage
- Add error recovery (save checkpoint on crash)

**Tests:**
- Training runs for multiple epochs
- Checkpoints save correctly
- Loss decreases over time
- Generated audio quality improves

**Integration:**
- Complete training pipeline operational

**Demo:**
- Show training logs, saved checkpoints, and sample generated audio
