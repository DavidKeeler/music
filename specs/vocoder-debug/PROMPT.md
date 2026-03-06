# Objective
Debug, validate, and enhance the vocoder training pipeline to ensure stable finetuning of pretrained HiFi-GAN/MelGAN on music data.

# Context
Vocoder training script needs validation and enhancements:
- No verification that pretrained model loads correctly
- Inefficient Python generator-based dataset pipeline
- No gradient clipping or NaN detection
- No validation dataset or metrics
- No audio generation during training (can't assess quality)
- Missing learning rate scheduling and early stopping

# Key Requirements

## 1. Model Loading Validation
- Add checks in `main()` to verify generator loaded successfully
- Test mel spectrogram generation (verify 80 bins)
- Test forward pass with dummy input
- Print diagnostic information
- Provide actionable error messages if loading fails

## 2. Efficient Dataset Pipeline
- Rewrite `VocoderDataset.create_dataset()` in `src/music_generation/vocoder.py`
- Use tf.data operations instead of Python generator
- Return tuple: `(train_dataset, val_dataset)` with 10% validation split
- Use `map()` with `py_function` for audio loading
- Add proper `ensure_shape()` for type safety
- Use `repeat()` for training, `prefetch(AUTOTUNE)` for performance

## 3. Enhanced Training Loop
- Modify `VocoderTraining` in `src/music_generation/train_vocoder.py`
- Add `clip_norm=1.0` parameter for gradient clipping
- Use `tf.clip_by_global_norm()` to clip gradients
- Add NaN/Inf detection after loss computation
- Print warning and skip update if NaN/Inf detected
- Track gradient norm properly

## 4. Validation Dataset and Metrics
- Update `train_vocoder()` to accept train and val datasets
- Pass `validation_data` and `validation_steps` to `model.fit()`
- Update ModelCheckpoint to monitor `val_loss`
- Save best model based on validation loss

## 5. Audio Generation Callback
- Create `AudioGenerationCallback` class in `src/music_generation/train_vocoder.py`
- Generate audio samples every 5 epochs
- Save target and predicted audio to `{checkpoint_dir}/samples/`
- Use soundfile to write WAV files
- Print confirmation when samples generated

## 6. Learning Rate Scheduling and Early Stopping
- Add `ExponentialDecay` schedule (initial_lr, decay_steps=1000, decay_rate=0.98)
- Add `EarlyStopping` callback (monitor='val_loss', patience=10)
- Add `CSVLogger` for training history
- Update optimizer to use schedule

# Acceptance Criteria

**Given** vocoder training starts  
**When** model loading completes  
**Then** validation output confirms generator loaded, mel params verified, forward pass successful

**Given** dataset is created  
**When** processing audio files  
**Then** train/val split is created with 90/10 ratio, dataset uses tf.data pipeline

**Given** training runs  
**When** processing batches  
**Then** gradients are clipped, no NaN/Inf issues occur, gradient norm is tracked

**Given** epoch completes  
**When** validation runs  
**Then** validation loss is computed and logged to TensorBoard

**Given** epoch is multiple of 5  
**When** audio generation callback triggers  
**Then** target and predicted audio samples saved to disk

**Given** training completes  
**When** best model is determined  
**Then** model with lowest val_loss is saved and can be loaded for inference

# Implementation Notes

- All changes maintain backward compatibility where possible
- Focus on stability and monitoring over performance optimization
- Audio samples are critical for assessing vocoder quality (loss alone insufficient)
- Gradient clipping prevents training instability common in vocoder training
- See `specs/vocoder-debug/design.md` for detailed technical design
- See `specs/vocoder-debug/plan.md` for step-by-step implementation guide

# Files to Modify

1. `src/music_generation/train_vocoder.py` - Main training script (validation, callbacks, scheduling)
2. `src/music_generation/vocoder.py` - Dataset pipeline rewrite
3. `requirements.txt` - Ensure soundfile is present

# Testing

- Verify model loading with valid/invalid model names
- Test dataset creation with small file set (10 files)
- Run training for 10 epochs, verify validation metrics
- Listen to generated audio samples (quality assessment)
- Verify best model is saved correctly
- Check TensorBoard for train/val curves

# Reference
Detailed design and plan in `specs/vocoder-debug/`
