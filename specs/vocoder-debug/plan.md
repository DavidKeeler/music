# Implementation Plan

## Checklist
- [ ] Step 1: Add model loading validation
- [ ] Step 2: Rewrite dataset pipeline with tf.data
- [ ] Step 3: Enhance training loop with gradient clipping and NaN detection
- [ ] Step 4: Add validation dataset and metrics
- [ ] Step 5: Add audio generation callback
- [ ] Step 6: Add learning rate scheduling and early stopping

---

## Step 1: Add Model Loading Validation

**Objective:** Verify pretrained vocoder loads correctly and mel parameters are compatible.

**Implementation:**
Add validation checks in `main()` function in `src/music_generation/train_vocoder.py`:

1. Check generator is not None
2. Test mel spectrogram generation
3. Verify mel shape matches expected (80 bins)
4. Test forward pass with dummy input
5. Print diagnostic information

**Files to modify:**
- `src/music_generation/train_vocoder.py`

**Code changes:**
```python
def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load pretrained vocoder
    print("="*60)
    print("Loading pretrained vocoder...")
    vocoder_model = vocoder.load_pretrained_vocoder(args.model_name)
    
    if vocoder_model.generator is None:
        raise ValueError(
            "Failed to load pretrained vocoder. "
            "Provide --model_name or check TensorFlowTTS installation."
        )
    
    print(f"✓ Loaded generator: {type(vocoder_model.generator).__name__}")
    
    # Verify mel parameters
    print("\nVerifying mel spectrogram parameters...")
    from . import audio_utils
    test_audio = tf.random.normal([8192])
    test_mel = audio_utils.audio_to_mel_ljspeech(test_audio)
    print(f"  Mel shape: {test_mel.shape}")
    print(f"  Expected: [?, 80]")
    
    if test_mel.shape[-1] != 80:
        raise ValueError(
            f"Mel bins mismatch: got {test_mel.shape[-1]}, expected 80. "
            "Check audio_utils.audio_to_mel_ljspeech() configuration."
        )
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_mel_batch = tf.expand_dims(test_mel, 0)
    try:
        test_output = vocoder_model.generator(test_mel_batch, training=False)
        print(f"  Input mel: {test_mel_batch.shape}")
        print(f"  Output audio: {test_output.shape}")
        print("✓ Forward pass successful")
    except Exception as e:
        raise RuntimeError(f"Forward pass failed: {e}")
    
    print("="*60)
    print()
    
    # Continue with dataset and training...
```

**Tests:**
- Run with valid model name, verify passes
- Run with invalid model name, verify error message
- Verify mel shape check catches mismatches

**Integration:**
Non-breaking, adds validation before training starts

**Demo:**
Run training command, show validation output before training begins.

---

## Step 2: Rewrite Dataset Pipeline with tf.data

**Objective:** Replace Python generator with efficient tf.data pipeline and add train/val split.

**Implementation:**
Rewrite `VocoderDataset` in `src/music_generation/vocoder.py`:

1. Add `cache_dir` parameter
2. Implement `create_dataset()` returning (train_ds, val_ds) tuple
3. Use `tf.data.Dataset.from_tensor_slices()` for file paths
4. Use `map()` with `py_function` for audio loading
5. Add proper shape setting with `ensure_shape()`
6. Add validation split (default 10%)
7. Use `repeat()` for training dataset
8. Add `prefetch(AUTOTUNE)` for performance

**Files to modify:**
- `src/music_generation/vocoder.py`
- `src/music_generation/train_vocoder.py` (update dataset creation call)

**Code changes:**
See design.md Section 2 for full implementation.

Key changes:
- `create_dataset()` returns tuple: `(train_dataset, val_dataset)`
- Use native tf.data operations
- Add validation split parameter
- Remove Python generator approach

**Tests:**
1. Unit test: Create dataset with 10 files, verify train/val split
2. Integration test: Iterate through dataset, verify shapes
3. Performance test: Measure dataset creation time

**Integration:**
- Update `train_vocoder.py` to handle tuple return
- Update `main()` to pass both datasets to training

**Demo:**
Run dataset creation, show train/val split counts and iteration speed.

---

## Step 3: Enhance Training Loop

**Objective:** Add gradient clipping, NaN detection, and better metrics.

**Implementation:**
Modify `VocoderTraining` class in `src/music_generation/train_vocoder.py`:

1. Add `clip_norm` parameter to `__init__`
2. Use `tf.clip_by_global_norm()` for gradient clipping
3. Add NaN/Inf detection after loss computation
4. Print warning if NaN/Inf detected
5. Track gradient norm properly

**Files to modify:**
- `src/music_generation/train_vocoder.py`

**Code changes:**
```python
class VocoderTraining(tf.keras.Model):
    """Enhanced vocoder training wrapper."""
    
    def __init__(self, generator, stft_loss, clip_norm=1.0):
        super().__init__()
        self.generator = generator
        self.stft_loss = stft_loss
        self.clip_norm = clip_norm
        self.grad_norm_tracker = tf.keras.metrics.Mean(name="grad_norm")
    
    def train_step(self, data):
        mel, target_audio = data
        
        with tf.GradientTape() as tape:
            pred_audio = self.generator(mel, training=True)
            
            # Trim to same length
            min_len = tf.minimum(tf.shape(pred_audio)[-1], tf.shape(target_audio)[-1])
            pred_audio = pred_audio[..., :min_len]
            target_audio = target_audio[..., :min_len]
            
            loss = self.stft_loss(pred_audio, target_audio)
            
            # Check for NaN/Inf
            if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                tf.print("⚠️  WARNING: NaN/Inf loss detected! Skipping update.")
                return {"loss": loss, "grad_norm": 0.0}
        
        grads = tape.gradient(loss, self.generator.trainable_variables)
        
        # Clip gradients
        if self.clip_norm > 0:
            grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_norm)
        else:
            grad_squares = [tf.reduce_sum(g**2) for g in grads if g is not None]
            grad_norm = tf.sqrt(tf.add_n(grad_squares)) if grad_squares else 0.0
        
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        self.grad_norm_tracker.update_state(grad_norm)
        
        return {"loss": loss, "grad_norm": self.grad_norm_tracker.result()}
```

**Tests:**
1. Unit test: Verify gradient clipping works
2. Integration test: Train with NaN-inducing data, verify detection
3. Verify gradient norm is tracked correctly

**Integration:**
- Update `train_vocoder()` to pass `clip_norm` parameter
- Add command-line argument for clip_norm (optional)

**Demo:**
Run training, show gradient norm in logs, verify no NaN issues.

---

## Step 4: Add Validation Dataset and Metrics

**Objective:** Compute validation loss each epoch for monitoring.

**Implementation:**
Modify `train_vocoder()` in `src/music_generation/train_vocoder.py`:

1. Accept both train and val datasets
2. Pass `validation_data` to `model.fit()`
3. Add `validation_steps` parameter
4. Update checkpoint to monitor `val_loss`

**Files to modify:**
- `src/music_generation/train_vocoder.py`

**Code changes:**
```python
def train_vocoder(generator, train_dataset, val_dataset, epochs, lr, checkpoint_dir):
    """Train vocoder with validation."""
    
    # ... (STFT loss, training model setup)
    
    # Calculate steps
    steps_per_epoch = 100  # Adjust based on dataset size
    validation_steps = 20
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_dir / "best_model.h5"),
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',  # Monitor validation loss
            verbose=1
        ),
        # ... other callbacks
    ]
    
    training_model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,  # Add validation
        validation_steps=validation_steps,
        callbacks=callbacks
    )
```

**Tests:**
1. Verify validation loss is computed each epoch
2. Verify best model is saved based on val_loss
3. Check TensorBoard shows train and val curves

**Integration:**
- Update `main()` to pass val_dataset to `train_vocoder()`
- Ensure dataset creation returns both train and val

**Demo:**
Run training, show validation loss in logs and TensorBoard.

---

## Step 5: Add Audio Generation Callback

**Objective:** Generate audio samples during training to assess quality.

**Implementation:**
Create new callback class in `src/music_generation/train_vocoder.py`:

1. Create `AudioGenerationCallback` class
2. Generate audio every N epochs
3. Save target and predicted audio to disk
4. Add to callbacks list in `train_vocoder()`

**Files to modify:**
- `src/music_generation/train_vocoder.py`
- `requirements.txt` (ensure soundfile is present)

**Code changes:**
```python
class AudioGenerationCallback(tf.keras.callbacks.Callback):
    """Generate audio samples during training."""
    
    def __init__(self, val_dataset, output_dir, generator, frequency=5):
        super().__init__()
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = generator
        self.frequency = frequency
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency != 0:
            return
        
        # Get one batch
        for mel, target_audio in self.val_dataset.take(1):
            pred_audio = self.generator(mel, training=False)
            
            # Save first sample
            import soundfile as sf
            target_path = self.output_dir / f"epoch_{epoch+1:03d}_target.wav"
            pred_path = self.output_dir / f"epoch_{epoch+1:03d}_pred.wav"
            
            sf.write(str(target_path), target_audio[0].numpy(), 22050)
            sf.write(str(pred_path), pred_audio[0].numpy(), 22050)
            
            print(f"\n✓ Generated audio: {pred_path}")
            break

# In train_vocoder(), add to callbacks:
callbacks.append(
    AudioGenerationCallback(
        val_dataset,
        checkpoint_dir / "samples",
        generator,
        frequency=5
    )
)
```

**Tests:**
1. Verify audio files are created every 5 epochs
2. Verify audio files are valid WAV format
3. Listen to samples to assess quality

**Integration:**
- Ensure soundfile is in requirements.txt
- Create samples directory automatically

**Demo:**
Run training for 10 epochs, show generated audio files, play samples.

---

## Step 6: Add Learning Rate Scheduling and Early Stopping

**Objective:** Improve training convergence with LR decay and early stopping.

**Implementation:**
Modify `train_vocoder()` in `src/music_generation/train_vocoder.py`:

1. Create `ExponentialDecay` learning rate schedule
2. Add `EarlyStopping` callback
3. Add `CSVLogger` for training history
4. Update optimizer to use schedule

**Files to modify:**
- `src/music_generation/train_vocoder.py`

**Code changes:**
```python
def train_vocoder(generator, train_dataset, val_dataset, epochs, lr, checkpoint_dir):
    """Train vocoder with LR scheduling and early stopping."""
    
    # ... (STFT loss, training model)
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=1000,
        decay_rate=0.98,
        staircase=True
    )
    
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    )
    
    callbacks = [
        # ... (ModelCheckpoint, TensorBoard)
        tf.keras.callbacks.CSVLogger(
            str(checkpoint_dir / "training.csv"),
            append=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # ... (AudioGenerationCallback)
    ]
    
    # ... (fit)
```

**Tests:**
1. Verify learning rate decays over time
2. Verify early stopping triggers if no improvement
3. Verify CSV log is created with training history

**Integration:**
- Non-breaking, enhances existing training
- Early stopping may reduce total epochs

**Demo:**
Run training, show learning rate decay in TensorBoard, demonstrate early stopping.

---

## Summary

This plan transforms the vocoder training script from a basic finetuning loop into a robust, production-ready training pipeline with:

1. **Validation** - Model loading checks, mel parameter verification
2. **Efficiency** - tf.data pipeline, proper prefetching
3. **Stability** - Gradient clipping, NaN detection
4. **Monitoring** - Validation metrics, audio generation, TensorBoard
5. **Optimization** - LR scheduling, early stopping

**Estimated Impact:**
- Training stability: Significantly improved (gradient clipping, NaN detection)
- Dataset efficiency: 3-5x faster (tf.data vs Python generator)
- Monitoring: Comprehensive (validation, audio samples, metrics)
- Code quality: Production-ready with proper error handling

**Estimated Time:** ~2 hours for full implementation and testing
