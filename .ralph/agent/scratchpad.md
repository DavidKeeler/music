# Scratchpad: TensorFlow Pretrained Vocoder Finetuning

## Iteration 1 - Planning

### Understanding
The objective is to enhance the TensorFlow music generation implementation with three features:
1. **Gradient norm logging** - Add gradient norm metric to vocoder training
2. **SavedModel checkpoints** - Switch from weights-only to full model saving
3. **MelGAN integration** - Use TensorFlowTTS MelGAN instead of TF Hub HiFi-GAN
4. **Unified inference model** - Create MusicGenerationModel class combining MelGenerator + Vocoder

### Implementation Order (from plan.md)
1. Add gradient norm logging to VocoderTraining
2. Update checkpoint saving to SavedModel format
3. Update MelGAN loading and output handling
4. Create MusicGenerationModel inference class
5. Add unit tests
6. Add smoke tests

### Key Technical Details
- MelGAN outputs `[batch, samples, 1]` - must squeeze to `[batch, samples]`
- Use `TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")`
- Gradient norm: `tf.sqrt(sum([tf.reduce_sum(g**2) for g in grads if g is not None]))`
- SavedModel: change `save_weights_only=False` in ModelCheckpoint
- MusicGenerationModel: wraps mel_generator + vocoder, provides `generate()` and `from_checkpoints()`

### Files to Check
Need to examine existing implementation to understand current state:
- `src/music_generation/train_vocoder.py` - VocoderTraining class
- `src/music_generation/vocoder.py` - HiFiGANVocoder class, loading functions
- `src/music_generation/model.py` - MelGenerator class (for generate method)

### Plan
Will create tasks for each step in the implementation order.

## Iteration 1 - Task 1 Complete

### Task: Add gradient norm logging to VocoderTraining (task-1772397682-e602)

**What was done:**
- Added `grad_norm_tracker = tf.keras.metrics.Mean(name="grad_norm")` to `VocoderTraining.__init__`
- Computed gradient norm in `train_step()` after gradient computation: `grad_norm = tf.sqrt(sum([tf.reduce_sum(g**2) for g in grads if g is not None]))`
- Updated metric with `self.grad_norm_tracker.update_state(grad_norm)`
- Return both loss and grad_norm in metrics dict: `{"loss": loss, "grad_norm": self.grad_norm_tracker.result()}`
- Added `@property metrics` returning `[self.grad_norm_tracker]` for automatic reset between epochs

**Verification:**
- Syntax validation passed with `python3 -m py_compile`
- Code follows Keras custom train_step pattern from research

**Commit:** 20dba2e - "feat: add gradient norm logging to VocoderTraining"

**Next priority task:** Update checkpoint saving to SavedModel format (task-1772397682-1b6a)

## Iteration 2 - Task 2 Complete

### Task: Update checkpoint saving to SavedModel format (task-1772397682-1b6a)

**What was done:**
- Changed `save_weights_only=True` to `save_weights_only=False` in both `train.py` and `train_vocoder.py`
- Removed `.h5` extensions from checkpoint paths since SavedModel format saves as directories:
  - `train.py`: `mel_generator.weights.h5` → `mel_generator`
  - `train_vocoder.py`: `checkpoint_epoch_{epoch:02d}.h5` → `checkpoint_epoch_{epoch:02d}`
- Updated completion message in `train.py` from "Weights saved" to "Model saved"

**Verification:**
- Syntax validation passed with `python3 -m py_compile`
- Changes are minimal and focused on the checkpoint saving mechanism

**Commit:** a1061cf - "feat: switch checkpoint saving to SavedModel format"

**Next priority task:** Update MelGAN loading and output handling (task-1772397682-41f5)
