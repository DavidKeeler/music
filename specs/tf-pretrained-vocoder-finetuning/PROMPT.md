# PROMPT: TensorFlow Pretrained Vocoder Finetuning

## Objective

Enhance the TensorFlow music generation implementation with three features: (1) unified inference model class combining MelGenerator and Vocoder, (2) gradient norm logging in vocoder training, and (3) SavedModel checkpoint format. Use MelGAN from TensorFlowTTS as the pretrained vocoder.

## Key Requirements

1. **MusicGenerationModel Class** - Create `src/music_generation/inference.py` with a Keras Model that:
   - Wraps MelGenerator and Vocoder
   - Has `generate(seed_mel, num_frames, temperature)` method for autoregressive generation
   - Has `from_checkpoints(mel_checkpoint, vocoder_checkpoint)` class method
   - Handles shape conversions: mel `[time, 80]` → audio `[samples]`

2. **Gradient Norm Logging** - Update `VocoderTraining` in `src/music_generation/train_vocoder.py`:
   - Add `grad_norm_tracker = tf.keras.metrics.Mean(name="grad_norm")` in `__init__`
   - Compute `grad_norm = tf.sqrt(sum([tf.reduce_sum(g**2) for g in grads if g is not None]))` in `train_step`
   - Return `{"loss": loss, "grad_norm": self.grad_norm_tracker.result()}`
   - Add `@property metrics` returning `[self.grad_norm_tracker]`

3. **SavedModel Checkpoints** - Update checkpoint management:
   - Change `save_weights_only=False` in `train_vocoder()` ModelCheckpoint callback
   - Simplify `load_vocoder_from_checkpoint()` to just `tf.keras.models.load_model(checkpoint_path)`

4. **MelGAN Integration** - Update `src/music_generation/vocoder.py`:
   - Add `TensorFlowTTS>=1.0.0` to `requirements.txt`
   - Update `load_pretrained_vocoder()` to use `TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")`
   - Update `HiFiGANVocoder.call()` to use `self.generator.inference()` and squeeze output from `[batch, samples, 1]` to `[batch, samples]`

5. **Tests** - Add unit tests and smoke tests:
   - `tests/test_inference.py` - Test MusicGenerationModel initialization and generate method
   - Update `tests/test_train_vocoder.py` - Test gradient norm metric exists
   - `tests/test_training_smoke.py` - Smoke test training for 1 epoch and checkpoint loading

## Acceptance Criteria

**Given** vocoder training is running  
**When** I check the logs  
**Then** I should see both `loss` and `grad_norm` metrics

**Given** training completes  
**When** I check the checkpoint directory  
**Then** checkpoints should be directories (SavedModel format), not .h5 files

**Given** trained MelGenerator and Vocoder checkpoints  
**When** I call `MusicGenerationModel.from_checkpoints(mel_path, vocoder_path)`  
**Then** it should load both models successfully

**Given** a MusicGenerationModel instance  
**When** I call `generate(seed_mel, num_frames=200)`  
**Then** it should return a 1D audio tensor

**Given** all tests  
**When** I run `pytest tests/`  
**Then** all tests should pass

## Implementation Notes

- **Minimal changes only** - Don't modify code that already works
- **Keep existing class names** - `HiFiGANVocoder` name is fine (no rename needed)
- **MelGAN output shape** - Returns `[batch, samples, 1]`, must squeeze to `[batch, samples]`
- **Shape compatibility** - MelGenerator outputs `[batch, time, 80]`, Vocoder expects `[batch, time, 80]`
- **Follow plan.md** - Implement in order: grad norm → SavedModel → MelGAN → inference model → tests

## Reference

All design details, architecture diagrams, and implementation guidance are in:
- `specs/tf-pretrained-vocoder-finetuning/design.md`
- `specs/tf-pretrained-vocoder-finetuning/plan.md`
- `specs/tf-pretrained-vocoder-finetuning/research/`

## File Changes

**New:**
- `src/music_generation/inference.py` (~80 lines)
- `tests/test_inference.py` (~50 lines)
- `tests/test_training_smoke.py` (~50 lines)

**Modified:**
- `src/music_generation/train_vocoder.py` (~20 lines)
- `src/music_generation/vocoder.py` (~30 lines)
- `tests/test_train_vocoder.py` (~20 lines)
- `requirements.txt` (1 line)

**Total: ~250 lines of code**
