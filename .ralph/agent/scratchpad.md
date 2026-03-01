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
