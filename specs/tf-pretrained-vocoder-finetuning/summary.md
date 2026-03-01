# Summary: TensorFlow Pretrained Vocoder Finetuning

## Project Overview

Enhance the existing TensorFlow music generation implementation by adding three key features: (1) unified inference model class combining MelGenerator and Vocoder, (2) gradient norm logging in vocoder training, and (3) improved checkpoint management using SavedModel format. Uses MelGAN from TensorFlowTTS as the pretrained vocoder.

## Artifacts Created

### Requirements
- **rough-idea.md** - Initial concept: port PyTorch pretrained vocoder finetuning to TensorFlow
- **requirements.md** - 9 Q&A pairs clarifying the specification

### Research
- **melgan-api.md** - MelGAN model details, loading, and inference API
- **keras-custom-train-step.md** - Pattern for adding gradient norm as custom metric
- **savedmodel-format.md** - SavedModel format for saving/loading models
- **combining-models.md** - Keras patterns for combining multiple models
- **summary.md** - Consolidated research findings

### Design
- **design.md** - Complete design document with:
  - Architecture diagrams
  - Component interfaces
  - Data models
  - Error handling
  - Acceptance criteria
  - Testing strategy

### Implementation Plan
- **plan.md** - 6-step implementation plan:
  1. Add gradient norm logging to VocoderTraining
  2. Update checkpoint saving to SavedModel format
  3. Update MelGAN loading and output handling
  4. Create MusicGenerationModel inference class
  5. Add unit tests
  6. Add smoke tests

## Key Decisions

1. **Use MelGAN from TensorFlowTTS** - `tensorspeech/tts-melgan-ljspeech-en` pretrained model
2. **Enhance existing implementation** - Don't replace, just add missing features
3. **Keras-native patterns** - Model subclassing, custom train_step, metrics
4. **SavedModel format** - Full model saving instead of weights-only
5. **Gradient norm as metric** - Computed in train_step, automatically logged
6. **Model subclassing for inference** - MusicGenerationModel wraps both models
7. **Minimal changes** - Only ~200 lines of new/modified code

## Critical Implementation Details

### MelGAN Integration
- Load via `TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")`
- Use `inference()` method (not `call()`)
- Output shape: `[batch, samples, 1]` - needs squeezing to `[batch, samples]`
- Input shape: `[batch, time, 80]` mel spectrograms

### Gradient Norm Logging
```python
grad_norm = tf.sqrt(sum([tf.reduce_sum(g**2) for g in grads if g is not None]))
self.grad_norm_tracker.update_state(grad_norm)
```

### SavedModel Checkpoints
```python
# Save
model.save('path/to/checkpoint')

# Load
model = tf.keras.models.load_model('path/to/checkpoint')
```

### Inference Model Pattern
```python
class MusicGenerationModel(tf.keras.Model):
    def __init__(self, mel_generator, vocoder):
        super().__init__()
        self.mel_generator = mel_generator
        self.vocoder = vocoder
    
    def generate(self, seed_mel, num_frames, temperature=1.0):
        generated_mel = self.mel_generator.generate(seed_mel, num_frames, temperature)
        mel_batch = tf.expand_dims(generated_mel, 0)
        audio = self.vocoder(mel_batch, training=False)
        return tf.squeeze(audio, axis=0)
    
    @classmethod
    def from_checkpoints(cls, mel_checkpoint, vocoder_checkpoint):
        mel_generator = tf.keras.models.load_model(mel_checkpoint)
        vocoder = tf.keras.models.load_model(vocoder_checkpoint)
        return cls(mel_generator, vocoder)
```

## File Changes Summary

### New Files
- `src/music_generation/inference.py` - MusicGenerationModel class (~80 lines)
- `tests/test_inference.py` - Unit tests for inference model (~50 lines)
- `tests/test_training_smoke.py` - Smoke tests (~50 lines)

### Modified Files
- `src/music_generation/train_vocoder.py` - Add grad norm, SavedModel (~20 lines changed)
- `src/music_generation/vocoder.py` - Update MelGAN loading and call() (~30 lines changed)
- `tests/test_train_vocoder.py` - Add grad norm tests (~20 lines added)
- `requirements.txt` - Add TensorFlowTTS dependency (1 line)

### Total Changes
- ~150 lines new code
- ~50 lines modified code
- ~100 lines test code
- **~300 lines total**

## Dependencies

Add to `requirements.txt`:
```
TensorFlowTTS>=1.0.0
```

## Next Steps

### For Implementation

Follow the 6-step plan in `plan.md`:
1. Start with gradient norm logging (easiest, immediate value)
2. Update checkpoint format (improves management)
3. Integrate MelGAN (enables pretrained model)
4. Create inference model (ties everything together)
5. Add unit tests (validates components)
6. Add smoke tests (validates end-to-end)

### For Testing

After implementation:
1. Run unit tests: `pytest tests/test_inference.py tests/test_train_vocoder.py -v`
2. Run smoke tests: `pytest tests/test_training_smoke.py -v -s`
3. Train vocoder on small dataset (1-2 epochs)
4. Generate audio samples and verify quality
5. Check TensorBoard for grad_norm metric

### Example Usage

```python
from src.music_generation.inference import MusicGenerationModel
from src.music_generation.audio_utils import load_audio, audio_to_mel_ljspeech
import soundfile as sf

# Load models
model = MusicGenerationModel.from_checkpoints(
    'checkpoints/mel_generator',
    'checkpoints/vocoder/checkpoint_epoch_10'
)

# Load seed
seed_audio = load_audio('seed.wav')
seed_mel = audio_to_mel_ljspeech(seed_audio)

# Generate
audio = model.generate(seed_mel, num_frames=500)

# Save
sf.write('generated.wav', audio.numpy(), 22050)
```

## Success Metrics

- [ ] Gradient norm appears in training logs and TensorBoard
- [ ] Checkpoints save as directories (SavedModel format)
- [ ] MelGAN loads and generates audio without errors
- [ ] MusicGenerationModel generates audio end-to-end
- [ ] All unit tests pass
- [ ] Smoke tests complete without errors
- [ ] Generated audio quality is reasonable

## Comparison with PyTorch Version

**Similarities:**
- Same approach: load pretrained vocoder and finetune
- Same features: gradient norm logging, inference model class
- Same training strategy: generator-only with STFT loss

**Differences:**
- PyTorch uses SpeechBrain HiFi-GAN, TensorFlow uses TensorFlowTTS MelGAN
- PyTorch uses `.pth` checkpoints, TensorFlow uses SavedModel
- PyTorch uses `nn.Module`, TensorFlow uses `tf.keras.Model`
- Different APIs but same concepts

## Project Location

`/Users/davidkeeler/code/conducting3/specs/tf-pretrained-vocoder-finetuning/`
