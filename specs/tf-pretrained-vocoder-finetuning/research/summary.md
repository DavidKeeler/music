# Research Summary

## Overview

Research conducted to inform TensorFlow pretrained vocoder finetuning implementation.

## Key Findings

### 1. MelGAN Model (melgan-api.md)

**Model:** `tensorspeech/tts-melgan-ljspeech-en` from TensorFlowTTS

**Key Points:**
- Loads via `TFAutoModel.from_pretrained()`
- Inference method: `melgan.inference(mel)` 
- Input: `[batch, time, 80]` mel spectrograms
- Output: `[batch, samples, 1]` audio (needs squeezing)
- Trained on LJSpeech at 22,050 Hz
- Has `trainable_variables` for finetuning
- Risk of overfitting on small datasets

### 2. Custom train_step (keras-custom-train-step.md)

**Pattern for gradient norm logging:**
```python
class CustomModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.grad_norm_tracker = keras.metrics.Mean(name="grad_norm")
    
    def train_step(self, data):
        # ... compute gradients ...
        grad_norm = tf.sqrt(sum([tf.reduce_sum(g**2) for g in gradients if g is not None]))
        
        self.loss_tracker.update_state(loss)
        self.grad_norm_tracker.update_state(grad_norm)
        
        return {"loss": self.loss_tracker.result(), "grad_norm": self.grad_norm_tracker.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.grad_norm_tracker]
```

**Benefits:**
- Still use `fit()` with callbacks
- Automatic metric reset between epochs
- Works with TensorBoard

### 3. SavedModel Format (savedmodel-format.md)

**Recommended approach:**
```python
# Save full model (architecture + weights + optimizer)
model.save('path/to/model')

# Load and continue training
model = tf.keras.models.load_model('path/to/model')
model.fit(data, labels, epochs=10)
```

**Benefits:**
- Portable (no source code needed)
- Includes optimizer state for resuming training
- Production-ready format
- Easy versioning

### 4. Combining Models (combining-models.md)

**Recommended pattern for MelGenerator + Vocoder:**

Model subclassing with custom methods:
```python
class MusicGenerationModel(tf.keras.Model):
    def __init__(self, mel_generator, vocoder):
        super().__init__()
        self.mel_generator = mel_generator
        self.vocoder = vocoder
    
    def call(self, inputs, training=False):
        mel = self.mel_generator(inputs, training=training)
        return self.vocoder(mel, training=training)
    
    def generate(self, seed_mel, num_frames, temperature=1.0):
        """Custom autoregressive generation."""
        generated_mel = self.mel_generator.generate(seed_mel, num_frames, temperature)
        audio = self.vocoder(generated_mel, training=False)
        return audio
    
    @classmethod
    def from_checkpoints(cls, mel_checkpoint, vocoder_checkpoint):
        mel_generator = tf.keras.models.load_model(mel_checkpoint)
        vocoder = tf.keras.models.load_model(vocoder_checkpoint)
        return cls(mel_generator, vocoder)
```

**Key considerations:**
- Keep models separate (don't merge weights)
- Verify shape compatibility (80 mel bins)
- Use `training=False` for inference
- Models trained independently, combined only for inference

## Implementation Implications

1. **VocoderTraining class:** Add `grad_norm_tracker` metric in `train_step()`
2. **Checkpoint saving:** Switch from `save_weights_only=True` to full model save
3. **MelGAN loading:** Use `TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")`
4. **Inference model:** Create `MusicGenerationModel` class with `generate()` and `from_checkpoints()` methods
5. **Shape handling:** MelGAN expects `[batch, time, 80]`, outputs `[batch, samples, 1]`

## References

- MelGAN: https://huggingface.co/tensorspeech/tts-melgan-ljspeech-en
- Custom train_step: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
- SavedModel: https://www.tensorflow.org/guide/saved_model
- Model Subclassing: https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
