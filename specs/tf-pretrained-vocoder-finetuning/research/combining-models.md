# Combining Multiple Keras Models for Inference

## Overview

When building inference pipelines that combine multiple models (e.g., MelGenerator + Vocoder), there are several Keras-native patterns to consider.

## Pattern 1: Sequential Composition (Recommended for Our Use Case)

Create a wrapper model that calls models sequentially:

```python
class MusicGenerationModel(tf.keras.Model):
    """Combines MelGenerator and Vocoder for end-to-end generation."""
    
    def __init__(self, mel_generator, vocoder):
        super().__init__()
        self.mel_generator = mel_generator
        self.vocoder = vocoder
    
    def call(self, inputs, training=False):
        # Generate mel spectrogram
        mel = self.mel_generator(inputs, training=training)
        
        # Convert to audio
        audio = self.vocoder(mel, training=training)
        
        return audio
    
    def generate(self, seed_mel, num_frames, temperature=1.0):
        """Autoregressive generation with vocoding."""
        # Generate mel frames
        generated_mel = self.mel_generator.generate(
            seed_mel, num_frames, temperature
        )
        
        # Convert to audio
        audio = self.vocoder(generated_mel, training=False)
        
        return audio
```

## Pattern 2: Functional API

For models with fixed architectures:

```python
# Define inputs
seed_input = tf.keras.Input(shape=(None, 80))

# Connect models
mel_output = mel_generator(seed_input)
audio_output = vocoder(mel_output)

# Create combined model
combined_model = tf.keras.Model(inputs=seed_input, outputs=audio_output)
```

## Pattern 3: Model Subclassing with Custom Methods

Most flexible for complex inference logic:

```python
class MusicGenerationModel(tf.keras.Model):
    def __init__(self, mel_generator, vocoder):
        super().__init__()
        self.mel_generator = mel_generator
        self.vocoder = vocoder
    
    def call(self, inputs, training=False):
        """Standard forward pass."""
        mel = self.mel_generator(inputs, training=training)
        return self.vocoder(mel, training=training)
    
    def generate_from_seed(self, seed_mel, num_frames):
        """Custom generation method."""
        # Autoregressive mel generation
        mel = self._generate_mel_autoregressive(seed_mel, num_frames)
        
        # Vocode in chunks to avoid memory issues
        audio = self._vocode_in_chunks(mel, chunk_size=100)
        
        return audio
    
    def _generate_mel_autoregressive(self, seed, num_frames):
        # Implementation details
        pass
    
    def _vocode_in_chunks(self, mel, chunk_size):
        # Implementation details
        pass
```

## Best Practices

1. **Model Subclassing:** Use for complex inference logic with custom methods
2. **Keep models separate:** Don't merge weights, just compose at inference time
3. **Compatibility checks:** Verify output/input shapes match between models
4. **Training vs Inference:** Use `training=False` for inference
5. **Memory management:** Consider chunking for long sequences

## Loading from Checkpoints

```python
@classmethod
def from_checkpoints(cls, mel_checkpoint, vocoder_checkpoint):
    """Load both models from checkpoints."""
    mel_generator = tf.keras.models.load_model(mel_checkpoint)
    vocoder = tf.keras.models.load_model(vocoder_checkpoint)
    
    # Verify compatibility
    assert mel_generator.output_shape[-1] == 80, "Mel generator must output 80 mel bins"
    
    return cls(mel_generator, vocoder)
```

## Example Usage

```python
# Create combined model
model = MusicGenerationModel.from_checkpoints(
    'checkpoints/mel_generator',
    'checkpoints/vocoder'
)

# Generate audio
seed_mel = load_seed_audio()
audio = model.generate(seed_mel, num_frames=500)

# Save
save_audio(audio, 'generated.wav')
```

## Key Considerations for Our Use Case

- **MelGenerator** outputs `[batch, time, 80]` mel spectrograms
- **Vocoder** expects `[batch, time, 80]` input
- **Autoregressive generation** happens in MelGenerator, vocoder just converts
- **Separate training** - models trained independently, combined only for inference
- **No gradient flow** between models during inference

## References

- Keras Functional API: https://www.tensorflow.org/guide/keras/functional_api
- Model Subclassing: https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
