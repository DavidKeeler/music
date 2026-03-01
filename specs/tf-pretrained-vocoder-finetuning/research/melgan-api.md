# MelGAN API and Model Details

## Model Information

**Model:** `tensorspeech/tts-melgan-ljspeech-en`  
**Framework:** TensorFlowTTS  
**Training Data:** LJSpeech dataset (English, single speaker)  
**Sample Rate:** 22,050 Hz  
**Paper:** [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711)

## Loading the Model

```python
from tensorflow_tts.inference import TFAutoModel

melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")
```

The model auto-downloads from Hugging Face on first use.

## Inference API

```python
# Input: mel spectrogram [batch, time, mel_bins]
# Output: audio waveform [batch, samples, 1]
audio = melgan.inference(mel_outputs)

# Extract audio (squeeze channel dimension)
audio = audio[0, :, 0]  # [samples]
```

**Key Points:**
- Input shape: `[batch, time, mel_bins]` where `mel_bins=80`
- Output shape: `[batch, samples, 1]` (needs squeezing to `[batch, samples]`)
- Method: `inference()` for generation
- The model is a Keras Model, so it has `trainable_variables` for finetuning

## Finetuning Considerations

Based on TensorFlowTTS GitHub issues:

1. **Learning Rate:** Use lower learning rate than training from scratch (common for transfer learning)
2. **Overfitting Risk:** MelGAN can overfit quickly on small datasets - monitor validation loss
3. **Training Speed:** Can be slow on some GPUs, consider batch size adjustments
4. **Multi-band MelGAN:** Alternative model with better quality but similar API

## Model Architecture

MelGAN uses:
- Generator with transposed convolutions for upsampling
- Trained with adversarial loss + feature matching loss
- Pretrained version includes only the generator (no discriminators needed for inference)

## References

- Hugging Face: https://huggingface.co/tensorspeech/tts-melgan-ljspeech-en
- TensorFlowTTS: https://github.com/TensorSpeech/TensorFlowTTS
- Paper: https://arxiv.org/abs/1910.06711
