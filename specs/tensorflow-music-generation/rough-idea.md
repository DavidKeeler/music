# Rough Idea: TensorFlow Music Generation System

Port the existing PyTorch music generation system from `/Users/davidkeeler/code/conducting2` to TensorFlow/Keras.

## Source System

The PyTorch implementation in `conducting2` includes:
- Transformer-based mel spectrogram generator with autoregressive generation
- HiFi-GAN vocoder finetuning for mel-to-audio conversion
- Teacher forcing with exponential decay during training
- Custom STFT loss for vocoder training
- End-to-end inference pipeline combining mel generation and vocoding

## Target System

Rewrite in TensorFlow 2.x following the patterns from `/Users/davidkeeler/code/conducting/src/main/python/complex_music_model/`:
- Use `tf.keras.Model` for all models
- Use `model.fit()` with custom `train_step()` for training
- Use `tf.data.Dataset` for data pipelines
- Use TensorFlow audio operations for preprocessing
- Use Keras callbacks for checkpointing and logging

## Key Requirements

1. **Mel Generator**: Transformer model that generates mel spectrograms autoregressively
2. **Vocoder**: Neural vocoder (HiFi-GAN or alternative) that converts mels to audio
3. **Training**: Custom training loop with teacher forcing decay
4. **Inference**: End-to-end generation from input to audio waveform
5. **Compatibility**: Match the functionality of the PyTorch version

## Reference Implementations

- PyTorch source: `/Users/davidkeeler/code/conducting2/src/music_generation/`
- TensorFlow patterns: `/Users/davidkeeler/code/conducting/src/main/python/complex_music_model/`
