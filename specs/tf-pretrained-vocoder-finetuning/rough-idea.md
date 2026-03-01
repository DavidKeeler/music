# Rough Idea: TensorFlow Pretrained Vocoder Finetuning

Port the pretrained vocoder finetuning feature from the PyTorch implementation (conducting2) to TensorFlow (conducting3). Replace or enhance the current HiFi-GAN vocoder implementation with a pretrained TensorFlow Hub vocoder, add finetuning capability, and create an inference model class for end-to-end music generation.

## Context

- **Source:** `/Users/davidkeeler/code/conducting2/specs/pretrained-vocoder-finetuning/`
- **Target:** `/Users/davidkeeler/code/conducting3` (TensorFlow/Keras implementation)
- **Current State:** 
  - TensorFlow implementation uses HiFi-GAN from TensorFlow Hub
  - Located in `src/music_generation/vocoder.py`
  - Has basic vocoder wrapper but no finetuning capability
  - Has `train_vocoder.py` but may need updates
- **PyTorch Version:**
  - Uses SpeechBrain's `speechbrain/tts-hifigan-ljspeech` pretrained model
  - Finetunes generator-only with STFT loss
  - Has unified `MusicGenerationModel` inference class
  - Uses LJSpeech mel parameters for compatibility

## Goal

Improve audio quality and add finetuning capability by:
1. Loading a pretrained TensorFlow Hub HiFi-GAN vocoder
2. Enabling finetuning on music domain data
3. Creating a unified inference model class (similar to PyTorch version)
4. Ensuring mel parameter compatibility with pretrained model

## Key Differences from PyTorch

- TensorFlow Hub instead of SpeechBrain
- Keras Model API instead of PyTorch nn.Module
- tf.data.Dataset instead of PyTorch DataLoader
- TensorFlow-specific training loop patterns
- Different checkpoint format (.h5 or SavedModel)

## Reference Implementation

See `/Users/davidkeeler/code/conducting2/specs/pretrained-vocoder-finetuning/` for:
- Design patterns
- Architecture decisions
- Mel parameter requirements
- Training approach
- Inference model structure
