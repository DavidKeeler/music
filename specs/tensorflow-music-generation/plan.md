# Implementation Plan: TensorFlow Music Generation System

## Overview

Port PyTorch music generation system to TensorFlow/Keras following established patterns.

## Step 1: Project Setup
**Objective:** Create project structure and dependencies

**Tasks:**
- Create directory structure in `/Users/davidkeeler/code/conducting3/`
- Create `requirements.txt` with TensorFlow dependencies
- Create `README.md` with project overview
- Create empty `__init__.py` files

**Files created:**
- `requirements.txt`
- `README.md`
- `src/music_generation/__init__.py`
- `tests/__init__.py`

**Tests:** N/A (setup step)

**Demo:** Directory structure exists

---

## Step 2: Configuration Module
**Objective:** Port all hyperparameters to TensorFlow-compatible config

**Tasks:**
- Create `src/music_generation/config.py`
- Port model parameters from PyTorch config
- Port training parameters
- Port audio parameters (LJSpeech mel settings)
- Port teacher forcing parameters

**Reference:** `conducting2/src/music_generation/config.py`

**Tests:** `tests/test_config.py` - verify all constants exist and have correct types

**Demo:** Import config and print key parameters

---

## Step 3: Audio Utilities
**Objective:** Implement audio preprocessing with TensorFlow

**Tasks:**
- Create `src/music_generation/audio_utils.py`
- Implement `audio_to_mel()` using `tf.signal.stft()` and mel weight matrix
- Implement `audio_to_mel_ljspeech()` with LJSpeech parameters
- Implement audio loading with `tf.audio.decode_wav()`

**Reference:** 
- `conducting2/src/music_generation/audio_utils.py`
- `conducting/src/main/python/complex_music_model/train_audio.py` (audio_to_mel function)

**Tests:** `tests/test_audio_utils.py` - verify mel spectrogram shapes and values

**Demo:** Load audio file and convert to mel spectrogram

---

## Step 4: Custom Keras Layers
**Objective:** Implement reusable layer components

**Tasks:**
- Create `src/music_generation/layers.py`
- Implement `PositionalEncoding` layer
- Implement `FeedForward` layer (Dense + activation + Dense)
- Implement any custom attention if needed (or use built-in)

**Reference:** `conducting2/src/music_generation/model_components.py`

**Tests:** `tests/test_layers.py` - verify layer outputs and shapes

**Demo:** Create layers and pass dummy tensors through them

---

## Step 5: Mel Generator Model
**Objective:** Implement Transformer-based mel generator

**Tasks:**
- Create `src/music_generation/model.py`
- Implement `MelGenerator` as `tf.keras.Model`
- Use `tf.keras.layers.MultiHeadAttention` for attention
- Use custom layers from Step 4
- Implement encoder-decoder architecture
- Ensure tensor shapes are [batch, time, features]

**Reference:** `conducting2/src/music_generation/model.py`

**Tests:** `tests/test_model.py` - verify forward pass, output shapes, parameter count

**Demo:** Create model, pass dummy input, get output

---

## Step 6: Dataset Pipeline
**Objective:** Create tf.data.Dataset for training

**Tasks:**
- Create `src/music_generation/dataset.py`
- Implement `create_musicnet_dataset()` function
- Load audio files from directory
- Convert to mel spectrograms
- Create input/target pairs for autoregressive training
- Implement batching, shuffling, prefetching
- Implement caching for preprocessed data

**Reference:** 
- `conducting2/src/music_generation/dataset.py`
- `conducting/src/main/python/complex_music_model/train_audio.py` (make_batches function)

**Tests:** `tests/test_dataset.py` - verify dataset yields correct shapes

**Demo:** Create dataset and iterate through a few batches

---

## Step 7: Training with Teacher Forcing
**Objective:** Implement custom training loop with model.fit()

**Tasks:**
- Create `src/music_generation/train.py`
- Implement `MelGeneratorTraining` wrapper model with custom `train_step()`
- Implement teacher forcing with exponential decay
- Implement warmup + cosine learning rate schedule
- Implement training CLI with argparse
- Add callbacks: ModelCheckpoint, TensorBoard, CSVLogger
- Implement smoke test mode

**Reference:** `conducting/src/main/python/complex_music_model/train_audio.py`

**Tests:** `tests/test_train.py` - verify training step, teacher forcing decay

**Demo:** Run smoke test (1 epoch, 2 batches) and verify it completes

---

## Step 8: Vocoder Losses
**Objective:** Implement STFT loss for vocoder training

**Tasks:**
- Create `src/music_generation/losses.py`
- Implement `MultiResolutionSTFTLoss` using `tf.signal.stft()`
- Support multiple FFT sizes
- Compute L1 loss on magnitude spectrograms

**Reference:** `conducting2/src/music_generation/vocoder_losses.py`

**Tests:** `tests/test_losses.py` - verify loss computation

**Demo:** Compute loss between two audio tensors

---

## Step 9: Vocoder Model
**Objective:** Implement or load neural vocoder

**Tasks:**
- Create `src/music_generation/vocoder.py`
- Option A: Load pretrained vocoder from TensorFlow Hub
- Option B: Implement HiFi-GAN generator in Keras
- Implement `load_pretrained_vocoder()` function
- Ensure compatibility with LJSpeech mel parameters

**Reference:** `conducting2/src/music_generation/train_vocoder.py` (load_pretrained_vocoder)

**Tests:** `tests/test_vocoder.py` - verify vocoder loads and produces audio

**Demo:** Load vocoder, pass mel spectrogram, get audio waveform

---

## Step 10: Vocoder Training
**Objective:** Implement vocoder finetuning script

**Tasks:**
- Create `src/music_generation/train_vocoder.py`
- Implement dataset for vocoder (audio files → mel + audio pairs)
- Implement training loop with STFT loss
- Implement CLI with argparse
- Save checkpoints

**Reference:** `conducting2/src/music_generation/train_vocoder.py`

**Tests:** `tests/test_train_vocoder.py` - verify training step

**Demo:** Run smoke test for vocoder training

---

## Step 11: End-to-End Inference
**Objective:** Combine mel generator and vocoder for generation

**Tasks:**
- Create `src/music_generation/inference.py`
- Implement `MusicGenerationModel` class
- Implement `generate()` method (autoregressive mel generation + vocoding)
- Implement `from_checkpoints()` classmethod
- Handle tensor shape transformations

**Reference:** `conducting2/src/music_generation/inference.py`

**Tests:** `tests/test_inference.py` - verify end-to-end generation

**Demo:** Load checkpoints and generate audio

---

## Step 12: Documentation
**Objective:** Document usage and API

**Tasks:**
- Create `TRAINING.md` with training instructions
- Create `INFERENCE.md` with inference examples
- Update `README.md` with overview and quick start
- Add docstrings to all public functions

**Tests:** N/A

**Demo:** Follow documentation to train and generate

---

## Step 13: Integration Testing
**Objective:** Verify complete system works end-to-end

**Tasks:**
- Run full training on small dataset
- Verify checkpoints are saved
- Load checkpoints and run inference
- Verify audio quality
- Run all tests with `pytest`

**Tests:** All tests pass

**Demo:** Complete training → inference → audio output pipeline

---

## Checklist

- [ ] Step 1: Project Setup
- [ ] Step 2: Configuration Module
- [ ] Step 3: Audio Utilities
- [ ] Step 4: Custom Keras Layers
- [ ] Step 5: Mel Generator Model
- [ ] Step 6: Dataset Pipeline
- [ ] Step 7: Training with Teacher Forcing
- [ ] Step 8: Vocoder Losses
- [ ] Step 9: Vocoder Model
- [ ] Step 10: Vocoder Training
- [ ] Step 11: End-to-End Inference
- [ ] Step 12: Documentation
- [ ] Step 13: Integration Testing

---

## Notes

- Each step builds on previous steps
- Each step ends with working, tested code
- Follow TensorFlow patterns from `/Users/davidkeeler/code/conducting/`
- Reference PyTorch implementation from `/Users/davidkeeler/code/conducting2/`
- Maintain tensor shape convention: [batch, time, features]
