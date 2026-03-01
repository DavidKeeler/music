# Requirements Clarification

This document records questions and answers to refine the specification.

---

## Q1: Current Implementation Assessment

The TensorFlow implementation already has:
- `HiFiGANVocoder` class wrapping a pretrained model
- `VocoderTraining` class with custom train_step
- `audio_to_mel_ljspeech()` function for LJSpeech-compatible mels
- `train_vocoder.py` script with training loop

Should we:
A) Enhance the existing implementation (add better checkpoint management, inference model class, etc.)
B) Replace it entirely following the PyTorch pattern more closely
C) Keep the core but refactor specific components

**A1:** The current vocoder training DOES work - it loads a pretrained TensorFlow Hub model and finetunes it with STFT loss. The implementation is functional and follows similar patterns to the PyTorch version. We should **enhance the existing implementation (Option A)** by adding missing features rather than replacing it.

---

## Q2: Missing Features Priority

Compared to the PyTorch version, the TensorFlow implementation is missing:
1. Unified inference model class (like `MusicGenerationModel`)
2. Gradient norm logging
3. Checkpoint loading validation/testing
4. Better documentation

**A2:** Focus on these three features:
- Unified inference model class (like `MusicGenerationModel`)
- Gradient norm logging
- Checkpoint loading validation/testing

---

## Q3: Inference Model Interface

The PyTorch version has a `MusicGenerationModel` class that:
- Combines MelGenerator + Vocoder
- Has a `generate()` method for autoregressive mel generation + vocoding
- Has a `from_checkpoints()` class method to load both models

Should the TensorFlow version follow the same pattern exactly, or adapt it to be more "Keras-like" (e.g., using Keras Functional API, different method names)?

**A3:** Adapt it to be more "Keras-like" - use Keras Model patterns and conventions that feel natural in TensorFlow/Keras ecosystem.

---

## Q4: Gradient Norm Logging Implementation

For gradient norm logging, should we:
A) Add it as a custom metric in the `VocoderTraining.train_step()` method
B) Create a custom Keras callback that computes gradient norms
C) Log it directly in the training loop with TensorBoard

Which approach fits best with the existing TensorFlow implementation?

**A4:** Option A - Add it as a custom metric in the `VocoderTraining.train_step()` method. This is the most Keras-native approach and integrates cleanly with the existing custom train_step.

---

## Q5: Checkpoint Loading Strategy

The current `load_vocoder_from_checkpoint()` function supports both .h5 files and SavedModel directories. However, the training saves weights with `save_weights_only=True`. 

Should we:
A) Keep the current approach (save weights only, load with `load_weights()`)
B) Save the full model (SavedModel format) for easier loading
C) Support both formats with clear documentation on which to use when

**A5:** Option B - Save the full model in SavedModel format. This is more portable, easier to load, and the standard TensorFlow approach for model serialization.

---

## Q6: MelGenerator Integration

The inference model needs to combine MelGenerator + Vocoder. Looking at the existing codebase:
- Does `src/music_generation/model.py` contain the MelGenerator class?
- Should the inference model load the MelGenerator from a checkpoint, or accept it as a parameter?
- Should we verify compatibility between the two models (e.g., mel dimensions)?

**A6:** 
- Yes, `model.py` contains the MelGenerator class
- Accept it as a parameter (more flexible, Keras-like)
- Yes, verify compatibility between models (mel dimensions match)
- Also verify that training works end-to-end

---

## Q7: Testing Strategy

For validating the implementation, should we:
A) Add unit tests only (test individual components)
B) Add integration tests only (test end-to-end generation)
C) Add both unit and integration tests
D) Add tests plus a manual validation script for listening to generated audio

What level of testing is needed?

**A7:** Option A - Add unit tests for individual components, plus smoke tests for the training scripts to ensure they run without errors.

---

## Q8: Scope Confirmation

To summarize, the implementation should:
1. Add a Keras-style inference model class combining MelGenerator + Vocoder
2. Add gradient norm logging as a metric in `train_step()`
3. Update checkpoint saving to use SavedModel format instead of weights-only
4. Verify mel dimension compatibility between models
5. Add unit tests and smoke tests for training
6. Keep the existing TensorFlow Hub model loading as-is

Is this the complete scope, or are there other features from the PyTorch version we should include?

**A8:** Yes, this is the complete scope. The implementation follows the same pattern as the PyTorch version (load pretrained HiFi-GAN and finetune), but uses TensorFlow Hub models instead of SpeechBrain since they're different frameworks.

---

## Q9: Pretrained Vocoder Model Choice

The TensorFlow implementation has references to multiple vocoder models:
- Google SoundStream (TF Hub)
- MelGAN (TensorFlowTTS)
- Class is named `HiFiGANVocoder` but doesn't actually use HiFi-GAN

Which pretrained vocoder should we use?

**A9:** Use **MelGAN from TensorFlowTTS** (`tensorspeech/tts-melgan-ljspeech-en`). This is a TensorFlow-native pretrained vocoder that can be loaded and finetuned.

---

**Requirements clarification complete. Ready to proceed to research phase?**
