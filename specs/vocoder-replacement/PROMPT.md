# Vocoder Replacement Implementation

## Objective

Replace the broken TensorFlowTTS dependency with a working vocoder solution that enables mel-to-audio conversion for the music generation system.

## Context

The current system cannot perform vocoder training or audio generation because TensorFlowTTS has broken dependencies and is unmaintained. This blocks:
- Fine-tuning vocoder on MusicNet music dataset
- Converting generated mel spectrograms to audio
- End-to-end music generation pipeline

See `specs/vocoder-replacement/` for complete design and research.

## Key Requirements

1. **Mel normalization** - Add normalization layer using dataset statistics
2. **Griffin-Lim vocoder** - Debug/testing path (no training needed)
3. **HiFi-GAN vocoder** - Primary path (extract from TensorFlowTTS, fine-tune on music)
4. **Vocos wrapper** - Fallback path (PyTorch-based, optional)
5. **Unified interface** - Single API with automatic fallback
6. **Training support** - Enable fine-tuning on MusicNet
7. **Integration** - Update inference.py for end-to-end generation

## Acceptance Criteria

### AC1: Vocoder Inference
**Given** a mel spectrogram of shape [1, 100, 80]  
**When** passed to any vocoder backend  
**Then** output audio has shape [1, 25600]  
**And** audio values are in range [-1, 1]  
**And** audio is not silent (max absolute value > 0.01)

### AC2: Backend Fallback
**Given** HiFi-GAN fails to load  
**When** loading pretrained vocoder  
**Then** system falls back to Vocos or Griffin-Lim  
**And** user is notified via logging  
**And** audio generation still works

### AC3: End-to-End Generation
**Given** trained mel generator and vocoder  
**When** running inference with seed audio  
**Then** mel generator produces mel spectrogram  
**And** vocoder converts mel to audio  
**And** output audio is saved as WAV file  
**And** audio is listenable

### AC4: Training Support
**Given** pretrained HiFi-GAN vocoder  
**When** running train_vocoder.py on MusicNet  
**Then** training completes without errors  
**And** checkpoints are saved  
**And** mel reconstruction loss decreases

### AC5: Test Coverage
**Given** vocoder implementation  
**When** running pytest tests/test_vocoder*.py  
**Then** all tests pass  
**And** coverage includes shape validation, backend switching, and integration

## Implementation Plan

Follow `specs/vocoder-replacement/plan.md` for detailed step-by-step implementation:

1. Add mel normalization (MelNormalizer class)
2. Implement Griffin-Lim vocoder (librosa-based)
3. Extract HiFi-GAN from TensorFlowTTS repo
4. Create HiFi-GAN wrapper with pretrained loading
5. Implement Vocos wrapper (optional PyTorch fallback)
6. Update unified vocoder interface with fallback chain
7. Update train_vocoder.py for fine-tuning
8. Add comprehensive test coverage
9. Update inference.py integration
10. Fine-tune on MusicNet dataset

## Technical Constraints

- Python 3.9-3.12
- TensorFlow 2.13+ (primary)
- PyTorch optional (for Vocos fallback)
- Mel format: log-mel, 80 bins, 22,050 Hz, fmax=11025
- Must maintain compatibility with existing mel generator

## Reference Files

- Design: `specs/vocoder-replacement/design.md`
- Plan: `specs/vocoder-replacement/plan.md`
- Research: `specs/vocoder-replacement/research/`
- Current code: `src/music_generation/vocoder.py` (needs replacement)

## Success Indicators

- ✅ Can load and run vocoder inference
- ✅ Can fine-tune vocoder on music data
- ✅ End-to-end generation produces audio
- ✅ All tests pass
- ✅ Audio quality is acceptable for music

## Notes

- Start with Griffin-Lim for quick validation
- HiFi-GAN extraction is the main work item
- Vocos is optional but recommended as fallback
- Fine-tuning requires GPU (5-10 hours)
- Current mel format already uses log compression (verified)
