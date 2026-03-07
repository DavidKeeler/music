# Scratchpad: Vocoder Replacement Implementation

## Current Understanding

The objective is to replace the broken TensorFlowTTS dependency with a working vocoder solution. The current `vocoder.py` uses TensorFlowTTS which has broken dependencies and is unmaintained.

### Current State Analysis

1. **Existing Code**: `src/music_generation/vocoder.py` has:
   - `HiFiGANVocoder` wrapper class
   - `load_pretrained_vocoder()` using TensorFlowTTS
   - `VocoderDataset` for fine-tuning
   - Depends on broken TensorFlowTTS package

2. **Design Documents**: Complete design and plan exist in `specs/vocoder-replacement/`
   - 10-step implementation plan
   - Detailed architecture design
   - Research findings

3. **Test Coverage**: One test exists (`test_vocoder.py`) for training, but no inference tests

### Implementation Strategy

Following the plan in `specs/vocoder-replacement/plan.md`, the approach is:

1. **Quick wins first**: Mel normalization + Griffin-Lim (Steps 1-2)
2. **Main work**: Extract HiFi-GAN from TensorFlowTTS repo (Steps 3-4)
3. **Optional fallback**: Vocos wrapper (Step 5)
4. **Integration**: Update unified interface (Step 6)
5. **Training support**: Update train_vocoder.py (Step 7)
6. **Testing**: Comprehensive test coverage (Step 8)
7. **End-to-end**: Update inference.py (Step 9)
8. **Fine-tuning**: Train on MusicNet (Step 10)

### Key Technical Decisions

- **Primary path**: HiFi-GAN (TensorFlow) - extract from TensorFlowTTS repo
- **Fallback path**: Vocos (PyTorch) - optional but recommended
- **Debug path**: Griffin-Lim (librosa) - for testing only
- **Mel format**: Already log-compressed, just needs normalization
- **Sample rate**: 22,050 Hz (music-focused)
- **Mel bins**: 80
- **Hop length**: 256

### Plan Narrative

The implementation will proceed incrementally:

**Phase 1 (Steps 1-2)**: Get something working quickly
- Add mel normalization using dataset statistics
- Implement Griffin-Lim for immediate testing capability
- This unblocks mel generator testing even if neural vocoder takes time

**Phase 2 (Steps 3-4)**: Main vocoder implementation
- Clone TensorFlowTTS repo and extract HiFi-GAN code
- Remove internal dependencies, make it standalone
- Create wrapper with consistent API
- Load pretrained weights from Hugging Face

**Phase 3 (Step 5)**: Optional fallback
- Add Vocos wrapper for PyTorch-based fallback
- Provides state-of-the-art quality if HiFi-GAN has issues

**Phase 4 (Steps 6-9)**: Integration and testing
- Update unified vocoder interface with fallback chain
- Update train_vocoder.py for fine-tuning support
- Add comprehensive test coverage
- Update inference.py for end-to-end generation

**Phase 5 (Step 10)**: Fine-tuning
- Fine-tune on MusicNet dataset (requires GPU, 5-10 hours)
- Adapt pretrained speech model to music domain

### Next Actions

Will create tasks for each step and start with Step 1 (mel normalization).

## Iteration 1 Complete

**Task:** Step 1 - Add mel normalization (MelNormalizer class)

**Implementation:**
- Added `MelNormalizer` class to `audio_utils.py`
- Provides `normalize()` and `denormalize()` methods
- Includes `from_dataset()` classmethod to compute statistics from cached mels
- Uses streaming approach to compute mean/std without loading all data into memory

**Tests:**
- Created `tests/test_mel_normalizer.py` with 6 tests
- All tests pass (34/34 total including existing tests)
- Coverage includes: normalize, denormalize, roundtrip, from_dataset, error handling, batch processing

**Committed:** 471d3fe

**Next:** Step 2 - Implement Griffin-Lim vocoder (now unblocked)
