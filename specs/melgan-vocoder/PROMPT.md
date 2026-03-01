# Set Up MelGAN Vocoder for Music Generation

## Objective

Integrate MelGAN vocoder from TensorFlowTTS into conducting3 to convert mel spectrograms to audio waveforms. Keep the stack pure TensorFlow (no PyTorch dependencies).

## Key Requirements

1. Install TensorFlowTTS library
2. Update `src/music_generation/vocoder.py` to use pretrained MelGAN from Hugging Face
3. Ensure mel generation model parameters match vocoder requirements (22,050 Hz, 80 mel bins)
4. Create test script to verify vocoder works standalone
5. Document any parameter mismatches found in audio processing

## Technical Specifications

### Vocoder Model
- **Model:** MelGAN from TensorFlowTTS
- **Source:** https://huggingface.co/tensorspeech/tts-melgan-ljspeech-en
- **Sample rate:** 22,050 Hz
- **Mel bins:** 80
- **Input shape:** [batch, time, 80]
- **Output shape:** [batch, samples]

### Required Parameters Match
- Sample rate: 22,050 Hz
- n_fft: 1024
- hop_length: 256
- win_length: 1024
- n_mels: 80
- Normalization: LJSpeech-compatible log-mel

## Implementation Tasks

### 1. Add Dependency
Add to `requirements.txt`:
```
TensorFlowTTS>=1.0.0
soundfile>=0.12.0
```

### 2. Update Vocoder Implementation

Modify `src/music_generation/vocoder.py`:
- Import `TFAutoModel` from `tensorflow_tts.inference`
- Update `HiFiGANVocoder.__init__()` to load MelGAN via `TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")`
- Update `call()` method to handle MelGAN's input/output format
- Simplify `load_pretrained_vocoder()` to return MelGAN by default

### 3. Verify Audio Processing Compatibility

Check `src/music_generation/audio_utils.py` (or equivalent):
- Verify sample rate is 22,050 Hz
- Verify mel spectrogram uses 80 bins
- Verify FFT parameters match requirements
- Document any mismatches that need fixing

### 4. Create Test Script

Create `tests/test_vocoder_melgan.py`:
- Load vocoder using `load_pretrained_vocoder()`
- Generate dummy mel spectrogram [1, 100, 80]
- Run inference to generate audio
- Save output as WAV file
- Assert audio is not silent
- Print shapes and audio range for verification

## Acceptance Criteria

**Given** TensorFlowTTS is installed and MelGAN model is available  
**When** I run `python tests/test_vocoder_melgan.py`  
**Then** the test should pass and generate `test_outputs/melgan_test.wav` with audible content

**Given** the vocoder is loaded via `load_pretrained_vocoder()`  
**When** I pass a mel spectrogram of shape [batch, time, 80]  
**Then** it should return audio waveform of shape [batch, samples] without errors

**Given** the audio processing configuration  
**When** I review the mel generation parameters  
**Then** they should match MelGAN requirements (22,050 Hz, 80 bins, correct FFT params) OR mismatches should be documented

## References

- TensorFlowTTS repo: https://github.com/TensorSpeech/TensorFlowTTS
- MelGAN model: https://huggingface.co/tensorspeech/tts-melgan-ljspeech-en
- Setup guide: `/Users/davidkeeler/code/conducting3/VOCODER_SETUP.md`
