"""Configuration constants for music generation module.

This module centralizes all configuration parameters for the music generation
system, including audio processing, model architecture, training, and inference
settings. All magic numbers should be defined here rather than scattered
throughout the codebase.
"""
from pathlib import Path

# Audio Processing Parameters
# 22.05kHz sample rate for music generation
SAMPLE_RATE = 22050
# Number of mel frequency bins
N_MELS = 80
# STFT window length in samples
FRAME_LENGTH = 1024
# STFT hop length in samples (256 samples = ~11.6ms at 22.05kHz)
FRAME_STEP = 256

# LJSpeech Mel Parameters (for pretrained vocoder compatibility)
LJSPEECH_SAMPLE_RATE = 22050
LJSPEECH_HOP_LENGTH = 256
LJSPEECH_N_FFT = 1024
LJSPEECH_N_MELS = 80
LJSPEECH_F_MIN = 0.0
LJSPEECH_F_MAX = 8000.0

# Vocoder Training Parameters
VOCODER_SEGMENT_LENGTH = 8192
VOCODER_LR = 1e-4

# Model Architecture Parameters
# Transformer hidden dimension (reduced to 128 for memory optimization)
D_MODEL = 128
# Number of attention heads
NUM_HEADS = 4
# Number of transformer layers
NUM_LAYERS = 3
# Local attention window sizes per layer
WINDOW_SIZES = [128, 256, 512]

# Training Parameters
# Batch size for training (reduced to 4 for memory optimization)
BATCH_SIZE = 4
# Sequence length in frames (~6 seconds at 22.05kHz with 256 hop)
SEQ_LEN = 512
# Initial learning rate (with warmup + cosine decay)
LEARNING_RATE = 1e-4
# Number of training epochs (reduced from 20 to 3 for faster iteration on CPU)
NUM_EPOCHS = 3

# Teacher Forcing Schedule Parameters
# Initial teacher forcing ratio (1.0 = always use ground truth)
INITIAL_TF_RATIO = 1.0
# Minimum teacher forcing ratio (floor for exponential decay)
MIN_TF_RATIO = 0.05
# Exponential decay rate (k in ε(step) = max(ε_min, ε_initial * exp(-k * step)))
TF_DECAY_K = 1e-5
# Warmup steps before decay starts (ratio stays at INITIAL_TF_RATIO)
TF_WARMUP_STEPS = 0

# Inference Parameters
# Temperature for sampling (1.0 = no scaling)
TEMPERATURE = 1.0
# Nucleus sampling threshold
TOP_P = 0.9

# Path Constants
DATA_DIR = Path.home() / "data" / "music" / "musicnet" / "train_data"
CACHE_DIR = Path.home() / "data" / "music" / "musicnet" / "cache"
CHECKPOINT_DIR = Path.home() / "models" / "conducting"
OUTPUT_DIR = Path.home() / "output" / "music_generation"

# Mel Normalization Statistics
# These are computed by the dataset on first run and cached to disk
MEL_MEAN = None
MEL_STD = None
