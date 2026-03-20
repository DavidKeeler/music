"""Configuration constants for music generation module.

This module centralizes all configuration parameters for the music generation
system, including audio processing, model architecture, training, and inference
settings. All magic numbers should be defined here rather than scattered
throughout the codebase.
"""
from pathlib import Path

# Audio Processing Parameters
# 24kHz sample rate (matches Vocos vocoder)
SAMPLE_RATE = 24000
# Number of mel frequency bins (matches Vocos 24kHz)
N_MELS = 100
# STFT window length in samples
FRAME_LENGTH = 1024
# STFT hop length in samples (256 samples = ~10.7ms at 24kHz)
FRAME_STEP = 256

# Vocos Mel Parameters (for pretrained vocoder compatibility)
VOCOS_SAMPLE_RATE = 24000
VOCOS_HOP_LENGTH = 256
VOCOS_N_FFT = 1024
VOCOS_N_MELS = 100
VOCOS_F_MIN = 0.0
VOCOS_F_MAX = 12000.0

# Legacy LJSpeech aliases (kept for any remaining references)
LJSPEECH_SAMPLE_RATE = VOCOS_SAMPLE_RATE
LJSPEECH_HOP_LENGTH = VOCOS_HOP_LENGTH
LJSPEECH_N_FFT = VOCOS_N_FFT
LJSPEECH_N_MELS = VOCOS_N_MELS
LJSPEECH_F_MIN = VOCOS_F_MIN
LJSPEECH_F_MAX = VOCOS_F_MAX

# Vocoder Training Parameters
VOCODER_SEGMENT_LENGTH = 8192
VOCODER_LR = 1e-4

# Model Architecture Parameters
# Transformer hidden dimension (reduced to 128 for memory optimization)
D_MODEL = 256
# Number of attention heads
NUM_HEADS = 4
# Number of transformer layers
NUM_LAYERS = 3
# Local attention window sizes per layer (adjusted for TOKEN_SEQ_LEN=128)
WINDOW_SIZES = [32, 64, 128]

# Latent Conditioning (VAE)
LATENT_DIM = 64
KL_BETA = 0.005

# Dilated Convolution Configuration
# Dilation rates for causal conv layers (one CausalConvBlock per rate)
# Receptive field = 1 + 2*sum(rates) frames
# Default [1, 2, 4, 8, 16] gives 63 frames (~672ms at 24kHz with hop=256)
CONV_DILATION_RATES = [1, 2, 4, 8, 16]

# Training Parameters
# Batch size for training (reduced to 4 for memory optimization)
BATCH_SIZE = 4
# Sequence length in frames (~6 seconds at 22.05kHz with 256 hop)
SEQ_LEN = 512
# Initial learning rate (with warmup + cosine decay)
LEARNING_RATE = 1e-4
# Number of training epochs (reduced from 20 to 3 for faster iteration on CPU)
NUM_EPOCHS = 3

# Learned Tokenizer Configuration
# Compression ratio for mel tokenizer (must be power of 2)
# Controls temporal downsampling: T_tokens = T / TOKEN_COMPRESSION_RATIO
# Implemented as successive stride-2 causal convolutions
TOKEN_COMPRESSION_RATIO = 4
assert TOKEN_COMPRESSION_RATIO > 0 and (TOKEN_COMPRESSION_RATIO & (TOKEN_COMPRESSION_RATIO - 1)) == 0, \
    f"TOKEN_COMPRESSION_RATIO must be a power of 2, got {TOKEN_COMPRESSION_RATIO}"

import math
# Number of stride-2 conv layers in tokenizer/detokenizer
TOKEN_NUM_CONV_LAYERS = int(math.log2(TOKEN_COMPRESSION_RATIO))
# Token sequence length after compression
TOKEN_SEQ_LEN = SEQ_LEN // TOKEN_COMPRESSION_RATIO  # 512 // 4 = 128

# Backward-compat aliases (removed in Step 7 cleanup)
REDUCTION_FACTOR = TOKEN_COMPRESSION_RATIO
GROUPED_MEL_DIM = N_MELS * REDUCTION_FACTOR
EFFECTIVE_SEQ_LEN = TOKEN_SEQ_LEN

# Teacher Forcing Schedule Parameters
# Initial teacher forcing ratio (1.0 = always use ground truth)
INITIAL_TF_RATIO = 1.0
# Minimum teacher forcing ratio (floor for exponential decay)
MIN_TF_RATIO = 0.05
# Exponential decay rate (k in ε(step) = max(ε_min, ε_initial * exp(-k * step)))
TF_DECAY_K = 5e-7
# Warmup steps before decay starts (ratio stays at INITIAL_TF_RATIO)
TF_WARMUP_STEPS = 500000

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
