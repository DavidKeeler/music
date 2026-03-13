"""Unit tests for reduction factor implementation."""
import tensorflow as tf
import numpy as np
from pathlib import Path

from src.music_generation.config import (
    REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN, N_MELS, SEQ_LEN
)


class TestConfigValues:
    """Test reduction factor configuration values."""
    
    def test_reduction_factor_exists(self):
        """Verify REDUCTION_FACTOR is defined."""
        assert REDUCTION_FACTOR == 4
    
    def test_grouped_mel_dim(self):
        """Verify GROUPED_MEL_DIM = N_MELS * REDUCTION_FACTOR."""
        assert GROUPED_MEL_DIM == N_MELS * REDUCTION_FACTOR
        assert GROUPED_MEL_DIM == 320
    
    def test_effective_seq_len(self):
        """Verify EFFECTIVE_SEQ_LEN = SEQ_LEN // REDUCTION_FACTOR."""
        assert EFFECTIVE_SEQ_LEN == SEQ_LEN // REDUCTION_FACTOR
        assert EFFECTIVE_SEQ_LEN == 128


class TestDatasetTruncation:
    """Test dataset truncation to R-divisible length."""
    
    def test_truncation_exact_divisible(self):
        """Test truncation when length is exactly divisible by R."""
        mel = tf.random.normal([512, 80])
        truncated_length = mel.shape[0] - (mel.shape[0] % REDUCTION_FACTOR)
        assert truncated_length == 512
        mel_truncated = mel[:truncated_length]
        assert mel_truncated.shape[0] == 512
    
    def test_truncation_not_divisible(self):
        """Test truncation when length is not divisible by R."""
        mel = tf.random.normal([515, 80])
        truncated_length = mel.shape[0] - (mel.shape[0] % REDUCTION_FACTOR)
        assert truncated_length == 512
        mel_truncated = mel[:truncated_length]
        assert mel_truncated.shape[0] == 512
    
    def test_truncation_preserves_values(self):
        """Test that truncation preserves the first N frames."""
        mel = tf.random.normal([515, 80])
        truncated_length = mel.shape[0] - (mel.shape[0] % REDUCTION_FACTOR)
        mel_truncated = mel[:truncated_length]
        # First frame should be identical
        tf.debugging.assert_near(mel[0], mel_truncated[0])


class TestDatasetReshaping:
    """Test dataset reshaping to grouped frames."""
    
    def test_reshape_basic(self):
        """Test basic reshape from [T, 80] to [T/R, R*80]."""
        mel = tf.random.normal([512, 80])
        mel_grouped = tf.reshape(mel, [-1, GROUPED_MEL_DIM])
        assert mel_grouped.shape == (128, 320)
    
    def test_reshape_preserves_values(self):
        """Test that reshape preserves all values."""
        mel = tf.random.normal([512, 80])
        mel_grouped = tf.reshape(mel, [-1, GROUPED_MEL_DIM])
        # Flatten both and compare
        mel_flat = tf.reshape(mel, [-1])
        grouped_flat = tf.reshape(mel_grouped, [-1])
        tf.debugging.assert_near(mel_flat, grouped_flat)
    
    def test_reshape_frame_grouping(self):
        """Test that consecutive frames are grouped correctly."""
        # Create mel with identifiable pattern
        mel = tf.range(512 * 80, dtype=tf.float32)
        mel = tf.reshape(mel, [512, 80])
        mel_grouped = tf.reshape(mel, [-1, GROUPED_MEL_DIM])
        
        # First grouped frame should contain first 4 frames
        first_grouped = mel_grouped[0]
        first_four_frames = tf.reshape(mel[:4], [-1])
        tf.debugging.assert_near(first_grouped, first_four_frames)


class TestDatasetSequencePairs:
    """Test input/target sequence pair generation."""
    
    def test_sequence_offset(self):
        """Test that target is input shifted by 1 grouped frame."""
        mel_grouped = tf.random.normal([200, 320])
        
        # Simulate sequence extraction
        start = 0
        input_seq = mel_grouped[start:start + EFFECTIVE_SEQ_LEN]
        target_seq = mel_grouped[start + 1:start + EFFECTIVE_SEQ_LEN + 1]
        
        assert input_seq.shape == (128, 320)
        assert target_seq.shape == (128, 320)
        
        # Target[0] should equal Input[1]
        tf.debugging.assert_near(target_seq[0], input_seq[1])
    
    def test_sequence_length(self):
        """Test that sequences have correct length."""
        mel_grouped = tf.random.normal([200, 320])
        
        input_seq = mel_grouped[0:EFFECTIVE_SEQ_LEN]
        target_seq = mel_grouped[1:EFFECTIVE_SEQ_LEN + 1]
        
        assert input_seq.shape[0] == EFFECTIVE_SEQ_LEN
        assert target_seq.shape[0] == EFFECTIVE_SEQ_LEN


class TestModelProjections:
    """Test model input/output projections handle grouped frames."""
    
    def test_model_input_shape(self):
        """Test model accepts grouped frame input [B, T/R, R*80]."""
        from src.music_generation.model import MelGenerator
        
        model = MelGenerator()
        dummy_input = tf.random.normal([2, EFFECTIVE_SEQ_LEN, GROUPED_MEL_DIM])
        output = model(dummy_input, training=False)
        
        # Should process without errors
        assert output.shape[0] == 2
        assert output.shape[1] == EFFECTIVE_SEQ_LEN
        assert output.shape[2] == GROUPED_MEL_DIM
    
    def test_model_output_shape(self):
        """Test model outputs grouped frames [B, T/R, R*80]."""
        from src.music_generation.model import MelGenerator
        
        model = MelGenerator()
        dummy_input = tf.random.normal([4, EFFECTIVE_SEQ_LEN, GROUPED_MEL_DIM])
        output = model(dummy_input, training=False)
        
        assert output.shape == (4, EFFECTIVE_SEQ_LEN, GROUPED_MEL_DIM)


class TestLossComputation:
    """Test training loss computation with grouped frames."""
    
    def test_loss_reshaping(self):
        """Test loss computation reshapes to per-frame format."""
        # Simulate predictions and targets
        preds = tf.random.normal([2, EFFECTIVE_SEQ_LEN - 1, GROUPED_MEL_DIM])
        targets = tf.random.normal([2, EFFECTIVE_SEQ_LEN - 1, GROUPED_MEL_DIM])
        
        # Reshape to [B, T/R-1, R, 80]
        batch_size = tf.shape(preds)[0]
        seq_len = tf.shape(preds)[1]
        preds_frames = tf.reshape(preds, [batch_size, seq_len, REDUCTION_FACTOR, N_MELS])
        target_frames = tf.reshape(targets, [batch_size, seq_len, REDUCTION_FACTOR, N_MELS])
        
        assert preds_frames.shape[2] == REDUCTION_FACTOR
        assert preds_frames.shape[3] == N_MELS
        assert target_frames.shape[2] == REDUCTION_FACTOR
        assert target_frames.shape[3] == N_MELS
    
    def test_loss_scalar_output(self):
        """Test loss computation produces scalar value."""
        # Simulate predictions and targets
        preds = tf.random.normal([2, EFFECTIVE_SEQ_LEN - 1, GROUPED_MEL_DIM])
        targets = tf.random.normal([2, EFFECTIVE_SEQ_LEN - 1, GROUPED_MEL_DIM])
        
        # Reshape and compute loss
        batch_size = tf.shape(preds)[0]
        seq_len = tf.shape(preds)[1]
        preds_frames = tf.reshape(preds, [batch_size, seq_len, REDUCTION_FACTOR, N_MELS])
        target_frames = tf.reshape(targets, [batch_size, seq_len, REDUCTION_FACTOR, N_MELS])
        
        loss = tf.reduce_mean(tf.square(preds_frames - target_frames))
        
        # Loss should be scalar
        assert loss.shape == ()
        assert tf.math.is_finite(loss)


class TestGenerationWithReduction:
    """Test generation method handles grouped frames correctly."""
    
    def test_seed_truncation(self):
        """Test seed is truncated to R-divisible length."""
        # Seed with non-divisible length (101 % 4 = 1)
        seed = tf.random.normal([101, N_MELS])
        
        # Truncate
        seed_len = int(seed.shape[0])
        truncated_len = seed_len - (seed_len % REDUCTION_FACTOR)
        seed_truncated = seed[:truncated_len]
        
        assert seed_truncated.shape[0] == 100  # 101 - (101 % 4) = 100
        assert int(seed_truncated.shape[0]) % REDUCTION_FACTOR == 0
    
    def test_seed_reshaping(self):
        """Test seed is reshaped to grouped format."""
        seed = tf.random.normal([96, N_MELS])
        
        # Reshape to grouped
        seed_grouped = tf.reshape(seed, [-1, GROUPED_MEL_DIM])
        
        assert seed_grouped.shape == (24, GROUPED_MEL_DIM)
    
    def test_generation_steps(self):
        """Test correct number of generation steps for target frames."""
        num_frames = 500
        num_steps = (num_frames + REDUCTION_FACTOR - 1) // REDUCTION_FACTOR
        
        assert num_steps == 125  # ceil(500 / 4)
    
    def test_output_reshaping(self):
        """Test generated grouped frames are reshaped back to individual frames."""
        # Simulate generated grouped frames
        generated_grouped = tf.random.normal([125, GROUPED_MEL_DIM])
        
        # Reshape to individual frames
        generated = tf.reshape(generated_grouped, [-1, N_MELS])
        
        assert generated.shape == (500, N_MELS)
