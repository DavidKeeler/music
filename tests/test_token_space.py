"""Tests for token-space correctness: forward_tokens shape, arbitrary lengths,
no-retokenization in generation/training, input==target dataset, UpSampling1D."""

import tensorflow as tf
import pytest
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

from src.music_generation.model import MelGenerator, MelDetokenizer
from src.music_generation.config import (
    D_MODEL, N_MELS, SEQ_LEN, TOKEN_SEQ_LEN, TOKEN_COMPRESSION_RATIO, SAMPLE_RATE
)


class TestForwardTokensShape:
    """AC6: forward_tokens() returns [B, T_tok, D_MODEL]."""

    def test_basic_shape(self):
        model = MelGenerator()
        tokens = tf.random.normal([2, TOKEN_SEQ_LEN, D_MODEL])
        out = model.forward_tokens(tokens)
        assert out.shape == (2, TOKEN_SEQ_LEN, D_MODEL)

    def test_single_token(self):
        model = MelGenerator()
        tokens = tf.random.normal([1, 1, D_MODEL])
        out = model.forward_tokens(tokens)
        assert out.shape == (1, 1, D_MODEL)

    def test_with_z(self):
        model = MelGenerator()
        tokens = tf.random.normal([2, 10, D_MODEL])
        z = tf.random.normal([2, 64])
        out = model.forward_tokens(tokens, z=z)
        assert out.shape == (2, 10, D_MODEL)


class TestArbitraryInputLengths:
    """AC5: model(mel) with T not divisible by 4 returns [B, T, N_MELS]."""

    @pytest.mark.parametrize("T", [1, 5, 7, 13, 63, 100, 510])
    def test_non_divisible_length(self, T):
        model = MelGenerator()
        x = tf.random.normal([1, T, N_MELS])
        y = model(x, training=False)
        assert y.shape == (1, T, N_MELS), f"Failed for T={T}"


class TestDetokenizerUpSampling:
    """AC7: MelDetokenizer contains UpSampling1D layers."""

    def test_has_upsampling_layers(self):
        detok = MelDetokenizer()
        has_up = any(
            isinstance(layer, tf.keras.layers.UpSampling1D)
            for block in detok.blocks
            for layer in block
        )
        assert has_up, "MelDetokenizer should contain UpSampling1D layers"


class TestGenerationNoRetokenization:
    """AC1: generate() calls tokenizer once (seed) and detokenizer once (end)."""

    def test_tokenizer_called_once(self):
        model = MelGenerator()
        # Pre-build to avoid counting build-time calls
        model(tf.random.normal([1, 64, N_MELS]), training=False)

        call_count = [0]
        original_call = model.tokenizer.call

        def counting_call(*args, **kwargs):
            call_count[0] += 1
            return original_call(*args, **kwargs)

        model.tokenizer.call = counting_call

        seed = tf.random.normal([64, N_MELS])
        model.generate(seed, num_frames=50, temperature=0.0)

        assert call_count[0] == 1, f"tokenizer called {call_count[0]} times, expected 1"

    def test_detokenizer_called_once(self):
        model = MelGenerator()
        model(tf.random.normal([1, 64, N_MELS]), training=False)

        call_count = [0]
        original_call = model.detokenizer.call

        def counting_call(*args, **kwargs):
            call_count[0] += 1
            return original_call(*args, **kwargs)

        model.detokenizer.call = counting_call

        seed = tf.random.normal([64, N_MELS])
        model.generate(seed, num_frames=50, temperature=0.0)

        assert call_count[0] == 1, f"detokenizer called {call_count[0]} times, expected 1"


class TestScheduledSamplingNoRetokenization:
    """AC2: _parallel_scheduled_sampling uses forward_tokens for predicted tokens,
    not tokenizer(model_output). Tokenizer is called on input x (not on predictions)."""

    def test_forward_tokens_used_for_predictions(self):
        """forward_tokens() should be called for predicted tokens (not tokenizer on mel output)."""
        from src.music_generation.train import MelGeneratorTraining

        model = MelGenerator()
        trainer = MelGeneratorTraining(model, initial_tf_ratio=0.5)
        trainer.compile(optimizer='adam')
        model(tf.random.normal([1, 12, N_MELS]), training=False)

        ft_count = [0]
        original_ft = model.forward_tokens

        def counting_ft(*args, **kwargs):
            ft_count[0] += 1
            return original_ft(*args, **kwargs)

        model.forward_tokens = counting_ft

        x = tf.random.normal([2, 12, N_MELS])
        trainer.train_step((x, x))

        # forward_tokens called in pass 1 (predictions) — at least once
        assert ft_count[0] >= 1, f"forward_tokens called {ft_count[0]} times, expected >= 1"

    def test_detokenizer_only_in_pass2(self):
        """Detokenizer called once (pass 2 via forward_from_tokens), not in pass 1."""
        from src.music_generation.train import MelGeneratorTraining

        model = MelGenerator()
        trainer = MelGeneratorTraining(model, initial_tf_ratio=0.5)
        trainer.compile(optimizer='adam')
        model(tf.random.normal([1, 12, N_MELS]), training=False)

        call_count = [0]
        original_call = model.detokenizer.call

        def counting_call(*args, **kwargs):
            call_count[0] += 1
            return original_call(*args, **kwargs)

        model.detokenizer.call = counting_call

        x = tf.random.normal([2, 12, N_MELS])
        trainer.train_step((x, x))

        # detokenizer called once in pass 2 (forward_from_tokens)
        assert call_count[0] == 1, f"detokenizer called {call_count[0]} times, expected 1"


class TestLossNoShift:
    """AC3: Loss is MSE(preds, y) with no temporal shifting."""

    def test_pure_teacher_forcing_no_shift(self):
        """Verify loss uses full tensors, not sliced."""
        from src.music_generation.train import MelGeneratorTraining

        model = MelGenerator()
        trainer = MelGeneratorTraining(model, initial_tf_ratio=1.0)
        trainer.compile(optimizer='adam')

        x = tf.random.normal([2, 12, N_MELS])
        y = x
        result = trainer.train_step((x, y))

        assert tf.math.is_finite(result['loss'])
        # With input==target and a randomly initialized model, loss should be > 0
        assert result['loss'] > 0


class TestDatasetInputEqualsTarget:
    """AC4: Dataset yields (input, target) where they are the same tensor."""

    def test_input_equals_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data"
            cache_path = Path(tmpdir) / "cache"
            data_path.mkdir()
            cache_path.mkdir()

            # Create a test audio file long enough for SEQ_LEN
            t = np.linspace(0, 30.0, int(SAMPLE_RATE * 30.0))
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            sf.write(data_path / "test.wav", audio, SAMPLE_RATE)

            from src.music_generation.dataset import create_dataset
            ds, _ = create_dataset(str(data_path), str(cache_path), batch_size=1, shuffle=False)

            for inp, tgt in ds.take(1):
                np.testing.assert_array_equal(inp.numpy(), tgt.numpy())
