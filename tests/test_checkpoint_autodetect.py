"""Tests for checkpoint auto-detection in train_pose.py."""

import tempfile
import os
import numpy as np
import tensorflow as tf
import pytest

from src.music_generation.model import MelGenerator
from src.music_generation.train import MelGeneratorTraining
from src.music_generation.train_pose import (
    PoseConditionedTraining,
    load_checkpoint_with_autodetect,
    _normalize_var_path,
    _copy_matching_weights,
)
from src.music_generation.config import N_MELS, POSE_FEATURE_DIM, D_MODEL


@pytest.fixture
def phase1_weights_path():
    """Save a Phase 1 (audio-only) checkpoint and return (path, base_model)."""
    base = MelGenerator()
    wrapper = MelGeneratorTraining(base)
    wrapper.compile(optimizer='adam')
    x = tf.random.normal([1, 64, N_MELS])
    _ = wrapper(x, training=False)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, 'phase1.weights.h5')
    wrapper.save_weights(path)
    return path, base


@pytest.fixture
def phase2_model():
    """Build a Phase 2 (pose-conditioned) model."""
    base = MelGenerator()
    wrapper = PoseConditionedTraining(base)
    wrapper.compile(optimizer='adam')
    x = tf.random.normal([1, 64, N_MELS])
    pose = tf.random.normal([1, 64, POSE_FEATURE_DIM])
    _ = wrapper((x, x, pose), training=False)
    return wrapper


class TestNormalizeVarPath:
    """Test path normalization for weight matching."""

    def test_strips_wrapper_prefix(self):
        """mel_generator_training/mel_generator/z_proj/kernel -> z_proj/kernel"""
        class FakeVar:
            path = "mel_generator_training/mel_generator/z_proj/kernel"
        assert _normalize_var_path(FakeVar()) == "z_proj/kernel"

    def test_strips_numbered_suffix(self):
        """pose_conditioned_training/mel_generator_1/z_proj/kernel -> z_proj/kernel"""
        class FakeVar:
            path = "pose_conditioned_training/mel_generator_1/z_proj/kernel"
        assert _normalize_var_path(FakeVar()) == "z_proj/kernel"

    def test_preserves_numbered_sublayers(self):
        """conv_block_0 numbering preserved (only mel_generator suffix stripped)."""
        class FakeVar:
            path = "wrapper/mel_generator_2/conv_block_0/norm/gamma"
        assert _normalize_var_path(FakeVar()) == "conv_block_0/norm/gamma"


class TestPhase1Loading:
    """Test loading Phase 1 checkpoint into Phase 2 model."""

    def test_audio_weights_transferred(self, phase1_weights_path, phase2_model):
        """Audio-only weights from Phase 1 are copied to Phase 2."""
        path, phase1_base = phase1_weights_path
        orig_z_proj = phase1_base.z_proj.get_weights()[0].copy()

        load_checkpoint_with_autodetect(phase2_model, path)

        loaded_z_proj = phase2_model.base_model.z_proj.get_weights()[0]
        np.testing.assert_allclose(orig_z_proj, loaded_z_proj, atol=1e-6)

    def test_pose_layers_stay_initialized(self, phase1_weights_path, phase2_model):
        """Pose-specific layers keep their random init after Phase 1 load."""
        path, _ = phase1_weights_path
        pose_proj_before = phase2_model.base_model.pose_proj.get_weights()[0].copy()

        load_checkpoint_with_autodetect(phase2_model, path)

        pose_proj_after = phase2_model.base_model.pose_proj.get_weights()[0]
        np.testing.assert_allclose(pose_proj_before, pose_proj_after)

    def test_training_runs_after_load(self, phase1_weights_path, phase2_model):
        """Training step succeeds after loading Phase 1 checkpoint."""
        path, _ = phase1_weights_path
        load_checkpoint_with_autodetect(phase2_model, path)

        x = tf.random.normal([2, 64, N_MELS])
        pose = tf.random.normal([2, 64, POSE_FEATURE_DIM])
        result = phase2_model.train_step((x, x, pose))

        assert 'loss' in result
        assert not tf.math.is_nan(result['loss'])


class TestCopyMatchingWeights:
    """Test the weight copying helper."""

    def test_matching_weights_copied(self):
        """Weights with matching normalized paths and shapes are copied."""
        src = MelGenerator()
        tgt = MelGenerator()
        x = tf.random.normal([1, 64, N_MELS])
        _ = src(x, training=False)
        _ = tgt(x, training=False)

        # Weights should differ before copy
        assert not np.allclose(
            src.z_proj.get_weights()[0], tgt.z_proj.get_weights()[0]
        )

        _copy_matching_weights(src, tgt)

        np.testing.assert_allclose(
            src.z_proj.get_weights()[0], tgt.z_proj.get_weights()[0], atol=1e-6
        )
