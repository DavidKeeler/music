"""Tests for MusicNetDataset with caching and normalization."""
import pytest
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil
import numpy as np
import soundfile as sf

from src.music_generation.dataset import MusicNetDataset, create_dataset
from src.music_generation.config import SAMPLE_RATE, SEQ_LEN, N_MELS


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with test audio files."""
    temp_dir = tempfile.mkdtemp()
    data_path = Path(temp_dir) / "data"
    cache_path = Path(temp_dir) / "cache"
    data_path.mkdir()
    cache_path.mkdir()
    
    duration = 10.0
    for i in range(3):
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        freq = 440 * (i + 1)
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
        
        filepath = data_path / f"test_{i}.wav"
        sf.write(filepath, audio, SAMPLE_RATE)
    
    yield data_path, cache_path
    shutil.rmtree(temp_dir)


def test_dataset_computes_statistics_on_first_run(temp_data_dir):
    """Test that dataset computes and caches statistics on first run."""
    data_path, cache_path = temp_data_dir
    stats_path = cache_path / "stats.json"
    
    assert not stats_path.exists()
    
    dataset = MusicNetDataset(data_path, cache_path)
    
    assert stats_path.exists()


def test_create_dataset_returns_tf_dataset(temp_data_dir):
    """Test that create_dataset returns tf.data.Dataset."""
    data_path, cache_path = temp_data_dir
    
    ds = create_dataset(data_path, cache_path, batch_size=2, shuffle=False)
    
    assert isinstance(ds, tf.data.Dataset)
    
    # Get first batch
    for input_batch, target_batch in ds.take(1):
        assert input_batch.shape[0] <= 2  # batch size
        assert input_batch.shape[2] == N_MELS
        assert target_batch.shape[0] <= 2
        assert target_batch.shape[2] == N_MELS
