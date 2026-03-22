"""TensorFlow Dataset for MusicNet with caching and normalization."""
import tensorflow as tf
from pathlib import Path
import json
import logging
import numpy as np

from .audio_utils import load_audio, audio_to_mel, normalize_mel
from .config import SEQ_LEN, N_MELS, POSE_FEATURE_DIM

logger = logging.getLogger(__name__)


class MusicNetDataset:
    """Dataset for MusicNet audio files with lazy loading and caching.
    
    On first run:
    - Computes global MEL_MEAN and MEL_STD from all files
    - Caches statistics to {cache_dir}/stats.json
    - Converts all audio to mel spectrograms
    - Caches mels to {cache_dir}/{filename}.npy
    
    On subsequent runs:
    - Loads statistics from cache
    - Builds sequence indices without loading mels (lazy loading)
    - Loads mels on-demand during iteration
    
    Returns (input_mel, target_mel) pairs where input and target are the same slice.
    """
    
    def __init__(self, data_dir: Path, cache_dir: Path):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing .wav files
            cache_dir: Directory for cached mels and statistics
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        self.audio_files = sorted(self.data_dir.glob("*.wav"))
        if not self.audio_files:
            raise ValueError(f"No .wav files found in {data_dir}")
        
        # Load or compute statistics
        self.stats_path = self.cache_dir / "stats.json"
        if self.stats_path.exists():
            logger.info("Loading cached statistics")
            with open(self.stats_path, 'r') as f:
                stats = json.load(f)
            self.mel_mean = stats['mean']
            self.mel_std = stats['std']
        else:
            logger.info("Computing statistics from all files")
            self.mel_mean, self.mel_std = self._compute_statistics()
            with open(self.stats_path, 'w') as f:
                json.dump({'mean': float(self.mel_mean), 'std': float(self.mel_std)}, f)
        
        # Build file indices - skip length checking, handle during iteration
        self.file_indices = list(range(len(self.audio_files)))
    
    def _compute_statistics(self):
        """Compute global mean and std from all audio files using streaming computation."""
        total_sum = 0.0
        total_sum_sq = 0.0
        total_count = 0
        
        for audio_file in self.audio_files:
            logger.info(f"Processing {audio_file.name}")
            waveform = load_audio(audio_file)
            mel = audio_to_mel(waveform)
            
            # Accumulate statistics without storing all data
            total_sum += tf.reduce_sum(mel).numpy()
            total_sum_sq += tf.reduce_sum(tf.square(mel)).numpy()
            total_count += tf.size(mel).numpy()
        
        # Compute mean and std from accumulated values
        mean = total_sum / total_count
        variance = (total_sum_sq / total_count) - (mean ** 2)
        std = variance ** 0.5
        
        logger.info(f"Computed statistics: mean={mean:.4f}, std={std:.4f}")
        return mean, std
    
    def _get_mel_length(self, audio_file: Path) -> int:
        """Get mel spectrogram length without loading full data."""
        cache_file = self.cache_dir / f"{audio_file.stem}.npy"
        
        if cache_file.exists():
            # Load only to get shape
            import numpy as np
            mel = np.load(cache_file)
            return mel.shape[0]
        else:
            # Need to compute mel to get length
            waveform = load_audio(audio_file)
            mel = audio_to_mel(waveform)
            # Cache it for later
            import numpy as np
            np.save(cache_file, mel.numpy())
            return mel.shape[0]
    
    def _load_or_compute_mel(self, audio_file: Path):
        """Load mel from cache or compute and cache it."""
        cache_file = self.cache_dir / f"{audio_file.stem}.npy"
        
        if cache_file.exists():
            import numpy as np
            mel = tf.constant(np.load(cache_file), dtype=tf.float32)
        else:
            logger.info(f"Computing mel for {audio_file.name}")
            waveform = load_audio(audio_file)
            mel = audio_to_mel(waveform)
            import numpy as np
            np.save(cache_file, mel.numpy())
        
        # Normalize
        mel_normalized = normalize_mel(mel, self.mel_mean, self.mel_std)
        return mel_normalized
    
    def _generator(self):
        """Generator function for tf.data.Dataset."""
        for file_idx in self.file_indices:
            audio_file = self.audio_files[file_idx]
            
            # Load mel on-demand
            mel = self._load_or_compute_mel(audio_file)
            mel_length = mel.shape[0]
            
            # Skip if too short
            if mel_length < SEQ_LEN:
                continue
            
            # Generate sequences with stride on raw mel frames
            stride = SEQ_LEN // 2
            for start_frame in range(0, mel_length - SEQ_LEN + 1, stride):
                input_mel = mel[start_frame:start_frame + SEQ_LEN]
                
                yield input_mel.numpy(), input_mel.numpy()
    
    def _pose_generator(self):
        """Generator yielding (mel, mel, pose) triples for files with matching pose data."""
        for file_idx in self.file_indices:
            audio_file = self.audio_files[file_idx]
            pose_file = audio_file.parent / f"{audio_file.stem}_pose.npy"

            if not pose_file.exists():
                logger.warning(f"No pose file for {audio_file.name}, skipping")
                continue

            mel = self._load_or_compute_mel(audio_file)
            pose = np.load(pose_file).astype(np.float32)
            mel_length = mel.shape[0]

            # Align lengths
            min_len = min(mel_length, pose.shape[0])
            if min_len < SEQ_LEN:
                continue

            stride = SEQ_LEN // 2
            for start in range(0, min_len - SEQ_LEN + 1, stride):
                m = mel[start:start + SEQ_LEN]
                p = pose[start:start + SEQ_LEN]
                yield m.numpy(), m.numpy(), p

    def __len__(self) -> int:
        """Return approximate number of sequences."""
        return len(self.file_indices) * 10  # Rough estimate


def create_dataset(data_dir: str, cache_dir: str, batch_size: int, shuffle: bool = True):
    """Create tf.data.Dataset for training.
    
    Args:
        data_dir: Directory containing .wav files
        cache_dir: Directory for cached mels and statistics
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        
    Returns:
        Tuple of (tf.data.Dataset, steps_per_epoch).
        Dataset repeats infinitely; use steps_per_epoch with model.fit().
    """
    dataset_obj = MusicNetDataset(Path(data_dir), Path(cache_dir))
    
    def _make_ds():
        return tf.data.Dataset.from_generator(
            dataset_obj._generator,
            output_signature=(
                tf.TensorSpec(shape=(SEQ_LEN, N_MELS), dtype=tf.float32),
                tf.TensorSpec(shape=(SEQ_LEN, N_MELS), dtype=tf.float32)
            )
        )
    
    # Count total samples from one pass (uses cached mels)
    num_samples = _make_ds().reduce(0, lambda count, _: count + 1).numpy()
    steps_per_epoch = num_samples // batch_size
    logger.info(f"Dataset: {num_samples} samples, {steps_per_epoch} steps/epoch (batch_size={batch_size})")
    
    ds = _make_ds()
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds, steps_per_epoch


def create_pose_dataset(data_dir: str, cache_dir: str, batch_size: int, shuffle: bool = True):
    """Create tf.data.Dataset yielding (mel, mel, pose) triples.

    Only includes audio files that have a matching ``{stem}_pose.npy``.

    Returns:
        Tuple of (tf.data.Dataset, steps_per_epoch).
    """
    dataset_obj = MusicNetDataset(Path(data_dir), Path(cache_dir))

    def _make_ds():
        return tf.data.Dataset.from_generator(
            dataset_obj._pose_generator,
            output_signature=(
                tf.TensorSpec(shape=(SEQ_LEN, N_MELS), dtype=tf.float32),
                tf.TensorSpec(shape=(SEQ_LEN, N_MELS), dtype=tf.float32),
                tf.TensorSpec(shape=(SEQ_LEN, POSE_FEATURE_DIM), dtype=tf.float32),
            )
        )

    num_samples = _make_ds().reduce(0, lambda count, _: count + 1).numpy()
    if num_samples == 0:
        raise ValueError(f"No audio files with matching _pose.npy found in {data_dir}")
    steps_per_epoch = max(1, num_samples // batch_size)
    logger.info(f"Pose dataset: {num_samples} samples, {steps_per_epoch} steps/epoch (batch_size={batch_size})")

    ds = _make_ds()
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, steps_per_epoch
