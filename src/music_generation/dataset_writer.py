"""TFRecord serialization for paired mel/keypoint samples."""
import tensorflow as tf
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(values) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_list_feature(values) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def write_sample(writer: tf.io.TFRecordWriter, mel: np.ndarray,
                 keypoints: np.ndarray, time_signature: tuple,
                 tempo: float, fps: int, source_file: str):
    """Write a single (mel, keypoints) sample to a TFRecord writer.

    Args:
        writer: Open TFRecordWriter.
        mel: float32 array [T_mel, 80].
        keypoints: float32 array [T_kp, 17, 3].
        time_signature: (numerator, denominator) tuple.
        tempo: BPM as float.
        fps: Keypoint frame rate.
        source_file: Source filename for provenance.
    """
    feature = {
        'mel': _float_list_feature(mel.astype(np.float32).flatten()),
        'keypoints': _float_list_feature(keypoints.astype(np.float32).flatten()),
        'mel_frames': _int64_feature(mel.shape[0]),
        'kp_frames': _int64_feature(keypoints.shape[0]),
        'time_signature': _int64_list_feature([int(time_signature[0]), int(time_signature[1])]),
        'tempo': _float_feature(float(tempo)),
        'fps': _int64_feature(int(fps)),
        'source_file': _bytes_feature(source_file.encode('utf-8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


FEATURE_DESCRIPTION = {
    'mel': tf.io.VarLenFeature(tf.float32),
    'keypoints': tf.io.VarLenFeature(tf.float32),
    'mel_frames': tf.io.FixedLenFeature([], tf.int64),
    'kp_frames': tf.io.FixedLenFeature([], tf.int64),
    'time_signature': tf.io.FixedLenFeature([2], tf.int64),
    'tempo': tf.io.FixedLenFeature([], tf.float32),
    'fps': tf.io.FixedLenFeature([], tf.int64),
    'source_file': tf.io.FixedLenFeature([], tf.string),
}


def parse_example(serialized: tf.Tensor) -> dict:
    """Parse a single TFRecord example into tensors.

    Returns dict with:
        mel: [T_mel, 80]
        keypoints: [T_kp, 17, 3]
        time_signature: [2]
        tempo: scalar
        fps: scalar
        source_file: scalar string
    """
    parsed = tf.io.parse_single_example(serialized, FEATURE_DESCRIPTION)
    mel_frames = parsed['mel_frames']
    kp_frames = parsed['kp_frames']
    mel = tf.reshape(tf.sparse.to_dense(parsed['mel']), [mel_frames, 80])
    keypoints = tf.reshape(tf.sparse.to_dense(parsed['keypoints']), [kp_frames, 17, 3])
    return {
        'mel': mel,
        'keypoints': keypoints,
        'time_signature': parsed['time_signature'],
        'tempo': parsed['tempo'],
        'fps': parsed['fps'],
        'source_file': parsed['source_file'],
    }


def load_dataset(tfrecord_path: str) -> tf.data.Dataset:
    """Load a TFRecord file as a parsed tf.data.Dataset."""
    return tf.data.TFRecordDataset(tfrecord_path).map(parse_example)
