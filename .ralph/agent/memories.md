# Memories

## Patterns

### mem-1772397744-50ee
> Keras custom train_step pattern: add metrics as tf.keras.metrics.Mean in __init__, update in train_step, return dict with metric.result(), add @property metrics for auto-reset
<!-- tags: tensorflow, keras, training | created: 2026-03-01 -->

### mem-1772303418-a7d1
> Tests ported to TensorFlow: use tf.random.normal, tf.math.is_nan/is_inf + tf.reduce_any, tf.reduce_all for assertions. Dataset tests use tf.data.Dataset iteration instead of PyTorch DataLoader.
<!-- tags: testing, tensorflow | created: 2026-02-28 -->

### mem-1772303281-2eb7
> inference.py uses MusicGenerationModel wrapping mel_generator + vocoder. generate() method does autoregressive mel generation then vocoding. from_checkpoints() supports .h5 and SavedModel formats.
<!-- tags: inference, tensorflow | created: 2026-02-28 -->

### mem-1772303223-5490
> train_vocoder.py uses VocoderTraining wrapper with custom train_step() for STFT loss. Uses model.fit() with ModelCheckpoint and TensorBoard callbacks. Trims audio lengths with tf.minimum() before loss computation.
<!-- tags: vocoder, training, tensorflow | created: 2026-02-28 -->

### mem-1772303146-1bd5
> Vocoder uses TF Hub wrapper pattern: HiFiGANVocoder wraps pretrained models, handles shape conversion [B,80,T]↔[B,T,80], VocoderDataset for finetuning with random audio crops
<!-- tags: vocoder, tensorflow, tfhub | created: 2026-02-28 -->

### mem-1772303077-feb0
> Vocoder losses ported to TensorFlow: all inherit from tf.keras.layers.Layer. STFT uses tf.signal.stft with frame_length/frame_step/fft_length params. Loss functions use tf.reduce_mean for aggregation.
<!-- tags: vocoder, tensorflow, losses | created: 2026-02-28 -->

### mem-1772303019-592d
> train.py uses MelGeneratorTraining wrapper with custom train_step() for teacher forcing. Autoregressive loop with tf.random.uniform() to mix ground truth and predictions. Uses model.fit() with WarmupCosineSchedule and callbacks.
<!-- tags: training, tensorflow, teacher-forcing | created: 2026-02-28 -->

### mem-1772302955-cdfb
> TensorFlow dataset uses tf.data.Dataset.from_generator with lazy loading. Generator yields numpy arrays, converted to tensors automatically. Cache mels as .npy, stats as JSON.
<!-- tags: dataset, tensorflow, caching | created: 2026-02-28 -->

### mem-1772302885-7280
> MelGenerator ported to Keras: uses tf.keras.Model with Dense layers for projections, maintains [B,T,C] shape throughout, includes generate() for autoregressive inference
<!-- tags: model, keras, tensorflow | created: 2026-02-28 -->

### mem-1772302727-a749
> TensorFlow audio processing uses tf.signal.stft + tf.signal.linear_to_mel_weight_matrix instead of torchaudio. Tensor shapes are [time, features] vs PyTorch [features, time].
<!-- tags: audio, tensorflow, porting | created: 2026-02-28 -->

### mem-1772302622-7f31
> Project uses src/music_generation/ structure with config.py for all hyperparameters. TensorFlow port follows PyTorch config exactly with added teacher forcing params.
<!-- tags: structure, config | created: 2026-02-28 -->

## Decisions

## Fixes

### mem-1772302816-36ab
> Python 3.14 too new for TensorFlow. Need Python 3.9-3.12 for tensorflow-macos on Apple Silicon. Use pyenv or conda to manage Python versions.
<!-- tags: tensorflow, python, environment | created: 2026-02-28 -->

## Context

### mem-1772303344-1aed
> requirements.txt uses tensorflow>=2.13.0, tensorflow-hub>=0.14.0 for pretrained vocoder, soundfile/librosa for audio I/O. README.md documents training workflow, inference API, and architecture.
<!-- tags: documentation, dependencies | created: 2026-02-28 -->
