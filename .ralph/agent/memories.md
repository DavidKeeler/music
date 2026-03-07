# Memories

## Patterns

### mem-1772814961-0a07
> test_parallel_training.py: 3 tests verify parallel training - test_train_step_shapes (correct shapes), test_single_forward_pass (single pass not loop), test_loss_decreases (convergence). Uses SimpleMelGenerator mock to avoid dependencies.
<!-- tags: testing, parallel-training | created: 2026-03-06 -->

### mem-1772814831-fa94
> Fixed prefetch value (2) replaces AUTOTUNE for predictable memory behavior. Applied to both dataset.py and vocoder.py dataset pipelines.
<!-- tags: dataset, memory, tensorflow | created: 2026-03-06 -->

### mem-1772814706-5076
> scripts/profile_memory.py calls train() with only essential params: data_dir, cache_dir, checkpoint_dir, epochs, batch_size. No teacher forcing params needed for parallel training.
<!-- tags: profiling, training | created: 2026-03-06 -->

### mem-1772814618-c42c
> Parallel Transformer training: Single forward pass with causal masking replaces autoregressive loop. Loss compares preds[:, :-1, :] with y[:, 1:, :]. Memory scales O(model) not O(seq_len × model). ~100x improvement.
<!-- tags: training, transformer, memory | created: 2026-03-06 -->

### mem-1772782717-285b
> scripts/profile_memory.py uses tracemalloc.start(), get_traced_memory() for (current, peak), snapshot.statistics('lineno') for top allocations. format_bytes() helper for human-readable output. No external deps.
<!-- tags: memory, profiling, python | created: 2026-03-06 -->

### mem-1772782576-716d
> check_batch_size() warns if batch_size > 8 and RAM < 16GB. Suggests batch_size=2 for <8GB, batch_size=4 for 8-16GB. Uses psutil.virtual_memory().total for RAM detection. Called at start of train() before dataset loading.
<!-- tags: memory, training, batch-size | created: 2026-03-06 -->

### mem-1772780012-9f4e
> Fixed-window autoregressive: Add context_len=64 param to MelGeneratorTraining, truncate ar_input to last 64 frames in loop. Reduces O(n²) to O(n) memory. Maintains quality (Transformer has positional encoding).
<!-- tags: memory, training, tensorflow | created: 2026-03-06 -->

### mem-1772775165-b054
> Streaming statistics: Replace batch concatenation with running sum/sum_sq/count. Compute mean=sum/count, variance=(sum_sq/count)-mean², std=sqrt(variance). O(1) memory vs O(n). Numerically equivalent.
<!-- tags: memory, tensorflow, dataset | created: 2026-03-06 -->

### mem-1772775091-4132
> configure_memory() enables TF memory growth for GPUs via tf.config.experimental.set_memory_growth(). Called at start of main() before model creation. Prevents upfront allocation of all GPU memory.
<!-- tags: tensorflow, memory, training | created: 2026-03-06 -->

### mem-1772726787-86b1
> test_causality_and_windows.py: 6 tests verify causality (output at t depends only on inputs <=t via tensor_scatter_nd_update), window constraints (each transformer uses WINDOW_SIZES=[128,256,512]), and relative position bias shape [num_heads, window_size]
<!-- tags: testing, causality, transformer | created: 2026-03-05 -->

### mem-1772726647-35a3
> MelGenerator uses 3 explicit transformer layers (transformer1/2/3) with window_size params from WINDOW_SIZES=[128,256,512]. No loop - explicit sequential calls in forward pass.
<!-- tags: model, transformer, architecture | created: 2026-03-05 -->

### mem-1772726514-4d08
> LocalWindowAttention uses learned relative position bias [num_heads, window_size]. Compute rel_distances = positions[:, None] - positions[None, :], clip to [0, window_size-1], gather from bias tensor on axis=1, broadcast over batch with [None, :, :, :]
<!-- tags: attention, transformer, tensorflow | created: 2026-03-05 -->

### mem-1772726366-ad6e
> Config updated: NUM_LAYERS=3, WINDOW_SIZES=[128,256,512] replaces old WINDOW_SIZE=128. Each transformer layer uses different window size.
<!-- tags: config, architecture | created: 2026-03-05 -->

### mem-1772412522-b88c
> MusicGenerationModel test coverage: test_music_generation_model_initialization (init), test_music_generation_model_call (forward pass), test_music_generation_model_generate (autoregressive), test_checkpoint_save_and_load (from_checkpoints)
<!-- tags: testing, tensorflow, inference | created: 2026-03-02 -->

### mem-1772412464-6ef7
> test_checkpoint_save_and_load smoke test: creates minimal models, saves as SavedModel, loads via MusicGenerationModel.from_checkpoints(), verifies generate() output
<!-- tags: testing, tensorflow, checkpoints | created: 2026-03-02 -->

### mem-1772412365-7318
> MusicGenerationModel inference class: combines mel_generator + vocoder, has call() for forward pass and generate() for autoregressive generation, from_checkpoints() supports both .h5 and SavedModel formats
<!-- tags: inference, tensorflow | created: 2026-03-02 -->

### mem-1772412282-5ea2
> MelGAN vocoder already fully implemented: uses TFAutoModel.from_pretrained('tensorspeech/tts-melgan-ljspeech-en'), calls inference() method, squeezes output from [batch, samples, 1] to [batch, samples], handles shape conversion to [batch, time, 80]
<!-- tags: vocoder, melgan, tensorflow | created: 2026-03-02 -->

### mem-1772412210-da12
> SavedModel format: ModelCheckpoint uses save_weights_only=False to save full model (architecture + weights + optimizer state). Checkpoint paths must be strings for directory format. Loading functions support both .h5 and SavedModel via Path.suffix check.
<!-- tags: tensorflow, checkpoints, savedmodel | created: 2026-03-02 -->

### mem-1772412097-3f5b
> VocoderTraining already has gradient norm logging: grad_norm_tracker metric, computes L2 norm in train_step, returns in result dict, has metrics property
<!-- tags: vocoder, training, tensorflow | created: 2026-03-02 -->

### mem-1772398290-5df2
> Unit tests use minimal mock models: SimpleGenerator/SimpleVocoder classes defined inline to avoid complex dependencies. Tests focus on interface contracts, not full integration.
<!-- tags: testing, tensorflow | created: 2026-03-01 -->

### mem-1772397908-88bd
> MelGAN vocoder integration: uses TFAutoModel.from_pretrained('tensorspeech/tts-melgan-ljspeech-en'), calls inference() method, outputs [batch, samples, 1] requiring squeeze to [batch, samples]
<!-- tags: vocoder, tensorflow, melgan | created: 2026-03-01 -->

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

### mem-1772907778-0464
> tf.signal.frame crashes on Apple Silicon Metal with 'New volume mismatch' error. Use tf.gather + tf.einsum instead for windowed operations. Vectorized gather with indices = tf.range(T)[:, None] + tf.range(window_size)[None, :] works correctly.
<!-- tags: tensorflow, metal, apple-silicon | created: 2026-03-07 -->

### mem-1772302816-36ab
> Python 3.14 too new for TensorFlow. Need Python 3.9-3.12 for tensorflow-macos on Apple Silicon. Use pyenv or conda to manage Python versions.
<!-- tags: tensorflow, python, environment | created: 2026-02-28 -->

## Context

### mem-1772303344-1aed
> requirements.txt uses tensorflow>=2.13.0, tensorflow-hub>=0.14.0 for pretrained vocoder, soundfile/librosa for audio I/O. README.md documents training workflow, inference API, and architecture.
<!-- tags: documentation, dependencies | created: 2026-02-28 -->

### mem-1772983986-behavior
> CRITICAL BEHAVIOR CORRECTION: When running ralph plan or PDD workflow, NEVER make code changes or system modifications without explicit user approval. Always present the plan/proposal first and wait for confirmation before executing any changes. Planning phase is for design and documentation only - implementation comes later.
<!-- tags: behavior, planning, pdd, ralph | created: 2026-03-07 -->
