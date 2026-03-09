# Rough Idea

Fix teacher forcing implementation issues: (1) max_context=64 is too small for SEQ_LEN=512, should be configurable; (2) TensorArray loop is memory-heavy, explore tf.scan or vectorization. Need proper testing.

## Context

The teacher forcing schedule was recently implemented in `src/music_generation/train.py`. During review, two issues were identified:

1. **max_context fixed at 64**: If SEQ_LEN > 64, we're not using the full context during autoregressive generation. This could reduce model performance. Solution: max_context = SEQ_LEN or make it configurable.

2. **Memory and performance**: Looping over tf.range(seq_len) with TensorArray is memory-heavy. Could be faster with tf.scan or vectorized tricks, but okay for prototyping.

## Goal

Fix these implementation issues with proper testing to ensure:
- Full context is available during autoregressive training
- Memory usage is optimized
- Performance is acceptable
- All changes are tested
