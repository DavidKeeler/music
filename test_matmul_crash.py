"""Test to isolate the matmul crash issue"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

import tensorflow as tf
import math

# Force CPU execution
tf.config.set_visible_devices([], 'GPU')

# Simulate the shapes from LocalWindowAttention
B, H, T, window_size, head_dim = 2, 4, 10, 8, 32

# Create test tensors
q = tf.random.normal([B, H, T, head_dim])
k_padded = tf.random.normal([B, H, T + window_size - 1, head_dim])

# Try using tf.signal.frame
print("Testing tf.signal.frame approach on CPU...")
try:
    k_windows = tf.signal.frame(k_padded, window_size, 1, axis=2)
    print(f"k_windows shape: {k_windows.shape}")
    q_expanded = tf.expand_dims(q, axis=3)
    scores = tf.matmul(q_expanded, k_windows, transpose_b=True) / math.sqrt(head_dim)
    print(f"scores shape: {scores.shape}")
    print("SUCCESS: tf.signal.frame worked on CPU!")
except Exception as e:
    print(f"FAILED: {e}")
