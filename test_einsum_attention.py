"""Test einsum-based attention without windowing"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Try without setting visible devices

import tensorflow as tf
import math

# Simulate the shapes
B, H, T, window_size, head_dim = 2, 4, 10, 8, 32

q = tf.random.normal([B, H, T, head_dim])
k_padded = tf.random.normal([B, H, T + window_size - 1, head_dim])
v_padded = tf.random.normal([B, H, T + window_size - 1, head_dim])

print("Testing einsum-based windowed attention...")
try:
    # For each query position t, compute attention over k[t:t+window_size]
    # We'll use a loop to avoid creating the full windowed tensor
    scores_list = []
    for t in range(T):
        # Get window for this position
        k_window = k_padded[:, :, t:t+window_size, :]  # [B, H, window_size, head_dim]
        q_t = q[:, :, t:t+1, :]  # [B, H, 1, head_dim]
        
        # Compute attention scores
        score_t = tf.einsum('bhid,bhjd->bhij', q_t, k_window) / math.sqrt(head_dim)  # [B, H, 1, window_size]
        scores_list.append(score_t)
    
    scores = tf.concat(scores_list, axis=2)  # [B, H, T, window_size]
    print(f"scores shape: {scores.shape}")
    print("SUCCESS: Loop-based einsum worked!")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
