# Rough Idea: Update Simple Audio Model

## Overview
Upgrade the current TensorFlow mel autoregressive model architecture.

## Goal
Upgrade the current architecture to:
* 3 transformer layers
* Window sizes: 128, 256, 512 (increasing with depth)
* Add relative positional encoding (causal)
* Keep everything strictly causal
* No global attention
* No non-causal pooling
* Maintain compatibility with current training loop

## Requirements

### 1️⃣ Modify Transformer Stack
In MelGenerator:
* Replace current NUM_LAYERS usage.
* Explicitly define 3 TransformerBlocks with different window sizes.

Required structure:
- Layer 1 (shallow): window_size = 128
- Layer 2: window_size = 256
- Layer 3 (deep): window_size = 512

Order must be: conv → conv → 128 → 256 → 512 → conv_head → output

### 2️⃣ Add Relative Positional Bias (Causal)
Implement relative positional bias inside LocalWindowAttention.

Requirements:
* Use learned relative position bias per head.
* Only support distances from 0 to window_size - 1.
* Causal only (no future positions).
* Shape: [num_heads, window_size]
* For each query position i and key position j:
    * distance = i - j
    * If 0 ≤ distance < window_size: add bias[head, distance]
    * Else masked

Implementation guidance:
* Precompute relative distance matrix for current T.
* Clip distances to window_size - 1.
* Apply mask before softmax.
* Add bias to attention scores before masking.

DO NOT:
* Use absolute positional embeddings.
* Use sinusoidal encoding.
* Use rotary embeddings.

Only learned relative bias.

### 3️⃣ Optimize Attention Mask
Current implementation builds full T×T mask.

Replace with:
* Use banded masking logic based on window_size.
* Avoid allocating large dense masks if possible.
* Keep strictly causal.

But correctness > micro-optimization.

### 4️⃣ Maintain Autoregressive Generation
Do not change:
* generate() method API
* Sliding SEQ_LEN context logic
* Teacher forcing training loop

Model must still:
* Accept full sequences during training
* Work frame-by-frame during inference

### 5️⃣ Keep Model Size Stable
DO NOT:
* Increase D_MODEL
* Increase NUM_HEADS
* Add extra layers
* Change optimizer or training loop

Only modify attention + transformer stack definition.

### 6️⃣ Code Quality Constraints
* Must remain fully compatible with TensorFlow 2.x
* No external libraries
* No PyTorch-style operations
* Must run in graph mode

## Deliverables
Provide:
1. Updated LocalWindowAttention with relative bias
2. Updated TransformerBlock if needed
3. Updated MelGenerator transformer stack definition
4. Clear comments explaining causality handling
