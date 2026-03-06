# Parallel Training Fix for Transformer

## Problem Statement

The current training loop has a critical memory issue: it runs SEQ_LEN forward passes inside a single GradientTape, causing memory to scale as O(SEQ_LEN × model_activation_memory). For SEQ_LEN=512, this is effectively running 512 transformers simultaneously in the computation graph.

## Current Broken Implementation

```python
def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
        ar_input = x[:, :1, :]
        preds = []
        
        for t in range(1, seq_len):  # 512 iterations!
            pred = self.base_model(ar_input, training=True)[:, -1:, :]
            preds.append(pred)
            # ... teacher forcing logic
            ar_input = tf.concat([ar_input, next_input], axis=1)
        
        pred_seq = tf.concat(preds, axis=1)
        loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
```

**Memory usage:** O(SEQ_LEN × model_memory) ≈ 100x too much

## Root Cause

This is accidentally implementing RNN-style backpropagation through time with a Transformer inside each step. Transformers should be trained fully parallel using the causal mask in attention to prevent future information leakage.

## Correct Implementation

```python
def train_step(self, data):
    x, y = data
    
    with tf.GradientTape() as tape:
        preds = self.base_model(x, training=True)  # Single forward pass
        loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
    
    grads = tape.gradient(loss, self.base_model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
    
    return {"loss": loss}
```

**Memory usage:** O(model_memory) - normal Transformer training

## Why This Works

The causal mask in the attention mechanism already ensures that:
- Output at position t only depends on inputs at positions ≤ t
- No future information leakage
- Full sequence can be processed in parallel during training

Teacher forcing is implicit: we always feed ground truth inputs (x) and predict next frames (y).

## Secondary Issues

### 1. Attention Matrix Memory
Still builds [B, H, T, T] attention matrices even with windowing.

For SEQ_LEN=1024, HEADS=8: memory = B × 8 × 1024²

**Fix:** Implement proper local attention (future optimization, not critical)

### 2. Dataset Prefetching
`prefetch(tf.data.AUTOTUNE)` can over-allocate.

**Fix:** Use `prefetch(2)` for bounded memory

### 3. Model Size
If still hitting memory issues after fixing training loop:
- Reduce D_MODEL
- Reduce NUM_HEADS  
- Reduce SEQ_LEN
- Reduce BATCH_SIZE

## Expected Impact

**Memory reduction:** ~100x (from O(SEQ_LEN × model) to O(model))

**Training speed:** Significantly faster (1 forward pass vs SEQ_LEN forward passes)

**Model quality:** Unchanged (this is the correct training method)

## Components to Remove

1. Teacher forcing ratio (`tf_ratio`, `initial_tf_ratio`, `decay_k`, `min_tf_ratio`)
2. Autoregressive input loop (`ar_input`, concatenation)
3. Context length truncation (`context_len`)
4. Teacher forcing random sampling logic

## Components to Keep

1. Model architecture (causal attention already correct)
2. Loss function (MAE between predictions and targets)
3. Optimizer and learning rate schedule
4. Checkpointing and callbacks

## Testing Requirements

1. Verify training completes without OOM on MacBook Air
2. Verify loss decreases over epochs
3. Verify generated audio quality is acceptable
4. Profile memory usage (should be ~100x lower)
5. Compare training speed (should be much faster)
6. Verify model convergence is equivalent or better

## Desired Outcome

A correctly implemented parallel Transformer training loop that:
- Uses O(model) memory instead of O(SEQ_LEN × model)
- Trains significantly faster
- Maintains or improves model quality
- Works on resource-constrained hardware
