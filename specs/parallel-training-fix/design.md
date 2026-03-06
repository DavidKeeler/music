# Parallel Training Fix Design

## Overview

Replace the incorrect autoregressive training loop with proper parallel Transformer training. The current implementation runs SEQ_LEN forward passes per batch inside a single GradientTape, causing ~100x memory overhead. The fix uses a single forward pass with causal masking, which is the standard training method for autoregressive Transformers.

## Detailed Requirements

### Functional Requirements
1. Training must use single forward pass per batch (parallel processing)
2. Causal masking must prevent future information leakage
3. Loss must compare predictions at t with targets at t+1
4. Model quality must be maintained or improved
5. Training must complete on MacBook Air without OOM
6. Generated audio quality must be acceptable

### Non-Functional Requirements
1. Memory usage must be O(model) not O(SEQ_LEN × model)
2. Training speed must be significantly faster
3. Code must be simpler (remove teacher forcing complexity)
4. Implementation must follow standard Transformer training practices

### Constraints
1. Cannot change model architecture (causal attention already correct)
2. Must maintain checkpoint compatibility (model weights unchanged)
3. Must work with existing dataset pipeline
4. Must use TensorFlow 2.13+ APIs

## Architecture Overview

```mermaid
graph TD
    A[Input Sequence x] --> B[Single Forward Pass]
    B --> C[Predictions preds]
    C --> D[Shift and Compare]
    D --> E[Loss: preds[:, :-1] vs y[:, 1:]]
    E --> F[Backprop]
    
    G[Causal Mask] --> B
    G --> H[Prevents Future Leakage]
```

**Key insight:** Causal attention mask handles autoregression during training. No explicit loop needed.

## Components and Interfaces

### 1. Simplified Training Wrapper
**Location:** `src/music_generation/train.py` - `MelGeneratorTraining`

**Current (Broken):**
```python
class MelGeneratorTraining(tf.keras.Model):
    def __init__(self, base_model, initial_tf_ratio, decay_k, min_ratio, context_len):
        # Complex teacher forcing setup
        
    def train_step(self, data):
        # 512 forward passes in loop
        for t in range(1, seq_len):
            pred = self.base_model(ar_input, training=True)
            # ... teacher forcing logic
```

**New (Correct):**
```python
class MelGeneratorTraining(tf.keras.Model):
    """Simplified training wrapper with parallel processing."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            # Single forward pass - causal mask prevents future leakage
            preds = self.base_model(x, training=True)
            
            # Compare predictions at t with targets at t+1
            # preds[:, :-1] = predictions for positions 0 to T-2
            # y[:, 1:] = targets for positions 1 to T-1
            loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
        
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {"loss": loss}
```

**Changes:**
- Remove `initial_tf_ratio`, `decay_k`, `min_ratio`, `context_len` parameters
- Remove `compute_tf_ratio()` method
- Replace loop with single forward pass
- Simplify loss computation
- Remove teacher forcing logic

### 2. Update Training Function
**Location:** `src/music_generation/train.py` - `train()`

**Remove:**
```python
# Delete these parameters from function signature and calls
initial_tf_ratio=INITIAL_TF_RATIO
tf_decay_k=TF_DECAY_K
min_tf_ratio=MIN_TF_RATIO
```

**Update model creation:**
```python
# Old:
model = MelGeneratorTraining(
    base_model, 
    initial_tf_ratio=INITIAL_TF_RATIO,
    decay_k=TF_DECAY_K,
    min_ratio=MIN_TF_RATIO,
    context_len=64
)

# New:
model = MelGeneratorTraining(base_model)
```

### 3. Update Config
**Location:** `src/music_generation/config.py`

**Remove (no longer needed):**
```python
# Teacher forcing parameters
INITIAL_TF_RATIO = 1.0
TF_DECAY_K = 1e-5
MIN_TF_RATIO = 0.05
```

**Keep:**
- All model architecture parameters (D_MODEL, NUM_HEADS, etc.)
- Training parameters (BATCH_SIZE, LEARNING_RATE, etc.)
- Dataset parameters (SEQ_LEN, N_MELS, etc.)

### 4. Fix Dataset Prefetching
**Location:** `src/music_generation/dataset.py` - `create_dataset()`

**Change:**
```python
# Old:
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# New:
dataset = dataset.prefetch(2)  # Bounded prefetch to prevent over-allocation
```

### 5. Update Memory Profiling Script
**Location:** `scripts/profile_memory.py`

**Remove teacher forcing parameters:**
```python
# Delete these arguments:
parser.add_argument('--initial_tf_ratio', ...)
parser.add_argument('--tf_decay_k', ...)
parser.add_argument('--min_tf_ratio', ...)

# Remove from train() call:
train(
    data_dir=args.data_dir,
    cache_dir=args.cache_dir,
    checkpoint_dir=args.checkpoint_dir,
    epochs=args.epochs,
    batch_size=args.batch_size
    # Remove: initial_tf_ratio, tf_decay_k, min_tf_ratio
)
```

## Data Models

No changes to data models. Input/output shapes remain:
- Input: [batch, seq_len, n_mels]
- Output: [batch, seq_len, n_mels]
- Loss computed on shifted sequences

## Error Handling

### Shape Mismatch Detection
```python
def train_step(self, data):
    x, y = data
    
    # Verify shapes
    tf.debugging.assert_equal(
        tf.shape(x), tf.shape(y),
        message="Input and target shapes must match"
    )
    
    with tf.GradientTape() as tape:
        preds = self.base_model(x, training=True)
        
        # Verify prediction shape
        tf.debugging.assert_equal(
            tf.shape(preds), tf.shape(x),
            message="Prediction shape must match input shape"
        )
        
        loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
    
    # ... rest of training
```

### NaN/Inf Detection
```python
def train_step(self, data):
    # ... forward pass
    
    if tf.math.is_nan(loss) or tf.math.is_inf(loss):
        tf.print("⚠️  WARNING: NaN/Inf loss detected!")
        return {"loss": loss}
    
    # ... gradient update
```

## Acceptance Criteria

**Given** the training loop is fixed  
**When** training on MacBook Air with batch_size=4  
**Then** training completes without OOM errors

**Given** parallel training is implemented  
**When** profiling memory usage  
**Then** peak memory is ~100x lower than before

**Given** single forward pass per batch  
**When** measuring training speed  
**Then** throughput (samples/sec) is significantly higher

**Given** causal masking prevents future leakage  
**When** training for 50 epochs  
**Then** loss decreases steadily and model converges

**Given** model is trained with parallel method  
**When** generating audio samples  
**Then** quality is equivalent or better than before

**Given** teacher forcing is removed  
**When** comparing loss curves  
**Then** convergence is similar or faster

## Testing Strategy

### Unit Tests
1. Test train_step with small batch (verify shapes, no errors)
2. Test loss computation (verify correct shifting)
3. Test gradient flow (verify gradients are computed)
4. Test NaN/Inf detection

### Integration Tests
1. Train for 5 epochs on small dataset (10 files)
2. Verify loss decreases
3. Verify checkpoints save/load correctly
4. Generate audio sample and verify quality

### Memory Tests
1. Profile memory before and after fix
2. Verify peak memory reduction (~100x)
3. Verify no memory growth over epochs
4. Test with different batch sizes (2, 4, 8)

### Performance Tests
1. Measure training time per epoch (before/after)
2. Measure throughput (samples/sec)
3. Verify speedup is significant (10-50x expected)

### Quality Tests
1. Train for 50 epochs
2. Generate audio samples at epochs 10, 25, 50
3. Listen to samples and assess quality
4. Compare to previous implementation (if available)

## Appendices

### Why Parallel Training Works

**Causal Attention Mask:**
```
Position:  0  1  2  3
    0:     ✓  ✗  ✗  ✗
    1:     ✓  ✓  ✗  ✗
    2:     ✓  ✓  ✓  ✗
    3:     ✓  ✓  ✓  ✓
```

At position t, attention can only see positions ≤ t. This is enforced by the mask in `TransformerBlock`.

**Training:**
- Input: [0, 1, 2, 3]
- Predictions: [pred_0, pred_1, pred_2, pred_3]
- Targets: [1, 2, 3, 4]
- Loss: |pred_0 - 1| + |pred_1 - 2| + |pred_2 - 3| + |pred_3 - 4|

Each prediction is computed with only past context (enforced by mask).

**Inference:**
- Start with seed [0]
- Predict next: pred_1
- Append: [0, pred_1]
- Predict next: pred_2
- Append: [0, pred_1, pred_2]
- Continue...

### Teacher Forcing vs Parallel Training

**Teacher Forcing (what we had):**
- Explicitly feed ground truth at each step
- Requires loop with conditional logic
- Memory: O(SEQ_LEN × model)
- Used in RNNs where hidden state is sequential

**Parallel Training (what we need):**
- Feed full ground truth sequence
- Single forward pass
- Memory: O(model)
- Standard for Transformers (causal mask handles autoregression)

### Memory Calculation

**Before (broken):**
```
Memory = SEQ_LEN × (model_params + activations)
       = 512 × (50MB + 100MB)
       = 76.8 GB per batch
```

**After (correct):**
```
Memory = model_params + activations
       = 50MB + 100MB
       = 150 MB per batch
```

**Reduction:** ~500x

### Alternative Approaches Considered

**1. Keep Teacher Forcing with Truncated BPTT**
- Split sequence into chunks
- Still requires loop
- More complex
- **Rejected:** Parallel training is simpler and correct

**2. Gradient Checkpointing**
- Recompute activations during backward pass
- Trades compute for memory
- **Rejected:** Fix root cause instead

**3. Reduce Sequence Length**
- Train on shorter sequences
- **Rejected:** Doesn't fix the bug, just masks it

### Migration Notes

**Breaking Changes:**
- Teacher forcing parameters removed from config
- `MelGeneratorTraining` constructor signature changed
- Training behavior changed (but model quality should improve)

**Backward Compatibility:**
- Model architecture unchanged
- Checkpoint format unchanged (can load old weights)
- Inference code unchanged
- Dataset pipeline unchanged (except prefetch)

**Recommended Migration:**
1. Delete old checkpoints (trained with broken method)
2. Retrain from scratch with correct method
3. Compare quality to verify improvement

### Future Optimizations

**1. Efficient Local Attention**
Current: O(T²) attention matrix
Future: O(T × window) with proper local attention

**2. Mixed Precision Training**
Use float16 for 50% memory reduction

**3. Gradient Accumulation**
Simulate larger batch sizes

**4. Flash Attention**
Optimized attention kernel (if available in TensorFlow)

These are secondary optimizations. Fix the training loop first.
