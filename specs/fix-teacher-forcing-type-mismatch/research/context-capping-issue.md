# Research: Context Capping Issue (Line 211)

## The Problem

### Current Code (conducting3/)
```python
for t in range(1, seq_len):
    # Line 211: Artificially cap context at 128 frames
    context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]  # MAX_CONTEXT_FRAMES = 128
    pred = self.base_model(context, training=True)
    next_frame_pred = pred[:, -1:, :]
    
    # ... teacher forcing logic ...
    ar_input = tf.concat([ar_input, next_input], axis=1)
```

### Original Code (conducting/)
```python
for t in range(1, seq_len):
    # Passes FULL ar_input to model (no capping)
    pred = self.base_model(ar_input, training=True)[:, -1:, :]
    
    # ... teacher forcing logic ...
    ar_input = tf.concat([ar_input, next_input], axis=1)
```

---

## What Happens Step-by-Step

### Scenario: Training with seq_len=512

#### Original Behavior (conducting/)
| Iteration | ar_input size | Model sees | Model can use |
|-----------|---------------|------------|---------------|
| 1 | [batch, 1, 80] | 1 frame | 1 frame |
| 50 | [batch, 50, 80] | 50 frames | 50 frames |
| 128 | [batch, 128, 80] | 128 frames | 128 frames |
| 200 | [batch, 200, 80] | **200 frames** | **200 frames** |
| 512 | [batch, 512, 80] | **512 frames** | **512 frames** |

**Result:** Model has access to ALL previous context at every step.

#### Current Behavior (conducting3/)
| Iteration | ar_input size | Model sees | Model can use |
|-----------|---------------|------------|---------------|
| 1 | [batch, 1, 80] | 1 frame | 1 frame |
| 50 | [batch, 50, 80] | 50 frames | 50 frames |
| 128 | [batch, 128, 80] | 128 frames | 128 frames |
| 200 | [batch, 200, 80] | **128 frames** ❌ | **Only last 128** |
| 512 | [batch, 512, 80] | **128 frames** ❌ | **Only last 128** |

**Result:** Model is artificially limited to last 128 frames, even though it can handle more.

---

## Why This Is Bad

### 1. Loss of Long-Range Context

At iteration 512, the model is trying to predict frame 512 but can only see frames 384-512 (last 128).

**What it's missing:**
- Frames 1-383 (384 frames of context!)
- Musical themes from the beginning
- Long-term harmonic progressions
- Structural patterns

**Example:** If generating a 23-second sequence (512 frames × 11.6ms):
- Original: Model sees full 23 seconds of context
- Current: Model only sees last 1.5 seconds of context

### 2. Inconsistent with Model Architecture

The model has THREE transformer layers with window sizes:
- Layer 1: window_size = 128
- Layer 2: window_size = 256
- Layer 3: window_size = 512

**Layer 3 is designed to attend to 512 frames**, but we're only giving it 128!

### 3. Inconsistent Training vs Inference

**During parallel training (tf_ratio=1.0):**
```python
preds = self.base_model(x, training=True)  # x is [batch, 512, 80]
```
Model sees ALL 512 frames.

**During autoregressive training (tf_ratio<1.0):**
```python
context = ar_input[:, -128:, :]  # Only 128 frames
pred = self.base_model(context, training=True)
```
Model sees only 128 frames.

**This is a train/test mismatch!** The model learns different behaviors depending on tf_ratio.

---

## Why Was This Added?

Looking at the comment in the code:
```python
# Predict next frame - use last MAX_CONTEXT_FRAMES for efficiency
context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]
```

**Hypothesis:** Someone thought this would save memory or computation.

**Reality:**
- **Memory:** Doesn't help! `ar_input` still grows to full size inside GradientTape
- **Computation:** Minimal savings - slicing is cheap, model forward pass dominates

---

## What Should Happen

### Remove the capping (match original):
```python
for t in range(1, seq_len):
    # Pass full ar_input to model
    pred = self.base_model(ar_input, training=True)
    next_frame_pred = pred[:, -1:, :]
    
    # ... rest of logic ...
```

**Benefits:**
1. Model has full context (like original)
2. Consistent with parallel training path
3. Matches model's architectural capacity (512-frame windows)
4. No artificial limitations

**Drawbacks:**
- None! The model is designed to handle this.

---

## Memory Implications

**Common misconception:** "Capping saves memory"

**Reality:**
```python
ar_input = tf.concat([ar_input, next_input], axis=1)  # Grows to [batch, 512, 80]
```

`ar_input` grows to full size regardless of capping. The capping only affects what we PASS to the model, not what we STORE.

**Memory usage:**
- With capping: ar_input grows to 512 frames (168 MB in GradientTape)
- Without capping: ar_input grows to 512 frames (168 MB in GradientTape)

**Same memory usage!** The capping doesn't help.

---

## Computational Implications

**Model forward pass cost:**
- With capping at 128: Process 128 frames
- Without capping: Process up to 512 frames

**But remember the windowed attention:**
- Each position only attends to its local window
- Window sizes: 128, 256, 512
- Computational complexity: O(T × window_size), not O(T²)

**Cost comparison at iteration 512:**
- With capping: O(128 × 512) = 65,536 operations
- Without capping: O(512 × 512) = 262,144 operations

**4x more computation** - but this is the CORRECT behavior! The model needs this context to make good predictions.

---

## Comparison to Original

### Original Code Performance
- User confirmed: worked with longer sequences
- No capping
- Model had full context
- Generated good music

### Current Code Performance
- Artificially limited to 128 frames
- Inconsistent with model architecture
- Train/test mismatch
- Likely worse generation quality for long sequences

---

## The Fix

**Remove the capping line:**

```python
# BEFORE (current - buggy)
for t in range(1, seq_len):
    context = ar_input[:, -MAX_CONTEXT_FRAMES:, :]  # ❌ Remove this
    pred = self.base_model(context, training=True)
    next_frame_pred = pred[:, -1:, :]
    # ...

# AFTER (fixed - matches original)
for t in range(1, seq_len):
    pred = self.base_model(ar_input, training=True)  # ✓ Pass full context
    next_frame_pred = pred[:, -1:, :]
    # ...
```

**This makes the code identical to the original's behavior.**

---

## Summary

| Aspect | Original (conducting/) | Current (conducting3/) | Should Be |
|--------|----------------------|----------------------|-----------|
| **Context at iter 512** | 512 frames | 128 frames ❌ | 512 frames ✓ |
| **Matches model capacity** | Yes ✓ | No ❌ | Yes ✓ |
| **Consistent train/test** | Yes ✓ | No ❌ | Yes ✓ |
| **Memory usage** | 168 MB | 168 MB | 168 MB |
| **Computation** | Full | Reduced ❌ | Full ✓ |

**Conclusion:** The context capping is a mistake that limits model performance without providing any benefits. It should be removed to match the original behavior.
