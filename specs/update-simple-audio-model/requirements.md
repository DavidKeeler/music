# Requirements Clarification

This document records questions and answers to refine the requirements for updating the simple audio model.

---

## Q1: Relative Positional Bias Scope

The current `LocalWindowAttention` class is initialized with `window_size` as a parameter. When implementing relative positional bias with shape `[num_heads, window_size]`, should each of the three transformer layers (with window sizes 128, 256, 512) have its own independent learned bias parameters, or should there be a shared bias mechanism across layers?

In other words:
- **Option A**: Each layer has its own bias tensor sized to its specific window (Layer 1: [4, 128], Layer 2: [4, 256], Layer 3: [4, 512])
- **Option B**: Use a shared bias mechanism with some form of interpolation or truncation for different window sizes

**A1**: Option A - Each layer should have its own independent learned bias tensor. This is the correct approach because:
- Different window sizes require different bias ranges (can't share a 128-length bias for a 512-window)
- Only 3 layers total, so parameter overhead is minimal (3,584 total parameters)
- Allows each layer to learn positional patterns appropriate for its receptive field
- Follows research best practices (T5 shares across layers for efficiency with many layers, but we have only 3 with different windows)

---

## Q2: Relative Position Bias Initialization

For the learned relative position bias tensors `[num_heads, window_size]` in each layer, what initialization strategy should be used?

Options:
- **Option A**: Zero initialization (bias starts at 0, network learns from scratch)
- **Option B**: Small random initialization (e.g., normal distribution with small std)
- **Option C**: Specific initialization pattern (e.g., negative values to create initial recency bias)

**A2**: Option B - Small random initialization. This helps break symmetry and allows different heads to learn diverse patterns from the start.

---

## Q3: Mask Optimization Strategy

The current implementation builds full T×T masks for both causal and window constraints. Should we:

- **Option A**: Keep the current approach (full T×T masks, simple and correct)
- **Option B**: Implement banded/sparse masking to avoid allocating large dense masks
- **Option C**: Precompute masks for common sequence lengths and cache them

**A3**: Option A - Keep the current approach with full T×T masks. Correctness over micro-optimization. For our sequence lengths (T ≤ 512), memory overhead is acceptable and implementation remains simple and maintainable.

---

## Q4: Transformer Stack Definition

The requirement specifies explicitly defining 3 TransformerBlocks with different window sizes rather than using a loop. Should the MelGenerator class:

- **Option A**: Create 3 separate named attributes (self.transformer_layer1, self.transformer_layer2, self.transformer_layer3)
- **Option B**: Use a list comprehension but with explicit window sizes (self.transformer_blocks = [TransformerBlock(..., window_size=ws) for ws in [128, 256, 512]])
- **Option C**: Define them inline without storing in a list (call them sequentially in the forward pass)

**A4**: Option A - Create 3 separate named attributes. This makes the architecture explicit and clear, with each layer's window size visible in the initialization code.

---

## Q5: Relative Position Bias Application Order

When computing attention scores, the relative position bias should be added before masking and softmax. Should the order be:

- **Option A**: scores = (Q·K^T / sqrt(d)) + bias, then add mask, then softmax
- **Option B**: scores = Q·K^T / sqrt(d), add mask, add bias, then softmax
- **Option C**: Different order (please specify)

**A5**: Option A - Add bias immediately after scaled dot product, then apply mask, then softmax. This is the standard order used in T5 and other relative position bias implementations.

---

## Q6: Config File Updates

The current config.py has `NUM_LAYERS = 1` and `WINDOW_SIZE = 128`. Since we're hardcoding 3 layers with specific window sizes, should we:

- **Option A**: Remove NUM_LAYERS and WINDOW_SIZE from config.py (no longer used)
- **Option B**: Keep them but add a comment that they're deprecated/unused
- **Option C**: Update them to reflect the new architecture (e.g., NUM_LAYERS = 3, WINDOW_SIZES = [128, 256, 512])

**A6**: Option C - Update config.py to reflect the new architecture. This keeps the config as the single source of truth for model hyperparameters and makes the architecture visible without reading the model code.

---

## Q7: Backward Compatibility

The current model has checkpoints trained with 1 layer and window_size=128. After this update, those checkpoints will be incompatible. Should we:

- **Option A**: Accept breaking change (old checkpoints won't load, start training from scratch)
- **Option B**: Add checkpoint migration logic to convert old format to new format
- **Option C**: Keep old model code and create a new model class (e.g., MelGeneratorV2)

**A7**: Option A - Accept breaking change. Old checkpoints are not needed, start fresh with the new architecture.

---

## Requirements Clarification Complete

All key decisions have been made. Ready to proceed to design phase.
