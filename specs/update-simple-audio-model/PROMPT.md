# PROMPT: Update Simple Audio Model Architecture

## Objective

Upgrade the TensorFlow mel autoregressive model from 1 transformer layer to 3 layers with increasing window sizes (128, 256, 512) and add learned relative positional encoding.

## Key Requirements

1. Add relative position bias to `LocalWindowAttention` - learned tensor `[num_heads, window_size]` per layer
2. Update `MelGenerator` to use 3 explicitly named transformer layers (not a loop)
3. Update `config.py` constants: `NUM_LAYERS=3`, `WINDOW_SIZES=[128,256,512]`
4. Maintain strict causality and compatibility with existing training loop
5. Initialize bias with small random values (`stddev=0.01`)

## Acceptance Criteria

**Given:** Modified model with 3 transformer layers and relative position bias  
**When:** Processing mel spectrograms and training  
**Then:**
- Forward pass produces correct shapes: `[B, T, 80]` → `[B, T, 80]`
- Causality holds: modifying future positions doesn't affect past outputs
- Training completes without errors (no NaN/Inf)
- Autoregressive generation works unchanged
- Relative position bias tensors are trainable and updated during training

**Given:** Causality test with sequence length 100  
**When:** Modifying positions 50-100 to random noise  
**Then:** Outputs at positions 0-49 remain identical (diff < 1e-5)

**Given:** Training loop execution  
**When:** Running one training step  
**Then:**
- Gradients exist for all parameters including `relative_position_bias`
- No NaN or Inf in gradients
- Loss is computed successfully

## Reference

Complete specification in `specs/update-simple-audio-model/`:
- `design.md` - Full architecture and component details
- `plan.md` - 5-step implementation guide
- `research/` - Technical background on relative position encoding and TensorFlow patterns

## Implementation Notes

**Relative position bias application:**
```python
# In LocalWindowAttention.call()
positions = tf.range(T)
rel_distances = positions[:, None] - positions[None, :]
rel_distances = tf.clip_by_value(rel_distances, 0, self.window_size - 1)
bias = tf.gather(self.relative_position_bias, rel_distances, axis=1)
scores = scores + bias[None, :, :, :]  # Add before mask
```

**MelGenerator transformer stack:**
```python
# In __init__
self.transformer_layer1 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=128)
self.transformer_layer2 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=256)
self.transformer_layer3 = TransformerBlock(D_MODEL, NUM_HEADS, window_size=512)

# In call()
x = self.transformer_layer1(x, training=training)
x = self.transformer_layer2(x, training=training)
x = self.transformer_layer3(x, training=training)
```

**Files to modify:**
1. `src/music_generation/config.py`
2. `src/music_generation/layers.py`
3. `src/music_generation/model.py`
