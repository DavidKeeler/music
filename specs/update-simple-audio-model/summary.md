# Summary: Update Simple Audio Model

## Project Overview

Upgrade the TensorFlow mel autoregressive model from a single-layer transformer to a 3-layer architecture with increasing window sizes (128, 256, 512) and learned relative positional encoding.

## Artifacts Created

### 1. rough-idea.md
Original requirements and constraints for the architecture update.

### 2. requirements.md
Q&A record documenting key design decisions:
- Per-layer independent relative position bias
- Small random initialization for bias weights
- Simple full T×T masking (no sparse optimization)
- 3 explicitly named transformer layers
- Bias application order (after scaling, before mask)
- Config updates to reflect new architecture
- Accept breaking changes to old checkpoints

### 3. research/
Research findings on implementation approaches:

**relative-positional-encoding.md:**
- Survey of RPE methods (T5, ALiBi, NoPE)
- Recommendation: T5-style learned bias per layer
- Per-layer bias shape: [num_heads, window_size]
- Total parameters: 3,584 (negligible overhead)

**tensorflow-attention-patterns.md:**
- TensorFlow 2.x best practices for causal attention
- Efficient masking with tf.linalg.band_part
- Relative bias application via tf.gather
- NaN handling in softmax
- Graph mode compatibility patterns

### 4. design.md
Comprehensive design document including:
- Detailed architecture overview with data flow diagrams
- Component specifications (LocalWindowAttention, TransformerBlock, MelGenerator)
- Relative position bias implementation details
- Masking logic (causal + window)
- Error handling strategies
- Acceptance criteria (Given-When-Then format)
- Testing strategy
- Appendices: technology choices, research summary, alternatives, parameter count

### 5. plan.md
5-step implementation plan:
1. Update config.py with new constants
2. Add relative position bias to LocalWindowAttention
3. Update MelGenerator with 3 explicit transformer layers
4. Test causality and window constraints
5. Verify training compatibility

Each step includes:
- Objective
- Implementation details with code snippets
- Files to modify
- Test requirements
- Integration notes
- Demo commands/output

## Key Design Decisions

### Architecture
- **3 transformer layers** with window sizes: 128, 256, 512
- **Learned relative position bias** per layer (not shared)
- **Strict causality** maintained throughout
- **No global attention** - purely local windowed
- **Model size stable** - only +3.5K parameters for bias

### Implementation Approach
- **Per-layer bias**: Each layer has independent [num_heads, window_size] tensor
- **Simple masking**: Full T×T masks (correctness over micro-optimization)
- **Explicit layers**: 3 named attributes (transformer_layer1/2/3) not loop
- **Small random init**: Break symmetry, allow diverse head patterns
- **Breaking changes**: Old checkpoints incompatible, retrain from scratch

### Technical Patterns
- Use `tf.gather` for efficient bias indexing
- Combine causal + window masks with `tf.maximum`
- Handle NaN in softmax with `tf.where`
- Use `tf.shape` for dynamic shapes (graph mode)
- Apply bias after scaling, before mask

## Files to Modify

1. **src/music_generation/config.py**
   - Update NUM_LAYERS: 1 → 3
   - Remove WINDOW_SIZE
   - Add WINDOW_SIZES = [128, 256, 512]

2. **src/music_generation/layers.py**
   - Add relative_position_bias weight to LocalWindowAttention
   - Implement bias application in call() method

3. **src/music_generation/model.py**
   - Replace loop-based transformer stack with 3 explicit layers
   - Update __init__ and call() methods

4. **tests/test_causality.py** (new)
   - Causality verification tests
   - Window constraint tests

## No Changes Required

- Training loop (train.py)
- Dataset pipeline (dataset.py)
- Inference API (inference.py)
- Vocoder (vocoder.py)
- Loss functions (losses.py)
- Audio utilities (audio_utils.py)

## Next Steps

### Option 1: Manual Implementation
Follow the implementation plan step-by-step:
```bash
# Step 1: Update config
# Edit src/music_generation/config.py

# Step 2: Update attention layer
# Edit src/music_generation/layers.py

# Step 3: Update model
# Edit src/music_generation/model.py

# Step 4: Test
pytest tests/test_causality.py

# Step 5: Train
python -m src.music_generation.train --epochs 1
```

### Option 2: Autonomous Implementation with Ralph
Create a PROMPT.md for Ralph to implement autonomously using the spec-driven preset.

## Success Criteria

The implementation is complete when:
1. ✓ Model builds without errors
2. ✓ Forward pass produces correct output shapes
3. ✓ Causality tests pass (future doesn't affect past)
4. ✓ Window constraints are respected
5. ✓ Training loop runs without modification
6. ✓ Autoregressive generation works
7. ✓ Checkpoints save/load successfully
8. ✓ No NaN/Inf in losses or gradients

## Estimated Effort

- **Implementation**: 2-3 hours (manual) or 30 minutes (with Ralph)
- **Testing**: 1 hour
- **Training**: Depends on dataset size and hardware
- **Total**: ~4 hours manual, ~2 hours with Ralph

## Risk Assessment

**Low Risk:**
- Well-defined requirements
- Minimal code changes
- No external dependencies
- Existing training loop unchanged

**Potential Issues:**
- Shape mismatches (mitigated by comprehensive tests)
- NaN in attention (handled with tf.where)
- Causality bugs (caught by tests)

## References

- Design document: `specs/update-simple-audio-model/design.md`
- Implementation plan: `specs/update-simple-audio-model/plan.md`
- Research: `specs/update-simple-audio-model/research/`
- Original codebase: `src/music_generation/`
