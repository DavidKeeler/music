# Research Summary

## Key Findings

### 1. Schedule Type Recommendation

**Exponential decay** is recommended as the single implementation option:

```
ε(i) = max(ε_min, ε_initial * exp(-k * i))
```

**Rationale:**
- Most widely used in practice across speech synthesis and sequence generation
- Smooth, continuous transition reduces training instability
- Simple to implement and tune
- Good balance between rapid initial decay and gradual final approach
- Proven effective in curriculum scheduling research

**Alternative considered:** Inverse sigmoid is theoretically optimal but more complex to tune. Exponential provides similar benefits with simpler parameterization.

### 2. Recommended Parameters

Based on research and common practices:

- **Initial ratio (ε_initial)**: 1.0 (pure teacher forcing)
- **Minimum ratio (ε_min)**: 0.05 (maintains some stability)
- **Decay rate (k)**: 1e-5 to 1e-4 (requires tuning based on dataset size)
- **Start step**: 0 or after warmup period (e.g., 5-10 epochs)

### 3. Implementation Approach

**Custom training loop** is recommended over TensorFlow Addons:

**Reasons:**
- Current codebase already uses custom training loop
- TensorFlow Addons is in maintenance mode
- Transformer-based model (not RNN-based seq2seq)
- More control over scheduling logic
- No additional dependencies

**Implementation strategy:**
- Add `tf_ratio` as non-trainable variable
- Implement exponential decay schedule function
- Modify training step to conditionally use ground truth vs. predictions
- Use probabilistic masking for efficiency

### 4. Integration Points

Modifications needed in `src/music_generation/train.py`:

1. Add teacher forcing ratio variable and schedule
2. Modify autoregressive training loop
3. Add conditional sampling logic
4. Track tf_ratio as metric
5. Update config with schedule parameters

### 5. Expected Benefits

- **Reduced exposure bias**: Model learns to handle its own predictions
- **Better autoregressive inference**: Less error accumulation
- **Improved generation quality**: More coherent long-term sequences
- **Smoother transitions**: Exponential decay prevents training instability

### 6. Potential Risks

- **Training instability**: If decay too rapid
- **Slower convergence**: Initially, as model adapts to mixed inputs
- **Hyperparameter sensitivity**: Decay rate requires tuning

**Mitigation:**
- Start with conservative decay rate (1e-5)
- Monitor loss curves for instability
- Keep minimum ratio > 0
- Allow warmup period before starting decay

## Research Files

1. `scheduled-sampling-overview.md` - Comprehensive background on scheduled sampling
2. `tensorflow-implementation.md` - TensorFlow/Keras implementation patterns

## Next Steps

Proceed to requirements clarification to finalize:
- Exact parameter values
- Integration approach
- Testing strategy
- Acceptance criteria
