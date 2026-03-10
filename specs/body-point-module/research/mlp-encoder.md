# MLP Encoder Design for Pose Features

## Overview

The MLP (Multi-Layer Perceptron) encoder transforms normalized skeleton features into embeddings. For our application, it takes 85-dimensional pose features and outputs 128-dimensional embeddings.

## Architecture Patterns

### 1. Simple Two-Layer MLP (Baseline)

```
Input [85] 
  → Dense(256, activation='relu')
  → Dense(128)
  → Output [128]
```

**Properties**:
- 2 layers
- Hidden dimension: 256 (2x output)
- Output dimension: 128 (matches D_MODEL)
- Parameters: ~44K

**Pros**: Simple, fast, good baseline
**Cons**: Limited expressiveness

### 2. Three-Layer MLP (More Expressive)

```
Input [85]
  → Dense(256, activation='relu')
  → Dense(256, activation='relu')
  → Dense(128)
  → Output [128]
```

**Properties**:
- 3 layers
- Consistent hidden dimension: 256
- Parameters: ~88K

**Pros**: More capacity for complex patterns
**Cons**: Slightly slower, more parameters

### 3. MLP with Normalization (Recommended)

```
Input [85]
  → Dense(256)
  → LayerNorm
  → GELU
  → Dense(256)
  → LayerNorm
  → GELU
  → Dense(128)
  → Output [128]
```

**Properties**:
- Layer normalization after each dense layer
- GELU activation (smoother than ReLU)
- Normalizes activations for stable training

**Pros**: Better training stability, faster convergence
**Cons**: Slightly more computation

### 4. Residual MLP (Advanced)

```
Input [85]
  → Dense(256)  # Project to hidden dim
  → [
      Dense(256) → LayerNorm → GELU
      → Dense(256) → LayerNorm
      → Add(residual)
      → GELU
    ] × 2 blocks
  → Dense(128)  # Project to output dim
  → Output [128]
```

**Properties**:
- Residual connections for gradient flow
- Multiple residual blocks
- More parameters but better optimization

**Pros**: Handles deeper networks, better gradients
**Cons**: More complex, more parameters

## Design Choices

### Hidden Dimension

**Rule of thumb**: 2-4x the output dimension

For output dim = 128:
- 256: Standard (2x)
- 512: Larger capacity (4x)
- 128: Minimal (1x, bottleneck)

**Recommendation**: 256 (good balance)

### Number of Layers

| Layers | Use Case | Parameters |
|--------|----------|------------|
| 2 | Simple baseline | ~44K |
| 3 | Standard | ~88K |
| 4+ | Complex patterns | 130K+ |

**Recommendation**: 2-3 layers for pose encoding

### Activation Functions

**ReLU**: `max(0, x)`
- Standard, fast
- Can cause dead neurons

**GELU**: `x * Φ(x)` (smooth approximation)
- Smoother gradients
- Better for transformers
- Used in BERT, GPT

**LeakyReLU**: `max(0.01x, x)`
- Prevents dead neurons
- Good alternative to ReLU

**Recommendation**: GELU (matches transformer architecture style)

### Normalization

**LayerNorm**: Normalizes across features
- Stabilizes training
- Reduces internal covariate shift
- Standard in transformers

**BatchNorm**: Normalizes across batch
- Not suitable for batch_size=1 streaming
- Requires running statistics

**Recommendation**: LayerNorm (compatible with streaming)

### Dropout

**Purpose**: Regularization to prevent overfitting

```python
Dense(256) → LayerNorm → GELU → Dropout(0.1) → Dense(128)
```

**Rate**: 0.1-0.2 typical
- 0.1: Light regularization
- 0.2: Moderate regularization
- 0.5: Heavy regularization

**Recommendation**: Start without dropout, add if overfitting occurs

## Recommended Configuration

For body point module (85 → 128):

```python
def build_pose_encoder():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation('gelu'),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation('gelu'),
        tf.keras.layers.Dense(128)
    ])
```

**Properties**:
- Input: [85] (17 joints × 5 features)
- Hidden: [256, 256]
- Output: [128] (matches D_MODEL)
- Activations: GELU
- Normalization: LayerNorm
- No dropout initially

**Parameters**: ~88K (lightweight for CPU)

## Integration with Temporal Layer

The MLP encoder processes individual frames, then temporal convolution aggregates over time:

```
Frame features [85]
  ↓
MLP Encoder
  ↓
Frame embedding [128]
  ↓
Stack buffer [8, 128]
  ↓
Temporal Conv1D
  ↓
Temporal embedding [128]
```

**Alternative order** (process buffer first):

```
Frame features [85]
  ↓
Stack buffer [8, 85]
  ↓
Temporal Conv1D
  ↓
Temporal features [8, 128]
  ↓
Take last timestep [128]
  ↓
MLP Encoder
  ↓
Final embedding [128]
```

**Recommendation**: First approach (MLP then temporal)
- MLP processes spatial structure (joints)
- Temporal conv processes motion patterns
- More intuitive separation of concerns

## Weight Initialization

**Default (Glorot/Xavier)**: Good for most cases

**He initialization**: Better for ReLU
```python
Dense(256, kernel_initializer='he_normal')
```

**Recommendation**: Use default (Glorot) with GELU

## Bias Terms

**Include bias**: Standard practice
```python
Dense(256, use_bias=True)  # Default
```

**No bias with LayerNorm**: Some architectures skip bias before normalization
```python
Dense(256, use_bias=False)
LayerNormalization()
```

**Recommendation**: Use default (include bias)

## Summary

**Recommended MLP encoder**:
- Architecture: 85 → 256 → 256 → 128
- Layers: 3 (2 hidden + 1 output)
- Activation: GELU
- Normalization: LayerNorm after each hidden layer
- Dropout: None initially
- Parameters: ~88K
- Suitable for CPU inference

This configuration balances expressiveness with computational efficiency for real-time streaming.

## References

- GELU activation used in BERT and GPT models
- LayerNorm standard in transformer architectures
- 2-4x hidden dimension is common practice
- Residual connections help with deeper networks

*Content was rephrased for compliance with licensing restrictions*
