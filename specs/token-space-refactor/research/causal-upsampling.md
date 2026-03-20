# Causal UpSampling1D + Conv1D Research

## Context

The `MelDetokenizer` has `TOKEN_NUM_CONV_LAYERS=2` upsample stages. Each stage currently does:
```python
x = tf.repeat(x, repeats=2, axis=1)  # nearest-neighbor upsample
x = CausalConv1D(filters, kernel_size=3)(x)
x = LayerNorm()(x)
x = gelu(x)
```

The goal is to replace `tf.repeat` with `UpSampling1D(size=2)` + Conv1D (or reuse `CausalConv1D`).

## Findings

### (1) Does UpSampling1D(size=2) preserve causality?

**Yes.** `UpSampling1D(size=2)` repeats each temporal step `size` times along the time axis. From the Keras docs:

> "Repeats each temporal step `size` times along the time axis."

For input `[A, B, C]`, output is `[A, A, B, B, C, C]`. This means:
- `output[0]` and `output[1]` come from `input[0]`
- `output[2]` and `output[3]` come from `input[1]`
- `output[2k]` and `output[2k+1]` come from `input[k]`

So `output[t]` depends only on `input[t // 2]`, which depends only on inputs at index `≤ t//2`. **Strict causality is preserved** — this is identical to `tf.repeat(x, repeats=2, axis=1)`.

### (2) Padding scheme for Conv1D after upsampling

The existing `CausalConv1D` in `layers.py` already handles this correctly. It left-pads with `(kernel_size - 1) * dilation_rate` zeros then applies `padding='valid'` Conv1D. This ensures `output[t]` depends only on `input[t]` and earlier.

The combination `UpSampling1D(2)` → `CausalConv1D(filters, kernel_size=3)` preserves strict causality because:
1. Upsampling preserves causality (shown above)
2. CausalConv1D preserves causality (left-pad + valid conv)
3. Composition of causal operations is causal

No additional padding scheme is needed beyond what `CausalConv1D` already provides.

### (3) Minimal code example

```python
# Replace in MelDetokenizer.call():
for conv, norm in self.blocks:
    x = tf.keras.layers.UpSampling1D(size=2)(x)  # [B, T*2, D]
    x = conv(x)    # CausalConv1D, preserves length
    x = norm(x)
    x = tf.nn.gelu(x)
```

Or as layer construction (preferred — avoid recreating UpSampling1D each call):

```python
# In __init__:
self.blocks = []
for i in range(n):
    up = tf.keras.layers.UpSampling1D(size=2)
    conv = CausalConv1D(filters[i], kernel_size=3)
    norm = tf.keras.layers.LayerNormalization()
    self.blocks.append((up, conv, norm))

# In call:
for up, conv, norm in self.blocks:
    x = up(x)
    x = conv(x)
    x = norm(x)
    x = tf.nn.gelu(x)
```

### (4) Output length confirmation

**Yes.** For input `[B, T_tok, D]`, `UpSampling1D(size=2)` produces `[B, T_tok * 2, D]`.

From the docs: input shape `(batch_size, steps, features)` → output shape `(batch_size, upsampled_steps, features)` where `upsampled_steps = steps * size`.

With 2 stages (`TOKEN_NUM_CONV_LAYERS=2`):
- Input: `[B, 128, D_MODEL]`
- After stage 1: `[B, 256, filters[0]]`
- After stage 2: `[B, 512, filters[1]]`
- After final Dense proj: `[B, 512, N_MELS]`

This matches `T_tok * TOKEN_COMPRESSION_RATIO = 128 * 4 = 512 = SEQ_LEN`. ✓

## Recommendation

The replacement is a direct drop-in. `UpSampling1D(2)` is functionally identical to `tf.repeat(x, repeats=2, axis=1)` but is a proper Keras layer (serializable, named in model summary). Reuse the existing `CausalConv1D` from `layers.py` — no new padding logic needed.
