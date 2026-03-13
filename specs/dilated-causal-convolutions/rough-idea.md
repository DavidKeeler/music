### Dilated Causal Convolutions — Implementation Context

This document describes how to integrate **dilated causal convolutions** into the mel generator architecture. The goal is to increase the temporal receptive field of the convolution stack without increasing kernel size or parameter count.

---

## Motivation

The current convolution layers use:

```
kernel_size = 3
dilation_rate = 1
```

This results in a very small receptive field.

With two stacked layers:

```
conv1: 3 frames
conv2: 5 frames total
```

Given the mel hop size:

```
hop_length = 256 samples
sample_rate = 22050 Hz
```

each frame represents approximately:

```
11.6 ms
```

So the convolution stack currently observes only:

```
≈ 58 ms
```

of temporal context. This is too short to capture meaningful acoustic or musical structures such as note attacks, phoneme transitions, or rhythmic patterns.

Dilated convolutions allow the receptive field to grow rapidly while keeping the network lightweight.

---

## Dilated Convolution Concept

A **dilated convolution** introduces spacing between kernel elements.

For a convolution with:

```
kernel_size = 3
dilation = d
```

the effective receptive field becomes:

```
2 * d + 1
```

Example:

| dilation | frames used |
| -------- | ----------- |
| 1        | t-2, t-1, t |
| 2        | t-4, t-2, t |
| 4        | t-8, t-4, t |

This expands temporal coverage without increasing parameters.

---

## Receptive Field Expansion

Stacking dilated layers causes the receptive field to grow approximately exponentially.

Recommended dilation stack:

```
dilation = 1
dilation = 2
dilation = 4
```

For kernel size 3, the receptive field becomes:

```
1 + 2*(1 + 2 + 4) = 15 frames
```

In time:

```
15 × 11.6 ms ≈ 174 ms
```

This allows the model to observe nearly 0.2 seconds of context before the transformer layers.

---

## Integration in the Mel Generator

The current architecture contains three convolution stages:

```
conv1
conv2
conv_head
```

These should be assigned progressively larger dilation rates.

Recommended configuration:

```
conv1: dilation = 1
conv2: dilation = 2
conv_head: dilation = 4
```

This improves local temporal modeling before the transformer stack processes longer dependencies.

---

## Required Code Changes

The existing `CausalConvBlock` already supports a dilation parameter:

```
CausalConvBlock(filters, kernel_size, dilation_rate, residual=True)
```

Only the initialization of the layers needs to be modified.

Example:

```
self.conv1 = CausalConvBlock(
    D_MODEL,
    kernel_size=3,
    dilation_rate=1,
    residual=True
)

self.conv2 = CausalConvBlock(
    D_MODEL,
    kernel_size=3,
    dilation_rate=2,
    residual=True
)

self.conv_head = CausalConvBlock(
    D_MODEL,
    kernel_size=3,
    dilation_rate=4,
    residual=True
)
```

No other code changes are required because the causal padding already accounts for dilation.

---

## Causal Padding Behavior

The causal convolution layer computes left padding as:

```
padding = (kernel_size - 1) * dilation_rate
```

This ensures that output at timestep `t` only depends on inputs at times:

```
≤ t
```

Therefore the autoregressive property of the model remains intact.

---

## Expected Effects

Adding dilation to the convolution layers should produce the following improvements:

* increased temporal receptive field before attention
* improved modeling of short-term audio structure
* more stable autoregressive generation
* better convergence during training

The parameter count and computational cost remain nearly unchanged.

---

## Summary

Dilated causal convolutions allow the model to capture longer temporal dependencies using small kernels. By assigning progressively larger dilation rates across the convolution stack, the mel generator gains a significantly larger receptive field without increasing model complexity.
