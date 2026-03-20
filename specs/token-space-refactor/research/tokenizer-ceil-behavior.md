# MelTokenizer Ceil Behavior Analysis

## Summary

The MelTokenizer produces `ceil(T/4)` output tokens for input length T, **not** `floor(T/4)`.
This is because causal left-padding before each stride-2 convolution effectively rounds up.
The MelDetokenizer always produces `T_tok * 4` frames, creating a mismatch of `(4 - T%4) % 4`
extra frames for non-divisible inputs.

## MelTokenizer Mechanics

Each of the 2 stride-2 conv layers does:
1. Left-pad by `kernel_size - 1 = 2` (kernel_size=3)
2. Conv1D stride=2, padding='valid'

Valid convolution output length: `floor((L_in - kernel_size) / stride) + 1`

So per layer: `floor((T + 2 - 3) / 2) + 1 = floor((T - 1) / 2) + 1 = ceil(T / 2)`

Two layers of `ceil(T/2)` gives `ceil(ceil(T/2) / 2)` which equals `ceil(T/4)`.

## Traced Examples

### T=10 (not divisible by 4)

| Layer | Input len | After pad (+2) | Valid conv formula | Output len |
|-------|-----------|-----------------|-------------------|------------|
| 1     | 10        | 12              | floor((12-3)/2)+1 = 5 | 5 |
| 2     | 5         | 7               | floor((7-3)/2)+1 = 3  | 3 |

Result: 3 tokens. `ceil(10/4) = 3`. ✓

Detokenizer output: `3 * 4 = 12` frames. Mismatch: **+2 extra frames**.

### T=13 (not divisible by 4)

| Layer | Input len | After pad (+2) | Valid conv formula | Output len |
|-------|-----------|-----------------|-------------------|------------|
| 1     | 13        | 15              | floor((15-3)/2)+1 = 7 | 7 |
| 2     | 7         | 9               | floor((9-3)/2)+1 = 4  | 4 |

Result: 4 tokens. `ceil(13/4) = 4`. ✓

Detokenizer output: `4 * 4 = 16` frames. Mismatch: **+3 extra frames**.

### T=15 (not divisible by 4)

| Layer | Input len | After pad (+2) | Valid conv formula | Output len |
|-------|-----------|-----------------|-------------------|------------|
| 1     | 15        | 17              | floor((17-3)/2)+1 = 8 | 8 |
| 2     | 8         | 10              | floor((10-3)/2)+1 = 4 | 4 |

Result: 4 tokens. `ceil(15/4) = 4`. ✓

Detokenizer output: `4 * 4 = 16` frames. Mismatch: **+1 extra frame**.

### T=12 (divisible by 4, control)

| Layer | Input len | After pad (+2) | Valid conv formula | Output len |
|-------|-----------|-----------------|-------------------|------------|
| 1     | 12        | 14              | floor((14-3)/2)+1 = 6 | 6 |
| 2     | 6         | 8               | floor((8-3)/2)+1 = 3  | 3 |

Result: 3 tokens. `ceil(12/4) = 3 = floor(12/4)`. ✓ No mismatch.

### T=16 (divisible by 4, control)

| Layer | Input len | After pad (+2) | Valid conv formula | Output len |
|-------|-----------|-----------------|-------------------|------------|
| 1     | 16        | 18              | floor((18-3)/2)+1 = 8 | 8 |
| 2     | 8         | 10              | floor((10-3)/2)+1 = 4 | 4 |

Result: 4 tokens. `ceil(16/4) = 4 = floor(16/4)`. ✓ No mismatch.

## MelDetokenizer Confirmation

The detokenizer does `tf.repeat(x, repeats=2, axis=1)` per layer (2 layers).
This is a pure doubling: `T_tok → T_tok*2 → T_tok*4`. The CausalConv1D layers
(kernel_size=3, stride=1, causal left-pad) preserve length. So detokenizer output
is always exactly `T_tok * 4`.

## Mismatch Formula

For input length T:
- Tokenizer output: `ceil(T/4)` tokens
- Detokenizer output: `ceil(T/4) * 4` frames
- Extra frames: `(4 - T%4) % 4` (i.e., 0 when T%4==0, else `4 - T%4`)

## Why This Matters

The current code assumes T is always divisible by 4 (docstring says "T is divisible by
TOKEN_COMPRESSION_RATIO", and dataset.py truncates to enforce this). But if we want to
support arbitrary-length inputs, we need to:

1. **Tokenizer**: Pad input to next multiple of 4 before encoding, OR accept the ceil behavior as-is.
2. **Detokenizer**: Trim output to original length T after decoding.

The ceil behavior is inherent to the causal left-padding + stride-2 architecture and cannot
be changed without modifying the convolution structure.

## Recommended Approach

Pad input to next multiple of 4 in the tokenizer, trim detokenizer output to original length.
This keeps the convolution math clean (always exact `T_padded/4` tokens) and the trim is a
simple slice operation.
