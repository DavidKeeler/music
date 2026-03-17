# Rough Idea: Deeper Dilated Causal Conv Refactor

Refactor the MelGenerator convolution stack to support a deeper dilated causal architecture and increase the receptive field to ~0.7 seconds.

## Goals

* Replace the current fixed 3-layer causal conv setup with a flexible stack driven by `CONV_DILATION_RATES`
* Use dilation rates: [1, 2, 4, 8, 16]
* Achieve a receptive field of ~63 frames (~670 ms at 24kHz, hop=256)
* Preserve strict causality and residual connections
* Keep all other architecture components unchanged
* Decrease internal layer sizes so the model doesn't get too big with the additional layers

## Required Changes

### 1. Config
- Set: `CONV_DILATION_RATES = [1, 2, 4, 8, 16]`

### 2. MelGenerator.__init__
- Remove hardcoded `self.conv1`, `self.conv2`, `self.conv_head`
- Replace with dynamic list comprehension over `CONV_DILATION_RATES`:
  ```python
  self.conv_layers = [
      CausalConvBlock(D_MODEL, kernel_size=3, dilation_rate=d, residual=True)
      for d in CONV_DILATION_RATES
  ]
  ```

### 3. Receptive Field Logging
- Replace existing logic with:
  ```python
  receptive_field_frames = 1 + 2 * sum(CONV_DILATION_RATES)
  frame_duration_ms = (FRAME_STEP / SAMPLE_RATE) * 1000
  receptive_field_ms = receptive_field_frames * frame_duration_ms
  ```

### 4. MelGenerator.call()
- Replace sequential conv1/conv2/conv_head calls with loop:
  ```python
  for conv in self.conv_layers:
      x = conv(x, training=training)
  ```

### 5. Do NOT Change
- Transformer blocks
- Latent conditioning (z)
- Input/output projections
- Training logic
- Reduction factor or sequence length

## Expected Outcome
- Increased local temporal modeling (~670 ms vs ~160 ms)
- Improved note continuity and harmonic stability
- Reduced burden on transformer layers for short-term structure
- Smaller internal dimension to offset parameter increase from deeper stack
