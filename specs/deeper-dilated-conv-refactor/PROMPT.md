# PROMPT: Deeper Dilated Causal Conv Refactor

## Objective

Refactor MelGenerator's hardcoded 3-layer conv stack into a dynamic config-driven dilated causal architecture with 5 layers, increasing the receptive field from ~160 ms to ~672 ms.

## Key Requirements

- Change `CONV_DILATION_RATES` in `config.py` from `[1, 2, 4]` to `[1, 2, 4, 8, 16]`
- In `MelGenerator.__init__`: replace `self.conv1`, `self.conv2`, `self.conv_head` with `self.conv_layers` built dynamically from `CONV_DILATION_RATES`
- In `MelGenerator.call()`: replace individual conv calls with a loop over `self.conv_layers`, all before the transformer stack
- Simplify receptive field logging to use `sum(CONV_DILATION_RATES)`
- Remove dilation rate padding/truncation logic and warnings
- `D_MODEL` stays at 128, no other architecture changes

## Acceptance Criteria

**Given** `CONV_DILATION_RATES = [1, 2, 4, 8, 16]`
**When** MelGenerator is instantiated
**Then** `self.conv_layers` contains 5 CausalConvBlock instances with matching dilation rates

**Given** a MelGenerator with 5 dilated conv layers
**When** `call()` is invoked with input `[B, 128, 400]`
**Then** output shape is `[B, 128, 400]` and all convs run before transformer blocks

**Given** `CONV_DILATION_RATES = [1, 2, 4, 8, 16]`
**When** receptive field is logged
**Then** log shows 63 frames and ~672.0 ms

**Given** any arbitrary `CONV_DILATION_RATES` list
**When** MelGenerator is instantiated
**Then** conv stack matches provided rates with no errors

**Given** the refactored model
**When** `generate()` is called
**Then** autoregressive generation works with correct output shape

## Reference

See `specs/deeper-dilated-conv-refactor/` for design doc and implementation plan.
