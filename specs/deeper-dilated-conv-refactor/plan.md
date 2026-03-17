# Implementation Plan: Deeper Dilated Causal Conv Refactor

## Checklist

- [ ] Step 1: Update config
- [ ] Step 2: Refactor MelGenerator
- [ ] Step 3: Verify

## Step 1: Update Config

**Objective:** Change `CONV_DILATION_RATES` to the new 5-element list.

**Implementation:**
- In `config.py`, change `CONV_DILATION_RATES = [1, 2, 4]` to `CONV_DILATION_RATES = [1, 2, 4, 8, 16]`
- Update the comment to reflect the new default and receptive field

**Test:** Import config, assert `CONV_DILATION_RATES == [1, 2, 4, 8, 16]`

**Demo:** Config loads with new values.

## Step 2: Refactor MelGenerator

**Objective:** Replace hardcoded conv layers with dynamic config-driven stack.

**Implementation:**
- In `model.py` `__init__`:
  - Remove `self.conv1`, `self.conv2`, `self.conv_head`
  - Remove dilation rate padding/truncation logic and warnings
  - Add `self.conv_layers` list comprehension over `CONV_DILATION_RATES`
  - Simplify receptive field logging to use `sum(CONV_DILATION_RATES)`
- In `model.py` `call()`:
  - Replace `self.conv1(x)` / `self.conv2(x)` calls with `for conv in self.conv_layers` loop
  - Remove `self.conv_head(x)` call after transformers
- Update class docstring to reflect new architecture

**Test:**
- Instantiate MelGenerator, verify `len(model.conv_layers) == 5`
- Forward pass with shape `[2, 128, 400]`, verify output shape matches
- Verify `generate()` produces correct output shape

**Demo:** Model instantiates, logs receptive field of 63 frames (~672 ms), forward pass succeeds.

## Step 3: Verify

**Objective:** Run existing tests to confirm nothing is broken.

**Implementation:**
- Run `pytest tests/`
- Spot-check parameter count is ~1.4M (up from ~1.3M)

**Demo:** All tests pass, parameter count confirmed.
