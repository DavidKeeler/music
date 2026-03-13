# Implementation Plan

## Checklist

- [ ] Step 1: Add CONV_DILATION_RATES to config.py
- [ ] Step 2: Update MelGenerator imports
- [ ] Step 3: Modify MelGenerator.__init__() with dilation logic
- [ ] Step 4: Create test file structure
- [ ] Step 5: Implement receptive field calculation tests
- [ ] Step 6: Implement model initialization tests
- [ ] Step 7: Implement integration tests
- [ ] Step 8: Run test suite and verify all pass
- [ ] Step 9: Manual verification with training script

---

## Step 1: Add CONV_DILATION_RATES to config.py

**Objective**: Add the configurable dilation rates parameter to the configuration module.

**Implementation**:
- Open `src/music_generation/config.py`
- Locate the `WINDOW_SIZES` definition (around line 35)
- Add the following after `WINDOW_SIZES`:

```python
# Dilated Convolution Configuration
# Dilation rates for conv1, conv2, conv_head layers
# Progressive dilation increases receptive field: 1 + 2*sum(rates) frames
# Default [1, 2, 4] gives 15 frames (~174ms at 22.05kHz with hop=256)
CONV_DILATION_RATES = [1, 2, 4]
```

**Tests**: None (configuration constant)

**Integration**: This constant will be imported by model.py in Step 2

**Demo**: Run `python -c "from src.music_generation.config import CONV_DILATION_RATES; print(CONV_DILATION_RATES)"` to verify the constant is accessible

---

## Step 2: Update MelGenerator imports

**Objective**: Import the new configuration constants needed for dilation and logging.

**Implementation**:
- Open `src/music_generation/model.py`
- Locate the import statement (lines 3-7)
- Modify to include `CONV_DILATION_RATES`, `FRAME_STEP`, and `SAMPLE_RATE`:

```python
from src.music_generation.config import (
    D_MODEL, NUM_HEADS, SEQ_LEN, N_MELS, WINDOW_SIZES,
    REDUCTION_FACTOR, GROUPED_MEL_DIM, EFFECTIVE_SEQ_LEN,
    CONV_DILATION_RATES, FRAME_STEP, SAMPLE_RATE
)
```

**Tests**: None (import statement)

**Integration**: These imports enable Step 3 implementation

**Demo**: Run `python -c "from src.music_generation.model import MelGenerator"` to verify no import errors

---

## Step 3: Modify MelGenerator.__init__() with dilation logic

**Objective**: Apply progressive dilation rates to the three convolution layers and log receptive field information.

**Implementation**:
- Open `src/music_generation/model.py`
- Locate `MelGenerator.__init__()` method (starts at line 33)
- Replace the entire `__init__` method with the implementation from design.md (Components section)
- Key changes:
  - Extract dilation rates with padding/truncation logic
  - Add warning logging for mismatched list lengths
  - Calculate and log receptive field
  - Pass `dilation_rate` parameter to CausalConvBlock constructors

**Tests**: Covered by Steps 5-7

**Integration**: This is the core change that enables dilated convolutions

**Demo**: Instantiate model and verify log output shows correct dilation rates and receptive field

---

## Step 4: Create test file structure

**Objective**: Set up the test file for dilated convolution testing.

**Implementation**:
- Create `tests/test_dilated_convolutions.py`
- Add imports and test class structure:

```python
"""Tests for dilated causal convolutions."""

import pytest
import tensorflow as tf
import logging
from unittest.mock import patch
from src.music_generation.model import MelGenerator
from src.music_generation.config import FRAME_STEP, SAMPLE_RATE


class TestReceptiveField:
    """Test receptive field calculations."""
    pass


class TestModelInitialization:
    """Test model initialization with various dilation configurations."""
    pass


class TestIntegration:
    """Integration tests for dilated convolutions."""
    pass
```

**Tests**: None (structure only)

**Integration**: Provides foundation for Steps 5-7

**Demo**: Run `pytest tests/test_dilated_convolutions.py -v` (should show 0 tests collected)

---

## Step 5: Implement receptive field calculation tests

**Objective**: Verify the receptive field formula works correctly for various dilation configurations.

**Implementation**:
- Add test methods to `TestReceptiveField` class:
  - `test_rf_default_config`: Verify [1,2,4] gives 15 frames
  - `test_rf_no_dilation`: Verify [1,1,1] gives 5 frames
  - `test_rf_aggressive`: Verify [2,4,8] gives 29 frames
  - `test_rf_to_milliseconds`: Verify time conversion is correct

**Tests**: Run `pytest tests/test_dilated_convolutions.py::TestReceptiveField -v`

**Integration**: These tests validate the formula used in Step 3 logging

**Demo**: All receptive field tests pass

---

## Step 6: Implement model initialization tests

**Objective**: Verify model initializes correctly with various dilation configurations and handles edge cases.

**Implementation**:
- Add test methods to `TestModelInitialization` class:
  - `test_init_default_dilations`: Model initializes with [1,2,4]
  - `test_init_custom_dilations`: Model initializes with custom rates
  - `test_init_short_list`: Warning logged when list too short
  - `test_init_long_list`: Warning logged when list too long
  - `test_init_empty_list`: Warning logged and defaults to [1,1,1]
  - `test_logging_output`: Verify log messages contain correct RF info

**Tests**: Run `pytest tests/test_dilated_convolutions.py::TestModelInitialization -v`

**Integration**: These tests verify Step 3 implementation handles all edge cases

**Demo**: All initialization tests pass, warnings are logged correctly

---

## Step 7: Implement integration tests

**Objective**: Verify the model works end-to-end with dilated convolutions.

**Implementation**:
- Add test methods to `TestIntegration` class:
  - `test_forward_pass_shape`: Output shape matches input shape
  - `test_causality_preserved`: Output at time t doesn't depend on future
  - `test_generation`: Autoregressive generation works
  - `test_training_step`: Model can perform a training step

**Tests**: Run `pytest tests/test_dilated_convolutions.py::TestIntegration -v`

**Integration**: These tests verify the entire model pipeline works with dilated convolutions

**Demo**: All integration tests pass, model generates audio successfully

---

## Step 8: Run test suite and verify all pass

**Objective**: Execute the complete test suite and ensure all tests pass.

**Implementation**:
- Run full test suite: `pytest tests/test_dilated_convolutions.py -v`
- Verify all tests pass (expected: ~15 tests)
- Run existing test suite to ensure no regressions: `pytest tests/ -v`
- Fix any failures

**Tests**: All tests in test_dilated_convolutions.py pass

**Integration**: Confirms implementation is correct and doesn't break existing functionality

**Demo**: Test output shows all green, no failures or errors

---

## Step 9: Manual verification with training script

**Objective**: Verify the implementation works in a real training scenario.

**Implementation**:
- Run training script with minimal configuration:
  ```bash
  python -m src.music_generation.train \
    --data_dir ~/data/music/musicnet/train_data \
    --cache_dir ./cache \
    --checkpoint_dir ./checkpoints_dilated \
    --epochs 1 \
    --batch_size 2
  ```
- Verify log output shows:
  - "MelGenerator initialized with dilation rates [1, 2, 4]"
  - "Total receptive field: 15 frames (~174.0 ms)"
- Verify training completes at least 1 epoch without errors
- Verify no NaN losses or crashes

**Tests**: Manual observation of training logs and metrics

**Integration**: Final validation that the implementation works in production

**Demo**: Training runs successfully, logs show correct receptive field, no errors
