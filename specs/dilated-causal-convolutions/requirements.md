# Requirements Clarification

This document records questions and answers to refine the implementation requirements.

---

## Q1: Scope of Changes

The rough idea mentions modifying three convolution layers (conv1, conv2, conv_head) with dilation rates [1, 2, 4]. Should this change:

- **Only affect the MelGenerator model**, or
- **Also apply to any other models in the codebase that use CausalConvBlock**?

Additionally, should the dilation rates be:
- **Hardcoded** in the model architecture, or
- **Configurable** via config.py parameters?

### A1:
- Changes apply to **MelGenerator only**
- Dilation rates should be **configurable** via config.py

---

## Q2: Configuration Format

For the configurable dilation rates, what format would you prefer in config.py?

Option A - **List of dilation rates**:
```python
CONV_DILATION_RATES = [1, 2, 4]  # Applied to conv1, conv2, conv_head in order
```

Option B - **Individual parameters**:
```python
CONV1_DILATION = 1
CONV2_DILATION = 2
CONV_HEAD_DILATION = 4
```

Option C - **Dictionary mapping**:
```python
CONV_DILATIONS = {
    'conv1': 1,
    'conv2': 2,
    'conv_head': 4
}
```

Which approach fits best with the existing config.py style?

### A2:
**Option A** - List of dilation rates `CONV_DILATION_RATES = [1, 2, 4]`

---

## Q3: Backward Compatibility

Existing checkpoints were trained with dilation_rate=1 for all convolution layers. When loading old checkpoints with the new dilated convolution configuration:

- Should the system **fail with a clear error message** indicating architecture mismatch?
- Should there be a **compatibility mode** that detects old checkpoints and adjusts?
- Should we **require retraining from scratch** and document this as a breaking change?

What's your preferred approach for handling existing checkpoints?

### A3:
**Retrain from scratch** - This is a breaking change that requires new training runs.

---

## Q4: Validation and Testing

How should we validate that the dilated convolutions are working correctly?

Should the implementation include:

- **Unit tests** for the receptive field calculation (verify 15 frames with [1,2,4] dilations)?
- **Integration tests** that compare model output shapes before/after the change?
- **Manual verification** during training (e.g., logging receptive field size at startup)?
- **Ablation experiments** comparing [1,1,1] vs [1,2,4] dilation rates on a small dataset?

What level of testing is appropriate for this change?

### A4:
- **Unit tests** for receptive field calculation
- **Integration tests** for model output shapes

---

## Q5: Error Handling

If the configuration list `CONV_DILATION_RATES` has the wrong length (not exactly 3 elements), how should the system respond?

- **Fail immediately** at model initialization with a clear error?
- **Pad with default values** (e.g., fill missing values with 1)?
- **Use only the first N values** and ignore extras?

What's the desired behavior?

---
