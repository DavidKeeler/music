# Requirements

This document tracks the requirements clarification process through Q&A.

---

## Q1: Reduction Factor Value

What reduction factor (R) do you want to start with, and should it be configurable?

Options:
- Fixed value (e.g., R=4 as mentioned in the rough idea)
- Configurable parameter in config.py
- Support for dynamic/adaptive reduction during training

**A1:** Configurable parameter in config.py

---

## Q2: Dataset Preprocessing

How should the dataset handle sequences that aren't evenly divisible by R?

Options:
- Pad sequences to make them divisible by R
- Truncate the remainder frames
- Process remainder frames separately with R=1
- Skip/filter out sequences that don't divide evenly

**A2:** Truncate the remainder frames

---

## Q3: Model Architecture Changes

The reduction factor affects multiple parts of the model. Which components need modification?

Should we update:
- Only the output projection layer (Dense(D_MODEL → R × N_MELS))?
- Input embedding layer to match the grouped frame structure?
- Both input and output to maintain consistency?
- Positional encoding to reflect the reduced sequence length?

**A3:** Both input and output to maintain consistency

---

## Q4: Teacher Forcing Strategy

During training with teacher forcing, how should the model consume ground truth frames?

Options:
- Feed R ground truth frames at each step (consistent with reduction factor)
- Continue feeding single frames but predict R frames (asymmetric)
- Gradually transition from single-frame to R-frame teacher forcing

**A4:** Feed R ground truth frames at each step (consistent with reduction factor)

---

## Q5: Inference Generation

During autoregressive generation at inference time, how should the model handle the predicted R frames?

Options:
- Use all R predicted frames as context for the next prediction
- Use only the last predicted frame as context
- Use a sliding window of the most recent frames
- Average or pool the R frames into a single representation

**A5:** Use all R predicted frames as context for the next prediction

---

## Q6: Backward Compatibility

Should the implementation support loading and converting existing checkpoints trained with R=1?

Options:
- Yes, provide checkpoint conversion utilities
- Yes, but only for inference (not continued training)
- No, require retraining from scratch
- Support both old (R=1) and new (R>1) models simultaneously

**A6:** No, require retraining from scratch

---

## Q7: Loss Computation

How should the loss be computed for the R predicted frames?

Options:
- Compute loss independently for each of the R frames and average
- Flatten all R frames and compute a single loss value
- Weight recent frames more heavily than earlier frames in the group
- Use a combination of frame-wise and sequence-wise losses

**A7:** Compute loss independently for each of the R frames and average

---

## Q8: Validation and Metrics

Should validation metrics be computed at the reduced sequence level or expanded back to individual frames?

Options:
- Compute metrics on grouped frames (faster, matches training)
- Expand predictions back to individual frames for metrics (more interpretable)
- Provide both grouped and expanded metrics
- Only track loss, no frame-level metrics

**A8:** Compute metrics on grouped frames (faster, matches training)

---

## Q9: Default Reduction Factor

What should be the default value for the reduction factor in config.py?

Options:
- R=2 (conservative, 2x speedup)
- R=4 (as mentioned in rough idea, 4x speedup)
- R=8 (aggressive, 8x speedup)
- Keep R=1 as default for backward compatibility

**A9:** R=4 (as mentioned in rough idea, 4x speedup)

---

## Q10: Testing Strategy

What level of testing is required for this feature?

Options:
- Unit tests only (shape transformations, layer dimensions)
- Unit tests + integration tests (full training step, inference generation)
- Unit tests + integration tests + regression tests (compare R=1 vs R>1 convergence)
- Minimal testing, rely on manual validation

**A10:** Unit tests only (shape transformations, layer dimensions)

---

## Q11: Error Handling

How should the system handle invalid reduction factor configurations?

Options:
- Validate at config load time and raise errors for invalid values
- Validate at model initialization and provide helpful error messages
- Auto-correct invalid values (e.g., R=0 → R=1)
- Allow any positive integer and let TensorFlow handle dimension errors

**A11:** We don't handle that case (allow any positive integer and let TensorFlow handle dimension errors)

---

## Q12: Documentation Requirements

What documentation should accompany this feature?

Options:
- Update README.md with reduction factor explanation and usage
- Add inline code comments only
- Create a separate design document explaining the technique
- Minimal documentation, code should be self-explanatory

**A12:** Inline code comments only

---

## Requirements Clarification Complete

The user has indicated that requirements clarification is complete and we should proceed to design.

