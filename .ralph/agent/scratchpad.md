# Scratchpad: TensorFlow Pretrained Vocoder Finetuning

## Iteration 1 - Understanding and Planning

### Objective
Enhance TensorFlow music generation with:
1. Unified inference model (MusicGenerationModel) combining MelGenerator + Vocoder
2. Gradient norm logging in vocoder training
3. SavedModel checkpoint format
4. MelGAN integration from TensorFlowTTS

### Analysis
The plan.md provides a clear 6-step implementation order:
1. Add gradient norm logging to VocoderTraining (easiest, immediate value)
2. Update checkpoint saving to SavedModel format
3. Update MelGAN loading and output handling
4. Create MusicGenerationModel inference class
5. Add unit tests
6. Add smoke tests

Each step is independent and testable. Total estimated: ~250 lines of code.

### Implementation Strategy
Following the plan's order makes sense because:
- Step 1 (grad norm) is isolated and easy to verify
- Step 2 (SavedModel) improves checkpoint management for later steps
- Step 3 (MelGAN) enables pretrained model use
- Step 4 (inference model) ties everything together
- Steps 5-6 (tests) validate the implementation

I'll create tasks for each step and implement them one at a time.


## Task 1: Gradient Norm Logging - COMPLETE

Verified that VocoderTraining already has gradient norm logging fully implemented:
- ✅ grad_norm_tracker metric in __init__
- ✅ Gradient norm computation in train_step using tf.sqrt(tf.add_n([tf.reduce_sum(g**2) for g in grads]))
- ✅ Returns {"loss": loss, "grad_norm": self.grad_norm_tracker.result()}
- ✅ @property metrics returning [self.grad_norm_tracker]

No changes needed. Task closed.
