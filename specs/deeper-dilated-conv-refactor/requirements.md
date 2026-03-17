# Requirements: Deeper Dilated Causal Conv Refactor

## Questions & Answers

### Q1: What should D_MODEL be reduced to?

Currently `D_MODEL = 128` with 3 conv layers. You're moving to 5 conv layers (dilations [1,2,4,8,16]). The conv parameter count roughly scales linearly with layer count, so 5 layers ≈ 1.67× the conv params. 

Some options:
- **96** — ~25% reduction, brings total conv params below current levels even with 5 layers
- **112** — ~12% reduction, roughly parameter-neutral overall
- **64** — ~50% reduction, aggressive shrink, significantly smaller model

**A1:** Keep `D_MODEL = 128` unchanged. The model is currently ~1.3M parameters — adding 2 extra conv layers is negligible. No need to reduce internal dimensions.

### Q2: Should the conv layers all go before the transformer stack, or should any go after?

Currently the architecture is: `conv1 → conv2 → transformer1 → transformer2 → transformer3 → conv_head`. So 2 convs before transformers, 1 after.

Your spec says replace all three with a loop of 5. Should all 5 go before the transformers, or do you want to preserve the pre/post split (e.g., 4 before + 1 after, or all 5 before)?

**A2:** All 5 conv layers go before the transformer stack. No conv after transformers. Architecture becomes: `input_proj → [conv × 5] → [transformer × 3] → output_proj`.

### Q4: Are the dilation rates [1, 2, 4, 8, 16] fixed, or should the design support arbitrary rates from config?

**A4:** Yes, arbitrary rates from config. The model dynamically builds conv layers from whatever `CONV_DILATION_RATES` contains — same pattern as the current config-driven approach.

### Q5: Any concerns about training stability with the deeper conv stack?

**A5:** No changes to training hyperparameters. Train with current settings and evaluate.

