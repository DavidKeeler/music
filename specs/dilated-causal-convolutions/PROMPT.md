# Implement Dilated Causal Convolutions

## Objective

Implement dilated causal convolutions in the MelGenerator model to increase temporal receptive field from ~58ms to ~174ms without adding parameters. Apply progressive dilation rates [1, 2, 4] to three convolution layers.

## Key Requirements

1. Add `CONV_DILATION_RATES = [1, 2, 4]` to `src/music_generation/config.py`
2. Modify `MelGenerator.__init__()` in `src/music_generation/model.py` to:
   - Import `CONV_DILATION_RATES`, `FRAME_STEP`, `SAMPLE_RATE` from config
   - Extract dilation rates with padding/truncation for mismatched list lengths
   - Log warnings for list length mismatches
   - Calculate and log receptive field at initialization
   - Pass `dilation_rate` parameter to `CausalConvBlock` constructors (conv1, conv2, conv_head)
3. Create `tests/test_dilated_convolutions.py` with:
   - Unit tests for receptive field calculation
   - Model initialization tests with various configurations
   - Integration tests for forward pass, causality, and generation
4. Verify all tests pass
5. No changes to `train.py` or `layers.py` required

## Acceptance Criteria

**Given** a MelGenerator model with dilated convolutions  
**When** the model is initialized with `CONV_DILATION_RATES = [1, 2, 4]`  
**Then** the model should:
- Initialize without errors
- Log "MelGenerator initialized with dilation rates [1, 2, 4]"
- Log "Total receptive field: 15 frames (~174.0 ms)"
- Apply dilation_rate=1 to conv1, dilation_rate=2 to conv2, dilation_rate=4 to conv_head

**Given** a MelGenerator model  
**When** forward pass is called with input shape [B, T, 320]  
**Then** output shape should be [B, T, 320]

**Given** a MelGenerator model  
**When** `CONV_DILATION_RATES = [1, 2]` (too short)  
**Then** model should initialize with dilations [1, 2, 1] and log warning

**Given** a MelGenerator model  
**When** `CONV_DILATION_RATES = [1, 2, 4, 8, 16]` (too long)  
**Then** model should initialize with dilations [1, 2, 4] and log warning

**Given** a test suite for dilated convolutions  
**When** all tests are run with `pytest tests/test_dilated_convolutions.py -v`  
**Then** all tests should pass

## Reference

Complete specification available in `specs/dilated-causal-convolutions/`:
- `design.md` - Detailed design with architecture diagrams and complete code
- `plan.md` - 9-step implementation plan
- `requirements.md` - Requirements clarification Q&A

## Implementation Notes

- Receptive field formula: `RF = 1 + 2 * sum(dilations)`
- Frame duration: `FRAME_STEP / SAMPLE_RATE * 1000` ms
- `CausalConvBlock` already supports `dilation_rate` parameter
- Causal padding automatically accounts for dilation: `padding = (kernel_size - 1) * dilation_rate`
