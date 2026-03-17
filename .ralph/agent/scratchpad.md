# Deeper Dilated Conv Refactor

## Understanding

Current state:
- `config.py`: `CONV_DILATION_RATES = [1, 2, 4]` (3 elements)
- `model.py`: Hardcoded `self.conv1`, `self.conv2`, `self.conv_head` with padding/truncation logic
- `conv1` and `conv2` run before transformers, `conv_head` runs after transformers
- Tests check for `conv1`, `conv2`, `conv_head` attributes and warn on list length mismatch

Target state:
- `config.py`: `CONV_DILATION_RATES = [1, 2, 4, 8, 16]` (5 elements)
- `model.py`: Dynamic `self.conv_layers` list, all before transformers, no padding/truncation/warnings
- Tests updated to check `conv_layers` list, no length mismatch tests needed

## Plan

Single task: config change + model refactor + test update. It's all one cohesive change.

Key changes:
1. config.py: Update CONV_DILATION_RATES and comment
2. model.py __init__: Replace conv1/conv2/conv_head with conv_layers loop, simplify RF logging, remove padding/warnings
3. model.py call(): Replace individual conv calls with loop, all before transformers
4. model.py docstring: Update architecture description
5. tests: Update to check conv_layers, remove too_short/too_long tests, add arbitrary rates test

## Iteration 1 - Complete

All changes implemented and committed:
1. config.py: CONV_DILATION_RATES = [1, 2, 4, 8, 16], updated comment
2. model.py: self.conv_layers dynamic list, loop in call(), simplified RF logging, removed padding/truncation/warnings
3. tests: Updated for conv_layers, added arbitrary rates test, RF logging test, removed too_short/too_long tests
4. All 11 dilated conv tests pass. 30 pre-existing failures from parallel loop (N_MELS 80→100 mismatch in other test files).

All 5 acceptance criteria met:
- ✅ 5 CausalConvBlock instances with matching dilation rates
- ✅ Output shape [B, 128, 400] preserved, all convs before transformers
- ✅ Log shows 63 frames ~672.0 ms
- ✅ Arbitrary rates work with no errors
- ✅ generate() works with correct output shape
