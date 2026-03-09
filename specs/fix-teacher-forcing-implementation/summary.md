# Summary: Fix Teacher Forcing Implementation

## Overview

Make the autoregressive context window size configurable to enable full sequence context during training, improving the model's ability to learn long-range dependencies. Includes comprehensive testing for memory safety and correctness.

## Artifacts

- **rough-idea.md** - Initial problem statement
- **research/loop-performance.md** - Analysis of TensorFlow loop performance (tf.scan vs TensorArray)
- **design.md** - Detailed design with architecture, components, and acceptance criteria
- **plan.md** - 7-step implementation plan with TDD approach

## Key Changes

1. Add `MAX_CONTEXT_FRAMES = SEQ_LEN` to `config.py` (default: 512)
2. Update `train.py` to use configurable context instead of hardcoded 64
3. Comprehensive test suite (unit, integration, memory profiling)
4. Documentation updates

## Impact

- **Performance**: Full context (512 frames) enables better long-range learning
- **Flexibility**: Configurable for memory-constrained environments
- **Memory**: Constant footprint maintained via sliding window
- **Risk**: Low - minimal code changes, extensive testing

## Estimated Effort

4-6 hours total:
- Implementation: 1 hour (2 lines of code)
- Testing: 2-3 hours (unit, integration, memory tests)
- Documentation: 1 hour
- Validation: 1-2 hours

## Next Steps

Create PROMPT.md for Ralph to implement autonomously, or implement manually following the plan.
