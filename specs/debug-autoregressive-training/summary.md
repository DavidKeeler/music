# Summary: Debug Autoregressive Training

## Problem
Training fails with `OperatorNotAllowedInGraphError` when using `tf.Variable` in Python `if` statement.

## Root Cause
Line 179 in `train_step()`: `if self.tf_ratio >= 1.0 - 1e-6:` uses symbolic tensor in Python conditional.

## Solution
Replace Python `if` with `tf.cond` for graph-compatible conditional execution.

## Artifacts Created

1. **rough-idea.md** - Problem statement and context
2. **requirements.md** - Q&A clarifying bug symptoms and constraints
3. **design.md** - Detailed fix design with architecture diagrams and acceptance criteria
4. **plan.md** - 5-step implementation plan with test requirements

## Key Design Decisions

- **Minimal changes:** Only modify `train_step()` method
- **Extract branches:** Create nested functions for each code path
- **Use tf.cond:** Graph-compatible conditional that handles symbolic tensors
- **Preserve logic:** No changes to teacher forcing or autoregressive algorithms
- **Fix NaN/Inf detection:** Simplify nested `tf.cond` to avoid type mismatch errors

## Implementation Steps

1. Extract pure teacher forcing logic into function
2. Extract autoregressive logic into function  
3. Replace Python if with tf.cond
4. Verify training runs successfully
5. Test tf_ratio transition behavior

## Success Criteria

✓ `python3 -m src.music_generation.train` runs without errors  
✓ Training completes at least one epoch  
✓ Metrics are logged correctly  
✓ Smooth transition between teacher forcing and autoregressive paths

## Next Steps

After this fix is implemented:
- Investigate memory-efficient autoregressive implementation from `/Users/davidkeeler/code/conducting`
- Consider creating a follow-up spec for memory optimization

## Files Modified

- `src/music_generation/train.py` - `MelGeneratorTraining.train_step()` method only

## Estimated Effort

- Implementation: 15-30 minutes
- Testing: 10-15 minutes (1 epoch run)
- Total: ~30-45 minutes
