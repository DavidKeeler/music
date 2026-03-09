# PROMPT: Fix Teacher Forcing Type Mismatch and Memory Issue

## Objective

Fix two critical bugs in `src/music_generation/train.py` that prevent training from running:
1. Type mismatch error (int64/int32) in `update_tf_ratio()` at line 125
2. Memory explosion (168 MB → 0.65 MB) in autoregressive loop at line 220

## Key Requirements

- Make exactly TWO one-line changes to `src/music_generation/train.py`
- Do NOT modify context capping logic (intentional design)
- Maintain gradient flow through predictions
- Preserve checkpoint compatibility
- Add comprehensive tests

## Acceptance Criteria

### Fix 1: Type Mismatch (Line 125)

**Given** the MelGeneratorTraining model is initialized with any warmup_steps value  
**When** training begins and update_tf_ratio() is called  
**Then** no TypeError should occur

**Given** warmup_steps = 100  
**When** update_tf_ratio() is called at step 50  
**Then** tf_ratio should equal initial_tf_ratio (still in warmup)

**Given** warmup_steps = 100  
**When** update_tf_ratio() is called at step 150  
**Then** tf_ratio should decay based on step_after_warmup = 50

### Fix 2: Memory Issue (Line 220)

**Given** training with SEQ_LEN=512 and BATCH_SIZE=4  
**When** autoregressive training runs (tf_ratio < 1.0)  
**Then** memory usage should be ~0.65 MB (not 168 MB)

**Given** the memory fix is applied  
**When** gradients are computed  
**Then** gradients should flow through preds to model parameters

**Given** the memory fix is applied  
**When** loss is computed on pred_seq  
**Then** loss value should be identical to before fix

### Integration

**Given** both fixes are applied  
**When** training runs for 100 steps  
**Then** no OOM errors and no type errors should occur

**Given** both fixes are applied  
**When** checkpoints are saved and loaded  
**Then** training should resume correctly

## Reference

See `specs/fix-teacher-forcing-type-mismatch/` for:
- `design.md` - Complete technical specification
- `plan.md` - Step-by-step implementation guide
- `research/` - Detailed analysis of both issues

## Implementation Notes

**Line 125 fix:**
```python
# Cast warmup_steps to int64 for type compatibility
step_after_warmup = tf.maximum(0, self.training_step - tf.cast(self.warmup_steps, tf.int64))
```

**Line 220 fix:**
```python
# Stop gradient: ar_input is just a container, doesn't need gradients
ar_input = tf.concat([ar_input, tf.stop_gradient(next_input)], axis=1)
```

## Testing Requirements

Create `tests/test_train.py` with:
1. `test_update_tf_ratio_no_type_error()` - Verify no TypeError
2. `test_warmup_behavior()` - Verify warmup maintains initial ratio
3. `test_gradients_flow_correctly()` - Verify gradients exist
4. `test_memory_usage_reasonable()` - Verify memory < 100 MB
5. `test_full_training_step()` - Verify end-to-end training
6. `test_checkpoint_save_load()` - Verify checkpoint compatibility

All tests must pass.
