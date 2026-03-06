# Vocoder Training Debug and Enhancement Summary

## Project Overview

Comprehensive plan to debug, validate, and enhance the vocoder training pipeline for stable finetuning of pretrained HiFi-GAN/MelGAN on music data.

## Problems Identified

Six areas needing improvement:
1. **No model validation** - No checks that pretrained model loads correctly
2. **Inefficient dataset** - Python generator instead of tf.data pipeline
3. **Training instability** - No gradient clipping or NaN detection
4. **No validation** - No validation dataset or metrics
5. **No quality assessment** - Can't hear generated audio during training
6. **Missing optimizations** - No LR scheduling or early stopping

## Solution Approach

Six-step enhancement plan:
1. Add model loading validation with diagnostic output
2. Rewrite dataset pipeline using tf.data with train/val split
3. Add gradient clipping and NaN/Inf detection
4. Add validation dataset and metrics
5. Create audio generation callback (samples every 5 epochs)
6. Add learning rate scheduling and early stopping

## Artifacts Created

```
specs/vocoder-debug/
├── rough-idea.md          # Problem statement and investigation areas
├── requirements.md        # (Empty - no additional requirements)
├── research/              # (Empty - no external research needed)
├── design.md              # Detailed technical design with implementations
├── plan.md                # 6-step implementation plan
├── PROMPT.md              # Ralph autonomous implementation spec
└── summary.md             # This file
```

## Key Design Decisions

**Dataset Pipeline:** tf.data with py_function
- Native TensorFlow operations for efficiency
- 90/10 train/val split
- Proper prefetching and shape handling

**Training Stability:** Gradient clipping + NaN detection
- Global norm clipping at 1.0
- Skip updates if NaN/Inf detected
- Track gradient norms for monitoring

**Quality Assessment:** Audio generation callback
- Generate samples every 5 epochs
- Save both target and predicted audio
- Essential for vocoder quality evaluation (loss alone insufficient)

**Optimization:** Exponential LR decay + early stopping
- Start at 1e-4, decay 0.98 every 1000 steps
- Early stopping with patience=10
- Restore best weights automatically

## Expected Impact

- **Stability:** Significantly improved (gradient clipping, NaN detection)
- **Efficiency:** 3-5x faster dataset pipeline
- **Monitoring:** Comprehensive (validation, audio samples, metrics)
- **Quality:** Better convergence (LR scheduling, early stopping)
- **Maintainability:** Production-ready with proper error handling

## Next Steps

### Option 1: Ralph Autonomous Implementation

```bash
ralph run -P specs/vocoder-debug/PROMPT.md
```

### Option 2: Manual Implementation

Follow 6-step plan in `specs/vocoder-debug/plan.md`:
1. Model loading validation (15 min)
2. Dataset pipeline rewrite (30 min)
3. Enhanced training loop (20 min)
4. Validation metrics (15 min)
5. Audio generation callback (20 min)
6. LR scheduling and early stopping (15 min)

**Total estimated time:** ~2 hours

## Testing After Implementation

```bash
# Run with validation and monitoring
python -m src.music_generation.train_vocoder \
  --data_dir ~/data/music/musicnet/train_data \
  --checkpoint_dir ./vocoder_checkpoints \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4
```

**Verify:**
- Model loading validation output appears
- Train/val split is created
- Validation loss logged each epoch
- Audio samples generated every 5 epochs
- Best model saved based on val_loss
- TensorBoard shows train/val curves
- No NaN/Inf issues during training

**Listen to audio samples:**
```bash
# Compare target vs predicted over epochs
ls vocoder_checkpoints/samples/
# epoch_005_target.wav, epoch_005_pred.wav, etc.
```

## References

- Original issue: Vocoder training needs debugging and validation
- Design document: `specs/vocoder-debug/design.md`
- Implementation plan: `specs/vocoder-debug/plan.md`
- Related code: `src/music_generation/train_vocoder.py`, `src/music_generation/vocoder.py`
