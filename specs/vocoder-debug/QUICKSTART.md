# Quick Reference: Vocoder Training Debug

## Problem
Vocoder training needs validation and enhancements:
- No model loading verification
- Inefficient Python generator dataset
- No gradient clipping or NaN detection
- No validation metrics
- No audio generation during training
- Missing LR scheduling

## Solution Summary
1. Model loading validation
2. tf.data pipeline with train/val split
3. Gradient clipping + NaN detection
4. Validation dataset and metrics
5. Audio generation callback (every 5 epochs)
6. LR scheduling + early stopping

## To Implement with Ralph

```bash
ralph run -P specs/vocoder-debug/PROMPT.md
```

## To Implement Manually

Follow 6 steps in `specs/vocoder-debug/plan.md` (~2 hours total)

## To Test After Implementation

```bash
python -m src.music_generation.train_vocoder \
  --data_dir ~/data/music/musicnet/train_data \
  --checkpoint_dir ./vocoder_checkpoints \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4
```

**Check for:**
- ✓ Model loading validation output
- ✓ Train/val split (90/10)
- ✓ Validation loss each epoch
- ✓ Audio samples in `vocoder_checkpoints/samples/`
- ✓ Best model saved based on val_loss
- ✓ No NaN/Inf issues

**Listen to samples:**
```bash
ls vocoder_checkpoints/samples/
# epoch_005_target.wav, epoch_005_pred.wav
# epoch_010_target.wav, epoch_010_pred.wav
# ... quality should improve over epochs
```

## Expected Results
- Training stability: Significantly improved
- Dataset efficiency: 3-5x faster
- Monitoring: Comprehensive (validation, audio, metrics)
- Quality: Better convergence

## Files Changed
- `src/music_generation/train_vocoder.py` (validation, callbacks, scheduling)
- `src/music_generation/vocoder.py` (dataset pipeline)
- `requirements.txt` (ensure soundfile present)
