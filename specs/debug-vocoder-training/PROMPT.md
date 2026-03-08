# Debug Vocoder Training

## Objective

Fix and debug the vocoder training pipeline to run successfully end-to-end. The training script currently fails with missing dependencies and may have additional runtime issues.

## Key Requirements

- Install missing `huggingface_hub` dependency and update requirements.txt
- Verify pretrained HiFiGAN model loads correctly on Metal GPU
- Validate data pipeline with MusicNet dataset at `~/data/music/musicnet`
- Execute single training step to catch runtime errors
- Run full training with monitoring and checkpointing

## Current Error

```
ModuleNotFoundError: No module named 'huggingface_hub'
```

Command that fails:
```bash
python3 -m src.music_generation.train_vocoder \
    --data_dir ~/data/music/musicnet \
    --checkpoint_dir ~/models/conducting/vocoder_checkpoints \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0002
```

## Acceptance Criteria

**Given** the vocoder training script with all dependencies installed  
**When** the training command is executed  
**Then** training runs without errors for at least one full epoch

**Given** a single training step  
**When** forward and backward passes complete  
**Then** loss is finite and weights update correctly

**Given** the full training run  
**When** training completes or is interrupted  
**Then** checkpoints are saved and training can resume

## Reference

See `specs/debug-vocoder-training/` for detailed implementation plan.

## Environment

- Python 3.12 in venv
- macOS Apple M2 with Metal GPU
- TensorFlow with Metal plugin
- Project root: `/Users/davidkeeler/code/conducting3`
