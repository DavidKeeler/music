# Scratchpad: Debug Vocoder Training

## Iteration 1 - Understanding the Problem

### Current Error
The vocoder training script fails with:
```
ModuleNotFoundError: No module named 'huggingface_hub'
```

### Command that fails:
```bash
python3 -m src.music_generation.train_vocoder \
    --data_dir ~/data/music/musicnet \
    --checkpoint_dir ~/models/conducting/vocoder_checkpoints \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0002
```

### Environment
- Python 3.12 in venv
- macOS Apple M2 with Metal GPU
- TensorFlow with Metal plugin
- Project root: `/Users/davidkeeler/code/conducting3`

### Existing Tasks
- task-1772938382-abe0: "Run single training step" (blocked by closed task - needs unblocking)
- task-1772938384-4b59: "Monitor full training execution" (blocked by task-1772938382-abe0)

### Plan
1. Install missing `huggingface_hub` dependency
2. Update requirements.txt
3. Unblock task-1772938382-abe0 (its blocker is closed)
4. Run single training step to catch runtime errors
5. Monitor full training execution

### Discovery
- `huggingface_hub` is already installed (v0.36.2)
- Memory shows HiFiGAN pretrained weights are unavailable (401 error from Hugging Face)
- Fallback mechanism uses Griffin-Lim for testing
- Model can be trained from scratch

### Task Analysis
The existing tasks assume dependencies are installed. But there's a blocker issue:
- task-1772938382-abe0 is blocked by task-1772938379-2286 (which is CLOSED)
- This is a stale dependency that needs to be removed

### Plan
1. Create prerequisite task: Verify vocoder can initialize (with fallback to Griffin-Lim)
2. Unblock task-1772938382-abe0 by removing the closed blocker
3. Execute task-1772938382-abe0: Run single training step
4. Execute task-1772938384-4b59: Monitor full training

### Next Action
Create the prerequisite task for vocoder initialization verification.
