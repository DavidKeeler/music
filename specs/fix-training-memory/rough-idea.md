# Fix Training Memory Issue

Training freezes the laptop. Suspected memory problem. Need to diagnose and fix.

**Command that freezes:**
```bash
python -m src.music_generation.train \
  --data_dir ~/data/music/musicnet/train_data \
  --cache_dir ./cache \
  --checkpoint_dir ./checkpoints \
  --epochs 100 \
  --batch_size 16
```

**Hardware:** MacBook Air
