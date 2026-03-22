# Pose-Conditioned Training

Integrate the BodyPointModule into the MelGenerator training pipeline. Two changes:

1. **Model update**: Add cross-attention layers to MelGenerator (transformer2 and transformer3) so it can accept pose conditioning via a projected pose sequence. Use gated conditioning (alpha parameter) to gradually introduce pose influence.

2. **New training script**: A separate `train_pose.py` that runs the full end-to-end pose-conditioned training following a 3-phase strategy:
   - Phase 1: Audio-only pretraining (existing `train.py`)
   - Phase 2: Introduce pose conditioning with alpha ramping 0.0 → 0.3, optionally freezing early layers
   - Phase 3: Joint training with alpha → 1.0 and conditioning dropout

The existing `train.py` remains unchanged for audio-only training.

## Knowledge Base Reference

See the full design spec: "Incorporating Pose (BodyPoint) Conditioning into MelGenerator" which covers:
- Cross-attention placement (transformer2/transformer3 only)
- Modified transformer block with gated cross-attention
- Pose projection and temporal alignment (Conv1D downsampling to token rate)
- 3-phase training strategy with alpha scheduling
- Conditioning dropout (p=0.1–0.2)
- Loss functions (mel reconstruction + optional pose-audio alignment)
