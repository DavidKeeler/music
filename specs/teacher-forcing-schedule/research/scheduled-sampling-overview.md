# Scheduled Sampling Overview

## Problem: Exposure Bias

During training with pure teacher forcing, the model receives ground truth previous frames as input. During inference, it must use its own predictions. This train-test mismatch causes:

- **Error accumulation**: Small prediction errors compound over time
- **Distribution shift**: Model sees sequences outside its training distribution
- **Poor autoregressive quality**: Model hasn't learned to handle its own mistakes

## Solution: Scheduled Sampling

Proposed by Bengio et al. (2015) in "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" [1], scheduled sampling gradually transitions from teacher forcing to using model predictions during training.

### Core Mechanism

At each timestep during training, with probability ε (teacher forcing ratio):
- Use ground truth previous frame (teacher forcing)
- Otherwise, use model's own prediction

The probability ε starts at 1.0 (pure teacher forcing) and decreases during training according to a schedule.

## Schedule Types

The original paper and subsequent research identified several schedule types:

### 1. Linear Decay
```
ε(i) = max(ε_min, ε_initial - k * i)
```
- Simple, predictable decrease
- k controls decay rate
- Reaches minimum at fixed step

### 2. Exponential Decay
```
ε(i) = max(ε_min, ε_initial * exp(-k * i))
```
- Rapid initial decrease, then slower
- More gradual transition
- Common in practice

### 3. Inverse Sigmoid
```
ε(i) = k / (k + exp(i / k))
```
- Smooth S-curve transition
- Gentle at both ends, steep in middle
- Recommended in original paper

### 4. Step Decay
```
ε drops at specific milestones (e.g., 1.0 → 0.5 → 0.2 → 0.05)
```
- Discrete transitions
- Easier to tune
- Less smooth than continuous schedules

## Research Findings

### What Works Best

According to curriculum scheduling research [2]:
- **Inverse sigmoid** is most commonly recommended for smooth transitions
- **Exponential decay** is widely used in practice for its simplicity
- **Linear decay** can work but may be too abrupt
- Fixed schedules (predetermined curves) are more robust than adaptive ones

### Application to Speech/Music

From Tacotron 2 and mel spectrogram generation research:
- Tacotron 2 paper [3] uses pure teacher forcing (no schedule)
- Community implementations note this causes inference quality issues [4]
- Mel spectrogram generation benefits from scheduled sampling due to autoregressive nature
- Continuous outputs (mel frames) may be more sensitive to exposure bias than discrete tokens

### Key Parameters

1. **Initial ratio (ε_initial)**: Usually 1.0 (pure teacher forcing)
2. **Minimum ratio (ε_min)**: Typically 0.0-0.1 (some teacher forcing helps stability)
3. **Decay rate (k)**: Controls transition speed, requires tuning
4. **Schedule type**: Inverse sigmoid or exponential most common

## Implementation Considerations

### Training Stability
- Start with pure teacher forcing (ε=1.0) for initial epochs
- Gradual decay prevents training instability
- Minimum ratio > 0 can help maintain stability

### Computational Cost
- Minimal overhead: just a probability check per timestep
- Model predictions needed during training (already computed in forward pass)
- No additional backward passes required

### Integration Points
- Modify training loop to sample from model predictions
- Track teacher forcing ratio as a metric
- Log ratio alongside loss for monitoring

## Limitations

Content rephrased for compliance with licensing restrictions:

The scheduled sampling approach has theoretical inconsistencies. When using model-generated history, the training target remains the original sequence's next token, which may not fit the generated context. This trains the model to partially ignore history and generate marginal token distributions. However, empirical results show improvements despite this theoretical issue.

## References

[1] Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. https://arxiv.org/abs/1506.03099

[2] Curriculum Scheduling of Recurrence. https://www.emergentmind.com/topics/curriculum-scheduling-of-recurrence

[3] Shen, J., et al. (2017). Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions. https://arxiv.org/abs/1712.05884

[4] NVIDIA Tacotron2 Issue #533: Teacher forcing. https://github.com/NVIDIA/tacotron2/issues/533
