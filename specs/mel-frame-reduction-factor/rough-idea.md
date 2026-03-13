### Predicting Multiple Mel Frames per Step (Reduction Factor)

To improve training speed, stability, and generation quality, the mel generator should predict **multiple frames at each autoregressive step** instead of a single frame. This technique is commonly called a **reduction factor** or **frame grouping**.

#### Motivation

With the current setup, the model predicts one frame at a time:

```
mel[t] → predict mel[t+1]
```

At a hop length of 256 samples and a sample rate of 22,050 Hz, each mel frame represents ~11.6 ms of audio. Predicting single frames forces the model to learn extremely small temporal changes, which leads to:

* slow convergence
* unstable autoregressive generation
* excessive sequence lengths

Predicting multiple frames per step solves these issues by increasing the temporal granularity of each prediction.

---

#### Reduction Factor

Define a reduction factor `R` (typically 2–4).
At each step the model predicts `R` consecutive mel frames:

```
mel[t] → predict mel[t+1 : t+R]
```

For example, with:

```
R = 4
N_MELS = 80
```

the output dimension becomes:

```
80 × 4 = 320
```

Instead of predicting `[B, T, 80]`, the model predicts:

```
[B, T, R × 80]
```

These frames are then reshaped to reconstruct the full sequence.

---

#### Benefits

1. **Shorter effective sequence length**

Original sequence length:

```
T = 512 frames
```

With `R = 4`:

```
T' = 512 / 4 = 128 tokens
```

This reduces transformer compute and memory significantly.

2. **Faster convergence**

Predicting larger temporal chunks provides stronger training signals and reduces the difficulty of modeling extremely small frame-to-frame changes.

3. **Improved autoregressive stability**

Prediction errors accumulate less frequently during generation since each step covers a longer segment of audio.

---

#### Model Changes

**Output projection**

The final projection layer must produce `R × N_MELS` channels:

```
Dense(D_MODEL → R × N_MELS)
```
