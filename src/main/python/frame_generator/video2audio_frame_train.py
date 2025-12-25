"""
video2audio_frame_train.py

Minimal, self-contained implementation of:
  - Video feature encoder (stub)
  - Transformer audio generator that predicts overlapping time frames
  - Overlap-Add (OLA) layer (Hann window) to rebuild waveform from frames
  - Multi-resolution STFT loss + waveform L1 loss
  - Simple training loop (replace synthetic dataset with your own)

Requirements:
  - Python 3.8+
  - TensorFlow 2.10+ (tested on 2.10/2.11/2.12)
  - Numpy
Run:
  python video2audio_frame_train.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

# ----------------- HYPERPARAMETERS -----------------
BATCH_SIZE = 8
VIDEO_SEQ_LEN = 16       # number of video tokens / frames (stub)
VIDEO_DIM = 128
FRAME_LENGTH = 512
HOP = 128
FRAMES_PER_EXAMPLE = 8
WAVEFORM_LEN = HOP * (FRAMES_PER_EXAMPLE - 1) + FRAME_LENGTH
D_MODEL = 256
NUM_LAYERS = 3
NUM_HEADS = 4
FF_DIM = 512
LR = 1e-4
STEPS_PER_EPOCH = 200
EPOCHS = 10
CHECKPOINT_DIR = "checkpoints/vid2aud"

# ----------------- HELPERS -----------------
def hann_window(n):
    return 0.5 - 0.5 * tf.cos(2.0 * np.pi * tf.range(n, dtype=tf.float32) / tf.cast(n - 1, tf.float32))

WINDOW = hann_window(FRAME_LENGTH)  # shape (FRAME_LENGTH,)

# ----------------- OLA LAYER -----------------
class OverlapAdd(layers.Layer):
    def __init__(self, frame_length, hop, window, **kwargs):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.hop = hop
        self.window = tf.reshape(window, (1, 1, frame_length))  # (1,1,frame_length)

    def call(self, frames):
        # frames: [batch, T_hops, frame_length]
        frames_win = frames * self.window
        out = tf.signal.overlap_and_add(frames_win, frame_step=self.hop)  # [batch, L]
        win_sq = tf.reshape(self.window[0,0,:]**2, (1, 1, self.frame_length))
        denom_frames = tf.tile(win_sq, [tf.shape(frames)[0], tf.shape(frames)[1], 1])
        denom = tf.signal.overlap_and_add(denom_frames, frame_step=self.hop)
        denom = tf.maximum(denom, 1e-6)
        out = out / denom
        return out

# ----------------- TRANSFORMER BLOCK -----------------
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(d_model)])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.att(x, x)
        x = self.norm1(x + self.dropout(attn, training=training))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out, training=training))
        return x

# ----------------- VIDEO ENCODER (stub) -----------------
def build_video_encoder(seq_len, video_dim, d_model):
    inp = layers.Input(shape=(seq_len, video_dim), name="video_tokens")
    x = layers.Dense(d_model)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(d_model)(x)
    return Model(inp, x, name="video_encoder")

# ----------------- FULL MODEL -----------------
def build_model():
    video_in = layers.Input(shape=(VIDEO_SEQ_LEN, VIDEO_DIM), name="video_input")
    frame_pos = layers.Input(shape=(FRAMES_PER_EXAMPLE,), dtype=tf.int32, name="frame_pos")

    video_encoder = build_video_encoder(VIDEO_SEQ_LEN, VIDEO_DIM, D_MODEL)
    vid_feat = video_encoder(video_in)                          # [B, video_seq_len, D_MODEL]
    pooled = layers.GlobalAveragePooling1D()(vid_feat)          # [B, D_MODEL]
    token = layers.Dense(D_MODEL)(pooled)                       # [B, D_MODEL]
    tokens = layers.RepeatVector(FRAMES_PER_EXAMPLE)(token)     # [B, T, D_MODEL]

    pos_emb = layers.Embedding(input_dim=FRAMES_PER_EXAMPLE, output_dim=D_MODEL)
    pos_vecs = pos_emb(frame_pos)                               # [B, T, D_MODEL]
    x = tokens + pos_vecs

    for _ in range(NUM_LAYERS):
        x = TransformerBlock(D_MODEL, NUM_HEADS, FF_DIM)(x)

    frame_out = layers.Dense(FRAME_LENGTH, name="frame_out")(x)  # [B, T, frame_length]
    frame_out = layers.Activation('tanh')(frame_out)

    ola = OverlapAdd(FRAME_LENGTH, HOP, WINDOW)
    waveform = ola(frame_out)                                   # [B, WAVEFORM_LEN]
    return Model(inputs=[video_in, frame_pos], outputs=waveform, name="video2audio_frame_model")

# ----------------- LOSSES -----------------
def multi_res_stft_loss(y_true, y_pred, fft_sizes=(512, 1024), hop_sizes=(128, 256)):
    loss = 0.0
    for n_fft, hop in zip(fft_sizes, hop_sizes):
        st_true = tf.signal.stft(y_true, frame_length=n_fft, frame_step=hop, fft_length=n_fft,
                                 window_fn=tf.signal.hann_window)
        st_pred = tf.signal.stft(y_pred, frame_length=n_fft, frame_step=hop, fft_length=n_fft,
                                 window_fn=tf.signal.hann_window)
        mag_true = tf.abs(st_true)
        mag_pred = tf.abs(st_pred)
        loss += tf.reduce_mean(tf.abs(mag_true - mag_pred))
    return loss / len(fft_sizes)

# ----------------- DATA (REPLACE with your dataset) -----------------
def synthetic_dataset(batch_size):
    while True:
        video = np.random.randn(batch_size, VIDEO_SEQ_LEN, VIDEO_DIM).astype(np.float32)
        waveform = np.random.randn(batch_size, WAVEFORM_LEN).astype(np.float32) * 0.1
        waveform = np.tanh(waveform).astype(np.float32)
        frame_pos = np.tile(np.arange(FRAMES_PER_EXAMPLE, dtype=np.int32)[None, :], (batch_size, 1))
        yield {"video_input": video, "frame_pos": frame_pos}, waveform

def create_tf_dataset(batch_size):
    ds = tf.data.Dataset.from_generator(lambda: synthetic_dataset(batch_size),
                                        output_types=({"video_input": tf.float32, "frame_pos": tf.int32}, tf.float32),
                                        output_shapes=({"video_input": (batch_size, VIDEO_SEQ_LEN, VIDEO_DIM),
                                                        "frame_pos": (batch_size, FRAMES_PER_EXAMPLE)},
                                                       (batch_size, WAVEFORM_LEN)))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------- TRAIN LOOP -----------------
def train():
    model = build_model()
    optimizer = optimizers.Adam(learning_rate=LR)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=5)

    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            waveform_loss = tf.reduce_mean(tf.abs(targets - preds))
            stft_loss = multi_res_stft_loss(targets, preds)
            loss = waveform_loss + 0.5 * stft_loss
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, waveform_loss, stft_loss

    ds = create_tf_dataset(BATCH_SIZE)
    it = iter(ds)
    print("Model summary:")
    model.summary()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for step in range(STEPS_PER_EPOCH):
            batch_inputs, batch_targets = next(it)
            loss, wav_l1, stft_l = train_step(batch_inputs, batch_targets)
            if (step + 1) % 50 == 0 or step == 0:
                print(f" step {step+1}/{STEPS_PER_EPOCH}  loss={loss.numpy():.6f}  wav_l1={wav_l1.numpy():.6f}  stft={stft_l.numpy():.6f}")
            ckpt.step.assign_add(1)
        # save checkpoint each epoch
        manager.save()
    print("Training done. Last checkpoint saved in", CHECKPOINT_DIR)

if __name__ == "__main__":
    train()
