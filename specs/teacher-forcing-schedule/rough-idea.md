# Rough Idea

We're using pure teacher forcing in the mel training. This is going to be a problem when we do full autoregressive inference. We need to slowly ramp down the teacher forcing during training.

## Context

The current music generation system uses a Transformer-based mel spectrogram generator trained with teacher forcing. During training, the model receives ground truth previous frames, but during inference, it must use its own predictions. This train-test mismatch (exposure bias) can lead to error accumulation and poor generation quality.

## Goal

Implement a teacher forcing schedule that gradually reduces the teacher forcing ratio during training, allowing the model to learn to handle its own predictions and improve autoregressive inference quality.
