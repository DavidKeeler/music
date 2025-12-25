import tensorflow as tf
from audio_transformer_freq import AudioTransformerFreq

print("Creating model...")
model = AudioTransformerFreq()

print("Creating input...")
x = tf.zeros((1, 32, 128))

print("Running forward pass...")
y = model(x)

print(f"Output shape: {y.shape}")
print("Success!")
