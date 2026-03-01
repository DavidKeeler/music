# TensorFlow SavedModel Format

## Overview

SavedModel is TensorFlow's recommended format for saving and loading models. It includes the model architecture, weights, optimizer state, and training configuration in a portable format.

## Saving Models

### Full Model (Recommended)

```python
# Save entire model
model.save('path/to/model')  # Creates a directory

# Or explicitly specify format
model.save('path/to/model', save_format='tf')
```

This creates a directory structure:
```
model/
├── saved_model.pb          # Model architecture and graph
├── variables/
│   ├── variables.data-*    # Model weights
│   └── variables.index
└── assets/                 # Additional files (if any)
```

### Weights Only

```python
# Save only weights
model.save_weights('path/to/weights.h5')

# Or in TensorFlow format
model.save_weights('path/to/weights')
```

## Loading Models

### Full Model

```python
# Load complete model (architecture + weights + optimizer)
model = tf.keras.models.load_model('path/to/model')

# Model is ready to use immediately
predictions = model.predict(data)

# Can continue training
model.fit(data, labels, epochs=5)
```

### Weights Only

```python
# Must recreate model architecture first
model = create_model()  # Your model definition

# Then load weights
model.load_weights('path/to/weights.h5')
```

## Benefits of SavedModel Format

1. **Portable:** No need for original source code
2. **Complete:** Includes architecture, weights, optimizer state
3. **Production-ready:** Works with TensorFlow Serving, TensorFlow Lite
4. **Versioning:** Easy to version and deploy
5. **Finetuning-friendly:** Can load and continue training immediately

## Checkpointing During Training

```python
# Save model every epoch
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:02d}',
    save_freq='epoch',
    save_weights_only=False  # Save full model
)

model.fit(data, labels, epochs=10, callbacks=[checkpoint_callback])
```

## Best Practices for Finetuning

1. **Use SavedModel format** - Easier to load and resume training
2. **Save full model** - Not just weights, includes optimizer state
3. **Version checkpoints** - Use epoch numbers or timestamps in filenames
4. **Test loading** - Verify checkpoints can be loaded before long training runs

## Example: Finetuning Workflow

```python
# Save pretrained model
pretrained_model.save('models/pretrained')

# Later: Load and finetune
model = tf.keras.models.load_model('models/pretrained')

# Optionally freeze some layers
for layer in model.layers[:5]:
    layer.trainable = False

# Compile with new optimizer/learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='mse')

# Continue training
model.fit(new_data, new_labels, epochs=10)

# Save finetuned model
model.save('models/finetuned')
```

## References

- TensorFlow Guide: https://www.tensorflow.org/guide/saved_model
- Keras Serialization: https://www.tensorflow.org/guide/keras/serialization_and_saving
