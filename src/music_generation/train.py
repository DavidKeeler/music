"""Training script with teacher forcing for mel generator."""
import argparse
import logging
import tensorflow as tf
from pathlib import Path
import psutil

from .config import (
    DATA_DIR, CACHE_DIR, CHECKPOINT_DIR,
    BATCH_SIZE, SEQ_LEN, LEARNING_RATE, NUM_EPOCHS
)
from .dataset import create_dataset
from .model import MelGenerator

logger = logging.getLogger(__name__)


def configure_memory():
    """Configure TensorFlow memory settings for resource-constrained environments."""
    # Force CPU-only on macOS to avoid GPU freezing issues
    tf.config.set_visible_devices([], 'GPU')
    logger.info("Configured TensorFlow to use CPU only")


def check_batch_size(batch_size):
    """Warn if batch size is too large for available system memory."""
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    if batch_size > 8 and ram_gb < 16:
        if ram_gb < 8:
            suggested = 2
        else:
            suggested = 4
        
        logger.warning(
            f"Batch size {batch_size} may be too large for {ram_gb:.1f}GB RAM. "
            f"Consider using batch_size={suggested} to avoid OOM errors."
        )



class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup and cosine decay."""
    
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-6):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))
        progress = (step - self.warmup_steps) / tf.maximum(1.0, self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + tf.cos(3.14159 * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
    
    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr
        }


class MelGeneratorTraining(tf.keras.Model):
    """Simplified training wrapper with parallel processing."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def get_config(self):
        """Return config for serialization."""
        return {
            "base_model": tf.keras.utils.serialize_keras_object(self.base_model)
        }
    
    @classmethod
    def from_config(cls, config):
        """Reconstruct from config."""
        base_model = tf.keras.utils.deserialize_keras_object(config["base_model"])
        return cls(base_model)
    
    def train_step(self, data):
        x, y = data
        
        # Shape validation
        tf.debugging.assert_equal(tf.shape(x), tf.shape(y))
        
        with tf.GradientTape() as tape:
            preds = self.base_model(x, training=True)
            
            # Validate prediction shape
            tf.debugging.assert_equal(tf.shape(preds), tf.shape(x))
            
            # Loss: compare predictions at t with targets at t+1
            loss = tf.reduce_mean(tf.abs(preds[:, :-1, :] - y[:, 1:, :]))
            
            # NaN detection (use tf.cond for graph compatibility)
            def warn_nan():
                tf.print("⚠️  WARNING: NaN/Inf loss detected!")
                return tf.constant(0)
            
            tf.cond(
                tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss)),
                warn_nan,
                lambda: tf.constant(0)
            )
        
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {"loss": loss}


def train(data_dir, cache_dir, checkpoint_dir, batch_size=BATCH_SIZE, 
          epochs=NUM_EPOCHS, lr=LEARNING_RATE, resume_from=None):
    """Train the mel generator model."""
    
    check_batch_size(batch_size)
    
    print(f"Loading dataset from {data_dir}")
    dataset = create_dataset(data_dir, cache_dir, batch_size)
    
    # Validate dataset shapes
    print("Validating dataset shapes...")
    for x, y in dataset.take(1):
        print(f"  Input shape: {x.shape}")
        print(f"  Target shape: {y.shape}")
        tf.debugging.assert_equal(tf.shape(x)[0], batch_size, message="Batch size mismatch")
        tf.debugging.assert_equal(tf.shape(x)[2], 80, message="Mel channels should be 80")
        tf.debugging.assert_equal(tf.shape(x), tf.shape(y), message="Input and target shapes must match")
    print("✓ Dataset validation passed")
    
    steps_per_epoch = 100  # Adjust based on dataset size
    total_steps = steps_per_epoch * epochs
    
    lr_schedule = WarmupCosineSchedule(lr, warmup_steps=5 * steps_per_epoch, total_steps=total_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5)
    
    base_model = MelGenerator()
    model = MelGeneratorTraining(base_model)
    model.compile(optimizer=optimizer)
    model.summary()
    
    # Verify model builds correctly with sample batch
    print("Verifying model build with sample batch...")
    for x, y in dataset.take(1):
        try:
            predictions = base_model(x, training=False)
            print(f"  Sample input shape: {x.shape}")
            print(f"  Sample output shape: {predictions.shape}")
            tf.debugging.assert_equal(tf.shape(predictions), tf.shape(x), 
                                     message="Output shape must match input shape")
            print("✓ Model build verification passed")
        except Exception as e:
            print(f"✗ Model build verification failed: {e}")
            raise
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "mel_generator.keras"
    
    if resume_from:
        print(f"Resuming from {resume_from}")
        base_model.load_weights(resume_from)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_weights_only=False, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir / "logs"),
    ]
    
    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    
    # Validate checkpoint was saved
    print("Validating checkpoint...")
    if checkpoint_path.exists():
        print(f"✓ Checkpoint saved successfully: {checkpoint_path}")
        print(f"  Checkpoint size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"✗ Checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError(f"Expected checkpoint at {checkpoint_path}")
    
    print(f"Training complete. Model saved to {checkpoint_path}")


def main():
    configure_memory()
    
    parser = argparse.ArgumentParser(description="Train mel generator")
    parser.add_argument('--data_dir', type=Path, default=DATA_DIR)
    parser.add_argument('--cache_dir', type=Path, default=CACHE_DIR)
    parser.add_argument('--checkpoint_dir', type=Path, default=CHECKPOINT_DIR)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--resume', type=Path, default=None)
    
    args = parser.parse_args()
    train(args.data_dir, args.cache_dir, args.checkpoint_dir, 
          args.batch_size, args.epochs, args.lr, args.resume)


if __name__ == '__main__':
    main()
