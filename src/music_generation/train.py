"""Training script with teacher forcing for mel generator."""
import argparse
import tensorflow as tf
from pathlib import Path

from .config import (
    DATA_DIR, CACHE_DIR, CHECKPOINT_DIR,
    BATCH_SIZE, SEQ_LEN, LEARNING_RATE, NUM_EPOCHS,
    INITIAL_TF_RATIO, TF_DECAY_K, MIN_TF_RATIO
)
from .dataset import create_dataset
from .model import MelGenerator


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


class MelGeneratorTraining(tf.keras.Model):
    """Training wrapper with teacher forcing."""
    
    def __init__(self, base_model, initial_tf_ratio=INITIAL_TF_RATIO, 
                 decay_k=TF_DECAY_K, min_ratio=MIN_TF_RATIO):
        super().__init__()
        self.base_model = base_model
        self.initial_tf_ratio = initial_tf_ratio
        self.decay_k = decay_k
        self.min_ratio = min_ratio
    
    def compute_tf_ratio(self):
        step = tf.cast(self.optimizer.iterations, tf.float32)
        ratio = self.initial_tf_ratio * tf.exp(-self.decay_k * step)
        return tf.maximum(self.min_ratio, ratio)
    
    def train_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        tf_ratio = self.compute_tf_ratio()
        
        with tf.GradientTape() as tape:
            ar_input = x[:, :1, :]
            preds = []
            
            for t in range(1, seq_len):
                pred = self.base_model(ar_input, training=True)[:, -1:, :]
                preds.append(pred)
                use_teacher = tf.random.uniform([batch_size, 1, 1]) < tf_ratio
                next_input = tf.where(use_teacher, x[:, t:t+1, :], pred)
                ar_input = tf.concat([ar_input, next_input], axis=1)
            
            pred_seq = tf.concat(preds, axis=1)
            loss = tf.reduce_mean(tf.abs(pred_seq - y[:, 1:, :]))
        
        grads = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        
        return {"loss": loss, "tf_ratio": tf_ratio}


def train(data_dir, cache_dir, checkpoint_dir, batch_size=BATCH_SIZE, 
          epochs=NUM_EPOCHS, lr=LEARNING_RATE, resume_from=None):
    """Train the mel generator model."""
    
    print(f"Loading dataset from {data_dir}")
    dataset = create_dataset(data_dir, cache_dir, batch_size)
    
    steps_per_epoch = 100  # Adjust based on dataset size
    total_steps = steps_per_epoch * epochs
    
    lr_schedule = WarmupCosineSchedule(lr, warmup_steps=5 * steps_per_epoch, total_steps=total_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5)
    
    base_model = MelGenerator()
    model = MelGeneratorTraining(base_model)
    model.compile(optimizer=optimizer)
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "mel_generator"
    
    if resume_from:
        print(f"Resuming from {resume_from}")
        base_model.load_weights(resume_from)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=checkpoint_dir / "logs"),
    ]
    
    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    
    print(f"Training complete. Model saved to {checkpoint_path}")


def main():
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
