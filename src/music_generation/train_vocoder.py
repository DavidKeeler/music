"""Pretrained vocoder finetuning script using HiFi-GAN."""

import argparse
import os
from pathlib import Path
import tensorflow as tf

# Suppress verbose TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
tf.get_logger().setLevel('ERROR')

from src.music_generation import vocoder
from src.music_generation import losses
from src.music_generation import config
from src.music_generation.hifigan.generator import TFHifiGANGenerator
from src.music_generation.hifigan.config import get_default_config


class VocoderTraining(tf.keras.Model):
    """Wrapper for vocoder training with custom train_step."""
    
    def __init__(self, generator, stft_loss):
        super().__init__()
        self.generator = generator
        self.stft_loss = stft_loss
        self.grad_norm_tracker = tf.keras.metrics.Mean(name="grad_norm")
    
    def call(self, mel_spectrogram, training=False):
        return self.generator(mel_spectrogram, training=training)
    
    def train_step(self, data):
        mel, target_audio = data
        
        with tf.GradientTape() as tape:
            # Generate audio from mel
            # For HiFiGAN, call hifigan directly to avoid @tf.function decorator
            # For other generators, call normally
            if hasattr(self.generator, 'hifigan'):
                pred_audio = self.generator.hifigan(mel, training=True)
            else:
                pred_audio = self.generator(mel, training=True)
            
            # Squeeze channel dimension if present [B, T, 1] -> [B, T]
            if len(pred_audio.shape) == 3 and pred_audio.shape[-1] == 1:
                pred_audio = tf.squeeze(pred_audio, axis=-1)
            
            # Trim to same length
            min_len = tf.minimum(tf.shape(pred_audio)[-1], tf.shape(target_audio)[-1])
            pred_audio = pred_audio[..., :min_len]
            target_audio = target_audio[..., :min_len]
            
            # Compute STFT loss
            loss = self.stft_loss(pred_audio, target_audio)
        
        # Compute gradients and update
        grads = tape.gradient(loss, self.generator.trainable_variables)
        
        # Debug: check if gradients are None
        if all(g is None for g in grads):
            raise ValueError(f"All gradients are None! Loss: {loss}, trainable_vars: {len(self.generator.trainable_variables)}")
        
        # Filter out None gradients
        grads_and_vars = [(g, v) for g, v in zip(grads, self.generator.trainable_variables) if g is not None]
        
        # Compute gradient norm
        grad_squares = [tf.reduce_sum(g**2) for g, _ in grads_and_vars]
        grad_norm = tf.sqrt(tf.add_n(grad_squares)) if grad_squares else tf.constant(0.0)
        
        self.optimizer.apply_gradients(grads_and_vars)
        
        # Update metrics
        self.grad_norm_tracker.update_state(grad_norm)
        
        return {"loss": loss, "grad_norm": self.grad_norm_tracker.result()}
    
    @property
    def metrics(self):
        return [self.grad_norm_tracker]


def train_vocoder(
    generator,
    train_dataset,
    epochs: int,
    lr: float,
    checkpoint_dir: str
):
    """Train vocoder with STFT loss.
    
    Args:
        generator: Pretrained HiFi-GAN generator
        train_dataset: tf.data.Dataset for training
        epochs: Number of epochs
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
    """
    # Create STFT loss
    stft_loss = losses.MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048]
    )
    
    # Wrap in training model
    training_model = VocoderTraining(generator, stft_loss)
    
    # Build model with dummy input
    dummy_mel = tf.random.normal([1, 100, 80])
    _ = training_model(dummy_mel, training=False)
    
    # Compile with optimizer
    training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    
    # Load latest checkpoint if exists
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir_path.glob("checkpoint_epoch_*.weights.h5"))
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")
        try:
            training_model.load_weights(str(latest_checkpoint))
        except (ValueError, Exception) as e:
            print(f"⚠ Could not load checkpoint (architecture changed?): {e}")
            print("Training from scratch.")
    
    # Callbacks
    checkpoint_path = checkpoint_dir_path / "checkpoint_epoch_{epoch:02d}.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            save_weights_only=True,
            save_freq='epoch'
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(checkpoint_dir_path / "logs"))
    ]
    
    # Train
    print("Starting training...")
    training_model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks
    )


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Finetune pretrained vocoder")
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Music data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/vocoder',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--backend', type=str, default='hifigan',
                        choices=['hifigan', 'vocos', 'griffin-lim', 'melgan'],
                        help='Vocoder backend to use')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained checkpoint (optional, downloads if not provided)')
    
    return parser.parse_args(args)


def main():
    """Main training function."""
    args = parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create or load vocoder
    print(f"Creating {args.backend} vocoder...")
    if args.backend == 'hifigan':
        # Create HiFiGAN from scratch (pretrained weights unavailable)
        hifigan_config = get_default_config()
        generator = TFHifiGANGenerator(hifigan_config)
        print("✓ HiFiGAN generator created from scratch")
    elif args.backend == 'vocos':
        vocoder_model = vocoder.load_vocos_vocoder()
        generator = vocoder_model.model if hasattr(vocoder_model, 'model') else vocoder_model
    else:
        # Use fallback chain for other backends
        vocoder_model = vocoder.load_pretrained_vocoder(
            backend=args.backend,
            enable_fallback=False
        )
        # Extract generator for training
        if hasattr(vocoder_model, 'generator') and vocoder_model.generator is not None:
            generator = vocoder_model.generator
        elif hasattr(vocoder_model, 'model'):
            generator = vocoder_model.model
        else:
            generator = vocoder_model
    
    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = vocoder.VocoderDataset(
        args.data_dir,
        segment_length=config.VOCODER_SEGMENT_LENGTH
    )
    train_dataset = dataset.create_dataset(batch_size=args.batch_size, shuffle=True)
    
    # Train
    train_vocoder(
        generator,
        train_dataset,
        args.epochs,
        args.lr,
        args.checkpoint_dir
    )
    
    print("Training complete!")


if __name__ == '__main__':
    main()
