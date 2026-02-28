"""Pretrained vocoder finetuning script using HiFi-GAN."""

import argparse
import os
from pathlib import Path
import tensorflow as tf
from src.music_generation import vocoder
from src.music_generation import losses
from src.music_generation import config


class VocoderTraining(tf.keras.Model):
    """Wrapper for vocoder training with custom train_step."""
    
    def __init__(self, generator, stft_loss):
        super().__init__()
        self.generator = generator
        self.stft_loss = stft_loss
    
    def call(self, mel_spectrogram, training=False):
        return self.generator(mel_spectrogram, training=training)
    
    def train_step(self, data):
        mel, target_audio = data
        
        with tf.GradientTape() as tape:
            # Generate audio from mel
            pred_audio = self.generator(mel, training=True)
            
            # Trim to same length
            min_len = tf.minimum(tf.shape(pred_audio)[-1], tf.shape(target_audio)[-1])
            pred_audio = pred_audio[..., :min_len]
            target_audio = target_audio[..., :min_len]
            
            # Compute STFT loss
            loss = self.stft_loss(pred_audio, target_audio)
        
        # Compute gradients and update
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        
        return {"loss": loss}


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
    
    # Compile with optimizer
    training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    
    # Callbacks
    checkpoint_path = Path(checkpoint_dir) / "checkpoint_epoch_{epoch:02d}.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            save_weights_only=True,
            save_freq='epoch'
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(Path(checkpoint_dir) / "logs"))
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
    parser = argparse.ArgumentParser(description="Finetune pretrained HiFi-GAN vocoder")
    
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
    parser.add_argument('--model_url', type=str, default=None,
                        help='TensorFlow Hub model URL (optional)')
    
    return parser.parse_args(args)


def main():
    """Main training function."""
    args = parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load pretrained vocoder
    print("Loading pretrained HiFi-GAN...")
    vocoder_model = vocoder.load_pretrained_vocoder(args.model_url)
    
    if vocoder_model.generator is None:
        raise ValueError(
            "No pretrained model loaded. Provide --model_url or implement custom generator loading."
        )
    
    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = vocoder.VocoderDataset(
        args.data_dir,
        segment_length=config.VOCODER_SEGMENT_LENGTH
    )
    train_dataset = dataset.create_dataset(batch_size=args.batch_size, shuffle=True)
    
    # Train
    train_vocoder(
        vocoder_model.generator,
        train_dataset,
        args.epochs,
        args.lr,
        args.checkpoint_dir
    )
    
    print("Training complete!")


if __name__ == '__main__':
    main()
