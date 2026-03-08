#!/usr/bin/env python3
"""Test single training step for vocoder training."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from pathlib import Path
from src.music_generation.vocoder import HiFiGANVocoder, load_pretrained_vocoder, VocoderDataset
from src.music_generation.losses import MultiResolutionSTFTLoss
from src.music_generation.train_vocoder import VocoderTraining

def test_single_training_step():
    """Run one training step and verify loss computation and weight updates."""
    
    print("=" * 60)
    print("Testing Single Training Step")
    print("=" * 60)
    
    # Load vocoder - build HiFiGAN from scratch for training test
    print("\n1. Building HiFiGAN vocoder from scratch...")
    from src.music_generation.hifigan.generator import TFHifiGANGenerator
    from src.music_generation.hifigan.config import get_default_config
    
    config = get_default_config()
    generator = TFHifiGANGenerator(config)
    
    # Build with dummy input
    dummy_mel = tf.random.normal([1, 100, 80])
    _ = generator(dummy_mel, training=False)
    print(f"   Built: TFHifiGANGenerator")
    
    # Create dataset
    print("\n2. Creating dataset...")
    data_dir = Path.home() / 'data' / 'music' / 'musicnet' / 'train_data'
    dataset = VocoderDataset(data_dir=data_dir)
    ds = dataset.create_dataset(batch_size=2, shuffle=True)
    
    # Get one batch
    print("   Getting one batch...")
    batch = next(iter(ds.take(1)))
    mel, audio = batch
    print(f"   Mel shape: {mel.shape}, Audio shape: {audio.shape}")
    
    # Create training wrapper
    print("\n3. Creating training wrapper...")
    loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048]
    )
    training_model = VocoderTraining(generator, loss_fn)
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    training_model.compile(optimizer=optimizer)
    
    # Get initial weights
    print("\n4. Capturing initial weights...")
    initial_weights = [w.numpy().copy() for w in training_model.trainable_weights]
    print(f"   Number of trainable weights: {len(initial_weights)}")
    
    # Run one training step
    print("\n5. Running training step...")
    
    # Debug: check if generator is being watched
    print(f"   Generator trainable: {len(training_model.generator.trainable_variables)} variables")
    
    result = training_model.train_step((mel, audio))
    
    print(f"   Loss: {result['loss'].numpy():.6f}")
    print(f"   Spectral loss: {result['spectral_loss'].numpy():.6f}")
    print(f"   Gradient norm: {result['grad_norm'].numpy():.6f}")
    
    # Verify loss is finite
    assert tf.math.is_finite(result['loss']), "Loss is not finite!"
    assert tf.math.is_finite(result['spectral_loss']), "Spectral loss is not finite!"
    assert tf.math.is_finite(result['grad_norm']), "Gradient norm is not finite!"
    print("   ✅ All losses are finite")
    
    # Verify weights updated
    print("\n6. Verifying weight updates...")
    weights_changed = False
    for i, (w_init, w_current) in enumerate(zip(initial_weights, training_model.trainable_weights)):
        w_curr_np = w_current.numpy()
        if not tf.reduce_all(tf.equal(w_init, w_curr_np)):
            weights_changed = True
            max_diff = tf.reduce_max(tf.abs(w_init - w_curr_np)).numpy()
            print(f"   Weight {i}: max diff = {max_diff:.6e}")
            break
    
    assert weights_changed, "Weights did not update!"
    print("   ✅ Weights updated successfully")
    
    print("\n" + "=" * 60)
    print("✅ Single training step test PASSED")
    print("=" * 60)

if __name__ == '__main__':
    test_single_training_step()
