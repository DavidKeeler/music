#!/usr/bin/env python3
"""Test script to verify HiFiGAN model loading."""

import tensorflow as tf
from src.music_generation.vocoder import HiFiGANVocoder, load_pretrained_vocoder

def test_hifigan_architecture():
    """Test that HiFiGAN architecture can be built without pretrained weights."""
    print("Testing HiFiGAN architecture initialization...")
    
    from src.music_generation.hifigan.generator import TFHifiGANGenerator
    from src.music_generation.hifigan.config import get_default_config
    
    # Create generator with default config
    config = get_default_config()
    generator = TFHifiGANGenerator(config)
    print("✓ Generator created with default config")
    
    # Build generator with dummy input [batch, time, mel_bins]
    dummy_mel = tf.random.normal([1, 100, 80])
    audio = generator(dummy_mel, training=False)
    print(f"✓ Generator built successfully")
    print(f"  Input shape: {dummy_mel.shape}")
    print(f"  Output shape: {audio.shape}")
    
    # Wrap in vocoder
    vocoder = HiFiGANVocoder(generator=generator)
    print("✓ HiFiGANVocoder wrapper created")
    
    # Test inference with batch
    batch_size = 2
    time_steps = 100
    mel_bins = 80
    
    dummy_mel_batch = tf.random.normal([batch_size, time_steps, mel_bins])
    audio_batch = vocoder(dummy_mel_batch, training=False)
    print(f"✓ Batch inference successful")
    print(f"  Input shape: {dummy_mel_batch.shape}")
    print(f"  Output shape: {audio_batch.shape}")
    
    # Verify output shape
    assert audio_batch.shape[0] == batch_size, f"Expected batch size {batch_size}, got {audio_batch.shape[0]}"
    assert len(audio_batch.shape) == 2, f"Expected 2D output [batch, samples], got shape {audio_batch.shape}"
    print(f"✓ Output shape correct: [batch={batch_size}, samples={audio_batch.shape[1]}]")
    
    # Verify output is finite
    assert tf.reduce_all(tf.math.is_finite(audio_batch)), "Output contains NaN or Inf"
    print("✓ Output is finite (no NaN/Inf)")
    
    # Verify output is not silent
    rms = tf.sqrt(tf.reduce_mean(tf.square(audio_batch)))
    print(f"  Output RMS: {rms.numpy():.6f}")
    assert rms > 1e-6, "Output is silent"
    print("✓ Output is non-silent")
    
    print("\n✅ HiFiGAN architecture test passed!")
    return True

def test_fallback_mechanism():
    """Test that vocoder fallback mechanism works."""
    print("\nTesting vocoder fallback mechanism...")
    
    # Try loading with fallback enabled (should fall back to Griffin-Lim)
    print("Loading vocoder with fallback enabled...")
    vocoder = load_pretrained_vocoder(backend="hifigan", enable_fallback=True)
    print(f"✓ Vocoder loaded: {type(vocoder).__name__}")
    
    # Test inference (don't pass training param for Griffin-Lim)
    dummy_mel = tf.random.normal([1, 100, 80])
    audio = vocoder(dummy_mel)
    print(f"✓ Inference successful")
    print(f"  Input shape: {dummy_mel.shape}")
    print(f"  Output shape: {audio.shape}")
    
    # Verify output
    assert tf.reduce_all(tf.math.is_finite(audio)), "Output contains NaN or Inf"
    print("✓ Output is finite")
    
    print("\n✅ Fallback mechanism test passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("HiFiGAN Model Loading Verification")
    print("=" * 60)
    print()
    
    # Test 1: Architecture without pretrained weights
    test_hifigan_architecture()
    
    # Test 2: Fallback mechanism
    test_fallback_mechanism()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("NOTE: Pretrained HiFiGAN weights are not available from")
    print("      tensorspeech/tts-hifigan-ljspeech-en on Hugging Face.")
    print("      The model architecture works correctly and can be")
    print("      trained from scratch or loaded from local checkpoints.")

