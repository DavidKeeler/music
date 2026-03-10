#!/usr/bin/env python3
"""Verify checkpoint compatibility with refactored training code."""

import sys
from pathlib import Path
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.music_generation.model import MelGenerator
from src.music_generation.train import MelGeneratorTraining
from src.music_generation import config


def verify_checkpoint(checkpoint_path: Path):
    """Verify that a checkpoint can be loaded and used."""
    print(f"\n{'='*60}")
    print(f"Verifying checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Try loading as full model first (includes training wrapper)
    print("Attempting to load as full model (with training wrapper)...")
    try:
        model = tf.keras.models.load_model(str(checkpoint_path))
        print(f"✓ Full model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Parameters: {model.count_params():,}")
        
        # Verify inference works
        print("\nVerifying model inference...")
        test_input = tf.random.normal([2, 50, config.N_MELS])
        output = model.base_model(test_input, training=False)
        print(f"✓ Inference successful: input {test_input.shape} -> output {output.shape}")
        
        # Verify training step works
        print("\nVerifying training step...")
        x = tf.random.normal([2, 50, config.N_MELS])
        y = tf.random.normal([2, 50, config.N_MELS])
        result = model.train_step((x, y))
        print(f"✓ Training step works: loss={result['loss']:.4f}, "
              f"grad_norm={result['grad_norm']:.4f}, tf_ratio={result['tf_ratio']:.4f}")
        
        # Verify variables are preserved
        print("\nVerifying training state variables...")
        assert hasattr(model, 'tf_ratio'), "Missing tf_ratio variable"
        assert hasattr(model, 'training_step'), "Missing training_step variable"
        print(f"✓ Variables preserved: tf_ratio={model.tf_ratio.numpy():.4f}, "
              f"training_step={model.training_step.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load as full model: {e}")
        print("\nAttempting to load weights only...")
        
        # Fall back to loading weights only
        try:
            base_model = MelGenerator()
            dummy_input = tf.random.normal([1, 10, config.N_MELS])
            _ = base_model(dummy_input, training=False)
            print(f"Created model with {base_model.count_params():,} parameters")
            
            base_model.load_weights(str(checkpoint_path))
            print("✓ Weights loaded successfully")
            
            # Verify inference
            test_input = tf.random.normal([2, 50, config.N_MELS])
            output = base_model(test_input, training=False)
            print(f"✓ Inference successful: input {test_input.shape} -> output {output.shape}")
            
            # Verify training wrapper works
            training_model = MelGeneratorTraining(
                base_model=base_model,
                initial_tf_ratio=config.INITIAL_TF_RATIO,
                min_tf_ratio=config.MIN_TF_RATIO,
                decay_k=config.TF_DECAY_K,
                warmup_steps=config.TF_WARMUP_STEPS,
            )
            training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
            
            x = tf.random.normal([2, 50, config.N_MELS])
            y = tf.random.normal([2, 50, config.N_MELS])
            result = training_model.train_step((x, y))
            print(f"✓ Training wrapper works: loss={result['loss']:.4f}")
            
            return True
            
        except Exception as e2:
            print(f"✗ Failed to load weights: {e2}")
            return False


def main():
    """Main verification routine."""
    checkpoint_paths = [
        Path("checkpoints/mel_generator.keras"),
        Path("checkpoints/model.weights.h5"),  # Extracted weights
    ]
    
    results = {}
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.exists():
            print(f"\nSkipping {checkpoint_path} (not found)")
            continue
        
        success = verify_checkpoint(checkpoint_path)
        results[checkpoint_path] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    for checkpoint_path, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {checkpoint_path}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All checkpoints are compatible with refactored code")
        return 0
    else:
        print("\n✗ Some checkpoints failed compatibility check")
        return 1


if __name__ == "__main__":
    sys.exit(main())
