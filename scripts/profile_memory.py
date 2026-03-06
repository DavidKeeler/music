#!/usr/bin/env python3
"""Memory profiling script for TensorFlow music generation training.

Usage:
    python scripts/profile_memory.py --data_dir ~/data/music/musicnet/train_data --epochs 2 --batch_size 4
"""

import argparse
import tracemalloc
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.music_generation.train import train


def format_bytes(size):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def profile_training(args):
    """Profile memory usage during training."""
    print("Starting memory profiling...")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 60)
    
    # Start memory tracking
    tracemalloc.start()
    
    # Get baseline memory
    baseline = tracemalloc.get_traced_memory()[0]
    print(f"Baseline memory: {format_bytes(baseline)}")
    
    try:
        # Run training
        train(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        print("-" * 60)
        print(f"Current memory: {format_bytes(current)}")
        print(f"Peak memory: {format_bytes(peak)}")
        print(f"Memory increase: {format_bytes(peak - baseline)}")
        
        # Get top memory allocations
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("\nTop 10 memory allocations:")
        for stat in top_stats[:10]:
            print(f"{stat}")
            
    finally:
        tracemalloc.stop()


def main():
    parser = argparse.ArgumentParser(description='Profile memory usage during training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching preprocessed data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    
    args = parser.parse_args()
    profile_training(args)


if __name__ == '__main__':
    main()
