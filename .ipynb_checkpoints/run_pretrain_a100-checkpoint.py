#!/usr/bin/env python3
"""
Optimized pre-training script for 40GB A100 GPU
Usage: python run_pretrain_a100.py --task santos --fp16
"""
import argparse
import subprocess
import sys

def get_a100_presets():
    """Return A100-optimized training presets"""
    return {
        'small': {
            'batch_size': 128,
            'accumulate_grad_batches': 2,
            'lr': 8e-5,  # scaled up for larger effective batch size
        },
        'medium': {
            'batch_size': 96,
            'accumulate_grad_batches': 4,
            'lr': 1e-4,
        },
        'large': {
            'batch_size': 64,
            'accumulate_grad_batches': 8,
            'lr': 1.5e-4,
        },
        'santos': {
            'batch_size': 128,
            'accumulate_grad_batches': 2,
            'lr': 8e-5,
        },
        'tus': {
            'batch_size': 96,
            'accumulate_grad_batches': 4,
            'lr': 1e-4,
        }
    }

def main():
    parser = argparse.ArgumentParser(description='A100-optimized pre-training')
    parser.add_argument("--task", type=str, default="santos", 
                       choices=['small', 'medium', 'large', 'santos', 'tus', 'demo'],
                       help="Task/dataset to train on")
    parser.add_argument("--preset", type=str, choices=['small', 'medium', 'large'],
                       help="Use predefined A100 preset (overrides task preset)")
    parser.add_argument("--custom", action="store_true",
                       help="Use custom parameters instead of presets")
    parser.add_argument("--batch_size", type=int, help="Custom batch size")
    parser.add_argument("--accumulate_grad_batches", type=int, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--run_id", type=int, default=0, help="Run ID")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--logdir", type=str, default="results/", help="Log directory")
    
    args = parser.parse_args()
    
    # Get presets
    presets = get_a100_presets()
    
    if args.custom:
        if not all([args.batch_size, args.accumulate_grad_batches, args.lr]):
            print("Error: --custom requires --batch_size, --accumulate_grad_batches, and --lr")
            sys.exit(1)
        config = {
            'batch_size': args.batch_size,
            'accumulate_grad_batches': args.accumulate_grad_batches,
            'lr': args.lr
        }
    else:
        preset_name = args.preset or args.task
        if preset_name not in presets:
            print(f"No preset found for {preset_name}, using 'small' preset")
            preset_name = 'small'
        config = presets[preset_name]
    
    # Build command
    cmd = [
        'python', 'run_pretrain.py',
        '--task', args.task,
        '--batch_size', str(config['batch_size']),
        '--accumulate_grad_batches', str(config['accumulate_grad_batches']),
        '--lr', str(config['lr']),
        '--n_epochs', str(args.n_epochs),
        '--run_id', str(args.run_id),
        '--logdir', args.logdir,
        '--save_model'
    ]
    
    if args.fp16:
        cmd.append('--fp16')
    
    print(f"Running A100-optimized training with config: {config}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Effective batch size: {config['batch_size'] * config['accumulate_grad_batches']}")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == '__main__':
    main()