#!/usr/bin/env python3
"""
Benchmark script to compare data loading performance
Usage: python benchmark_data_loading.py --task demo
"""
import argparse
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from sdd.dataset import PretrainTableDataset
from sdd.cached_dataset import CachedPretrainTableDataset
from sdd.memory_mapped_dataset import MemoryMappedPretrainDataset

def benchmark_dataset(dataset, name, batch_size=32, num_batches=50):
    """Benchmark dataset loading performance"""
    print(f"\n=== Benchmarking {name} ===")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=dataset.pad
    )
    
    # Warm up
    print("Warming up...")
    for i, batch in enumerate(dataloader):
        if i >= 5:  # 5 batches warmup
            break
    
    # Benchmark
    print(f"Benchmarking {num_batches} batches...")
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Simulate GPU processing time
        x_ori, x_aug, cls_indices = batch
        if torch.cuda.is_available():
            x_ori = x_ori.cuda()
            x_aug = x_aug.cuda()
        
        # Small delay to simulate model forward pass
        time.sleep(0.01)
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (i + 1) / elapsed
            print(f"  Batch {i+1}/{num_batches}, Speed: {batches_per_sec:.2f} batches/sec")
    
    total_time = time.time() - start_time
    avg_speed = num_batches / total_time
    
    print(f"Results for {name}:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average speed: {avg_speed:.2f} batches/sec")
    print(f"  Time per batch: {total_time/num_batches*1000:.1f} ms")
    
    return avg_speed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="demo")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--size", type=int, default=1000)
    
    args = parser.parse_args()
    
    # Create hyperparameters
    class HP:
        def __init__(self):
            self.task = args.task
            self.batch_size = args.batch_size
            self.max_len = 128
            self.size = args.size
            self.lr = 5e-5
            self.n_epochs = 20
            self.accumulate_grad_batches = 1
            self.lm = 'roberta'
            self.projector = 768
            self.augment_op = 'drop_col,sample_row'
            self.fp16 = False
            self.single_column = False
            self.table_order = 'column'
            self.sample_meth = 'head'
    
    hp = HP()
    
    # Determine path
    if args.task == "demo":
        path = 'data/demo/datalake'
    else:
        path = f'data/{args.task}/tables'
    
    print(f"Benchmarking data loading for task: {args.task}")
    print(f"Path: {path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    
    results = {}
    
    try:
        # 1. Standard dataset
        print("\n" + "="*50)
        print("Loading Standard Dataset...")
        start = time.time()
        standard_dataset = PretrainTableDataset.from_hp(path, hp)
        load_time = time.time() - start
        print(f"Load time: {load_time:.2f} seconds")
        
        results['standard'] = benchmark_dataset(
            standard_dataset, "Standard Dataset", args.batch_size, args.num_batches
        )
        
    except Exception as e:
        print(f"Standard dataset failed: {e}")
        results['standard'] = 0
    
    try:
        # 2. Memory-mapped dataset
        print("\n" + "="*50)
        print("Loading Memory-Mapped Dataset...")
        start = time.time()
        mmap_dataset = MemoryMappedPretrainDataset(path, hp, use_cache=True)
        load_time = time.time() - start
        print(f"Load time: {load_time:.2f} seconds")
        
        results['mmap'] = benchmark_dataset(
            mmap_dataset, "Memory-Mapped Dataset", args.batch_size, args.num_batches
        )
        
    except Exception as e:
        print(f"Memory-mapped dataset failed: {e}")
        results['mmap'] = 0
    
    try:
        # 3. Cached dataset
        print("\n" + "="*50)
        print("Loading Cached Dataset...")
        start = time.time()
        cached_dataset = CachedPretrainTableDataset.from_hp(path, hp)
        load_time = time.time() - start
        print(f"Load time: {load_time:.2f} seconds")
        
        results['cached'] = benchmark_dataset(
            cached_dataset, "Cached Dataset", args.batch_size, args.num_batches
        )
        
    except Exception as e:
        print(f"Cached dataset failed: {e}")
        results['cached'] = 0
    
    # Summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    for name, speed in results.items():
        if speed > 0:
            print(f"{name.title():15} : {speed:6.2f} batches/sec")
    
    if results['standard'] > 0:
        print("\nSpeedup vs Standard:")
        for name, speed in results.items():
            if name != 'standard' and speed > 0:
                speedup = speed / results['standard']
                print(f"{name.title():15} : {speedup:5.2f}x faster")
    
    print(f"\nRecommendation:")
    if results['cached'] > 0:
        print("Use --use_cached for best performance (requires preprocessing)")
    elif results['mmap'] > 0:
        print("Use --use_mmap for good performance without preprocessing")
    else:
        print("Use standard dataset with optimized DataLoader settings")

if __name__ == '__main__':
    main()