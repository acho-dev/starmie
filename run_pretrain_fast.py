#!/usr/bin/env python3
"""
Fast pre-training script that eliminates GPU idle time
Usage: python run_pretrain_fast.py --task santos --use_cached --fp16
"""
import argparse
import numpy as np
import random
import torch
import mlflow
import time
import logging

from sdd.cached_dataset import CachedPretrainTableDataset
from sdd.memory_mapped_dataset import MemoryMappedPretrainDataset  
from sdd.dataset import PretrainTableDataset
from sdd.pretrain import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fast pre-training with optimized data loading")
    parser.add_argument("--task", type=str, default="santos")
    parser.add_argument("--logdir", type=str, default="results_fast/")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=8e-5)  # Increased for larger effective batch
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='drop_col,sample_row')
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--table_order", type=str, default='column')
    parser.add_argument("--sample_meth", type=str, default='head')
    parser.add_argument("--mlflow_tag", type=str, default=None)
    
    # Data loading optimization options
    parser.add_argument("--use_cached", action="store_true", 
                       help="Use pre-cached dataset (fastest, requires preprocessing)")
    parser.add_argument("--use_mmap", action="store_true",
                       help="Use memory-mapped dataset (fast I/O)")
    parser.add_argument("--create_cache", action="store_true",
                       help="Create cache and exit (preprocessing step)")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Custom cache directory")

    hp = parser.parse_args()

    # MLflow logging
    for variable in ["task", "batch_size", "lr", "n_epochs", "augment_op", "sample_meth", "table_order", "accumulate_grad_batches"]:
        mlflow.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)

    # Set seed
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine data path
    if "santos" in hp.task:
        path = 'data/%s/datalake' % hp.task
        if hp.task == "santosLarge":
            path = 'data/santos-benchmark/real-benchmark/datalake'
    elif "tus" in hp.task:
        path = 'data/table-union-search-benchmark/small/benchmark'
        if hp.task == "tusLarge":
            path = 'data/table-union-search-benchmark/large/benchmark'
    elif hp.task == "demo":
        path = 'data/demo/datalake'
    else:
        path = 'data/%s/tables' % hp.task

    # Create dataset with optimization
    start_time = time.time()
    
    if hp.create_cache:
        logger.info("Creating cache and exiting...")
        trainset = CachedPretrainTableDataset.from_hp(path, hp)
        logger.info(f"Cache created in {time.time() - start_time:.2f} seconds")
        exit(0)
    
    elif hp.use_cached:
        logger.info("Using cached dataset...")
        trainset = CachedPretrainTableDataset.from_hp(path, hp)
        logger.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
        
    elif hp.use_mmap:
        logger.info("Using memory-mapped dataset...")
        trainset = MemoryMappedPretrainDataset(path, hp)
        logger.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
        
    else:
        logger.info("Using standard dataset...")
        trainset = PretrainTableDataset.from_hp(path, hp)
        logger.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

    # Log dataset type for tracking
    mlflow.log_param("dataset_type", 
                    "cached" if hp.use_cached else 
                    "mmap" if hp.use_mmap else "standard")

    # Start training
    logger.info("Starting training...")
    train_start = time.time()
    train(trainset, hp)
    train_time = time.time() - train_start
    
    logger.info(f"Training completed in {train_time:.2f} seconds")
    mlflow.log_metric("total_train_time", train_time)