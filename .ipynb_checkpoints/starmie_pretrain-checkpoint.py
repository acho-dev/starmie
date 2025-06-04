#!/usr/bin/env python3
"""
Starmie Pretrain - Train embeddings for CSV table discovery

Usage:
    python starmie_pretrain.py \
        --datalake_path /path/to/csv/files \
        --epochs 3 \
        --batch_size 64

This should be run before using starmie_search.py to search for similar tables.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import shutil

class StarmiePretrain:
    def __init__(self, datalake_path, cache_dir=None, results_dir=None):
        self.datalake_path = Path(datalake_path)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./starmie_cache")
        self.results_dir = Path(results_dir) if results_dir else Path("./results")
        self.dataset_name = f"dataset_{abs(hash(str(self.datalake_path)))}"
        self.setup_directories()
        
    def setup_directories(self):
        """Setup required directory structure"""
        self.cache_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.data_dir = Path("data") / self.dataset_name
        self.tables_dir = self.data_dir / "tables"
        self.datalake_dir = self.data_dir / "datalake"
        self.query_dir = self.data_dir / "query"
        self.vectors_dir = self.data_dir / "vectors"
        
        for dir_path in [self.tables_dir, self.datalake_dir, self.query_dir, self.vectors_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def prepare_datalake(self):
        """Copy or symlink CSV files to the expected structure"""
        print(f"📁 Preparing data lake from {self.datalake_path}")
        
        # Clear existing directories
        for dir_path in [self.tables_dir, self.datalake_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True)
        
        # Find all CSV files
        csv_files = list(self.datalake_path.glob("**/*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.datalake_path}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Copy or symlink CSV files to both tables/ and datalake/ directories
        for csv_file in csv_files:
            for dest_dir in [self.tables_dir, self.datalake_dir]:
                dest_file = dest_dir / csv_file.name
                # Avoid name conflicts
                counter = 1
                while dest_file.exists():
                    stem = csv_file.stem
                    dest_file = dest_dir / f"{stem}_{counter}.csv"
                    counter += 1
                
                try:
                    os.symlink(csv_file.absolute(), dest_file)
                except OSError:
                    # Fallback to copy if symlink fails
                    shutil.copy2(csv_file, dest_file)
    
    def is_trained(self):
        """Check if model is already trained for this dataset"""
        model_path = self.results_dir / self.dataset_name / f"model_drop_col_head_column_0.pt"
        vectors_exist = len(list(self.vectors_dir.glob("*.pkl"))) >= 2
        return model_path.exists() and vectors_exist
    
    def train_model(self, epochs=3, batch_size=64, max_tables=10000):
        """Train the Starmie model"""
        print(f"🔥 Training model on {self.dataset_name}")
        
        cmd = [
            "python", "run_pretrain.py",
            "--task", self.dataset_name,
            "--batch_size", str(batch_size),
            "--lr", "5e-5",
            "--lm", "roberta",
            "--n_epochs", str(epochs),
            "--max_len", "128",
            "--size", str(max_tables),
            "--projector", "768",
            "--save_model",
            "--augment_op", "drop_col",
            "--fp16",
            "--sample_meth", "head",
            "--table_order", "column",
            "--run_id", "0"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Model training completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
    
    def extract_vectors(self):
        """Extract vector embeddings"""
        print("🔍 Extracting vector embeddings")
        
        cmd = [
            "python", "extractVectors.py",
            "--benchmark", self.dataset_name,
            "--table_order", "column",
            "--run_id", "0",
            "--save_model"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Vector extraction completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Vector extraction failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Starmie CSV Table Pretraining")
    parser.add_argument("--datalake_path", required=True, help="Path to directory containing CSV files")
    parser.add_argument("--cache_dir", help="Directory for caching models and vectors")
    parser.add_argument("--results_dir", help="Directory for saving trained models")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--max_tables", type=int, default=10000, help="Maximum tables to use in training")
    parser.add_argument("--force_retrain", action="store_true", help="Force retraining even if model exists")
    
    args = parser.parse_args()
    
    try:
        # Initialize pretrainer
        pretrainer = StarmiePretrain(args.datalake_path, args.cache_dir, args.results_dir)
        
        # Check if already trained
        if not args.force_retrain and pretrainer.is_trained():
            print("✅ Model already trained. Use --force_retrain to retrain.")
            return
        
        # Prepare data
        pretrainer.prepare_datalake()
        
        # Train model
        pretrainer.train_model(args.epochs, args.batch_size, args.max_tables)
        
        # Extract vectors
        pretrainer.extract_vectors()
        
        print(f"\n🎯 Pretraining completed for dataset: {pretrainer.dataset_name}")
        print("You can now use starmie_search.py to search for similar tables.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()