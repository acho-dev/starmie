#!/usr/bin/env python3
"""
Starmie Search - A productized interface for CSV table discovery

Usage:
    # First, pretrain the model:
    python starmie_pretrain.py --datalake_path /path/to/csv/files
    
    # Then search:
    python starmie_search.py \
        --datalake_path /path/to/csv/files \
        --query_csv /path/to/query.csv \
        --top_k 10

    # Or with CSV content as text:
    python starmie_search.py \
        --datalake_path /path/to/csv/files \
        --query_text "name,age,city\nJohn,25,NYC\nJane,30,LA" \
        --top_k 10
"""

import argparse
import os
import sys
import tempfile
import glob
import json
import subprocess
from pathlib import Path
import shutil

class StarmieSearch:
    def __init__(self, datalake_path, cache_dir=None, results_dir=None):
        self.datalake_path = Path(datalake_path)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./starmie_cache")
        self.results_dir = Path(results_dir) if results_dir else Path("./results")
        
        # Use the same logic as starmie_pretrain.py to determine dataset name
        parent_dir = self.datalake_path.parent
        
        # Check if this looks like an existing dataset structure
        if (self.datalake_path.name in ['datalake', 'tables'] and 
            (parent_dir.parent.name == 'data' or 
             any((parent_dir / folder).exists() for folder in ['datalake', 'tables', 'query']))):
            # Use existing structure
            self.dataset_name = parent_dir.name
            self.data_dir = parent_dir
        else:
            # Create a dataset name based on the path
            self.dataset_name = f"dataset_{abs(hash(str(self.datalake_path)))}"
            self.data_dir = Path("data") / self.dataset_name
            
        self.setup_directories()
        
    def setup_directories(self):
        """Setup required directory structure"""
        self.cache_dir.mkdir(exist_ok=True)
        
        # Define directory paths based on the data_dir
        self.datalake_dir = self.datalake_path if self.datalake_path.name in ['datalake', 'tables'] else self.data_dir / "datalake"
        self.query_dir = self.data_dir / "query"
        self.vectors_dir = self.data_dir / "vectors"
        
        # Only create directories if they don't exist and we're not using existing structure
        if not self.data_dir.exists() or self.data_dir != self.datalake_path.parent:
            for dir_path in [self.datalake_dir, self.query_dir, self.vectors_dir]:
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
    
    def prepare_datalake(self):
        """Copy or symlink CSV files to the expected structure"""
        print(f"📁 Preparing data lake from {self.datalake_path}")
        
        # If we're using an existing dataset structure, don't modify it
        if self.data_dir == self.datalake_path.parent and self.datalake_path.name in ['datalake', 'tables']:
            print("Using existing dataset structure")
            csv_files = list(self.datalake_path.glob("*.csv"))
            print(f"Found {len(csv_files)} CSV files in existing structure")
            return
        
        # Clear existing datalake only if we're creating new structure
        if self.datalake_dir.exists() and self.datalake_dir != self.datalake_path:
            shutil.rmtree(self.datalake_dir)
        if not self.datalake_dir.exists():
            self.datalake_dir.mkdir(parents=True)
        
        # Find all CSV files
        csv_files = list(self.datalake_path.glob("**/*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.datalake_path}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Copy or symlink CSV files
        for csv_file in csv_files:
            if self.datalake_dir == self.datalake_path:
                continue  # Don't copy to itself
            dest_file = self.datalake_dir / csv_file.name
            # Avoid name conflicts
            counter = 1
            while dest_file.exists():
                stem = csv_file.stem
                dest_file = self.datalake_dir / f"{stem}_{counter}.csv"
                counter += 1
            
            try:
                os.symlink(csv_file.absolute(), dest_file)
            except OSError:
                # Fallback to copy if symlink fails
                shutil.copy2(csv_file, dest_file)
    
    def prepare_query(self, query_csv=None, query_text=None):
        """Prepare query CSV file"""
        # Ensure query directory exists
        self.query_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean query directory of any existing CSV files
        for existing_csv in self.query_dir.glob("*.csv"):
            existing_csv.unlink()
            print(f"🧹 Removed existing query file: {existing_csv.name}")
        
        if query_csv:
            query_path = Path(query_csv)
            if not query_path.exists():
                raise ValueError(f"Query CSV file not found: {query_csv}")
            
            dest_path = self.query_dir / "query.csv"
            shutil.copy2(query_path, dest_path)
            print(f"📄 Query CSV prepared: {query_path.name}")
            
        elif query_text:
            dest_path = self.query_dir / "query.csv"
            with open(dest_path, 'w') as f:
                f.write(query_text)
            print("📄 Query CSV created from text input")
            
        else:
            raise ValueError("Either query_csv or query_text must be provided")
    
    def is_trained(self):
        """Check if model is already trained for this dataset"""
        model_path = self.results_dir / self.dataset_name / f"model_drop_col_head_column_0.pt"
        vectors_exist = len(list(self.vectors_dir.glob("*.pkl"))) >= 2
        return model_path.exists() and vectors_exist
    
    def check_pretrained(self):
        """Check if model is pretrained and ready for search"""
        if not self.is_trained():
            raise ValueError(
                f"Model not found for dataset {self.dataset_name}.\n"
                f"Please run: python starmie_pretrain.py --datalake_path {self.datalake_path}"
            )
    
    def extract_query_vectors(self):
        """Extract vector embeddings for query"""
        print("🔍 Extracting query vector embeddings")
        
        # Remove existing query vectors to ensure fresh extraction
        query_vector_path = self.vectors_dir / "cl_query_drop_col_head_column_0.pkl"
        if query_vector_path.exists():
            query_vector_path.unlink()
            print("🔄 Removing existing query vectors")
        
        cmd = [
            "python", "extractVectors.py",
            "--benchmark", self.dataset_name,
            "--table_order", "column",
            "--run_id", "0",
            "--save_model"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Query vector extraction completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Query vector extraction failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
    
    def search(self, top_k=10, threshold=0.7, method="hnsw"):
        """Search for similar tables"""
        print(f"🔎 Searching for top {top_k} similar tables")
        
        if method == "linear":
            script = "test_naive_search.py"
            cmd = [
                "python", script,
                "--encoder", "cl",
                "--benchmark", self.dataset_name,
                "--augment_op", "drop_col",
                "--sample_meth", "head",
                "--matching", "linear",
                "--table_order", "column",
                "--run_id", "0",
                "--K", str(top_k),
                "--threshold", str(threshold)
            ]
        elif method == "hnsw":
            script = "test_hnsw_search.py"
            cmd = [
                "python", script,
                "--encoder", "cl",
                "--benchmark", self.dataset_name,
                "--augment_op", "drop_col",
                "--sample_meth", "head",
                "--table_order", "column",
                "--run_id", "0",
                "--K", str(top_k),
                "--threshold", str(threshold)
            ]
        elif method == "lsh":
            script = "test_lsh.py"
            cmd = [
                "python", script,
                "--encoder", "cl",
                "--benchmark", self.dataset_name,
                "--run_id", "0",
                "--num_func", "8",
                "--num_table", "100",
                "--K", str(top_k)
            ]
        elif method == "hnsw":
            script = "test_hnsw_search.py"
            cmd = [
                "python", script,
                "--encoder", "cl",
                "--benchmark", self.dataset_name,
                "--run_id", "0",
                "--K", str(top_k)
            ]
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return self.parse_results(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Search failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
    
    def parse_results(self, output):
        """Parse search results from output"""
        lines = output.strip().split('\n')
        results = []
        
        # Look for search results section
        in_results_section = False
        for line in lines:
            if "🎯 Search Results:" in line:
                in_results_section = True
                continue
            elif in_results_section and line.strip():
                if line.startswith("Query:"):
                    continue  # Skip query header
                elif line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")):
                    # Extract just the table name (remove the number prefix)
                    result = line.strip().split(". ", 1)
                    if len(result) > 1:
                        results.append(result[1])
                elif "No results found" in line:
                    results.append("No results found above threshold")
        
        # Fallback: look for any numbered results
        if not results:
            for line in lines:
                if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")):
                    result = line.strip().split(". ", 1)
                    if len(result) > 1:
                        results.append(result[1])
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Starmie CSV Table Search")
    parser.add_argument("--datalake_path", required=True, help="Path to directory containing CSV files")
    parser.add_argument("--query_csv", help="Path to query CSV file")
    parser.add_argument("--query_text", help="CSV content as text (alternative to --query_csv)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to return")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    parser.add_argument("--method", choices=["linear", "lsh", "hnsw"], default="hnsw", help="Search method")
    parser.add_argument("--cache_dir", help="Directory for caching models and vectors")
    parser.add_argument("--results_dir", help="Directory where trained models are saved")
    
    args = parser.parse_args()
    
    if not args.query_csv and not args.query_text:
        parser.error("Either --query_csv or --query_text must be provided")
    
    try:
        # Initialize search engine
        searcher = StarmieSearch(args.datalake_path, args.cache_dir, args.results_dir)
        
        # Prepare data
        searcher.prepare_datalake()
        searcher.prepare_query(args.query_csv, args.query_text)
        
        # Check if model is pretrained
        searcher.check_pretrained()
        
        # Extract query vectors
        searcher.extract_query_vectors()
        
        # Search
        results = searcher.search(args.top_k, args.threshold, args.method)
        
        # Display results
        print(f"\n🎯 Top {args.top_k} Similar Tables:")
        print("=" * 50)
        for i, result in enumerate(results[:args.top_k], 1):
            print(f"{i}. {result}")
        
        if not results:
            print("No results found. Try adjusting the threshold or check your data.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()