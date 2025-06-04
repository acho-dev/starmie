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
import time
import logging
import requests
from io import StringIO
from typing import List

class StarmieSearch:
    def __init__(self, datalake_path, cache_dir=None, results_dir=None, server_url=None):
        self.datalake_path = Path(datalake_path)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./starmie_cache")
        self.results_dir = Path(results_dir) if results_dir else Path("./results")
        self.server_url = server_url
        
        # Performance tracking
        self.timings = {}
        self.start_time = time.time()
        
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
    
    def _check_server_available(self) -> bool:
        """Check if the Starmie server is available"""
        if not self.server_url:
            return False
        
        try:
            response = requests.get(f"{self.server_url}/status", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return data.get('model_loaded', False)
        except:
            pass
        return False
    
    def _search_via_server(self, query_csv=None, query_text=None, top_k=10, threshold=0.7) -> List[str]:
        """Search using the Starmie server"""
        try:
            # Prepare CSV text
            if query_csv:
                with open(query_csv, 'r') as f:
                    csv_text = f.read()
            elif query_text:
                csv_text = query_text
            else:
                raise ValueError("Either query_csv or query_text must be provided")
            
            # Make request to server
            payload = {
                'csv_text': csv_text,
                'top_k': top_k,
                'threshold': threshold
            }
            
            response = requests.post(f"{self.server_url}/search", json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if data['success']:
                # Extract timing information
                if 'timing' in data:
                    self.timings['Server Search'] = data['timing']['total_time']
                    self.timings['Vector Extraction (Server)'] = data['timing']['vector_extraction']
                    self.timings['Search Execution (Server)'] = data['timing']['search']
                
                # Format results to match expected output
                results = [result['filename'] for result in data['results']]
                return results
            else:
                raise RuntimeError(f"Server search failed: {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Server search failed: {e}")
            raise
    
    def _time_step(self, step_name, func, *args, **kwargs):
        """Time a function execution and store the result"""
        print(f"⏱️  Starting: {step_name}")
        step_start = time.time()
        
        try:
            result = func(*args, **kwargs)
            step_end = time.time()
            elapsed = step_end - step_start
            self.timings[step_name] = elapsed
            print(f"✅ Completed: {step_name} ({elapsed:.2f}s)")
            return result
        except Exception as e:
            step_end = time.time()
            elapsed = step_end - step_start
            self.timings[step_name] = elapsed
            print(f"❌ Failed: {step_name} ({elapsed:.2f}s) - {e}")
            raise
    
    def _log_timing_summary(self):
        """Print a summary of all timing information"""
        total_time = time.time() - self.start_time
        print(f"\n📊 Performance Summary:")
        print("=" * 60)
        print(f"{'Step':<30} {'Time (s)':<10} {'% of Total':<10}")
        print("-" * 60)
        
        for step, elapsed in self.timings.items():
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            print(f"{step:<30} {elapsed:<10.2f} {percentage:<10.1f}%")
        
        print("-" * 60)
        print(f"{'TOTAL':<30} {total_time:<10.2f} {'100.0%':<10}")
        print("=" * 60)
        
        # Identify potential bottlenecks
        if self.timings:
            slowest_step = max(self.timings, key=self.timings.get)
            slowest_time = self.timings[slowest_step]
            if slowest_time > 1.0:  # More than 1 second
                print(f"🐌 Slowest step: {slowest_step} ({slowest_time:.2f}s)")
                
            if total_time > 10.0:  # More than 10 seconds total
                print("💡 Performance tips:")
                if "Model Loading" in self.timings and self.timings["Model Loading"] > 3.0:
                    print("   - Model loading is slow. Consider keeping the model in memory for repeated queries.")
                if "Query Vector Extraction" in self.timings and self.timings["Query Vector Extraction"] > 5.0:
                    print("   - Query vector extraction is slow. This happens every query and could be optimized.")
                if "Directory Setup" in self.timings and self.timings["Directory Setup"] > 2.0:
                    print("   - Directory setup is slow. File operations or symlinks might be causing delays.")
        
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
    
    def _prepare_datalake_impl(self):
        """Implementation of prepare_datalake"""
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
    
    def prepare_datalake(self):
        """Copy or symlink CSV files to the expected structure"""
        return self._time_step("Datalake Preparation", self._prepare_datalake_impl)
    
    def _prepare_query_impl(self, query_csv=None, query_text=None):
        """Implementation of prepare_query"""
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
    
    def prepare_query(self, query_csv=None, query_text=None):
        """Prepare query CSV file"""
        return self._time_step("Query Preparation", self._prepare_query_impl, query_csv, query_text)
    
    def _is_trained_impl(self):
        """Implementation of is_trained"""
        model_path = self.results_dir / self.dataset_name / f"model_drop_col_head_column_0.pt"
        vectors_exist = len(list(self.vectors_dir.glob("*.pkl"))) >= 2
        return model_path.exists() and vectors_exist
    
    def is_trained(self):
        """Check if model is already trained for this dataset"""
        return self._time_step("Model Check", self._is_trained_impl)
    
    def _check_pretrained_impl(self):
        """Implementation of check_pretrained"""
        if not self.is_trained():
            raise ValueError(
                f"Model not found for dataset {self.dataset_name}.\n"
                f"Please run: python starmie_pretrain.py --datalake_path {self.datalake_path}"
            )
    
    def check_pretrained(self):
        """Check if model is pretrained and ready for search"""
        return self._time_step("Pretrained Check", self._check_pretrained_impl)
    
    def _extract_query_vectors_impl(self):
        """Implementation of extract_query_vectors"""
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
            print(f"🔧 Running command: {' '.join(cmd)}")
            cmd_start = time.time()
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            cmd_end = time.time()
            print(f"✅ Query vector extraction completed (subprocess: {cmd_end - cmd_start:.2f}s)")
        except subprocess.CalledProcessError as e:
            print(f"❌ Query vector extraction failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
    
    def extract_query_vectors(self):
        """Extract vector embeddings for query"""
        return self._time_step("Query Vector Extraction", self._extract_query_vectors_impl)
    
    def _search_impl(self, top_k=10, threshold=0.7, method="hnsw"):
        """Implementation of search"""
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
            print(f"🔧 Running search command: {' '.join(cmd)}")
            cmd_start = time.time()
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            cmd_end = time.time()
            print(f"✅ Search subprocess completed ({cmd_end - cmd_start:.2f}s)")
            return self.parse_results(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Search failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
    
    def search(self, top_k=10, threshold=0.7, method="hnsw"):
        """Search for similar tables"""
        return self._time_step("Search Execution", self._search_impl, top_k, threshold, method)
    
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
    
    def search_smart(self, query_csv=None, query_text=None, top_k=10, threshold=0.7, method="hnsw") -> List[str]:
        """Smart search that uses server if available, otherwise falls back to local processing"""
        
        # Check if server is available
        if self._check_server_available():
            print("🚀 Using Starmie server for fast processing...")
            try:
                return self._time_step("Server Search", self._search_via_server, query_csv, query_text, top_k, threshold)
            except Exception as e:
                print(f"⚠️  Server search failed, falling back to local processing: {e}")
        
        # Fall back to local processing
        print("🔧 Using local processing...")
        
        # Prepare data (only if not using server)
        self.prepare_datalake()
        self.prepare_query(query_csv, query_text)
        
        # Check if model is pretrained
        self.check_pretrained()
        
        # Extract query vectors
        self.extract_query_vectors()
        
        # Search
        return self.search(top_k, threshold, method)

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
    parser.add_argument("--timing", action="store_true", help="Show detailed timing information")
    parser.add_argument("--server_url", help="URL of Starmie server (e.g., http://localhost:5000)")
    parser.add_argument("--force_local", action="store_true", help="Force local processing even if server is available")
    
    args = parser.parse_args()
    
    if not args.query_csv and not args.query_text:
        parser.error("Either --query_csv or --query_text must be provided")
    
    try:
        # Initialize search engine
        print("🚀 Initializing Starmie Search...")
        init_start = time.time()
        searcher = StarmieSearch(args.datalake_path, args.cache_dir, args.results_dir, 
                                server_url=args.server_url if not args.force_local else None)
        init_end = time.time()
        searcher.timings["Initialization"] = init_end - init_start
        print(f"✅ Completed: Initialization ({searcher.timings['Initialization']:.2f}s)")
        
        # Use smart search that automatically chooses between server and local
        if args.force_local:
            print("🔧 Forcing local processing...")
            # Prepare data
            searcher.prepare_datalake()
            searcher.prepare_query(args.query_csv, args.query_text)
            
            # Check if model is pretrained
            searcher.check_pretrained()
            
            # Extract query vectors
            searcher.extract_query_vectors()
            
            # Search
            results = searcher.search(args.top_k, args.threshold, args.method)
        else:
            # Smart search (server if available, local fallback)
            results = searcher.search_smart(args.query_csv, args.query_text, args.top_k, args.threshold, args.method)
        
        # Display results
        print(f"\n🎯 Top {args.top_k} Similar Tables:")
        print("=" * 50)
        for i, result in enumerate(results[:args.top_k], 1):
            print(f"{i}. {result}")
        
        if not results:
            print("No results found. Try adjusting the threshold or check your data.")
        
        # Show performance summary
        searcher._log_timing_summary()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'searcher' in locals():
            searcher._log_timing_summary()
        sys.exit(1)

if __name__ == "__main__":
    main()