import time
import pickle
import numpy as np
import pandas as pd
import json
import os
import glob
from datetime import datetime
from naive_search import NaiveSearcher
from lsh_search import LSHSearcher
from hnsw_search import HNSWSearcher
from extractVectors import extractVectors, get_df

class BenchmarkRunner:
    def __init__(self, dataset_name, encoder='sherlock', augment_op='drop_col', sample_meth='head', 
                 table_order='column', run_id=0, scale=1.0, use_pretrained_vectors=False):
        """
        Initialize benchmark runner
        
        Args:
            dataset_name (str): Name of dataset (e.g., 'demo', 'my_dataset')
            encoder (str): Encoder type ('sherlock', 'sato', 'cl')
            augment_op (str): Augmentation operation for custom encoders
            sample_meth (str): Sampling method for custom encoders  
            table_order (str): Table order for custom encoders
            run_id (int): Run ID for custom encoders
            scale (float): Fraction of tables to use (for scalability testing)
            use_pretrained_vectors (bool): Use existing vector files or extract from raw data
        """
        self.dataset_name = dataset_name
        self.encoder = encoder
        self.augment_op = augment_op
        self.sample_meth = sample_meth
        self.table_order = table_order
        self.run_id = run_id
        self.scale = scale
        self.results = []
        
        # Check data directories
        self.data_base = f"data/{dataset_name}"
        self.datalake_dir = f"{self.data_base}/datalake"
        self.query_dir = f"{self.data_base}/query"
        
        if not os.path.exists(self.datalake_dir) or not os.path.exists(self.query_dir):
            raise FileNotFoundError(f"Data directories not found: {self.datalake_dir}, {self.query_dir}")
        
        if use_pretrained_vectors:
            self._load_pretrained_vectors()
        else:
            self._extract_vectors_from_raw_data()
    
    def _load_pretrained_vectors(self):
        """Load pre-computed vector files"""
        # Construct paths based on encoder type
        if self.encoder in ['sherlock', 'sato']:
            # Pre-existing encoders
            self.table_path = f"{self.data_base}/{self.encoder}_datalake.pkl"
            self.query_path = f"{self.data_base}/{self.encoder}_query.pkl"
        else:
            # Custom encoders (e.g., 'cl')
            sampAug = f"{self.sample_meth}_{self.augment_op}"
            self.table_path = f"{self.data_base}/vectors/{self.encoder}_datalake_{sampAug}_{self.table_order}_{self.run_id}.pkl"
            self.query_path = f"{self.data_base}/vectors/{self.encoder}_query_{sampAug}_{self.table_order}_{self.run_id}.pkl"
        
        # Check if vector files exist
        if not os.path.exists(self.table_path) or not os.path.exists(self.query_path):
            print(f"Vector files not found!")
            print(f"Table path: {self.table_path}")
            print(f"Query path: {self.query_path}")
            raise FileNotFoundError("Required vector files not found")
        
        # Load queries
        with open(self.query_path, 'rb') as f:
            self.queries = pickle.load(f)
        print(f"Loaded {len(self.queries)} queries from {self.query_path}")
    
    def _extract_vectors_from_raw_data(self):
        """Extract vectors from raw CSV files combining datalake and query tables"""
        print("Loading raw data and extracting vectors...")
        
        # Load raw data from both directories
        print("Loading datalake tables...")
        datalake_dfs = get_df(self.datalake_dir)
        print(f"Loaded {len(datalake_dfs)} datalake tables")
        
        print("Loading query tables...")
        query_dfs = get_df(self.query_dir)  
        print(f"Loaded {len(query_dfs)} query tables")
        
        # Extract vectors separately to maintain separation
        if self.encoder not in ['sherlock', 'sato']:
            print(f"Extracting vectors using {self.encoder} encoder...")
            
            # Extract vectors for datalake tables
            datalake_df_list = list(datalake_dfs.values())
            datalake_vectors = extractVectors(datalake_df_list, self.dataset_name, self.augment_op, 
                                            self.sample_meth, self.table_order, self.run_id)
            
            # Extract vectors for query tables
            query_df_list = list(query_dfs.values())
            query_vectors = extractVectors(query_df_list, self.dataset_name, self.augment_op, 
                                         self.sample_meth, self.table_order, self.run_id)
            
            print(f"Extracted {len(datalake_vectors)} datalake vectors and {len(query_vectors)} query vectors")
            
            # Create table data with vectors
            table_data = []
            for i, (table_name, vectors) in enumerate(zip(datalake_dfs.keys(), datalake_vectors)):
                table_data.append((table_name, vectors))
            
            query_data = []
            for i, (table_name, vectors) in enumerate(zip(query_dfs.keys(), query_vectors)):
                query_data.append((table_name, vectors))
            
            # Apply scaling to datalake tables only
            if self.scale < 1.0:
                import random
                table_data = random.sample(table_data, int(self.scale * len(table_data)))
                print(f"Scaled down to {len(table_data)} datalake tables")
            
            # Store for use by search methods
            self.table_data = table_data
            self.queries = query_data
            
            print(f"Final: {len(table_data)} datalake tables and {len(query_data)} query tables")
            
        else:
            # For sherlock/sato, we need pre-computed vectors
            raise NotImplementedError(f"Raw data extraction not implemented for {self.encoder} encoder. Please use --use_pretrained_vectors flag.")
    
    def _create_temp_pickle_files(self):
        """Create temporary pickle files for compatibility with existing search classes"""
        if hasattr(self, 'table_data'):
            # Create temporary pickle files
            temp_table_path = f"temp_{self.dataset_name}_datalake.pkl"
            temp_query_path = f"temp_{self.dataset_name}_query.pkl"
            
            with open(temp_table_path, 'wb') as f:
                pickle.dump(self.table_data, f)
            with open(temp_query_path, 'wb') as f:
                pickle.dump(self.queries, f)
            
            self.table_path = temp_table_path
            self.query_path = temp_query_path
            
            # Flag for cleanup
            self._temp_files = [temp_table_path, temp_query_path]
        
    def cleanup_temp_files(self):
        """Clean up temporary pickle files"""
        if hasattr(self, '_temp_files'):
            for file_path in self._temp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up {file_path}")
    
    def run_naive_benchmark(self, K=10, threshold=0.6):
        """Run benchmark on naive search method"""
        print(f"Running Naive Search benchmark...")
        
        # Create temp files if needed
        self._create_temp_pickle_files()
        
        searcher = NaiveSearcher(self.table_path, self.scale)
        method_results = []
        
        total_time = 0
        for i, query in enumerate(self.queries):
            print(f"Processing query {i+1}/{len(self.queries)}")
            
            start_time = time.time()
            results = searcher.topk(self.encoder, query, K, threshold)
            end_time = time.time()
            
            query_time = end_time - start_time
            total_time += query_time
            
            method_results.append({
                'query_id': i,
                'query_name': query[0],
                'results': results,
                'time': query_time,
                'num_results': len(results)
            })
        
        avg_time = total_time / len(self.queries)
        
        return {
            'method': 'naive',
            'total_time': total_time,
            'avg_time': avg_time,
            'queries': method_results,
            'params': {'encoder': self.encoder, 'K': K, 'threshold': threshold}
        }
    
    def run_lsh_benchmark(self, hash_func_num=10, hash_table_num=10, K=10, N=5, threshold=0.6):
        """Run benchmark on LSH search method"""
        print(f"Running LSH Search benchmark...")
        
        # Create temp files if needed
        self._create_temp_pickle_files()
        
        searcher = LSHSearcher(self.table_path, hash_func_num, hash_table_num, self.scale)
        method_results = []
        
        total_time = 0
        total_candidates = 0
        
        for i, query in enumerate(self.queries):
            print(f"Processing query {i+1}/{len(self.queries)}")
            
            start_time = time.time()
            results, num_candidates = searcher.topk(self.encoder, query, K, N, threshold)
            end_time = time.time()
            
            query_time = end_time - start_time
            total_time += query_time
            total_candidates += num_candidates
            
            method_results.append({
                'query_id': i,
                'query_name': query[0],
                'results': results,
                'time': query_time,
                'num_results': len(results),
                'num_candidates': num_candidates
            })
        
        avg_time = total_time / len(self.queries)
        avg_candidates = total_candidates / len(self.queries)
        
        return {
            'method': 'lsh',
            'total_time': total_time,
            'avg_time': avg_time,
            'avg_candidates': avg_candidates,
            'queries': method_results,
            'params': {'hash_func_num': hash_func_num, 'hash_table_num': hash_table_num, 
                      'encoder': self.encoder, 'K': K, 'N': N, 'threshold': threshold}
        }
    
    def run_hnsw_benchmark(self, index_path="hnsw_index.bin", K=10, N=5, threshold=0.6):
        """Run benchmark on HNSW search method"""
        print(f"Running HNSW Search benchmark...")
        
        # Create temp files if needed
        self._create_temp_pickle_files()
        
        searcher = HNSWSearcher(self.table_path, index_path, self.scale)
        method_results = []
        
        total_time = 0
        total_candidates = 0
        
        for i, query in enumerate(self.queries):
            print(f"Processing query {i+1}/{len(self.queries)}")
            
            start_time = time.time()
            results, num_candidates = searcher.topk(self.encoder, query, K, N, threshold)
            end_time = time.time()
            
            query_time = end_time - start_time
            total_time += query_time
            total_candidates += num_candidates
            
            method_results.append({
                'query_id': i,
                'query_name': query[0],
                'results': results,
                'time': query_time,
                'num_results': len(results),
                'num_candidates': num_candidates
            })
        
        avg_time = total_time / len(self.queries)
        avg_candidates = total_candidates / len(self.queries)
        
        return {
            'method': 'hnsw',
            'total_time': total_time,
            'avg_time': avg_time,
            'avg_candidates': avg_candidates,
            'queries': method_results,
            'params': {'index_path': index_path, 'encoder': self.encoder, 'K': K, 'N': N, 'threshold': threshold}
        }
    
    def run_all_benchmarks(self, K=10, threshold=0.6, 
                          hash_func_num=10, hash_table_num=10, N=5, 
                          index_path="hnsw_index.bin"):
        """Run all three search methods and collect results"""
        print("="*60)
        print("STARTING COMPREHENSIVE BENCHMARK")
        print("="*60)
        
        all_results = []
        
        # Run Naive Search
        try:
            naive_results = self.run_naive_benchmark(K, threshold)
            all_results.append(naive_results)
        except Exception as e:
            print(f"Error running naive search: {e}")
        
        # Run LSH Search  
        try:
            lsh_results = self.run_lsh_benchmark(hash_func_num, hash_table_num, K, N, threshold)
            all_results.append(lsh_results)
        except Exception as e:
            print(f"Error running LSH search: {e}")
        
        # Run HNSW Search
        try:
            hnsw_results = self.run_hnsw_benchmark(index_path, K, N, threshold)
            all_results.append(hnsw_results)
        except Exception as e:
            print(f"Error running HNSW search: {e}")
        
        self.results = all_results
        
        # Cleanup temporary files
        self.cleanup_temp_files()
        
        return all_results
    
    def save_results(self, output_path=None):
        """Save benchmark results to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_results_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}")
        return output_path
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        if not self.results:
            print("No results to report. Run benchmarks first.")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK REPORT")
        print("="*80)
        
        # Summary table
        print("\nSUMMARY:")
        print("-" * 60)
        print(f"{'Method':<15} {'Avg Time (s)':<15} {'Total Time (s)':<15} {'Queries':<10}")
        print("-" * 60)
        
        for result in self.results:
            method = result['method'].upper()
            avg_time = f"{result['avg_time']:.4f}"
            total_time = f"{result['total_time']:.4f}"
            num_queries = len(result['queries'])
            print(f"{method:<15} {avg_time:<15} {total_time:<15} {num_queries:<10}")
        
        # Detailed results per method
        for result in self.results:
            print(f"\n{result['method'].upper()} SEARCH DETAILED RESULTS:")
            print("-" * 50)
            print(f"Parameters: {result['params']}")
            print(f"Total execution time: {result['total_time']:.4f} seconds")
            print(f"Average query time: {result['avg_time']:.4f} seconds")
            
            if 'avg_candidates' in result:
                print(f"Average candidates found: {result['avg_candidates']:.2f}")
            
            # Top 5 slowest queries
            queries_by_time = sorted(result['queries'], key=lambda x: x['time'], reverse=True)
            print(f"\nTop 5 slowest queries:")
            for i, query in enumerate(queries_by_time[:5]):
                print(f"  {i+1}. {query['query_name']}: {query['time']:.4f}s ({query['num_results']} results)")
            
            # Results distribution
            result_counts = [q['num_results'] for q in result['queries']]
            print(f"\nResults distribution:")
            print(f"  Min results: {min(result_counts)}")
            print(f"  Max results: {max(result_counts)}")
            print(f"  Avg results: {np.mean(result_counts):.2f}")
        
        # Speed comparison
        if len(self.results) > 1:
            print(f"\nSPEED COMPARISON:")
            print("-" * 30)
            baseline = min(self.results, key=lambda x: x['avg_time'])
            baseline_time = baseline['avg_time']
            
            for result in self.results:
                speedup = baseline_time / result['avg_time']
                if result['method'] == baseline['method']:
                    print(f"{result['method'].upper()}: BASELINE (fastest)")
                else:
                    print(f"{result['method'].upper()}: {speedup:.2f}x slower than {baseline['method']}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive benchmark on all search methods')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., demo, my_dataset)')
    parser.add_argument('--encoder', default='sherlock', help='Encoder type (sherlock, sato, cl)')
    parser.add_argument('--augment_op', default='drop_col', help='Augmentation operation for custom encoders')
    parser.add_argument('--sample_meth', default='head', help='Sampling method for custom encoders')
    parser.add_argument('--table_order', default='column', help='Table order for custom encoders')
    parser.add_argument('--run_id', type=int, default=0, help='Run ID for custom encoders')
    parser.add_argument('--scale', type=float, default=1.0, help='Fraction of tables to use')
    parser.add_argument('--K', type=int, default=10, help='Number of top results to return')
    parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold')
    parser.add_argument('--hash_func_num', type=int, default=10, help='Number of hash functions for LSH')
    parser.add_argument('--hash_table_num', type=int, default=10, help='Number of hash tables for LSH')
    parser.add_argument('--N', type=int, default=5, help='Number of candidates for LSH/HNSW')
    parser.add_argument('--index_path', default='hnsw_index.bin', help='Path for HNSW index')
    parser.add_argument('--output', help='Output file path for results')
    parser.add_argument('--use_pretrained_vectors', action='store_true', 
                       help='Use existing vector files instead of extracting from raw data')
    
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        dataset_name=args.dataset,
        encoder=args.encoder,
        augment_op=args.augment_op,
        sample_meth=args.sample_meth,
        table_order=args.table_order,
        run_id=args.run_id,
        scale=args.scale,
        use_pretrained_vectors=args.use_pretrained_vectors
    )
    
    # Run all benchmarks
    results = runner.run_all_benchmarks(
        K=args.K, 
        threshold=args.threshold,
        hash_func_num=args.hash_func_num,
        hash_table_num=args.hash_table_num,
        N=args.N,
        index_path=args.index_path
    )
    
    # Save results
    output_path = runner.save_results(args.output)
    
    # Generate report
    runner.generate_report()
    
    print(f"\nBenchmark complete! Results saved to {output_path}")

if __name__ == "__main__":
    main()