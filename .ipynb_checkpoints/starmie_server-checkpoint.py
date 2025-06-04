#!/usr/bin/env python3
"""
Starmie Model Server - Flask server for fast table search with persistent model loading

Usage:
    # Start the server:
    python starmie_server.py --datalake_path /path/to/csv/files --port 5000
    
    # Server will load the model once and keep it in memory for fast query processing
"""

import argparse
import os
import sys
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import threading
import logging

import pandas as pd
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template_string
from munkres import Munkres, make_cost_matrix, DISALLOWED

# Import Starmie components
from sdd.pretrain import load_checkpoint, inference_on_tables
from sdd.model import BarlowTwinsSimCLR
from sdd.dataset import PretrainTableDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StarmieModelService:
    """Service class that manages the pretrained model and provides vector extraction"""
    
    def __init__(self, datalake_path: str, results_dir: str = "./results"):
        self.datalake_path = Path(datalake_path)
        self.results_dir = Path(results_dir)
        
        # Determine dataset name using same logic as StarmieSearch
        parent_dir = self.datalake_path.parent
        if (self.datalake_path.name in ['datalake', 'tables'] and 
            (parent_dir.parent.name == 'data' or 
             any((parent_dir / folder).exists() for folder in ['datalake', 'tables', 'query']))):
            self.dataset_name = parent_dir.name
            self.data_dir = parent_dir
        else:
            self.dataset_name = f"dataset_{abs(hash(str(self.datalake_path)))}"
            self.data_dir = Path("data") / self.dataset_name
        
        self.vectors_dir = self.data_dir / "vectors"
        
        # Model components (loaded lazily)
        self.model = None
        self.trainset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.load_lock = threading.Lock()
        
        # Cache for datalake vectors (keep as column vectors, not averaged)
        self.datalake_column_vectors = None  # List of (filename, column_vectors) tuples
        self.datalake_filenames = None
        
        logger.info(f"Initialized StarmieModelService for dataset: {self.dataset_name}")
        logger.info(f"Using device: {self.device}")
    
    def _get_model_path(self) -> Path:
        """Get the path to the pretrained model"""
        model_path = self.results_dir / self.dataset_name / "model_drop_col_head_column_0.pt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pretrained model not found at {model_path}. "
                f"Please run: python starmie_pretrain.py --datalake_path {self.datalake_path}"
            )
        return model_path
    
    def load_model(self) -> bool:
        """Load the pretrained model into memory"""
        with self.load_lock:
            if self.model_loaded:
                return True
            
            try:
                start_time = time.time()
                logger.info("🔄 Loading pretrained model...")
                
                model_path = self._get_model_path()
                
                # Load checkpoint
                logger.info(f"Loading checkpoint from {model_path}")
                try:
                    # Try with weights_only parameter (newer PyTorch versions)
                    ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                except TypeError:
                    # Fallback for older PyTorch versions that don't support weights_only
                    ckpt = torch.load(model_path, map_location=self.device)
                
                # Load model and dataset
                # Pass the dataset name, not the full path, so load_checkpoint can resolve it correctly
                self.model, self.trainset = load_checkpoint(ckpt, current_dataset=self.dataset_name)
                self.model.eval()  # Set to evaluation mode
                
                load_time = time.time() - start_time
                logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
                
                # Load datalake vectors if available
                self._load_datalake_vectors()
                
                self.model_loaded = True
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                return False
    
    def _load_datalake_vectors(self):
        """Load precomputed datalake vectors (keep column vectors separate for pair-wise matching)"""
        try:
            vector_path = self.vectors_dir / "cl_datalake_drop_col_head_column_0.pkl"
            if vector_path.exists():
                import pickle
                with open(vector_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Store column vectors separately for each table
                self.datalake_column_vectors = []
                self.datalake_filenames = []
                
                for item in data:
                    filename = item[0]
                    vectors = item[1]  # Keep as column vectors for pair-wise matching
                    
                    if isinstance(vectors, np.ndarray):
                        if len(vectors.shape) == 1:
                            # Single vector, reshape to 2D
                            vectors = vectors.reshape(1, -1)
                        # Store as column vectors
                        self.datalake_column_vectors.append((filename, vectors))
                        self.datalake_filenames.append(filename)
                    else:
                        logger.warning(f"Unexpected vector format for {filename}: {type(vectors)}")
                
                logger.info(f"✅ Loaded datalake column vectors: {len(self.datalake_filenames)} tables")
            else:
                logger.warning(f"Datalake vectors not found at {vector_path}")
        except Exception as e:
            logger.error(f"Failed to load datalake vectors: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def extract_query_vectors(self, csv_data) -> np.ndarray:
        """Extract vectors for a query CSV"""
        if not self.model_loaded:
            success = self.load_model()
            if not success:
                raise RuntimeError("Failed to load model")
        
        try:
            start_time = time.time()
            
            # Ensure input is a DataFrame
            if isinstance(csv_data, str):
                df = pd.read_csv(csv_data)
            elif isinstance(csv_data, pd.DataFrame):
                df = csv_data
            else:
                raise ValueError("csv_data must be a file path or pandas DataFrame")
            
            # Extract vectors using the loaded model
            vectors = inference_on_tables([df], self.model, self.trainset, batch_size=1)
            
            extract_time = time.time() - start_time
            logger.info(f"✅ Query vectors extracted in {extract_time:.2f}s")
            
            # Ensure we return a numpy array
            result = vectors[0]
            if isinstance(result, list):
                result = np.array(result)
            
            logger.info(f"Query vectors shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to extract query vectors: {e}")
            raise
    
    def _pairwise_similarity_munkres(self, query_vectors: np.ndarray, table_vectors: np.ndarray) -> float:
        """Compute pair-wise similarity using Munkres algorithm for optimal bipartite matching"""
        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        query_normalized = query_vectors / (query_norm + 1e-8)
        
        table_norm = np.linalg.norm(table_vectors, axis=1, keepdims=True)
        table_normalized = table_vectors / (table_norm + 1e-8)
        
        # Compute cosine similarity matrix
        similarity_matrix = query_normalized @ table_normalized.T
        
        # Convert to cost matrix (Munkres minimizes, so use 1 - similarity)
        cost_matrix = 1.0 - similarity_matrix
        
        # Create cost matrix for Munkres (handle negative costs)
        max_cost = cost_matrix.max()
        munkres_matrix = make_cost_matrix(cost_matrix, lambda cost: cost if cost != DISALLOWED else max_cost + 1)
        
        # Solve assignment problem
        m = Munkres()
        try:
            indices = m.compute(munkres_matrix)
            
            # Calculate total similarity score
            total_similarity = 0.0
            for row, col in indices:
                if row < similarity_matrix.shape[0] and col < similarity_matrix.shape[1]:
                    total_similarity += similarity_matrix[row, col]
            
            # Normalize by the number of pairs matched
            if len(indices) > 0:
                return total_similarity / len(indices)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Munkres assignment failed: {e}, falling back to max similarity")
            # Fallback: return maximum similarity if Munkres fails
            return float(similarity_matrix.max())

    def search_similar_tables(self, query_vectors: np.ndarray, top_k: int = 10, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find similar tables using pair-wise matching with Munkres algorithm"""
        if self.datalake_column_vectors is None:
            raise RuntimeError("Datalake vectors not loaded")
        
        try:
            # Ensure query_vectors is a numpy array
            if isinstance(query_vectors, list):
                query_vectors = np.array(query_vectors)
            
            # Ensure query_vectors is 2D
            if len(query_vectors.shape) == 1:
                query_vectors = query_vectors.reshape(1, -1)
            
            logger.info(f"Query vectors shape: {query_vectors.shape}")
            logger.info(f"Number of datalake tables: {len(self.datalake_column_vectors)}")
            
            # Compute pair-wise similarities for each table
            similarities = []
            
            for filename, table_vectors in self.datalake_column_vectors:
                # Ensure table_vectors is 2D
                if len(table_vectors.shape) == 1:
                    table_vectors = table_vectors.reshape(1, -1)
                
                # Compute optimal pair-wise similarity using Munkres
                similarity = self._pairwise_similarity_munkres(query_vectors, table_vectors)
                similarities.append(similarity)
            
            similarities = np.array(similarities)
            
            logger.info(f"Similarities shape: {similarities.shape}")
            logger.info(f"Similarity range: {similarities.min():.4f} to {similarities.max():.4f}")
            
            # Filter by threshold and get top-k
            valid_indices = np.where(similarities >= threshold)[0]
            logger.info(f"Found {len(valid_indices)} results above threshold {threshold}")
            
            if len(valid_indices) == 0:
                return []
            
            # Sort by similarity (descending)
            sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]][:top_k]
            
            results = []
            for idx in sorted_indices:
                if idx < len(self.datalake_filenames):
                    filename = self.datalake_filenames[idx]
                    similarity = similarities[idx]
                    results.append((filename, float(similarity)))
                else:
                    logger.warning(f"Index {idx} out of range for filenames list")
            
            logger.info(f"Returning {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search similar tables: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

# Flask application
app = Flask(__name__)
model_service: Optional[StarmieModelService] = None

@app.route('/')
def index():
    """Simple web interface for testing"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Starmie Table Search Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; }
            textarea { width: 100%; height: 200px; font-family: monospace; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #007bff; }
            .status { margin: 10px 0; padding: 10px; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Starmie Table Search Server</h1>
            <p>Model Status: <span id="status">Loading...</span></p>
            
            <h3>Search for Similar Tables</h3>
            <form id="searchForm">
                <p>Enter CSV data (headers and a few rows):</p>
                <textarea id="csvData" placeholder="name,age,city
John,25,NYC
Jane,30,LA"></textarea>
                <br><br>
                <label>Top K: <input type="number" id="topK" value="10" min="1" max="50"></label>
                <label>Threshold: <input type="number" id="threshold" value="0.5" min="0" max="1" step="0.01"></label>
                <br><br>
                <button type="submit">Search</button>
            </form>
            
            <div id="results"></div>
        </div>
        
        <script>
            // Check model status
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').textContent = 
                        data.model_loaded ? '✅ Ready' : '❌ Loading...';
                });
            
            // Handle search form
            document.getElementById('searchForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<div class="status">🔍 Searching...</div>';
                
                try {
                    const response = await fetch('/search', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            csv_text: document.getElementById('csvData').value,
                            top_k: parseInt(document.getElementById('topK').value),
                            threshold: parseFloat(document.getElementById('threshold').value)
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        let html = '<h3>🎯 Search Results:</h3>';
                        if (data.results.length === 0) {
                            html += '<div class="status">No results found above threshold.</div>';
                        } else {
                            data.results.forEach((result, i) => {
                                html += `<div class="result">
                                    <strong>${i+1}. ${result.filename}</strong><br>
                                    Similarity: ${(result.similarity * 100).toFixed(1)}%
                                </div>`;
                            });
                        }
                        html += `<div class="status success">Search completed in ${data.timing.total_time.toFixed(2)}s 
                                (Vector extraction: ${data.timing.vector_extraction.toFixed(2)}s, 
                                Search: ${data.timing.search.toFixed(2)}s)</div>`;
                        resultsDiv.innerHTML = html;
                    } else {
                        resultsDiv.innerHTML = `<div class="status error">❌ Error: ${data.error}</div>`;
                    }
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="status error">❌ Network error: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return html

@app.route('/status')
def status():
    """Get server status"""
    return jsonify({
        'model_loaded': model_service.model_loaded if model_service else False,
        'dataset_name': model_service.dataset_name if model_service else None,
        'device': str(model_service.device) if model_service else None
    })

@app.route('/search', methods=['POST'])
def search():
    """Search for similar tables"""
    try:
        start_time = time.time()
        
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'})
        
        csv_text = data.get('csv_text', '')
        top_k = data.get('top_k', 10)
        threshold = data.get('threshold', 0.7)
        
        if not csv_text.strip():
            return jsonify({'success': False, 'error': 'No CSV data provided'})
        
        # Create temporary DataFrame
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(csv_text))
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid CSV format: {str(e)}'})
        
        # Extract query vectors
        vector_start = time.time()
        query_vectors = model_service.extract_query_vectors(df)
        vector_time = time.time() - vector_start
        
        # Search for similar tables
        search_start = time.time()
        results = model_service.search_similar_tables(query_vectors, top_k, threshold)
        search_time = time.time() - search_start
        
        total_time = time.time() - start_time
        
        # Format results
        formatted_results = [
            {'filename': filename, 'similarity': similarity}
            for filename, similarity in results
        ]
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'timing': {
                'total_time': total_time,
                'vector_extraction': vector_time,
                'search': search_time
            }
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/extract_vectors', methods=['POST'])
def extract_vectors():
    """Extract vectors for a CSV (utility endpoint)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'})
        
        csv_text = data.get('csv_text', '')
        if not csv_text.strip():
            return jsonify({'success': False, 'error': 'No CSV data provided'})
        
        # Create temporary DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(csv_text))
        
        # Extract vectors
        vectors = model_service.extract_query_vectors(df)
        
        return jsonify({
            'success': True,
            'vectors': vectors.tolist(),  # Convert numpy array to list for JSON
            'shape': vectors.shape
        })
        
    except Exception as e:
        logger.error(f"Vector extraction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

def main():
    parser = argparse.ArgumentParser(description="Starmie Model Server")
    parser.add_argument("--datalake_path", required=True, help="Path to directory containing CSV files")
    parser.add_argument("--results_dir", default="./results", help="Directory where trained models are saved")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--preload", action="store_true", help="Preload the model on startup")
    
    args = parser.parse_args()
    
    try:
        # Initialize model service
        global model_service
        model_service = StarmieModelService(args.datalake_path, args.results_dir)
        
        # Preload model if requested
        if args.preload:
            logger.info("🔄 Preloading model...")
            success = model_service.load_model()
            if not success:
                logger.error("❌ Failed to preload model")
                sys.exit(1)
            logger.info("✅ Model preloaded successfully")
        
        # Start Flask server
        logger.info(f"🚀 Starting Starmie server on {args.host}:{args.port}")
        logger.info(f"📊 Dataset: {model_service.dataset_name}")
        logger.info(f"🌐 Web interface: http://{args.host}:{args.port}")
        
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()