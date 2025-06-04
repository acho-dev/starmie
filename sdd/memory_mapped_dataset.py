import os
import mmap
import pickle
import pandas as pd
import numpy as np
from torch.utils import data
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class MemoryMappedTableCache:
    """Memory-mapped cache for fast table access"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.index_file = os.path.join(cache_dir, 'table_index.pkl')
        self.data_file = os.path.join(cache_dir, 'table_data.bin')
        self.table_index = {}
        self.mmap_file = None
        
    def create_cache(self, table_paths: List[str]):
        """Create memory-mapped cache from table files"""
        logger.info(f"Creating memory-mapped cache for {len(table_paths)} tables...")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Serialize all tables to binary format
        with open(self.data_file, 'wb') as f:
            offset = 0
            for i, table_path in enumerate(table_paths):
                print(f"💾 Caching table {i+1}/{len(table_paths)}: {os.path.basename(table_path)}")
                try:
                    # Read and serialize table
                    table = pd.read_csv(table_path, lineterminator='\n')
                    print(f"   📏 {len(table)} rows × {len(table.columns)} columns")
                    table_bytes = pickle.dumps(table)
                    print(f"   💿 Serialized: {len(table_bytes)/1024:.1f} KB")
                    
                    # Write to file and record position
                    f.write(table_bytes)
                    
                    self.table_index[i] = {
                        'offset': offset,
                        'size': len(table_bytes),
                        'path': table_path
                    }
                    
                    offset += len(table_bytes)
                    
                except Exception as e:
                    logger.warning(f"Failed to cache table {table_path}: {e}")
        
        # Save index
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.table_index, f)
        
        logger.info(f"Created memory-mapped cache with {len(self.table_index)} tables")
    
    def load_cache(self):
        """Load existing memory-mapped cache"""
        if not os.path.exists(self.index_file) or not os.path.exists(self.data_file):
            raise FileNotFoundError("Memory-mapped cache not found")
        
        # Load index
        with open(self.index_file, 'rb') as f:
            self.table_index = pickle.load(f)
        
        # Open memory-mapped file
        self.mmap_file = open(self.data_file, 'rb')
        self.mmap = mmap.mmap(self.mmap_file.fileno(), 0, access=mmap.ACCESS_READ)
        
        logger.info(f"Loaded memory-mapped cache with {len(self.table_index)} tables")
    
    def get_table(self, table_id: int) -> pd.DataFrame:
        """Get table by ID from memory-mapped cache"""
        if self.mmap is None:
            raise RuntimeError("Cache not loaded")
        
        if table_id not in self.table_index:
            raise KeyError(f"Table {table_id} not found in cache")
        
        index_entry = self.table_index[table_id]
        offset = index_entry['offset']
        size = index_entry['size']
        
        # Read from memory-mapped region
        self.mmap.seek(offset)
        table_bytes = self.mmap.read(size)
        
        # Deserialize table
        return pickle.loads(table_bytes)
    
    def close(self):
        """Close memory-mapped files"""
        if hasattr(self, 'mmap') and self.mmap:
            self.mmap.close()
        if self.mmap_file:
            self.mmap_file.close()
    
    def __del__(self):
        self.close()


class MemoryMappedPretrainDataset(data.Dataset):
    """Memory-mapped version of PretrainTableDataset for faster I/O"""
    
    def __init__(self, path: str, hp, use_cache: bool = True):
        self.path = path
        self.hp = hp
        
        # Initialize tokenizer and other components from original dataset
        from .dataset import PretrainTableDataset
        self.original_dataset = PretrainTableDataset.from_hp(path, hp)
        
        # Setup memory-mapped cache
        self.cache_dir = os.path.join(path, '.mmap_cache')
        self.table_cache = MemoryMappedTableCache(self.cache_dir)
        
        if use_cache:
            try:
                self.table_cache.load_cache()
                logger.info("Using existing memory-mapped cache")
            except FileNotFoundError:
                logger.info("Creating new memory-mapped cache...")
                self._create_cache()
        else:
            self._create_cache()
    
    def _create_cache(self):
        """Create memory-mapped cache from tables"""
        # Get table paths
        table_paths = []
        for table_name in self.original_dataset.tables:
            table_path = os.path.join(self.path, table_name)
            table_paths.append(table_path)
        
        # Create cache
        self.table_cache.create_cache(table_paths)
        self.table_cache.load_cache()
    
    def _read_table(self, table_id: int) -> pd.DataFrame:
        """Fast table reading using memory-mapped cache"""
        return self.table_cache.get_table(table_id)
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        """Get item with memory-mapped table access"""
        # Use original dataset logic but with faster table reading
        table_ori = self._read_table(idx)
        
        # Apply single column sampling
        if self.hp.single_column:
            col = np.random.choice(table_ori.columns)
            table_ori = table_ori[[col]]
        
        # Apply augmentation
        from .augment import augment
        if ',' in self.hp.augment_op:
            op1, op2 = self.hp.augment_op.split(',')
            table_tmp = table_ori.copy()
            table_ori = augment(table_tmp, op1)
            table_aug = augment(table_tmp.copy(), op2)
        else:
            table_aug = augment(table_ori.copy(), self.hp.augment_op)
        
        # Tokenize
        x_ori, mp_ori = self.original_dataset._tokenize(table_ori)
        x_aug, mp_aug = self.original_dataset._tokenize(table_aug)
        
        # Column mapping
        cls_indices = []
        for col in mp_ori:
            if col in mp_aug:
                cls_indices.append((mp_ori[col], mp_aug[col]))
        
        return x_ori, x_aug, cls_indices
    
    def pad(self, batch):
        """Use original dataset's pad function"""
        return self.original_dataset.pad(batch)
    
    def close(self):
        """Close memory-mapped resources"""
        self.table_cache.close()
    
    def __del__(self):
        self.close()