import os
import pickle
import torch
import random
import pandas as pd
from torch.utils import data
from transformers import AutoTokenizer
from .augment import augment
from .preprocessor import computeTfIdf, preprocess
from typing import List, Tuple, Dict, Any
import hashlib
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased', 
         'distilbert': 'distilbert-base-uncased'}


class CachedPretrainTableDataset(data.Dataset):
    """Cached version of PretrainTableDataset that pre-computes expensive operations"""
    
    def __init__(self, 
                 path: str,
                 augment_op: str = 'drop_col,sample_row',
                 lm: str = 'roberta',
                 max_len: int = 128,
                 size: int = 10000,
                 single_column: bool = False,
                 sample_meth: str = 'head',
                 table_order: str = 'column',
                 cache_dir: str = None):
        
        self.path = path
        self.augment_op = augment_op
        self.lm = lm
        self.max_len = max_len
        self.size = size
        self.single_column = single_column
        self.sample_meth = sample_meth
        self.table_order = table_order
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        
        # Setup caching
        if cache_dir is None:
            cache_dir = os.path.join(path, '.cache')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache key based on parameters
        self.cache_key = self._generate_cache_key()
        self.cache_file = os.path.join(cache_dir, f"cached_dataset_{self.cache_key}.pkl")
        
        # Load or create cached data
        self._load_or_create_cache()
    
    def _generate_cache_key(self) -> str:
        """Generate unique cache key based on dataset parameters"""
        params = {
            'path': self.path,
            'augment_op': self.augment_op,
            'lm': self.lm,
            'max_len': self.max_len,
            'size': self.size,
            'single_column': self.single_column,
            'sample_meth': self.sample_meth,
            'table_order': self.table_order
        }
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()[:16]
    
    def _load_or_create_cache(self):
        """Load cached data or create it if not exists"""
        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached dataset from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.cached_samples = cache_data['samples']
                self.tfidf_cache = cache_data.get('tfidf_cache', {})
            logger.info(f"Loaded {len(self.cached_samples)} cached samples")
        else:
            logger.info("Cache not found, creating new cache...")
            self._create_cache()
    
    def _create_cache(self):
        """Pre-process and cache all data"""
        # Get table list
        if os.path.isdir(self.path):
            tables = []
            for fn in os.listdir(self.path):
                if fn.endswith('.csv'):
                    tables.append(fn)
            self.tables = sorted(tables)[:self.size]
        else:
            raise ValueError(f"Path {self.path} is not a directory")
        
        logger.info(f"Pre-processing {len(self.tables)} tables...")
        
        self.cached_samples = []
        self.tfidf_cache = {}
        
        # Pre-compute TF-IDF for all tables if needed
        if "tfidf" in self.sample_meth:
            logger.info("Pre-computing TF-IDF statistics...")
            for i, table_fn in enumerate(tqdm(self.tables, desc="Computing TF-IDF")):
                table_path = os.path.join(self.path, table_fn)
                table = pd.read_csv(table_path, lineterminator='\n')
                self.tfidf_cache[i] = computeTfIdf(table)
        
        # Pre-process all samples
        logger.info("Pre-processing samples...")
        for idx in tqdm(range(len(self.tables)), desc="🚀 Processing tables", unit="table"):
            try:
                # Read table
                table_fn = self.tables[idx]
                table_path = os.path.join(self.path, table_fn)
                print(f"📊 Processing table {idx+1}/{len(self.tables)}: {table_fn}")
                table_ori = pd.read_csv(table_path, lineterminator='\n')
                print(f"   📏 Size: {len(table_ori)} rows × {len(table_ori.columns)} columns")
                
                # Apply single column sampling if needed
                if self.single_column and len(table_ori.columns) > 1:
                    # For caching, we'll store multiple column variants
                    col_samples = min(3, len(table_ori.columns))  # Cache up to 3 column variants
                    selected_columns = random.sample(list(table_ori.columns), col_samples)
                else:
                    selected_columns = [None]  # Use full table
                
                for col in selected_columns:
                    table_base = table_ori[[col]] if col else table_ori
                    
                    # Pre-compute augmented versions
                    if ',' in self.augment_op:
                        op1, op2 = self.augment_op.split(',')
                        table_aug1 = augment(table_base.copy(), op1)
                        table_aug2 = augment(table_base.copy(), op2)
                        aug_pairs = [(table_base, table_aug1), (table_base, table_aug2)]
                    else:
                        table_aug = augment(table_base.copy(), self.augment_op)
                        aug_pairs = [(table_base, table_aug)]
                    
                    # Pre-tokenize all variants
                    for table_ori_variant, table_aug_variant in aug_pairs:
                        try:
                            tfidf_dict = self.tfidf_cache.get(idx) if "tfidf" in self.sample_meth else None
                            
                            print(f"   🔤 Tokenizing variants...")
                            x_ori, mp_ori = self._tokenize(table_ori_variant, tfidf_dict)
                            x_aug, mp_aug = self._tokenize(table_aug_variant, tfidf_dict)
                            print(f"   ✅ Tokenized: {len(x_ori)} → {len(x_aug)} tokens")
                            
                            # Compute column mapping
                            cls_indices = []
                            for col_name in mp_ori:
                                if col_name in mp_aug:
                                    cls_indices.append((mp_ori[col_name], mp_aug[col_name]))
                            
                            # Store pre-processed sample
                            self.cached_samples.append({
                                'x_ori': x_ori,
                                'x_aug': x_aug,
                                'cls_indices': cls_indices,
                                'table_idx': idx
                            })
                        except Exception as e:
                            logger.warning(f"Failed to process variant for table {idx}: {e}")
                            continue
                            
            except Exception as e:
                logger.warning(f"Failed to process table {idx}: {e}")
                continue
        
        # Save cache
        logger.info(f"Saving {len(self.cached_samples)} samples to cache...")
        cache_data = {
            'samples': self.cached_samples,
            'tfidf_cache': self.tfidf_cache
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Cache created with {len(self.cached_samples)} samples")
    
    def _tokenize(self, table: pd.DataFrame, tfidf_dict: Dict = None) -> Tuple[List[int], Dict]:
        """Tokenize table (same logic as original but cleaner)"""
        if len(table) == 0:
            return [self.tokenizer.cls_token_id], {}
        
        max_tokens = 256 if 'head' not in self.sample_meth else min(256, len(table) * 5)
        
        res = []
        mapper = {}
        
        # Reorder table
        if self.table_order == 'column':
            table = table.T.reset_index().T  # column-wise
        
        for i, column in enumerate(table.columns):
            mapper[column] = len(res)
            res.append(self.tokenizer.cls_token_id)
            
            col_tokens = preprocess(table[column], tfidf_dict, max_tokens, self.sample_meth)
            col_text = ' '.join(map(str, col_tokens))
            
            budget = self.max_len - len(res)
            if budget > 0:
                encoded = self.tokenizer.encode(
                    text=col_text,
                    max_length=budget,
                    add_special_tokens=False,
                    truncation=True
                )
                res.extend(encoded)
        
        return res, mapper
    
    def __len__(self) -> int:
        return len(self.cached_samples)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        """Fast lookup from pre-computed cache"""
        sample = self.cached_samples[idx]
        return sample['x_ori'], sample['x_aug'], sample['cls_indices']
    
    def pad(self, batch):
        """Collate function for DataLoader"""
        x_ori_batch, x_aug_batch, cls_indices_batch = zip(*batch)
        
        # Pad sequences
        max_len_ori = max(len(x) for x in x_ori_batch)
        max_len_aug = max(len(x) for x in x_aug_batch)
        max_len = max(max_len_ori, max_len_aug)
        
        # Create padded tensors
        x_ori_padded = []
        x_aug_padded = []
        
        for x_ori, x_aug in zip(x_ori_batch, x_aug_batch):
            x_ori_pad = x_ori + [self.tokenizer.pad_token_id] * (max_len - len(x_ori))
            x_aug_pad = x_aug + [self.tokenizer.pad_token_id] * (max_len - len(x_aug))
            x_ori_padded.append(x_ori_pad)
            x_aug_padded.append(x_aug_pad)
        
        x_ori_tensor = torch.LongTensor(x_ori_padded)
        x_aug_tensor = torch.LongTensor(x_aug_padded)
        
        return x_ori_tensor, x_aug_tensor, cls_indices_batch
    
    @classmethod
    def from_hp(cls, path: str, hp) -> 'CachedPretrainTableDataset':
        """Create dataset from hyperparameters namespace"""
        return cls(
            path=path,
            augment_op=hp.augment_op,
            lm=hp.lm,
            max_len=hp.max_len,
            size=hp.size,
            single_column=hp.single_column,
            sample_meth=hp.sample_meth,
            table_order=getattr(hp, 'table_order', 'column'),
            cache_dir=getattr(hp, 'cache_dir', None)
        )