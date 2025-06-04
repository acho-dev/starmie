import time
import threading
from collections import defaultdict, deque
import torch
from torch.utils.data import DataLoader


class DataLoadingMonitor:
    """Monitor data loading performance during training"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.batch_times = deque(maxlen=window_size)
        self.table_loads = defaultdict(int)
        self.total_batches = 0
        self.start_time = None
        self.last_batch_time = None
        self.gpu_wait_time = 0
        self.data_load_time = 0
        
    def start_monitoring(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.last_batch_time = self.start_time
        print("🔍 Data loading monitor started")
        
    def batch_loaded(self, batch_info=None):
        """Record that a batch was loaded"""
        current_time = time.time()
        if self.last_batch_time:
            batch_time = current_time - self.last_batch_time
            self.batch_times.append(batch_time)
            
        self.last_batch_time = current_time
        self.total_batches += 1
        
        # Print stats every 10 batches
        if self.total_batches % 10 == 0:
            self._print_stats()
            
    def table_accessed(self, table_name):
        """Record table access"""
        self.table_loads[table_name] += 1
        
    def _print_stats(self):
        """Print current loading statistics"""
        if not self.batch_times:
            return
            
        avg_batch_time = sum(self.batch_times) / len(self.batch_times)
        recent_batch_time = self.batch_times[-1] if self.batch_times else 0
        batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
        
        print(f"📊 Batch {self.total_batches:4d} | "
              f"Speed: {batches_per_sec:5.1f} batch/s | "
              f"Avg time: {avg_batch_time*1000:5.1f}ms | "
              f"Last: {recent_batch_time*1000:5.1f}ms")
              
    def print_summary(self):
        """Print final summary"""
        if self.start_time:
            total_time = time.time() - self.start_time
            avg_speed = self.total_batches / total_time if total_time > 0 else 0
            
            print(f"\n📈 Data Loading Summary:")
            print(f"   Total batches: {self.total_batches}")
            print(f"   Total time: {total_time:.2f}s") 
            print(f"   Average speed: {avg_speed:.2f} batches/s")
            print(f"   Unique tables loaded: {len(self.table_loads)}")
            
            if self.table_loads:
                most_loaded = max(self.table_loads.items(), key=lambda x: x[1])
                print(f"   Most accessed table: {most_loaded[0]} ({most_loaded[1]} times)")


class MonitoredDataLoader:
    """DataLoader wrapper with monitoring"""
    
    def __init__(self, dataset, monitor=None, **dataloader_kwargs):
        self.dataset = dataset
        self.monitor = monitor or DataLoadingMonitor()
        self.dataloader = DataLoader(dataset, **dataloader_kwargs)
        
    def __iter__(self):
        self.monitor.start_monitoring()
        data_iter = iter(self.dataloader)
        
        for batch in data_iter:
            batch_start = time.time()
            
            # Record batch loading
            self.monitor.batch_loaded()
            
            yield batch
            
        self.monitor.print_summary()
        
    def __len__(self):
        return len(self.dataloader)


def create_monitored_dataloader(dataset, batch_size=32, **kwargs):
    """Create a monitored DataLoader"""
    monitor = DataLoadingMonitor()
    
    default_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,
        'collate_fn': dataset.pad
    }
    default_kwargs.update(kwargs)
    
    return MonitoredDataLoader(dataset, monitor, **default_kwargs)


# Patch dataset classes to include monitoring
def patch_dataset_for_monitoring(dataset_class):
    """Add monitoring to dataset class"""
    original_read_table = dataset_class._read_table
    
    def monitored_read_table(self, table_id):
        if hasattr(self, '_monitor'):
            table_name = self.tables[table_id] if hasattr(self, 'tables') else f"table_{table_id}"
            self._monitor.table_accessed(table_name)
        return original_read_table(self, table_id)
    
    dataset_class._read_table = monitored_read_table
    return dataset_class