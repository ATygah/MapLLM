import torch
from typing import List, Tuple, Dict, Optional
from .task_scheduler import TaskScheduler

class PE:
    """Processing Element for neural network computation."""
    def __init__(self, x: int, y: int, memory_size: int = 64 * 1024):
        self.x = x
        self.y = y
        self.memory_size = memory_size  # in bytes
        self.weights = None
        self.weight_shape = None
        self.layer_id = None
        self.split_dim = None  # The dimension along which weights are split
        self.row_start = None  # Start index of the split
        self.row_end = None  # End index of the split
        self.col_start = None  # Start index of the split
        self.col_end = None  # End index of the split
        
    def set_weights(self, weights: torch.Tensor, layer_id: int, split_dim: int, 
                    row_start: int = None, row_end: int = None, 
                    col_start: int = None, col_end: int = None):
        self.weights = weights
        self.weight_shape = weights.shape
        self.layer_id = layer_id
        self.split_dim = split_dim
        
        # Always set both row and column ranges regardless of split dimension
        self.row_start = row_start
        self.row_end = row_end
        self.col_start = col_start
        self.col_end = col_end
    
    def get_weights_size(self) -> int:
        """Get the size of weights in bytes."""
        if self.weights is None:
            return 0
        return self.weights.element_size() * self.weights.nelement()
    
    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform matrix multiplication with stored weights."""
        if self.weights is None:
            raise ValueError(f"PE({self.x}, {self.y}) has no weights assigned")
        
        # For a fully connected layer, we're doing input @ weights
        return torch.matmul(input_tensor, self.weights)
    
    def __repr__(self):
        return (f"PE({self.x}, {self.y}), Layer: {self.layer_id}, "
                f"Split: {self.split_dim}, Range: {self.row_start}:{self.row_end}, {self.col_start}:{self.col_end}")

class NoCTopology:
    """Manages the Network-on-Chip topology."""
    def __init__(self, rows: int, cols: int, pe_memory_size: int = 64 * 1024, 
                 channel_bandwidth: float = 32.0, data_type_bytes: int = 2):
        self.rows = rows
        self.cols = cols
        self.grid = {}
        self.pe_memory_size = pe_memory_size
        self.scheduler = TaskScheduler(channel_bandwidth=channel_bandwidth, 
                                      data_type_bytes=data_type_bytes)
        
        # Initialize grid
        for x in range(cols):
            for y in range(rows):
                self.grid[(x, y)] = PE(x, y, pe_memory_size)
    
    def get_pe(self, x: int, y: int) -> Optional[PE]:
        """Get PE at specified coordinates."""
        return self.grid.get((x, y)) 