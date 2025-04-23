import torch
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Set
import uuid
import math
from dataclasses import dataclass, field
from data_structs import dtype_size
import matplotlib.pyplot as plt

#TODO: Address the following issues:
# 1. Weight tensor dimension set in the TrafficTask class.
# 2. Do we need to maintain an input tensor?
# 3. Weight column spanning multiple PEs - partial sum needs to be computed.
# 4. Weight column doesn't fit in all the PEs along a Row or Column in the NoC so many rows/columns will be needed.
# 5. Can we create generic functions for sequence split, input dim split(partial sum), output dim split(many to many send)?

@dataclass
class TrafficTask:
    """Class representing a communication task between PEs."""
    task_id: str
    src_pe: Tuple[int, int]
    dest_pe: Tuple[int, int]
    tensor_shape: Tuple[int, ...]
    bytes_count: int
    cycle_count: int
    wait_ids: List[str] = field(default_factory=list)  # Changed from wait_id to wait_ids
    description: str = ""
    
    def __str__(self):
        wait_str = ", ".join(self.wait_ids) if self.wait_ids else "None"
        return (f"Task {self.task_id}: {self.src_pe} -> {self.dest_pe}, "
                f"Shape: {self.tensor_shape}, Bytes: {self.bytes_count}, "
                f"Cycles: {self.cycle_count}, Wait on: {wait_str}")

class TaskScheduler:
    """Manages and schedules communication tasks between PEs."""
    def __init__(self, channel_bandwidth: float = 32.0, data_type_bytes: int = 2):  # B/cycle
        self.tasks = {}
        self.traffic_table = []
        self.channel_bandwidth = channel_bandwidth  # B/cycle
        self.task_dependencies = {}
        self.data_type_bytes = data_type_bytes  # Size of data type in bytes
    
    def create_task(self, 
                   src_pe: Tuple[int, int],
                   dest_pe: Tuple[int, int],
                   tensor_shape: Tuple[int, ...],
                   wait_ids: Optional[List[str]] = None,  # Changed from wait_id
                   description: str = "") -> str:
        """Create a new communication task and return its ID."""
        task_id = str(uuid.uuid4())[:8]
        
        # Handle single wait_id for backward compatibility
        if not isinstance(wait_ids, list) and wait_ids is not None:
            wait_ids = [wait_ids]
        elif wait_ids is None:
            wait_ids = []
        
        # Calculate bytes and cycles - now using the parameterized data type size
        bytes_count = math.prod(tensor_shape) * self.data_type_bytes
        cycle_count = math.ceil(bytes_count / self.channel_bandwidth)
        
        task = TrafficTask(
            task_id=task_id,
            src_pe=src_pe,
            dest_pe=dest_pe,
            tensor_shape=tensor_shape,
            bytes_count=bytes_count,
            cycle_count=cycle_count,
            wait_ids=wait_ids,
            description=description
        )
        
        # TODO: What is the difference between tasks and traffic_table?     
        self.tasks[task_id] = task
        self.traffic_table.append(task)
        return task_id
    
    def get_traffic_table(self) -> pd.DataFrame:
        """Get the traffic table as a pandas DataFrame."""
        data = []
        for task in self.traffic_table:
            data.append({
                'task_id': task.task_id,
                'src_pe': f"({task.src_pe[0]}, {task.src_pe[1]})",
                'dest_pe': f"({task.dest_pe[0]}, {task.dest_pe[1]})" if task.dest_pe else "None",
                'tensor_shape': str(task.tensor_shape),
                'bytes': task.bytes_count,
                'cycles': task.cycle_count,
                'wait_ids': ", ".join(task.wait_ids) if task.wait_ids else "None",
                'description': task.description,
            })
        return pd.DataFrame(data)

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
        
        if split_dim in [0, 2]:  # Row or hybrid split
            self.row_start = row_start
            self.row_end = row_end
        
        if split_dim in [1, 2]:  # Column or hybrid split
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

class FCLayerMapper:
    """Maps fully connected layers to NoC topology."""
    def __init__(self, 
                 noc: NoCTopology, 
                 input_dim: int, 
                 output_dims: List[int],
                 seq_len: int,
                 mapping_strategy: str = "column_wise",
                 split_strategy: str = "column_split",
                 data_type: str = "float16"):
        """
        Initialize FCLayerMapper.
        
        Args:
            noc: The NoC topology to map layers onto
            input_dim: Input dimension of the network
            output_dims: List of output dimensions for each layer
            seq_len: Sequence length for inference
            mapping_strategy: How to map layers onto the NoC
                             - "column_wise": Each layer maps to a column
                             - "row_wise": Each layer maps to a row
                             - "grid_wise": Each layer maps to a 2D grid (optimized for hybrid split)
            split_strategy: How to split weight matrices
                           - "column_split": Split by columns (output dimension)
                           - "row_split": Split by rows (input dimension)
                           - "hybrid_split": Split by both dimensions
            data_type: The data type of the weights
        """
        self.noc = noc
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.seq_len = seq_len
        self.mapping_strategy = mapping_strategy
        self.split_strategy = split_strategy
        self.layer_count = len(output_dims)
        self.data_type = data_type
        self.data_type_bytes = dtype_size(self.data_type)
        
        # Maps (layer_id, pe_index) to PE coordinates
        self.layer_pe_map = {}
        # Maps PE coordinates to (layer_id, pe_index)
        self.pe_layer_map = {}
        
        # Validate and apply mapping strategy
        self._map_layers_to_pes()
    
    def _map_layers_to_pes(self):
        """Map layers to PEs based on splitting and mapping strategies."""
        # Calculate PE requirements for each layer based on split strategy
        layer_pe_requirements = self._calculate_pe_requirements()
        
        # Apply mapping strategy to place PEs on the NoC
        if self.mapping_strategy == "column_wise":
            self._map_to_columns(layer_pe_requirements)
        elif self.mapping_strategy == "row_wise":
            self._map_to_rows(layer_pe_requirements)
        elif self.mapping_strategy == "grid_wise":
            self._map_to_grid(layer_pe_requirements)
        else:
            raise ValueError(f"Unknown mapping strategy: {self.mapping_strategy}")
    
    def _calculate_pe_requirements(self):
        """
        Calculate how many PEs are needed for each layer and how the weight matrix
        should be split across them based on the split strategy.
        
        Returns:
            Dictionary mapping layer_id to a list of 
            (split_dim, split_start, split_end, col_split_start, col_split_end) tuples for each PE.
        """
        layer_pe_requirements = {}
        current_dim = self.input_dim
        
        for layer_id, output_dim in enumerate(self.output_dims):
            # Calculate weight matrix dimensions
            weight_matrix_rows = current_dim
            weight_matrix_cols = output_dim
            
            # Calculate weights per PE
            weights_per_pe = self.noc.pe_memory_size // self.data_type_bytes
            
            if self.split_strategy == "column_split":
                # Split along columns (output dimension)
                if weight_matrix_rows > weights_per_pe:
                    raise ValueError(f"Row size ({weight_matrix_rows}) exceeds PE capacity ({weights_per_pe})")
                
                cols_per_pe = max(1, weights_per_pe // weight_matrix_rows)
                pes_needed = math.ceil(weight_matrix_cols / cols_per_pe)
                
                # Calculate split ranges for each PE
                split_ranges = []
                
                for pe_idx in range(pes_needed):
                    col_start = pe_idx * cols_per_pe
                    col_end = min((pe_idx + 1) * cols_per_pe, weight_matrix_cols)
                    # split_dim=1 means splitting by columns
                    # Use full rows (0 to weight_matrix_rows) and specific columns
                    split_ranges.append((1, 0, weight_matrix_rows, col_start, col_end))
            
            elif self.split_strategy == "row_split":
                # Split along rows (input dimension)
                if weight_matrix_cols > weights_per_pe:
                    raise ValueError(f"Column size ({weight_matrix_cols}) exceeds PE capacity ({weights_per_pe})")
                
                rows_per_pe = max(1, weights_per_pe // weight_matrix_cols)
                pes_needed = math.ceil(weight_matrix_rows / rows_per_pe)
                
                # Calculate split ranges for each PE
                split_ranges = []
                for pe_idx in range(pes_needed):
                    row_start = pe_idx * rows_per_pe
                    row_end = min((pe_idx + 1) * rows_per_pe, weight_matrix_rows)
                    # split_dim=0 means splitting by rows
                    # Use specific rows and full columns (0 to weight_matrix_cols)
                    split_ranges.append((0, row_start, row_end, 0, weight_matrix_cols))
                
            elif self.split_strategy == "hybrid_split":
                # Split along both dimensions
                # For simplicity, we'll use a fixed grid division approach:
                # Determine how to divide the weight matrix into a grid of PEs
                
                # Find a balanced split across both dimensions
                # We want to minimize max(row_pes, col_pes) to keep the NoC more square-shaped
                best_row_pes = weight_matrix_rows
                best_col_pes = weight_matrix_cols
                best_max_pes = best_row_pes * best_col_pes
                
                # Try different splits to find a good balance
                for row_div in range(1, weight_matrix_rows + 1):
                    if weight_matrix_rows % row_div != 0:
                        continue  # Skip if not evenly divisible
                        
                    rows_per_pe = weight_matrix_rows // row_div
                    remaining_elements = weights_per_pe // rows_per_pe
                    
                    if remaining_elements == 0:
                        continue  # Can't fit even one row
                    
                    col_div = math.ceil(weight_matrix_cols / remaining_elements)
                    total_pes = row_div * col_div
                    
                    if total_pes < best_max_pes:
                        best_row_pes = row_div
                        best_col_pes = col_div
                        best_max_pes = total_pes
                
                # Calculate rows and columns per PE
                rows_per_pe = math.ceil(weight_matrix_rows / best_row_pes)
                cols_per_pe = math.ceil(weight_matrix_cols / best_col_pes)
                
                # Create the grid of PEs with both row and column splits
                split_ranges = []
                for row_idx in range(best_row_pes):
                    for col_idx in range(best_col_pes):
                        row_start = row_idx * rows_per_pe
                        row_end = min((row_idx + 1) * rows_per_pe, weight_matrix_rows)
                        
                        col_start = col_idx * cols_per_pe
                        col_end = min((col_idx + 1) * cols_per_pe, weight_matrix_cols)
                        
                        # For hybrid split, we store both row and column split information
                        split_ranges.append((2, row_start, row_end, col_start, col_end))
            else:
                raise ValueError(f"Unknown split strategy: {self.split_strategy}")
            
            layer_pe_requirements[layer_id] = split_ranges
            current_dim = output_dim
    
        return layer_pe_requirements
    
    def _map_to_columns(self, layer_pe_requirements):
        """
        Map layers to columns in the NoC.
        Each layer is mapped to a column, and multiple PEs in that column handle splits.
        For hybrid split, we arrange PEs in a 2D grid for each layer.
        
        Args:
            layer_pe_requirements: Dictionary mapping layer_id to a list of split information.
        """
        for layer_id, split_ranges in layer_pe_requirements.items():
            # Each layer gets a column
            col = layer_id
            
            # Check if we have enough rows
            if len(split_ranges) > self.noc.rows:
                raise ValueError(f"Layer {layer_id} requires {len(split_ranges)} PEs, "
                                 f"but NoC only has {self.noc.rows} rows")
            
            # For hybrid split, organize PEs in a grid
            if self.split_strategy == "hybrid_split":
                # Group PEs by their column split (each group handles a subset of output neurons)
                col_groups = {}
                for pe_idx, (split_type, row_start, row_end, col_start, col_end) in enumerate(split_ranges):
                    col_group = (col_start, col_end)
                    if col_group not in col_groups:
                        col_groups[col_group] = []
                    col_groups[col_group].append((pe_idx, (split_type, row_start, row_end, col_start, col_end)))
                
                # Place PEs in a more structured way
                pe_idx_map = {}  # Maps PE index to actual coordinates
                current_row = 0
                
                # For each column group, place all its PEs one after another
                for col_group, pe_list in col_groups.items():
                    for pe_idx, _ in pe_list:
                        pe_coords = (col, current_row)
                        pe_idx_map[pe_idx] = pe_coords
                        current_row += 1
            
            # Map each PE for this layer to the corresponding row in this column
            for pe_idx, split_info in enumerate(split_ranges):
                if self.split_strategy == "hybrid_split":
                    # For hybrid, use the pre-calculated coordinates
                    pe_coords = pe_idx_map[pe_idx]
                else:
                    # For non-hybrid, just place in sequence
                    pe_coords = (col, pe_idx)
                
                # Get the PE at these coordinates
                pe = self.noc.get_pe(*pe_coords)
                
                # Weight matrix dimensions
                if layer_id == 0:
                    input_dim = self.input_dim
                else:
                    input_dim = self.output_dims[layer_id]
                output_dim = self.output_dims[layer_id]
                
                # Unpack split information
                split_dim, row_start, row_end, col_start, col_end = split_info
                
                # Create weights tensor
                pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                
                # Update maps
                self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
            
    def _map_to_rows(self, layer_pe_requirements):
        """
        Map layers to rows in the NoC.
        Each layer is mapped to a row, and multiple PEs in that row handle splits.
        For hybrid split, we arrange PEs in a 2D grid for each layer.
        
        Args:
            layer_pe_requirements: Dictionary mapping layer_id to a list of split information.
        """
        for layer_id, split_ranges in layer_pe_requirements.items():
            # Each layer gets a row
            row = layer_id
            
            # Check if we have enough columns
            if len(split_ranges) > self.noc.cols:
                raise ValueError(f"Layer {layer_id} requires {len(split_ranges)} PEs, "
                                 f"but NoC only has {self.noc.cols} columns")
            
            # For hybrid split, organize PEs in a grid
            if self.split_strategy == "hybrid_split":
                # Group PEs by their column split (each group handles a subset of output neurons)
                col_groups = {}
                for pe_idx, (split_type, row_start, row_end, col_start, col_end) in enumerate(split_ranges):
                    col_group = (col_start, col_end)
                    if col_group not in col_groups:
                        col_groups[col_group] = []
                    col_groups[col_group].append((pe_idx, (split_type, row_start, row_end, col_start, col_end)))
                
                # Place PEs in a more structured way
                pe_idx_map = {}  # Maps PE index to actual coordinates
                current_col = 0
                
                # For each column group, place all its PEs one after another
                for col_group, pe_list in col_groups.items():
                    for pe_idx, _ in pe_list:
                        pe_coords = (current_col, row)
                        pe_idx_map[pe_idx] = pe_coords
                        current_col += 1
            
            # Map each PE for this layer to the corresponding column in this row
            for pe_idx, split_info in enumerate(split_ranges):
                if self.split_strategy == "hybrid_split":
                    # For hybrid, use the pre-calculated coordinates
                    pe_coords = pe_idx_map[pe_idx]
                else:
                    # For non-hybrid, just place in sequence
                    pe_coords = (pe_idx, row)
                
                # Get the PE at these coordinates
                pe = self.noc.get_pe(*pe_coords)
                
                # Weight matrix dimensions
                if layer_id == 0:
                    input_dim = self.input_dim
                else:
                    input_dim = self.output_dims[layer_id - 1]
                output_dim = self.output_dims[layer_id]
                
                # Unpack split information
                split_dim, row_start, row_end, col_start, col_end = split_info
                
                # Create weights tensor
                pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                
                # Update maps
                self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
    
    def _map_to_grid(self, layer_pe_requirements):
        """
        Map layers to 2D grids in the NoC.
        Each layer is mapped to a rectangular region, attempting to minimize total NoC dimensions.
        """
        # TODO: Add layer compaction techniques. Additional point for the paper coming for free. Priority: Low.
        # Track current placement position
        current_x = 0
        current_y = 0
        max_height_in_row = 0
        
        # For better packing, sort layers by their width (for hybrid split)
        # This places the widest layers first, which tends to use space more efficiently
        layer_widths = {}
        for layer_id, split_ranges in layer_pe_requirements.items():
            if self.split_strategy == "hybrid_split":
                # Count unique column groups
                col_groups = set()
                for _, _, _, col_start, col_end in split_ranges:
                    col_groups.add((col_start, col_end))
                layer_widths[layer_id] = len(col_groups)
            else:
                # For non-hybrid, estimate width as square root of PE count
                layer_widths[layer_id] = int(math.sqrt(len(split_ranges)))
        
        # Process layers in order of decreasing width for more efficient packing
        sorted_layers = sorted(layer_pe_requirements.keys(), 
                              key=lambda layer_id: layer_widths[layer_id], 
                              reverse=True)
        
        # Process layers in sequence
        for layer_id in sorted_layers:
            split_ranges = layer_pe_requirements[layer_id]
            
            # For hybrid split, determine an efficient grid layout
            if self.split_strategy == "hybrid_split":
                # Group PEs by their column and row ranges
                col_groups = {}  # Maps column range to list of PE indices
                row_groups = {}  # Maps row range to list of PE indices
                
                for pe_idx, (split_type, row_start, row_end, col_start, col_end) in enumerate(split_ranges):
                    col_group = (col_start, col_end)
                    row_group = (row_start, row_end)
                    
                    if col_group not in col_groups:
                        col_groups[col_group] = []
                    col_groups[col_group].append(pe_idx)
                    
                    if row_group not in row_groups:
                        row_groups[row_group] = []
                    row_groups[row_group].append(pe_idx)
                
                # Number of PEs in row and column dimensions for this layer
                num_row_groups = len(row_groups)
                num_col_groups = len(col_groups)
                
                # Check if we need to start a new row due to insufficient width
                if current_x + num_col_groups > self.noc.cols:
                    current_x = 0
                    current_y += max_height_in_row
                    max_height_in_row = 0
                
                # If starting a new row would put us out of bounds, we have a problem
                if current_y + num_row_groups > self.noc.rows:
                    raise ValueError(f"NoC dimensions ({self.noc.rows}x{self.noc.cols}) insufficient for all layers. "
                                   f"Layer {layer_id} needs {num_row_groups}x{num_col_groups} PEs, "
                                   f"but only {self.noc.rows-current_y}x{self.noc.cols-current_x} available.")
                
                # Create a grid placement for this layer
                # TODO: We are checking per layer right now.
                layer_grid_width = num_col_groups
                layer_grid_height = num_row_groups
                
                # Sort row and column groups by their ranges for consistent mapping
                sorted_row_groups = sorted(row_groups.items())
                sorted_col_groups = sorted(col_groups.items())
                
                # Keep track of PE positions in the grid
                pe_positions = {}  # Maps pe_idx to (row_idx, col_idx) within layer grid
                
                # Assign positions within the layer's grid
                for row_idx, (row_range, row_pe_indices) in enumerate(sorted_row_groups):
                    for col_idx, (col_range, col_pe_indices) in enumerate(sorted_col_groups):
                        # Find PEs that belong to both this row and column group
                        for intersect_pe_idx in set(row_pe_indices) & set(col_pe_indices):
                            pe_positions[intersect_pe_idx] = (row_idx, col_idx)
                
                # Map each PE to its position in the NoC
                for pe_idx, (split_type, row_start, row_end, col_start, col_end) in enumerate(split_ranges):
                    if pe_idx not in pe_positions:
                        # This shouldn't happen, but skip if not in the mapping
                        continue
                        
                    row_idx, col_idx = pe_positions[pe_idx]
                    
                    # Calculate absolute NoC coordinates
                    noc_x = current_x + col_idx
                    noc_y = current_y + row_idx
                    
                    # Final check to ensure coordinates are in bounds
                    if noc_x >= self.noc.cols or noc_y >= self.noc.rows:
                        raise ValueError(f"PE coordinates ({noc_x}, {noc_y}) out of bounds for NoC dimensions "
                                       f"({self.noc.rows}x{self.noc.cols}) for layer {layer_id}")
                    
                    pe_coords = (noc_x, noc_y)
                    pe = self.noc.get_pe(*pe_coords)
                    
                    # Create weights tensor
                    pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                    pe.set_weights(pe_weights, layer_id, split_type, row_start, row_end, col_start, col_end)
                    
                    # Update maps
                    self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                    self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
                
                # Update position trackers
                current_x += layer_grid_width
                max_height_in_row = max(max_height_in_row, layer_grid_height)
            else:
                # For non-hybrid splits, place PEs in a simple square-ish grid
                num_pes = len(split_ranges)
                
                # Calculate an efficient grid shape for this layer
                # Try to keep it relatively square
                grid_height = int(math.sqrt(num_pes * 1.5))
                grid_width = math.ceil(num_pes / grid_height)
                
                # Check if we need to start a new row
                if current_x + grid_width > self.noc.cols:
                    current_x = 0
                    current_y += max_height_in_row
                    max_height_in_row = 0
                
                # Map each PE
                for pe_idx, split_info in enumerate(split_ranges):
                    # Calculate position within the layer's grid
                    local_x = pe_idx % grid_width
                    local_y = pe_idx // grid_width
                    
                    # Calculate absolute NoC coordinates
                    noc_x = current_x + local_x
                    noc_y = current_y + local_y
                    
                    # Check if coordinates are valid
                    if noc_x >= self.noc.cols or noc_y >= self.noc.rows:
                        raise ValueError(f"NoC dimensions ({self.noc.rows}x{self.noc.cols}) too small for layer {layer_id}")
                    
                    pe_coords = (noc_x, noc_y)
                    pe = self.noc.get_pe(*pe_coords)
                    
                    # Unpack split information
                    split_dim, row_start, row_end, col_start, col_end = split_info
                    
                    # Create weights tensor
                    pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                    pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                    
                    # Update maps
                    self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                    self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
                
                # Update position trackers
                current_x += grid_width
                max_height_in_row = max(max_height_in_row, grid_height)
    
    def get_layer_pes(self, layer_id: int) -> List[Tuple[int, int]]:
        """Get all PE coordinates for a specific layer."""
        return [coords for (lid, _), coords in self.layer_pe_map.items() if lid == layer_id]
    
    def get_pe_details(self):
        """Get details of all mapped PEs."""
        details = []
        for pe_coords, (layer_id, pe_idx) in self.pe_layer_map.items():
            pe = self.noc.get_pe(*pe_coords)
            
            # Format the split range as a compact tensor shape representation
            if pe.split_dim == 0:  # Row split
                split_format = f"({pe.row_start}:{pe.row_end}, :)"  # All columns
            elif pe.split_dim == 1:  # Column split
                # Check for null values before formatting
                if pe.col_start is not None and pe.col_end is not None:
                    split_format = f"(:, {pe.col_start}:{pe.col_end})"  # All rows
                else:
                    split_format = "(:, None:None)"
            elif pe.split_dim == 2:  # Hybrid split
                # Check for null values before formatting
                row_part = f"{pe.row_start}:{pe.row_end}" if pe.row_start is not None and pe.row_end is not None else "None:None"
                col_part = f"{pe.col_start}:{pe.col_end}" if pe.col_start is not None and pe.col_end is not None else "None:None"
                split_format = f"({row_part}, {col_part})"
            else:
                split_format = "unknown split"
                
            details.append({
                'pe_coords': pe_coords,
                'layer_id': layer_id,
                'pe_idx': pe_idx,
                'split_dim': pe.split_dim,
                'weight_tensor': split_format,
                'weight_shape': pe.weight_shape
            })
        return pd.DataFrame(details)
    
    # TODO: Fix assign_aggregation_pes function
    # TODO: Fix get effective_noc_dimension function
    def assign_aggregation_pes(self, neural_network):
        """
        Assign dedicated PEs for aggregation when not reusing computation PEs.
        Updated to handle grid_wise mapping strategy and initialize column ranges.
        
        Args:
            neural_network: FCNeuralNetwork instance
            
        Returns:
            Tuple of (aggregation_pes, row_aggregation_pes, column_aggregation_pes) dictionaries
        """
        # Initialize aggregation PE dictionaries
        aggregation_pes = {}
        row_aggregation_pes = {}
        column_aggregation_pes = {}
        
        # For each layer, find a PE that's not used for computation
        all_computation_pes = set()
        for layer_id in range(len(neural_network.layer_dims)):
            layer_pes = self.get_layer_pes(layer_id)
            for pe_coords in layer_pes:
                all_computation_pes.add(pe_coords)
        
        # Keep track of all used PEs (computation + aggregation)
        all_used_pes = set(all_computation_pes)
        
        # For each layer, assign aggregation PEs
        for layer_id in range(len(neural_network.layer_dims)):
            # Check if this is a strategy that might need row-wise aggregation
            if neural_network.split_strategy == "hybrid_split" or neural_network.split_strategy == "row_split":
                # Group by column group to see if we actually have a row split
                pe_by_col_group = {}
                for pe_coords in self.get_layer_pes(layer_id):
                    pe = self.noc.get_pe(*pe_coords)
                    if hasattr(pe, 'col_start') and hasattr(pe, 'col_end') and pe.col_start is not None and pe.col_end is not None:
                        col_group = (pe.col_start, pe.col_end)
                        if col_group not in pe_by_col_group:
                            pe_by_col_group[col_group] = []
                        pe_by_col_group[col_group].append(pe_coords)
                
                # Only proceed with row aggregation if we have multiple column groups
                # or if any group doesn't cover the full output dimension
                needs_row_aggregation = False
                if len(pe_by_col_group) > 1:
                    needs_row_aggregation = True
                else:
                    # Check if the single column group covers the entire output dimension
                    for col_group in pe_by_col_group:
                        col_start, col_end = col_group
                        if col_start > 0 or col_end < neural_network.layer_dims[layer_id]:
                            needs_row_aggregation = True
                            break
                
                if needs_row_aggregation:
                    # Assign an aggregation PE for each column group
                    for col_group, pes in pe_by_col_group.items():
                        col_start, col_end = col_group
                        assigned = False
                        
                        # Search for an unused PE in each direction from the group's PEs
                        for pe_coords in pes:
                            x, y = pe_coords
                            
                            # Try nearby positions in all four directions
                            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                neighbor_coords = (x + dx, y + dy)
                                
                                # Check if this position is valid and unused
                                if (0 <= neighbor_coords[0] < self.noc.cols and 
                                    0 <= neighbor_coords[1] < self.noc.rows and
                                    neighbor_coords not in all_used_pes):
                                    
                                    if neural_network.split_strategy == "hybrid_split":
                                        row_aggregation_pes[(layer_id, col_group)] = neighbor_coords
                                    else:  # row_split
                                        aggregation_pes[layer_id] = neighbor_coords
                                    
                                    all_used_pes.add(neighbor_coords)
                                    
                                    # Initialize the aggregation PE with appropriate column range
                                    agg_pe = self.noc.get_pe(*neighbor_coords)
                                    agg_pe.col_start = col_start
                                    agg_pe.col_end = col_end
                                    
                                    assigned = True
                                    break
                            
                            if assigned:
                                break
                        
                        # If still no aggregation PE found, try any unused PE
                        if not assigned:
                            for x in range(self.noc.cols):
                                for y in range(self.noc.rows):
                                    if (x, y) not in all_used_pes:
                                        if neural_network.split_strategy == "hybrid_split":
                                            row_aggregation_pes[(layer_id, col_group)] = (x, y)
                                        else:  # row_split
                                            aggregation_pes[layer_id] = (x, y)
                                        
                                        all_used_pes.add((x, y))
                                        
                                        # Initialize the aggregation PE with appropriate column range
                                        agg_pe = self.noc.get_pe(x, y)
                                        agg_pe.col_start = col_start
                                        agg_pe.col_end = col_end
                                        
                                        assigned = True
                                        break
                                if assigned:
                                    break
                        
                        # If we couldn't find any unused PE, that's an error
                        if not assigned:
                            if neural_network.split_strategy == "hybrid_split":
                                raise ValueError(f"Could not find an unused PE for row aggregation in layer {layer_id}, "
                                                f"column group {col_group}. Consider increasing NoC dimensions or "
                                                f"using reuse_pe_for_aggregation=True.")
                            else:  # row_split
                                raise ValueError(f"Could not find an unused PE for aggregation in layer {layer_id}. "
                                               f"Consider increasing NoC dimensions or using reuse_pe_for_aggregation=True.")
                
                # Check if we need column aggregation for hybrid_split
                if neural_network.split_strategy == "hybrid_split":
                    # Check if we have multiple PEs working on different row ranges for any column
                    needs_column_aggregation = False
                    
                    # Group PEs by column to check for row splits
                    pes_by_col = {}
                    for pe_coords in self.get_layer_pes(layer_id):
                        pe = self.noc.get_pe(*pe_coords)
                        if hasattr(pe, 'col_start') and hasattr(pe, 'col_end'):
                            # For each column in the range, add this PE
                            for col in range(pe.col_start, pe.col_end):
                                if col not in pes_by_col:
                                    pes_by_col[col] = []
                                pes_by_col[col].append(pe_coords)
                    
                    # If any column has multiple PEs working on it, we need column aggregation
                    for col, pes in pes_by_col.items():
                        if len(pes) > 1:
                            needs_column_aggregation = True
                            break
                    
                    if needs_column_aggregation:
                        assigned = False
                        for x in range(self.noc.cols):
                            for y in range(self.noc.rows):
                                if (x, y) not in all_used_pes:
                                    column_aggregation_pes[layer_id] = (x, y)
                                    all_used_pes.add((x, y))
                                    
                                    # Initialize the column aggregation PE with full output range
                                    col_agg_pe = self.noc.get_pe(x, y)
                                    col_agg_pe.col_start = 0
                                    col_agg_pe.col_end = neural_network.layer_dims[layer_id]
                                    
                                    assigned = True
                                    break
                            if assigned:
                                break
                        
                        # If we couldn't find an unused PE for column aggregation
                        if not assigned:
                            raise ValueError(f"Could not find an unused PE for column aggregation in layer {layer_id}. "
                                           f"Consider increasing NoC dimensions or using reuse_pe_for_aggregation=True.")
            
            elif neural_network.split_strategy == "column_split":  # column_split only
                # For column_split, we need just one aggregation PE per layer
                assigned = False
                
                # First try to find a nearby unused PE
                layer_pes = self.get_layer_pes(layer_id)
                for pe_coords in layer_pes:
                    x, y = pe_coords
                    
                    # Try nearby positions in all four directions
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        neighbor_coords = (x + dx, y + dy)
                        
                        # Check if this position is valid and unused
                        if (0 <= neighbor_coords[0] < self.noc.cols and 
                            0 <= neighbor_coords[1] < self.noc.rows and
                            neighbor_coords not in all_used_pes):
                            
                            aggregation_pes[layer_id] = neighbor_coords
                            all_used_pes.add(neighbor_coords)
                            assigned = True
                            break
                    
                    if assigned:
                        break
                
                # If no nearby PE found, try any unused PE
                if not assigned:
                    for x in range(self.noc.cols):
                        for y in range(self.noc.rows):
                            if (x, y) not in all_used_pes:
                                aggregation_pes[layer_id] = (x, y)
                                all_used_pes.add((x, y))
                                assigned = True
                                break
                        if assigned:
                            break
                
                # If we couldn't find any unused PE, that's an error
                if not assigned:
                    raise ValueError(f"Could not find an unused PE for aggregation in layer {layer_id}. "
                                   f"Consider increasing NoC dimensions or using reuse_pe_for_aggregation=True.")
        
        return aggregation_pes, row_aggregation_pes, column_aggregation_pes

    def get_effective_noc_dimensions(self, nn):
        """
        Calculate the effective NoC dimensions based on actual PE usage.
        
        Args:
            nn: FCNeuralNetwork instance
            
        Returns:
            Tuple of (effective_rows, effective_cols, effective_grid_size)
        """
        # Get all active PEs
        active_pes = nn.active_pes
        
        print(f"Active PEs: {active_pes}")
        
        if not active_pes:
            return 0, 0, 0
            
        # Get unique row and column coordinates
        unique_rows = set(y for x, y in active_pes)
        unique_cols = set(x for x, y in active_pes)
        
        # Calculate the span of rows and columns
        effective_rows = len(unique_rows)
        effective_cols = len(unique_cols)
        
        # Calculate effective grid size
        effective_grid_size = effective_rows * effective_cols
        
        return effective_rows, effective_cols, effective_grid_size

class FCNeuralNetwork:
    """Fully Connected Neural Network on NoC."""
    def __init__(self, 
                 input_dim: int, 
                 layer_dims: List[int], 
                 seq_len: int = 1,
                 pe_memory_size: int = 64 * 1024,
                 mapping_strategy: str = "column_wise",
                 split_strategy: str = "column_split",
                 reuse_pe_for_aggregation: bool = True,
                 data_type: str = "float16",
                 channel_bandwidth: float = 32.0,
                 noc_rows: int = None,
                 noc_cols: int = None):
        """
        Initialize a Fully Connected Neural Network on NoC.
        
        Args:
            input_dim: Input dimension
            layer_dims: List of output dimensions for each layer
            seq_len: Sequence length for inference
            pe_memory_size: Memory size per PE in bytes
            mapping_strategy: How to map layers onto the NoC
                             - "column_wise": Each layer maps to a column
                             - "row_wise": Each layer maps to a row
                             - "grid_wise": Each layer maps to a 2D grid (optimized for hybrid split)
            split_strategy: How to split weight matrices
                           - "column_split": Split by columns (output dimension)
                           - "row_split": Split by rows (input dimension)
                           - "hybrid_split": Split by both rows and columns
            reuse_pe_for_aggregation: Whether to reuse computation PEs for aggregation
            data_type: The data type of the weights
            channel_bandwidth: The bandwidth of the channels in B/cycle
            noc_rows: Number of rows in the NoC grid (if None, will be calculated)
            noc_cols: Number of columns in the NoC grid (if None, will be calculated)
        """
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.seq_len = seq_len
        self.data_type = data_type
        self.split_strategy = split_strategy
        self.mapping_strategy = mapping_strategy
        self.reuse_pe_for_aggregation = reuse_pe_for_aggregation
        self.data_type_bytes = dtype_size(self.data_type)
        
        # Set to track active PEs (used for computation or aggregation)
        self.active_pes = set()
        
        # Calculate NoC dimensions based on strategies if not provided
        # TODO: _calculate_noc_dimension will exactly replicate the code for FCLayerMapper and aggregation PEs. 
        #       I have an idea for fixing the _noc_dimension issue. We can decouple pe_placement. First FCLayerMapper
        #       will run and map the PE without placing them, i.e, pe.set_weights and all. I have added that in
        #       the context of this chat. Then we can run aggregation_pes to find the aggregation PEs. THen we can
        #       initialize the NoCTopology with the found rows and columns. The rows and columns will be calculated
        #       similar to the get_effective_noc_dimension function.
        if noc_rows is None or noc_cols is None:
            noc_rows, noc_cols = self._calculate_noc_dimensions(
                pe_memory_size, mapping_strategy, split_strategy, reuse_pe_for_aggregation)
        
        # Create NoC topology, passing the data_type_bytes
        self.noc = NoCTopology(noc_rows, noc_cols, pe_memory_size, 
                              channel_bandwidth=channel_bandwidth,
                              data_type_bytes=self.data_type_bytes)
        
        # For hybrid strategy, we need additional mapping information
        self.row_aggregation_pes = {}  # Maps (layer_id, col_group) to row aggregation PE
        self.column_aggregation_pes = {}  # Maps layer_id to final aggregation PE
        
        # Map layers to PEs
        self.mapper = FCLayerMapper(
            self.noc, input_dim, layer_dims, seq_len, mapping_strategy, split_strategy, data_type)
        
        # Track dedicated aggregation PEs
        self.aggregation_pes = {}
        if (not reuse_pe_for_aggregation and 
            (split_strategy == "row_split" or split_strategy == "hybrid_split")):
            self.aggregation_pes, self.row_aggregation_pes, self.column_aggregation_pes = self.mapper.assign_aggregation_pes(self)
        
        # Register active PEs after mapping is complete
        self._register_active_pes()
    
    def _register_active_pes(self):
        """Register all active PEs (computation and aggregation) in the active_pes set."""
        # Add all computation PEs
        for pe_coords in self.mapper.pe_layer_map.keys():
            self.active_pes.add(pe_coords)
        print(f"Active PEs registered: {self.active_pes}")
        
        # Add aggregation PEs for column_split or row_split
        for pe_coords in self.aggregation_pes.values():
            self.active_pes.add(pe_coords)
        print(f"Active PEs registered row split: {self.active_pes}")
        
        # Add row aggregation PEs for hybrid_split
        for pe_coords in self.row_aggregation_pes.values():
            self.active_pes.add(pe_coords)
        print(f"Active PEs including row aggregation PEs: {self.active_pes}")
        
        # Add column aggregation PEs for hybrid_split
        for pe_coords in self.column_aggregation_pes.values():
            self.active_pes.add(pe_coords)
        print(f"Active PEs after including column aggregation PEs: {self.active_pes}")
    
    def _calculate_noc_dimensions(self, pe_memory_size, mapping_strategy, split_strategy, reuse_pe_for_aggregation):
        """Calculate dimensions of the NoC based on strategies."""
        # Calculate PE requirements for each layer
        layer_pe_counts = {}
        row_pe_counts = {}  # For hybrid strategy: PEs needed along input dimension
        col_pe_counts = {}  # For hybrid strategy: PEs needed along output dimension
        current_dim = self.input_dim
        
        for layer_id, output_dim in enumerate(self.layer_dims):
            # Calculate weight matrix dimensions
            weight_matrix_rows = current_dim
            weight_matrix_cols = output_dim
            
            # Calculate weights per PE
            weights_per_pe = pe_memory_size // self.data_type_bytes
            
            if split_strategy == "column_split":
                # Split along columns (output dimension)
                if weight_matrix_rows > weights_per_pe:
                    raise ValueError(f"Row size ({weight_matrix_rows}) exceeds PE capacity ({weights_per_pe})")
                
                cols_per_pe = max(1, weights_per_pe // weight_matrix_rows)
                pes_needed = math.ceil(weight_matrix_cols / cols_per_pe)
                
            elif split_strategy == "row_split":
                # Split along rows (input dimension)
                if weight_matrix_cols > weights_per_pe:
                    raise ValueError(f"Column size ({weight_matrix_cols}) exceeds PE capacity ({weights_per_pe})")
                
                rows_per_pe = max(1, weights_per_pe // weight_matrix_cols)
                pes_needed = math.ceil(weight_matrix_rows / rows_per_pe)
                
                # If we're not reusing PEs for aggregation, we need an extra PE per layer for aggregation
                if not reuse_pe_for_aggregation:
                    pes_needed += 1  # Add one PE per layer for aggregation
                
            elif split_strategy == "hybrid_split":
                # Split along both dimensions
                # Calculate how many elements can fit in a single PE
                elements_per_pe = weights_per_pe
                
                # Find a balanced split across both dimensions
                best_row_pes = 1
                best_col_pes = 1
                best_max_pes = float('inf')
                
                # Try different splits to find a good balance
                for row_div in range(1, weight_matrix_rows + 1):
                    rows_per_pe = math.ceil(weight_matrix_rows / row_div)
                    max_cols_per_pe = max(1, elements_per_pe // rows_per_pe)
                    col_div = math.ceil(weight_matrix_cols / max_cols_per_pe)
                    total_pes = row_div * col_div
                    
                    if total_pes < best_max_pes:
                        best_row_pes = row_div
                        best_col_pes = col_div
                        best_max_pes = total_pes
                
                # Store counts for hybrid approach
                row_pe_counts[layer_id] = best_row_pes
                col_pe_counts[layer_id] = best_col_pes
                
                # Total PEs needed = row_pes * col_pes + aggregation PEs
                pes_needed = best_row_pes * best_col_pes
                
                # For hybrid, we might need multiple aggregation PEs:
                # - One per column group for row aggregation
                # - One final for column concatenation
                if not reuse_pe_for_aggregation:
                    pes_needed += best_col_pes  # One per column group
                    pes_needed += 1  # One for final aggregation
            else:
                raise ValueError(f"Unknown split strategy: {split_strategy}")
            
            layer_pe_counts[layer_id] = pes_needed
            current_dim = output_dim
        
        # For grid_wise mapping, calculate an optimized grid layout
        if mapping_strategy == "grid_wise":
            # Use a large buffer factor if we need dedicated aggregation PEs
            buffer_factor = 3.0 if not reuse_pe_for_aggregation else 1.5
            
            if split_strategy == "hybrid_split":
                # Calculate total area needed by all layers
                total_area = 0
                layer_areas = {}
                
                for layer_id in range(len(self.layer_dims)):
                    if layer_id in row_pe_counts and layer_id in col_pe_counts:
                        layer_width = col_pe_counts[layer_id]
                        layer_height = row_pe_counts[layer_id]
                        layer_area = layer_width * layer_height
                        layer_areas[layer_id] = (layer_width, layer_height)
                        total_area += layer_area
                
                # Add buffer space to ensure we have enough room
                raw_area = total_area * buffer_factor
                
                # Calculate a reasonable aspect ratio (width/height) for the NoC
                aspect_ratio = 1.5  # Wider than tall
                
                # Calculate dimensions based on total area
                noc_width = int(math.sqrt(raw_area * aspect_ratio))
                noc_height = int(math.sqrt(raw_area / aspect_ratio))
                
                # Ensure minimum dimensions 
                noc_width = max(noc_width, 6)  # Increased minimum width
                noc_height = max(noc_height, 6)  # Increased minimum height
                
                # Also ensure we have enough width for the widest layer
                max_layer_width = max((width for width, _ in layer_areas.values()), default=2)
                noc_width = max(noc_width, max_layer_width)
                
                noc_rows = noc_height
                noc_cols = noc_width
            else:
                # For non-hybrid splits with grid_wise mapping
                total_pes = sum(layer_pe_counts.values())
                
                # Add buffer space
                total_pes = int(total_pes * buffer_factor)
                
                # Aim for a square-ish grid 
                grid_side = int(math.sqrt(total_pes))
                noc_rows = grid_side
                noc_cols = math.ceil(total_pes / grid_side)
                
                # Ensure minimum dimensions
                noc_rows = max(noc_rows, 4)
                noc_cols = max(noc_cols, 6)
            
            return noc_rows, noc_cols
        
        # For column_wise mapping
        elif mapping_strategy == "column_wise":
            if split_strategy == "hybrid_split":
                # For hybrid with column mapping, we need a 2D grid for each layer
                # Each layer gets a column, but we need multiple rows per layer
                max_layer_rows = 0
                for layer_id in range(len(self.layer_dims)):
                    if layer_id in row_pe_counts and layer_id in col_pe_counts:
                        layer_rows = row_pe_counts[layer_id] * col_pe_counts[layer_id]
                        if not reuse_pe_for_aggregation:
                            layer_rows += col_pe_counts[layer_id] + 1  # Add aggregation PEs
                        max_layer_rows = max(max_layer_rows, layer_rows)
                
                noc_cols = len(self.layer_dims)
                noc_rows = max_layer_rows
            else:
                # Standard column mapping
                noc_cols = len(self.layer_dims)
                noc_rows = max(layer_pe_counts.values())
                
                # Add buffer for non-reusable aggregation
                if not reuse_pe_for_aggregation and split_strategy == "row_split":
                    noc_rows += len(self.layer_dims)  # Add one row per layer for aggregation
            
            return noc_rows, noc_cols
        
        # For row_wise mapping
        elif mapping_strategy == "row_wise":
            if split_strategy == "hybrid_split":
                # For hybrid with row mapping, we need a 2D grid for each layer
                # Each layer gets a row, but we need multiple columns per layer
                max_layer_cols = 0
                for layer_id in range(len(self.layer_dims)):
                    if layer_id in row_pe_counts and layer_id in col_pe_counts:
                        layer_cols = row_pe_counts[layer_id] * col_pe_counts[layer_id]
                        if not reuse_pe_for_aggregation:
                            layer_cols += col_pe_counts[layer_id] + 1  # Add aggregation PEs
                        max_layer_cols = max(max_layer_cols, layer_cols)
                
                noc_rows = len(self.layer_dims)
                noc_cols = max_layer_cols
            else:
                # Standard row mapping
                noc_rows = len(self.layer_dims)
                noc_cols = max(layer_pe_counts.values())
                
                # Add buffer for non-reusable aggregation
                if not reuse_pe_for_aggregation and split_strategy == "row_split":
                    noc_cols += len(self.layer_dims)  # Add one column per layer for aggregation
            
            return noc_rows, noc_cols
        
        # If we somehow get here (shouldn't happen), return a safe default
        return max(8, len(self.layer_dims)), max(8, sum(layer_pe_counts.values()) // len(self.layer_dims))


    def run_inference(self, input_tensor: torch.Tensor) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[int, int], Optional[str]]]:
        """
        Run inference on the network and generate traffic table.
        
        Returns:
            Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples.
        """
        if input_tensor.shape[1] != self.input_dim:
            raise ValueError(f"Input tensor must have {self.input_dim} features, got {input_tensor.shape[1]}")
        
        # Select source_pe based on mapping strategy
        if self.mapping_strategy == "column_wise":
            source_pe = (-1, 0)  # Virtual PE at left side for column-wise mapping
        elif self.mapping_strategy == "row_wise":
            source_pe = (0, -1)  # Virtual PE at top for row-wise mapping
        elif self.mapping_strategy == "grid_wise":
            source_pe = (-1, -1)  # Virtual PE at top-left for grid-wise mapping
        else:
            raise ValueError(f"Unknown mapping strategy: {self.mapping_strategy}")
        
        # Select appropriate inference method based on split strategy
        if self.split_strategy == "column_split":
            return self._run_column_split_inference(input_tensor, source_pe)
        elif self.split_strategy == "row_split":
            return self._run_row_split_inference(input_tensor, source_pe)
        elif self.split_strategy == "hybrid_split":
            return self._run_hybrid_split_inference(input_tensor, source_pe)
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

    def _run_column_split_inference(self, input_tensor: torch.Tensor, source_pe: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[int, int], Optional[str]]]:
        """
        Run inference with column-split strategy.
        Weight matrices are split by columns (output neurons).
        No aggregation needed since each PE computes complete outputs for its subset of neurons.
        
        Args:
            input_tensor: Input tensor for inference
            source_pe: Coordinates of the virtual source PE (depends on mapping strategy)
        
        Returns:
            Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples.
        """
        seq_len, input_dim = input_tensor.shape
        
        # Maps destination PE coordinates to list of task IDs that send data to it
        pe_to_dependencies_map = {}
        
        # Keep track of computation tasks for each PE across all layers
        pe_computation_tasks = {}
        
        # First layer PEs - filter for active PEs only
        first_layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(0)
                          if pe_coords in self.active_pes]
        
        if not first_layer_pes:
            raise ValueError("No active PEs found for the first layer")
            
        pe_to_dependencies_map = {pe_coords: [] for pe_coords in first_layer_pes}
        
        # Send input to all PEs in the first layer
        for pe_coords in first_layer_pes:
            task_id = self.noc.scheduler.create_task(
                src_pe=source_pe,
                dest_pe=pe_coords,
                tensor_shape=(seq_len, input_dim),
                description=f"Input distribution to Layer 0 PE{pe_coords}"
            )
            pe_to_dependencies_map[pe_coords].append(task_id)
        
        # Process each layer
        for layer_id in range(len(self.layer_dims)):
            # Filter for active PEs only
            layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(layer_id)
                        if pe_coords in self.active_pes]
                        
            if not layer_pes:
                raise ValueError(f"No active PEs found for layer {layer_id}")
                
            next_pe_to_dependencies_map = {}
            
            # If not the last layer, initialize next layer PEs
            if layer_id < len(self.layer_dims) - 1:
                # Filter for active PEs only
                next_layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(layer_id + 1)
                                 if pe_coords in self.active_pes]
                                 
                if not next_layer_pes:
                    raise ValueError(f"No active PEs found for layer {layer_id + 1}")
                    
                for pe_coords in next_layer_pes:
                    next_pe_to_dependencies_map[pe_coords] = []
            
            # Compute outputs for this layer
            for pe_coords in layer_pes:
                pe = self.noc.get_pe(*pe_coords)
                
                # Skip PEs without proper weight shape
                if not hasattr(pe, 'weight_shape') or pe.weight_shape is None:
                    continue
                
                # This PE should wait for all tasks that sent data to it
                wait_ids = pe_to_dependencies_map.get(pe_coords, [])
                
                # PE computes its portion of the output
                compute_task_id = self.noc.scheduler.create_task(
                    src_pe=pe_coords,
                    dest_pe=pe_coords,  # Computing within the same PE
                    tensor_shape=(seq_len, pe.weight_shape[1]),
                    wait_ids=wait_ids,  # Wait for all input transfers
                    description=f"Layer {layer_id} PE{pe_coords} computation"
                )
                
                # Store the computation task ID for this PE
                pe_computation_tasks[pe_coords] = compute_task_id
                
                # If not the last layer, distribute outputs to next layer
                if layer_id < len(self.layer_dims) - 1:
                    # Filter for active PEs only
                    next_layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(layer_id + 1)
                                     if pe_coords in self.active_pes]
                    for next_pe_coords in next_layer_pes:
                        # Send output to each PE in the next layer
                        transfer_task_id = self.noc.scheduler.create_task(
                            src_pe=pe_coords,
                            dest_pe=next_pe_coords,
                            tensor_shape=(seq_len, pe.weight_shape[1]),
                            wait_ids=[compute_task_id],  # Wait for this PE's computation
                            description=f"Layer {layer_id} PE{pe_coords} -> Layer {layer_id+1} PE{next_pe_coords}"
                        )
                        
                        # Add this transfer as a dependency for the next layer's PE
                        next_pe_to_dependencies_map[next_pe_coords].append(transfer_task_id)
            
            # Update for next iteration (if not the last layer)
            if layer_id < len(self.layer_dims) - 1:
                pe_to_dependencies_map = next_pe_to_dependencies_map
        
        # Track the outputs from each PE in the final layer
        final_layer_id = len(self.layer_dims) - 1
        # Filter for active PEs only
        final_layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(final_layer_id)
                          if pe_coords in self.active_pes]
        
        # For each PE in the final layer, track its output
        pe_outputs = {}
        for pe_coords in final_layer_pes:
            pe = self.noc.get_pe(*pe_coords)
            
            # Skip PEs without proper col_start/col_end values
            if not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or pe.col_start is None or pe.col_end is None:
                continue
                
            # In column-split, each PE handles a specific set of output columns (neurons)
            # For column split (split_dim=1), col_range should use col_start and col_end
            col_range = (pe.col_start, pe.col_end)
            
            # Create a simulated output tensor for this PE
            pe_output = torch.zeros((seq_len, pe.col_end - pe.col_start))
            
            # Include this PE's output in the results
            pe_outputs[pe_coords] = (
                pe_output,
                col_range,  # Range of output neurons this PE computed
                pe_computation_tasks.get(pe_coords)
            )
        
        return pe_outputs

    def _run_row_split_inference(self, input_tensor: torch.Tensor, source_pe: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[int, int], Optional[str]]]:
        """
        Run inference with row-split strategy.
        Weight matrices are split by rows (input features).
        Partial results need to be summed across PEs to get the final output for each layer.
        
        Args:
            input_tensor: Input tensor for inference
            source_pe: Coordinates of the virtual source PE (depends on mapping strategy)
        
        Returns:
            Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples.
        """
        seq_len, input_dim = input_tensor.shape
        
        # For each layer, track compute tasks and aggregation tasks
        compute_task_ids = {}  # Maps layer_id -> PE coords -> task_id
        aggregation_task_ids = {}  # Maps layer_id -> task_id
        
        # Track aggregation PEs for each layer
        aggregation_pes = {}
        
        # Process each layer
        for layer_id in range(len(self.layer_dims)):
            # Get only active PEs for this layer
            layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(layer_id) 
                        if pe_coords in self.active_pes]
            
            if not layer_pes:
                raise ValueError(f"No active PEs found for layer {layer_id}")
                
            compute_task_ids[layer_id] = {}
            
            # Determine which PE to use for aggregation
            if self.reuse_pe_for_aggregation:
                # Reuse the first computation PE
                aggregation_pe = layer_pes[0]
            else:
                # Use the dedicated aggregation PE for this layer
                aggregation_pe = self.aggregation_pes[layer_id]
                if aggregation_pe not in self.active_pes:
                    raise ValueError(f"Aggregation PE {aggregation_pe} for layer {layer_id} is not active")
            
            aggregation_pes[layer_id] = aggregation_pe
            
            # Calculate layer dimensions
            if layer_id == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = self.layer_dims[layer_id - 1]
            layer_output_dim = self.layer_dims[layer_id]
            
            # Phase 1: Distribute input or previous layer output to compute PEs
            for pe_coords in layer_pes:
                pe = self.noc.get_pe(*pe_coords)
                
                # Skip PEs without proper row_start/row_end values
                if not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or pe.row_start is None or pe.row_end is None:
                    continue
                
                # In row-split, each PE needs only its subset of input features
                row_start, row_end = pe.row_start, pe.row_end
                
                # For first layer, input comes from source
                if layer_id == 0:
                    task_id = self.noc.scheduler.create_task(
                        src_pe=source_pe,
                        dest_pe=pe_coords,
                        tensor_shape=(seq_len, row_end - row_start),
                        description=f"Input features {row_start}-{row_end} to Layer {layer_id} PE{pe_coords}"
                    )
                else:
                    # For subsequent layers, input comes from previous layer's aggregation
                    prev_aggregation_pe = aggregation_pes[layer_id - 1]
                    prev_aggregation_task = aggregation_task_ids[layer_id - 1]
                    
                    task_id = self.noc.scheduler.create_task(
                        src_pe=prev_aggregation_pe,
                        dest_pe=pe_coords,
                        tensor_shape=(seq_len, row_end - row_start),
                        wait_ids=[prev_aggregation_task],
                        description=f"Layer {layer_id-1} output features {row_start}-{row_end} to Layer {layer_id} PE{pe_coords}"
                    )
                
                # Phase 2: Compute partial results on each PE
                compute_task_id = self.noc.scheduler.create_task(
                    src_pe=pe_coords,
                    dest_pe=pe_coords,
                    tensor_shape=(seq_len, layer_output_dim),
                    wait_ids=[task_id],
                    description=f"Layer {layer_id} PE{pe_coords} partial computation (rows {row_start}-{row_end})"
                )
                
                compute_task_ids[layer_id][pe_coords] = compute_task_id
            
            # Phase 3: Collect transfer tasks for aggregation
            transfer_task_ids = []
            
            # If we're reusing a computation PE, include its own computation
            if self.reuse_pe_for_aggregation and aggregation_pe in compute_task_ids[layer_id]:
                transfer_task_ids.append(compute_task_ids[layer_id][aggregation_pe])
            
            # Then, collect transfers from other PEs to the aggregation PE
            for pe_coords in layer_pes:
                # Skip the aggregation PE if we're reusing it (already included above)
                if pe_coords == aggregation_pe and self.reuse_pe_for_aggregation:
                    continue
                
                # Skip PEs without compute tasks
                if pe_coords not in compute_task_ids[layer_id]:
                    continue
                
                # Send partial results to aggregation PE
                transfer_task_id = self.noc.scheduler.create_task(
                    src_pe=pe_coords,
                    dest_pe=aggregation_pe,
                    tensor_shape=(seq_len, layer_output_dim),
                    wait_ids=[compute_task_ids[layer_id][pe_coords]],  # Wait for this PE's computation
                    description=f"Send partial results from Layer {layer_id} PE{pe_coords} to aggregation PE{aggregation_pe}"
                )
                transfer_task_ids.append(transfer_task_id)
            
            # Phase 4: Aggregate partial results - wait for all transfers to complete
            aggregation_task_id = self.noc.scheduler.create_task(
                src_pe=aggregation_pe,
                dest_pe=aggregation_pe,
                tensor_shape=(seq_len, layer_output_dim),
                wait_ids=transfer_task_ids,  # Wait for all transfers to arrive
                description=f"Aggregate results for Layer {layer_id} at PE{aggregation_pe}"
            )
            
            aggregation_task_ids[layer_id] = aggregation_task_id
        
        # Track the outputs from final layer
        final_layer_id = len(self.layer_dims) - 1
        # Get only active PEs for the final layer
        final_layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(final_layer_id) 
                          if pe_coords in self.active_pes]
        final_aggregation_pe = aggregation_pes[final_layer_id]
        
        # Create output dictionary with both PE-specific outputs and aggregated output
        pe_outputs = {}
        
        # Include each PE's partial output in the results
        for pe_coords in final_layer_pes:
            pe = self.noc.get_pe(*pe_coords)
            
            # Skip PEs without proper row_start/row_end values or compute tasks
            if (not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or 
                pe.row_start is None or pe.row_end is None or
                pe_coords not in compute_task_ids[final_layer_id]):
                continue
            
            # For row-split, each PE computes a partial output for all neurons
            # based on a subset of input features
            # For row split (split_dim=0), row_range correctly uses row_start and row_end
            row_range = (pe.row_start, pe.row_end)
            
            # Partial output tensor
            partial_output = torch.zeros((seq_len, self.layer_dims[final_layer_id]))
            
            # Include partial output in the results
            pe_outputs[pe_coords] = (
                partial_output,
                row_range,  # Range of input features this PE processed
                compute_task_ids[final_layer_id][pe_coords]
            )
        
        # Include the final aggregated output from the aggregation PE
        # This contains the complete output for all neurons in the layer
        aggregated_output = torch.zeros((seq_len, self.layer_dims[final_layer_id]))
        pe_outputs[final_aggregation_pe] = (
            aggregated_output,
            (0, self.layer_dims[final_layer_id]),  # Full range of output
            aggregation_task_ids[final_layer_id]
        )
        
        return pe_outputs

    def _run_hybrid_split_inference(self, input_tensor: torch.Tensor, source_pe: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[int, int], Optional[str]]]:
        """
        Run inference with hybrid-split strategy with direct data forwarding.
        Weight matrices are split by both rows and columns.
        After row aggregation, data is sent directly to the appropriate next layer PEs.
        
        Args:
            input_tensor: Input tensor for inference
            source_pe: Coordinates of the virtual source PE (depends on mapping strategy)
        
        Returns:
            Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples.
        """
        seq_len, input_dim = input_tensor.shape
        
        # For tracking tasks and outputs
        compute_task_ids = {}  # Maps layer_id -> PE coords -> task_id
        row_aggregation_task_ids = {}  # Maps layer_id -> col_group -> task_id
        compute_task_dependencies = {}  # Maps layer_id -> PE coords -> list of task_ids to wait for
        
        # Track PEs used for aggregation
        row_aggregation_pes = {}  # Maps (layer_id, col_group) -> PE coords
        
        # Group PEs by layer and column group
        pe_groups = {}  # Maps layer_id -> col_group -> list of PE coords
        
        # Group PEs by input row range for direct next-layer mapping
        pe_input_requirements = {}  # Maps layer_id -> (row_start, row_end) -> list of PE coords
        
        # Initialize PE groups - only process active PEs
        for pe_coords, (layer_id, pe_idx) in self.mapper.pe_layer_map.items():
            # Only process active PEs
            if pe_coords not in self.active_pes:
                continue
                
            pe = self.noc.get_pe(*pe_coords)
            # Skip PEs without proper col_start/col_end values
            if not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or pe.col_start is None or pe.col_end is None:
                continue
                
            # Group by column partition for row aggregation
            col_group = (pe.col_start, pe.col_end)
            
            if layer_id not in pe_groups:
                pe_groups[layer_id] = {}
                compute_task_ids[layer_id] = {}
                pe_input_requirements[layer_id] = {}
            
            if col_group not in pe_groups[layer_id]:
                pe_groups[layer_id][col_group] = []
            
            pe_groups[layer_id][col_group].append(pe_coords)
            
            # Also group by input row requirements for direct forwarding
            if hasattr(pe, 'row_start') and hasattr(pe, 'row_end') and pe.row_start is not None and pe.row_end is not None:
                row_range = (pe.row_start, pe.row_end)
                if row_range not in pe_input_requirements[layer_id]:
                    pe_input_requirements[layer_id][row_range] = []
                pe_input_requirements[layer_id][row_range].append(pe_coords)
        
        # Process each layer
        for layer_id in range(len(self.layer_dims)):
            if layer_id not in pe_groups:
                raise ValueError(f"No PEs found for layer {layer_id}")
            
            row_aggregation_task_ids[layer_id] = {}
            
            # Calculate layer dimensions
            if layer_id == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = self.layer_dims[layer_id - 1]
            layer_output_dim = self.layer_dims[layer_id]
            
            # Phase 1: Distribute inputs to all PEs in this layer
            for col_group, pe_coords_list in pe_groups[layer_id].items():
                # For each column group, check if we have multiple row ranges
                row_ranges = set()
                for pe_coords in pe_coords_list:
                    pe = self.noc.get_pe(*pe_coords)
                    if hasattr(pe, 'row_start') and hasattr(pe, 'row_end') and pe.row_start is not None and pe.row_end is not None:
                        row_ranges.add((pe.row_start, pe.row_end))
                
                has_row_split = len(row_ranges) > 1
                
                for pe_coords in pe_coords_list:
                    pe = self.noc.get_pe(*pe_coords)
                    
                    # Skip PEs without proper row_start/row_end values
                    if not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or pe.row_start is None or pe.row_end is None:
                        continue
                    
                    # Each PE needs only the input features it processes
                    row_start, row_end = pe.row_start, pe.row_end
                    
                    # Determine the source of inputs
                    if layer_id == 0:
                        # First layer gets input from the source
                        input_task_id = self.noc.scheduler.create_task(
                            src_pe=source_pe,
                            dest_pe=pe_coords,
                            tensor_shape=(seq_len, row_end - row_start),
                            description=f"Input features {row_start}-{row_end} to Layer {layer_id} PE{pe_coords}"
                        )
                        wait_ids = [input_task_id]
                    else:
                        # If not the first layer, we'll handle input dependencies later
                        wait_ids = compute_task_dependencies.get(layer_id, {}).get(pe_coords, [])
                    
                    # Phase 2: Compute results on each PE - only mark as "partial" if there's a row split
                    computation_type = "partial computation" if has_row_split else "computation"
                    compute_task_id = self.noc.scheduler.create_task(
                        src_pe=pe_coords,
                        dest_pe=pe_coords,
                        tensor_shape=(seq_len, pe.col_end - pe.col_start),
                        wait_ids=wait_ids,
                        description=f"Layer {layer_id} PE{pe_coords} {computation_type} (rows {pe.row_start}-{pe.row_end}, cols {pe.col_start}-{pe.col_end})"
                    )
                    
                    compute_task_ids[layer_id][pe_coords] = compute_task_id
                
                # Phase 3: Row aggregation - only if we have multiple row ranges
                if has_row_split:
                    # Determine row aggregation PE
                    if self.reuse_pe_for_aggregation:
                        row_agg_pe = pe_coords_list[0]
                    else:
                        row_agg_pe = self.row_aggregation_pes.get((layer_id, col_group))
                        if row_agg_pe is None:
                            raise ValueError(f"No row aggregation PE assigned for layer {layer_id}, column group {col_group}")
                    
                    row_aggregation_pes[(layer_id, col_group)] = row_agg_pe
                    
                    # Collect transfer tasks for row aggregation
                    row_transfer_task_ids = []
                    
                    # If reusing a PE, include its own computation
                    if self.reuse_pe_for_aggregation:
                        row_transfer_task_ids.append(compute_task_ids[layer_id][row_agg_pe])
                    
                    # Then, collect transfers from other PEs to the row aggregation PE
                    for pe_coords in pe_coords_list:
                        if pe_coords == row_agg_pe and self.reuse_pe_for_aggregation:
                            continue
                        
                        # Skip PEs without compute tasks
                        if pe_coords not in compute_task_ids[layer_id]:
                            continue
                            
                        transfer_task_id = self.noc.scheduler.create_task(
                            src_pe=pe_coords,
                            dest_pe=row_agg_pe,
                            tensor_shape=(seq_len, pe.col_end - pe.col_start),
                            wait_ids=[compute_task_ids[layer_id][pe_coords]],
                            description=f"Send partial row results from Layer {layer_id} PE{pe_coords} to row agg PE{row_agg_pe}"
                        )
                        row_transfer_task_ids.append(transfer_task_id)
                    
                    # Create row aggregation task
                    row_agg_task_id = self.noc.scheduler.create_task(
                        src_pe=row_agg_pe,
                        dest_pe=row_agg_pe,
                        tensor_shape=(seq_len, pe.col_end - pe.col_start),
                        wait_ids=row_transfer_task_ids,
                        description=f"Row aggregate for Layer {layer_id}, cols {col_group} at PE{row_agg_pe}"
                    )
                    
                    row_aggregation_task_ids[layer_id][col_group] = row_agg_task_id
                    source_pe_for_forwarding = row_agg_pe
                    source_task_for_forwarding = row_agg_task_id
                else:
                    # If no row split, use the first PE's compute task as the "source" for forwarding
                    source_pe_for_forwarding = pe_coords_list[0]
                    source_task_for_forwarding = compute_task_ids[layer_id][source_pe_for_forwarding]
                    row_aggregation_task_ids[layer_id][col_group] = source_task_for_forwarding
                
                # Phase 4: Direct forwarding to next layer
                # Skip this for the final layer
                if layer_id < len(self.layer_dims) - 1:
                    # Get PE for current column group
                    pe = self.noc.get_pe(*source_pe_for_forwarding)
                    
                    # Skip if PE doesn't have valid col_start/col_end
                    if not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or pe.col_start is None or pe.col_end is None:
                        continue
                        
                    col_start, col_end = pe.col_start, pe.col_end
                    
                    # Directly find row ranges in the next layer that might need our column data
                    for row_range, next_layer_pe_coords_list in pe_input_requirements.get(layer_id + 1, {}).items():
                        next_row_start, next_row_end = row_range
                        
                        # Check if this row range overlaps with our output columns
                        if not (next_row_end <= col_start or next_row_start >= col_end):
                            # Calculate the exact overlap
                            overlap_start = max(col_start, next_row_start)
                            overlap_end = min(col_end, next_row_end)
                            
                            # Send the overlapping portion to all PEs that need this row range
                            for next_pe_coords in next_layer_pe_coords_list:
                                # Only forward to active PEs
                                if next_pe_coords not in self.active_pes:
                                    continue
                                    
                                # Create forward task to send directly to next layer PE
                                forward_task_id = self.noc.scheduler.create_task(
                                    src_pe=source_pe_for_forwarding,
                                    dest_pe=next_pe_coords,
                                    tensor_shape=(seq_len, overlap_end - overlap_start),
                                    wait_ids=[source_task_for_forwarding],
                                    description=f"Direct forward from L{layer_id} PE{source_pe_for_forwarding} -> L{layer_id+1} PE{next_pe_coords} cols->rows {overlap_start}-{overlap_end}"
                                )
                                
                                # Store this forwarded data as a dependency
                                if layer_id + 1 not in compute_task_dependencies:
                                    compute_task_dependencies[layer_id + 1] = {}
                                if next_pe_coords not in compute_task_dependencies[layer_id + 1]:
                                    compute_task_dependencies[layer_id + 1][next_pe_coords] = []
                                
                                # Add this forward task to the dependencies
                                compute_task_dependencies[layer_id + 1][next_pe_coords].append(forward_task_id)
        
        # Track outputs from the final layer
        final_layer_id = len(self.layer_dims) - 1
        
        # Create output dictionary with results from all PEs
        pe_outputs = {}
        
        # Include each PE's output in the results
        for col_group, pe_coords_list in pe_groups[final_layer_id].items():
            for pe_coords in pe_coords_list:
                # Skip inactive PEs
                if pe_coords not in self.active_pes:
                    continue
                    
                pe = self.noc.get_pe(*pe_coords)
                
                # Skip PEs without valid row_start/row_end/col_start/col_end
                if (not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or 
                    not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or
                    pe.row_start is None or pe.row_end is None or 
                    pe.col_start is None or pe.col_end is None):
                    continue
                
                # Each PE computes a result for a subset of output neurons
                pe_outputs[pe_coords] = (
                    torch.zeros((seq_len, pe.col_end - pe.col_start)),  # Output tensor
                    ((pe.row_start, pe.row_end), (pe.col_start, pe.col_end)),  # Row and column ranges
                    compute_task_ids[final_layer_id].get(pe_coords)  # Task ID
                )
                
                # Include row aggregation outputs if they exist
                if col_group in row_aggregation_task_ids[final_layer_id]:
                    task_id = row_aggregation_task_ids[final_layer_id][col_group]
                    # Find the PE that did the aggregation
                    for pe_coords in pe_coords_list:
                        if (pe_coords in compute_task_ids[final_layer_id] and
                            compute_task_ids[final_layer_id][pe_coords] == task_id):
                            pe = self.noc.get_pe(*pe_coords)
                            
                            # Skip PEs without valid col_start/col_end
                            if not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or pe.col_start is None or pe.col_end is None:
                                continue
                                
                            pe_outputs[pe_coords] = (
                                torch.zeros((seq_len, pe.col_end - pe.col_start)),  # Output tensor
                                ((-1, -1), (pe.col_start, pe.col_end)),  # Row and column ranges (-1 means all rows aggregated)
                                task_id  # Task ID
                            )
                            break
        
        return pe_outputs
    
    def get_traffic_table(self) -> pd.DataFrame:
        """Get the traffic table."""
        return self.noc.scheduler.get_traffic_table()

    def print_pe_outputs(self, pe_outputs: Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[int, int], Optional[str]]]):
        """Print the outputs of each PE."""
        for pe_coords, (pe_output, output_range, computation_task_id) in pe_outputs.items():
            # Format the output range in a compact way
            if isinstance(output_range, tuple) and len(output_range) == 2 and isinstance(output_range[0], int):
                # Format for column_split: (start, end)
                # Check for None values
                start = output_range[0] if output_range[0] is not None else "None"
                end = output_range[1] if output_range[1] is not None else "None"
                range_str = f"(:, {start}:{end})"
            elif isinstance(output_range, tuple) and len(output_range) == 2 and isinstance(output_range[0], tuple):
                # Format for hybrid_split: ((row_start, row_end), (col_start, col_end))
                row_range, col_range = output_range
                
                # Handle row range
                if row_range == (-1, -1):
                    row_part = ":"  # All rows aggregated
                elif None in row_range:
                    row_part = "None:None"  # Handle None values
                else:
                    row_part = f"{row_range[0]}:{row_range[1]}"
                
                # Handle column range
                if col_range == (-1, -1):
                    col_part = ":"  # All columns aggregated
                elif None in col_range:
                    col_part = "None:None"  # Handle None values
                else:
                    col_part = f"{col_range[0]}:{col_range[1]}"
                
                range_str = f"({row_part}, {col_part})"
            else:
                # Default format if structure is not recognized
                range_str = str(output_range)
                
            print(f"PE{pe_coords} output: {pe_output.shape}, tensor slice: {range_str}, task_id: {computation_task_id}")

    def get_pe_utilization(self, use_effective_dimensions=False) -> dict:
        """
        Calculate the utilization of PEs in the NoC grid.
        
        Args:
            use_effective_dimensions: Whether to use effective dimensions based on actual PE coordinates
            
        Returns:
            A dictionary containing:
            - total_pes: Total number of PEs in the NoC grid (effective if use_effective_dimensions=True)
            - used_computation_pes: Number of PEs used for computation
            - used_aggregation_pes: Number of PEs used for aggregation
            - total_used_pes: Total number of PEs used (computation + aggregation)
            - computation_utilization: Percentage of NoC used for computation
            - total_utilization: Percentage of NoC used overall
            - effective_rows: Number of rows in the effective grid (if use_effective_dimensions=True)
            - effective_cols: Number of columns in the effective grid (if use_effective_dimensions=True)
        """
        # Count PEs used for computation (from the mapper)
        used_computation_pes = len(self.mapper.pe_layer_map)
        
        # Count PEs used for aggregation
        used_aggregation_pes = 0
        
        if not self.reuse_pe_for_aggregation:
            if self.split_strategy == "row_split":
                # For row_split, we have one aggregation PE per layer
                used_aggregation_pes = len(self.aggregation_pes)
            elif self.split_strategy == "hybrid_split":
                # For hybrid_split, we have row and column aggregation PEs
                used_aggregation_pes = len(self.row_aggregation_pes) + len(self.column_aggregation_pes)
                
                # Remove duplicates if the same PE is used for both row and column aggregation
                row_agg_pes = set(self.row_aggregation_pes.values())
                col_agg_pes = set(self.column_aggregation_pes.values())
                used_aggregation_pes = len(row_agg_pes.union(col_agg_pes))
        
        # Total used PEs
        total_used_pes = used_computation_pes + used_aggregation_pes
        
        # Calculate total PEs and utilization based on dimensions
        if use_effective_dimensions:
            # Get effective dimensions based on actual PE coordinates
            effective_rows, effective_cols, effective_total_pes = self.mapper.get_effective_noc_dimensions(self)
            total_pes = effective_total_pes
            # Avoid division by zero
            computation_utilization = (used_computation_pes / total_pes) * 100 if total_pes > 0 else 0
            total_utilization = (total_used_pes / total_pes) * 100 if total_pes > 0 else 0
            
            # Create result dictionary with effective dimensions
            result = {
                "total_pes": total_pes,
                "used_computation_pes": used_computation_pes,
                "used_aggregation_pes": used_aggregation_pes,
                "total_used_pes": total_used_pes,
                "computation_utilization": computation_utilization,
                "total_utilization": total_utilization,
                "effective_rows": effective_rows,
                "effective_cols": effective_cols
            }
        else:
            # Use allocated dimensions
            #TODO: This needs to be fixed. It's isn't able to figure out rows and cols for the noc effectively.
            total_pes = self.noc.rows * self.noc.cols
            computation_utilization = (used_computation_pes / total_pes) * 100 if total_pes > 0 else 0
            total_utilization = (total_used_pes / total_pes) * 100 if total_pes > 0 else 0
            
            # Create standard result dictionary
            result = {
                "total_pes": total_pes,
                "used_computation_pes": used_computation_pes,
                "used_aggregation_pes": used_aggregation_pes,
                "total_used_pes": total_used_pes,
                "computation_utilization": computation_utilization,
                "total_utilization": total_utilization
            }
        
        return result

def run_example(log_file="noc_simulation_legacy.log", data_type="float16", channel_bandwidth=32.0):
    """Example usage of the NoC implementation."""
    # Open log file for writing
    with open(log_file, "w") as f:
        # Parameters - modified to showcase grid-wise mapping
        input_dim = 3
        hidden_dim1 = 6
        hidden_dim2 = 6
        output_dim = 3
        seq_len = 1
        pe_memory_size = 24  # in bytes
        mapping_strategy = "column_wise"
        split_strategy = "column_split"
        reuse_pe_for_aggregation = False  # Set to False to demonstrate dedicated aggregation PEs
        
        # Create a deliberately oversized NoC grid
        noc_rows = 1000
        noc_cols = 1000
        
        # Log basic parameters
        f.write(f"NoC Simulation Parameters:\n")
        f.write(f"Input Dimension: {input_dim}\n")
        f.write(f"Hidden Dimension 1: {hidden_dim1} | Hidden Dimension 2: {hidden_dim2}\n")
        f.write(f"Output Dimension: {output_dim}\n")
        f.write(f"Sequence Length: {seq_len}\n")
        f.write(f"PE Memory Size: {pe_memory_size} bytes\n")
        f.write(f"Mapping Strategy: {mapping_strategy}\n")
        f.write(f"Split Strategy: {split_strategy}\n")
        f.write(f"Reuse PE for Aggregation: {reuse_pe_for_aggregation}\n")
        f.write(f"NoC Size: {noc_rows}x{noc_cols} ({noc_rows*noc_cols} PEs)\n")
        f.write(f"Data Type: {data_type}\n")
        f.write(f"Channel Bandwidth: {channel_bandwidth} B/cycle\n\n")
        
        # Create neural network with specified NoC dimensions
        nn = FCNeuralNetwork(
            input_dim=input_dim,
            layer_dims=[hidden_dim1, hidden_dim2, output_dim],
            seq_len=seq_len,
            pe_memory_size=pe_memory_size,
            split_strategy=split_strategy,
            mapping_strategy=mapping_strategy,
            reuse_pe_for_aggregation=reuse_pe_for_aggregation,
            data_type=data_type,
            channel_bandwidth=channel_bandwidth,
            noc_rows=noc_rows,
            noc_cols=noc_cols
        )
        
        # Calculate and log PE utilization
        pe_utilization = nn.get_pe_utilization()
        f.write("PE Utilization:\n")
        f.write(f"Total PEs in NoC: {pe_utilization['total_pes']}\n")
        f.write(f"PEs used for computation: {pe_utilization['used_computation_pes']} ({pe_utilization['computation_utilization']:.2f}%)\n")
        f.write(f"PEs used for aggregation: {pe_utilization['used_aggregation_pes']}\n")
        f.write(f"Total PEs used: {pe_utilization['total_used_pes']} ({pe_utilization['total_utilization']:.2f}%)\n\n")
        
        # Print utilization to console
        print("PE Utilization:")
        print(f"Total PEs in NoC: {pe_utilization['total_pes']}")
        print(f"PEs used for computation: {pe_utilization['used_computation_pes']} ({pe_utilization['computation_utilization']:.2f}%)")
        print(f"PEs used for aggregation: {pe_utilization['used_aggregation_pes']}")
        print(f"Total PEs used: {pe_utilization['total_used_pes']} ({pe_utilization['total_utilization']:.2f}%)")
        
        # Log PE mapping
        f.write("PE Mapping:\n")
        pe_details = nn.mapper.get_pe_details()
        f.write(pe_details.to_string() + "\n\n")
    
        # Run inference with random input
        input_tensor = torch.randn(seq_len, input_dim)
        pe_outputs = nn.run_inference(input_tensor)
        nn.print_pe_outputs(pe_outputs)
    
        # Log traffic table
        f.write("\nTraffic Table:\n")
        traffic_table = nn.get_traffic_table()
        f.write(traffic_table.to_string() + "\n\n")
        
        # Log summary statistics
        #total_bytes = traffic_table['bytes'].sum()
        #total_cycles = traffic_table['cycles'].sum()
        #f.write(f"\nTotal Communication: {total_bytes / (1024*1024):.2f} MB\n")
        #f.write(f"Total Cycles: {total_cycles}\n")
    
        # Count tasks by type
        task_types = traffic_table['description'].str.extract(r'(->|computation|collection)').value_counts()
        f.write("\nTask Distribution:\n")
        f.write(task_types.to_string() + "\n")
        
        # Print to console that log was saved
        print(f"Simulation results saved to {log_file}")

def analyze_pe_memory_impact(
    input_dim=768, 
    hidden_dim=3072, 
    output_dim=768, 
    seq_len=1,
    data_type="float16", 
    channel_bandwidth=32.0,
    memory_sizes=None,
    mapping_strategy="column_wise",
    split_strategy="hybrid_split",
    reuse_pe_for_aggregation=False,
    save_plot=True,
    plot_filename="pe_memory_analysis.png",
    use_effective_dimensions=True
):
    """
    Analyze and plot how NoC grid dimensions change with different PE memory sizes.
    
    Args:
        input_dim: Input dimension of the network
        hidden_dim: Hidden dimension of the network
        output_dim: Output dimension of the network
        seq_len: Sequence length for inference
        data_type: The data type of the weights
        channel_bandwidth: The bandwidth of the channels in B/cycle
        memory_sizes: List of PE memory sizes to test (in bytes)
        mapping_strategy: How to map layers onto the NoC
        split_strategy: How to split weight matrices
        reuse_pe_for_aggregation: Whether to reuse computation PEs for aggregation
        save_plot: Whether to save the plot to a file
        plot_filename: Name of the file to save the plot to
        use_effective_dimensions: Whether to use effective dimensions based on actual PE coordinates
        
    Returns:
        DataFrame with results for each memory size
    """
    if memory_sizes is None:
        # Default range of memory sizes to test (from 1KB to 1MB)
        memory_sizes = [
            1 * 1024,         # 1 KB
            2 * 1024,         # 2 KB
            4 * 1024,         # 4 KB
            8 * 1024,         # 8 KB
            16 * 1024,        # 16 KB
            32 * 1024,        # 32 KB
            64 * 1024,        # 64 KB
            128 * 1024,       # 128 KB
            256 * 1024,       # 256 KB
            512 * 1024,       # 512 KB
            1024 * 1024       # 1 MB
        ]
    
    results = []
    
    print(f"Analyzing impact of PE memory size on NoC grid dimensions...")
    print(f"Network: {input_dim}-{hidden_dim}-{output_dim}, {mapping_strategy} mapping, {split_strategy} split")
    print(f"Using {'effective' if use_effective_dimensions else 'allocated'} grid dimensions")
    
    for memory_size in memory_sizes:
        print(f"Testing PE memory size: {memory_size/1024:.1f} KB")
        
        # Create neural network with automatic NoC dimension calculation
        nn = FCNeuralNetwork(
            input_dim=input_dim,
            layer_dims=[hidden_dim, output_dim],
            seq_len=seq_len,
            pe_memory_size=memory_size,
            split_strategy=split_strategy,
            mapping_strategy=mapping_strategy,
            reuse_pe_for_aggregation=reuse_pe_for_aggregation,
            data_type=data_type,
            channel_bandwidth=channel_bandwidth,
            # Let the network calculate its own dimensions based on PE memory
            noc_rows=None,
            noc_cols=None
        )
        
        # Calculate PE utilization using the enhanced method
        pe_utilization = nn.get_pe_utilization(use_effective_dimensions=use_effective_dimensions)
        
        # Get dimensions and metrics from the utilization results
        if use_effective_dimensions:
            noc_rows = pe_utilization['effective_rows']
            noc_cols = pe_utilization['effective_cols']
            grid_size = pe_utilization['total_pes']
            utilization_pct = pe_utilization['total_utilization']
        else:
            noc_rows = nn.noc.rows
            noc_cols = nn.noc.cols
            grid_size = pe_utilization['total_pes']
            utilization_pct = pe_utilization['total_utilization']
        
        # Record results
        results.append({
            "pe_memory_kb": memory_size / 1024,
            "noc_rows": noc_rows,
            "noc_cols": noc_cols,
            "grid_size": grid_size,
            "computation_pes": pe_utilization['used_computation_pes'],
            "aggregation_pes": pe_utilization['used_aggregation_pes'],
            "total_used_pes": pe_utilization['total_used_pes'],
            "utilization_pct": utilization_pct
        })
        
        print(f"  Grid size: {noc_rows}x{noc_cols} = {grid_size} PEs")
        print(f"  Utilization: {pe_utilization['total_used_pes']} PEs ({utilization_pct:.2f}%)")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Grid dimensions vs memory size
    plt.subplot(2, 2, 1)
    plt.plot(df['pe_memory_kb'], df['noc_rows'], 'b-o', label='Rows')
    plt.plot(df['pe_memory_kb'], df['noc_cols'], 'r-o', label='Columns')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('PE Memory Size (KB)')
    plt.ylabel('Grid Dimension')
    plt.title('NoC Grid Dimensions vs PE Memory Size')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot 2: Total grid size vs memory size
    plt.subplot(2, 2, 2)
    plt.plot(df['pe_memory_kb'], df['grid_size'], 'g-o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('PE Memory Size (KB)')
    plt.ylabel('Total PEs in Grid')
    plt.title('NoC Grid Size vs PE Memory Size')
    plt.grid(True, which="both", ls="--")
    
    # Plot 3: PE usage breakdown
    plt.subplot(2, 2, 3)
    plt.plot(df['pe_memory_kb'], df['computation_pes'], 'b-o', label='Computation PEs')
    plt.plot(df['pe_memory_kb'], df['aggregation_pes'], 'r-o', label='Aggregation PEs')
    plt.plot(df['pe_memory_kb'], df['total_used_pes'], 'g-o', label='Total Used PEs')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('PE Memory Size (KB)')
    plt.ylabel('Number of PEs')
    plt.title('PE Usage Breakdown vs Memory Size')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot 4: Utilization percentage vs memory size
    plt.subplot(2, 2, 4)
    plt.plot(df['pe_memory_kb'], df['utilization_pct'], 'k-o')
    plt.xscale('log')
    plt.xlabel('PE Memory Size (KB)')
    plt.ylabel('Utilization (%)')
    plt.title('NoC Utilization vs PE Memory Size')
    plt.ylim(0, 100)
    plt.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved to {plot_filename}")
    
    plt.show()
    
    # Also save results to CSV for further analysis
    csv_filename = plot_filename.replace('.png', '.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    return df


def analyze_split_strategies(
    input_dim=768, 
    hidden_dim=3072, 
    output_dim=768, 
    seq_len=1,
    pe_memory_size=64 * 1024,  # 64 KB
    data_type="float16", 
    channel_bandwidth=32.0,
    mapping_strategy="column_wise",
    reuse_pe_for_aggregation=False,
    save_plot=True,
    plot_filename="split_strategy_analysis.png",
    use_effective_dimensions=True
):
    """
    Analyze and plot how different split strategies affect NoC grid dimensions.
    
    Args:
        input_dim: Input dimension of the network
        hidden_dim: Hidden dimension of the network
        output_dim: Output dimension of the network
        seq_len: Sequence length for inference
        pe_memory_size: PE memory size in bytes
        data_type: The data type of the weights
        channel_bandwidth: The bandwidth of the channels in B/cycle
        mapping_strategy: How to map layers onto the NoC
        reuse_pe_for_aggregation: Whether to reuse computation PEs for aggregation
        save_plot: Whether to save the plot to a file
        plot_filename: Name of the file to save the plot to
        use_effective_dimensions: Whether to use effective dimensions based on actual PE coordinates
        
    Returns:
        DataFrame with results for each split strategy
    """
    # Split strategies to test
    split_strategies = ["column_split", "row_split", "hybrid_split"]
    
    results = []
    
    print(f"Analyzing impact of split strategies on NoC grid dimensions...")
    print(f"Network: {input_dim}-{hidden_dim}-{output_dim}, {mapping_strategy} mapping")
    print(f"PE Memory: {pe_memory_size/1024:.1f} KB")
    print(f"Using {'effective' if use_effective_dimensions else 'allocated'} grid dimensions")
    
    for split_strategy in split_strategies:
        print(f"Testing split strategy: {split_strategy}")
        
        # Create neural network with automatic NoC dimension calculation
        nn = FCNeuralNetwork(
            input_dim=input_dim,
            layer_dims=[hidden_dim, output_dim],
            seq_len=seq_len,
            pe_memory_size=pe_memory_size,
            split_strategy=split_strategy,
            mapping_strategy=mapping_strategy,
            reuse_pe_for_aggregation=reuse_pe_for_aggregation,
            data_type=data_type,
            channel_bandwidth=channel_bandwidth,
            # Let the network calculate its own dimensions based on PE memory
            noc_rows=None,
            noc_cols=None
        )
        
        # Calculate PE utilization using the enhanced method
        pe_utilization = nn.get_pe_utilization(use_effective_dimensions=use_effective_dimensions)
        
        # Get dimensions and metrics from the utilization results
        if use_effective_dimensions:
            noc_rows = pe_utilization['effective_rows']
            noc_cols = pe_utilization['effective_cols']
            grid_size = pe_utilization['total_pes']
            utilization_pct = pe_utilization['total_utilization']
        else:
            noc_rows = nn.noc.rows
            noc_cols = nn.noc.cols
            grid_size = pe_utilization['total_pes']
            utilization_pct = pe_utilization['total_utilization']
        
        # Record results
        results.append({
            "split_strategy": split_strategy,
            "noc_rows": noc_rows,
            "noc_cols": noc_cols,
            "grid_size": grid_size,
            "computation_pes": pe_utilization['used_computation_pes'],
            "aggregation_pes": pe_utilization['used_aggregation_pes'],
            "total_used_pes": pe_utilization['total_used_pes'],
            "utilization_pct": utilization_pct
        })
        
        print(f"  Grid size: {noc_rows}x{noc_cols} = {grid_size} PEs")
        print(f"  Utilization: {pe_utilization['total_used_pes']} PEs ({utilization_pct:.2f}%)")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Bar plot of grid dimensions
    plt.subplot(2, 2, 1)
    x = range(len(split_strategies))
    width = 0.35
    plt.bar(x, df['noc_rows'], width, label='Rows')
    plt.bar([i + width for i in x], df['noc_cols'], width, label='Columns')
    plt.xlabel('Split Strategy')
    plt.ylabel('Grid Dimension')
    plt.title('NoC Grid Dimensions by Split Strategy')
    plt.xticks([i + width/2 for i in x], split_strategies)
    plt.legend()
    
    # Bar plot of total grid size
    plt.subplot(2, 2, 2)
    plt.bar(split_strategies, df['grid_size'])
    plt.xlabel('Split Strategy')
    plt.ylabel('Total PEs in Grid')
    plt.title('NoC Grid Size by Split Strategy')
    
    # Bar plot of PE usage breakdown
    plt.subplot(2, 2, 3)
    x = range(len(split_strategies))
    width = 0.3
    plt.bar(x, df['computation_pes'], width, label='Computation PEs')
    plt.bar([i + width for i in x], df['aggregation_pes'], width, label='Aggregation PEs')
    plt.bar([i + 2*width for i in x], df['total_used_pes'], width, label='Total Used PEs')
    plt.xlabel('Split Strategy')
    plt.ylabel('Number of PEs')
    plt.title('PE Usage Breakdown by Split Strategy')
    plt.xticks([i + width for i in x], split_strategies)
    plt.legend()
    
    # Bar plot of utilization percentage
    plt.subplot(2, 2, 4)
    plt.bar(split_strategies, df['utilization_pct'])
    plt.xlabel('Split Strategy')
    plt.ylabel('Utilization (%)')
    plt.title('NoC Utilization by Split Strategy')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved to {plot_filename}")
    
    plt.show()
    
    # Also save results to CSV for further analysis
    csv_filename = plot_filename.replace('.png', '.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    return df


def analyze_network_dimensions(
    base_dim=64,
    scaling_factors=None,
    seq_len=1,
    pe_memory_size=64 * 1024,  # 64 KB
    data_type="float16", 
    channel_bandwidth=32.0,
    mapping_strategy="column_wise",
    split_strategy="hybrid_split",
    reuse_pe_for_aggregation=False,
    save_plot=True,
    plot_filename="network_dimension_analysis.png",
    use_effective_dimensions=True
):
    """
    Analyze and plot how NoC grid dimensions change with different network dimensions.
    
    Args:
        base_dim: Base dimension to scale (scales to base_dim * factor)
        scaling_factors: List of scaling factors to test (e.g., [1, 2, 4, 8])
        seq_len: Sequence length for inference
        pe_memory_size: PE memory size in bytes (fixed)
        data_type: The data type of the weights
        channel_bandwidth: The bandwidth of the channels in B/cycle
        mapping_strategy: How to map layers onto the NoC
        split_strategy: How to split weight matrices
        reuse_pe_for_aggregation: Whether to reuse computation PEs for aggregation
        save_plot: Whether to save the plot to a file
        plot_filename: Name of the file to save the plot to
        use_effective_dimensions: Whether to use effective dimensions based on actual PE coordinates
        
    Returns:
        DataFrame with results for each network dimension
    """
    if scaling_factors is None:
        # Default range of scaling factors
        scaling_factors = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    
    results = []
    
    print(f"Analyzing impact of network dimensions on NoC grid size...")
    print(f"Base dimension: {base_dim}, {mapping_strategy} mapping, {split_strategy} split")
    print(f"PE Memory: {pe_memory_size/1024:.1f} KB (fixed)")
    print(f"Using {'effective' if use_effective_dimensions else 'allocated'} grid dimensions")
    
    for factor in scaling_factors:
        input_dim = base_dim * factor
        hidden_dim = base_dim * factor * 2  # Typically hidden dims are larger
        output_dim = base_dim * factor
        
        print(f"Testing network dimensions: {input_dim}-{hidden_dim}-{output_dim}")
        
        # Create neural network with automatic NoC dimension calculation
        nn = FCNeuralNetwork(
            input_dim=input_dim,
            layer_dims=[hidden_dim, output_dim],
            seq_len=seq_len,
            pe_memory_size=pe_memory_size,
            split_strategy=split_strategy,
            mapping_strategy=mapping_strategy,
            reuse_pe_for_aggregation=reuse_pe_for_aggregation,
            data_type=data_type,
            channel_bandwidth=channel_bandwidth,
            # Let the network calculate its own dimensions based on PE memory
            noc_rows=None,
            noc_cols=None
        )
        
        # Calculate PE utilization using the enhanced method
        pe_utilization = nn.get_pe_utilization(use_effective_dimensions=use_effective_dimensions)
        
        # Get dimensions and metrics from the utilization results
        if use_effective_dimensions:
            noc_rows = pe_utilization['effective_rows']
            noc_cols = pe_utilization['effective_cols']
            grid_size = pe_utilization['total_pes']
            utilization_pct = pe_utilization['total_utilization']
        else:
            noc_rows = nn.noc.rows
            noc_cols = nn.noc.cols
            grid_size = pe_utilization['total_pes']
            utilization_pct = pe_utilization['total_utilization']
        
        # Record results
        results.append({
            "scaling_factor": factor,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "noc_rows": noc_rows,
            "noc_cols": noc_cols,
            "grid_size": grid_size,
            "computation_pes": pe_utilization['used_computation_pes'],
            "aggregation_pes": pe_utilization['used_aggregation_pes'],
            "total_used_pes": pe_utilization['total_used_pes'],
            "utilization_pct": utilization_pct
        })
        
        print(f"  Grid size: {noc_rows}x{noc_cols} = {grid_size} PEs")
        print(f"  Utilization: {pe_utilization['total_used_pes']} PEs ({utilization_pct:.2f}%)")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Grid dimensions vs network size
    plt.subplot(2, 2, 1)
    plt.plot(df['scaling_factor'], df['noc_rows'], 'b-o', label='Rows')
    plt.plot(df['scaling_factor'], df['noc_cols'], 'r-o', label='Columns')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Network Dimension Scale Factor')
    plt.ylabel('Grid Dimension')
    plt.title('NoC Grid Dimensions vs Network Scale')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot 2: Total grid size vs network size
    plt.subplot(2, 2, 2)
    plt.plot(df['scaling_factor'], df['grid_size'], 'g-o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Network Dimension Scale Factor')
    plt.ylabel('Total PEs in Grid')
    plt.title('NoC Grid Size vs Network Scale')
    plt.grid(True, which="both", ls="--")
    
    # Plot 3: PE usage vs network size
    plt.subplot(2, 2, 3)
    plt.plot(df['scaling_factor'], df['computation_pes'], 'b-o', label='Computation PEs')
    plt.plot(df['scaling_factor'], df['aggregation_pes'], 'r-o', label='Aggregation PEs')
    plt.plot(df['scaling_factor'], df['total_used_pes'], 'g-o', label='Total Used PEs')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Network Dimension Scale Factor')
    plt.ylabel('Number of PEs')
    plt.title('PE Usage vs Network Scale')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot 4: Utilization vs network size
    plt.subplot(2, 2, 4)
    plt.plot(df['scaling_factor'], df['utilization_pct'], 'k-o')
    plt.xscale('log')
    plt.xlabel('Network Dimension Scale Factor')
    plt.ylabel('Utilization (%)')
    plt.title('NoC Utilization vs Network Scale')
    plt.ylim(0, 100)
    plt.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(plot_filename, dpi=300)
        print(f"Plot saved to {plot_filename}")
    
    plt.show()
    
    # Also save results to CSV for further analysis
    csv_filename = plot_filename.replace('.png', '.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    return df


def analyze_mapping_strategies(
    input_dim=768, 
    hidden_dim=3072, 
    output_dim=768, 
    seq_len=1,
    pe_memory_size=64 * 1024,  # 64 KB
    data_type="float16", 
    channel_bandwidth=32.0,
    split_strategy="hybrid_split",
    reuse_pe_for_aggregation=False,
    save_plot=True,
    plot_filename="mapping_strategy_analysis.png",
    use_effective_dimensions=True
):
    """
    Analyze and plot how different mapping strategies affect NoC grid dimensions.
    
    Args:
        input_dim: Input dimension of the network
        hidden_dim: Hidden dimension of the network
        output_dim: Output dimension of the network
        seq_len: Sequence length for inference
        pe_memory_size: PE memory size in bytes
        data_type: The data type of the weights
        channel_bandwidth: The bandwidth of the channels in B/cycle
        split_strategy: How to split weight matrices
        reuse_pe_for_aggregation: Whether to reuse computation PEs for aggregation
        save_plot: Whether to save the plot to a file
        plot_filename: Name of the file to save the plot to
        use_effective_dimensions: Whether to use effective dimensions based on actual PE coordinates
        
    Returns:
        DataFrame with results for each mapping strategy
    """
    # Mapping strategies to test
    mapping_strategies = ["column_wise", "row_wise", "grid_wise"]
    
    results = []
    
    print(f"Analyzing impact of mapping strategies on NoC grid dimensions...")
    print(f"Network: {input_dim}-{hidden_dim}-{output_dim}, {split_strategy} split")
    print(f"PE Memory: {pe_memory_size/1024:.1f} KB")
    print(f"Using {'effective' if use_effective_dimensions else 'allocated'} grid dimensions")
    
    for mapping_strategy in mapping_strategies:
        print(f"Testing mapping strategy: {mapping_strategy}")
        
        # Create neural network with automatic NoC dimension calculation
        try:
            nn = FCNeuralNetwork(
                input_dim=input_dim,
                layer_dims=[hidden_dim, output_dim],
                seq_len=seq_len,
                pe_memory_size=pe_memory_size,
                split_strategy=split_strategy,
                mapping_strategy=mapping_strategy,
                reuse_pe_for_aggregation=reuse_pe_for_aggregation,
                data_type=data_type,
                channel_bandwidth=channel_bandwidth,
                # Let the network calculate its own dimensions based on PE memory
                noc_rows=None,
                noc_cols=None
            )
            
            # Calculate PE utilization using the enhanced method
            pe_utilization = nn.get_pe_utilization(use_effective_dimensions=use_effective_dimensions)
            
            # Get dimensions and metrics from the utilization results
            if use_effective_dimensions:
                noc_rows = pe_utilization['effective_rows']
                noc_cols = pe_utilization['effective_cols']
                grid_size = pe_utilization['total_pes']
                utilization_pct = pe_utilization['total_utilization']
            else:
                noc_rows = nn.noc.rows
                noc_cols = nn.noc.cols
                grid_size = pe_utilization['total_pes']
                utilization_pct = pe_utilization['total_utilization']
            
            # Record results
            results.append({
                "mapping_strategy": mapping_strategy,
                "noc_rows": noc_rows,
                "noc_cols": noc_cols,
                "grid_size": grid_size,
                "computation_pes": pe_utilization['used_computation_pes'],
                "aggregation_pes": pe_utilization['used_aggregation_pes'],
                "total_used_pes": pe_utilization['total_used_pes'],
                "utilization_pct": utilization_pct
            })
            
            print(f"  Grid size: {noc_rows}x{noc_cols} = {grid_size} PEs")
            print(f"  Utilization: {pe_utilization['total_used_pes']} PEs ({utilization_pct:.2f}%)")
        
        except Exception as e:
            print(f"  Error with {mapping_strategy}: {str(e)}")
            # Add placeholder with error
            results.append({
                "mapping_strategy": mapping_strategy,
                "noc_rows": 0,
                "noc_cols": 0,
                "grid_size": 0,
                "computation_pes": 0,
                "aggregation_pes": 0,
                "total_used_pes": 0,
                "utilization_pct": 0,
                "error": str(e)
            })
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Filter out rows with errors for plotting
    if 'error' in df.columns:
        plot_df = df[df['error'].isna() | (df['error'] == '')]
    else:
        plot_df = df
    
    if len(plot_df) > 0:
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Bar plot of grid dimensions
        plt.subplot(2, 2, 1)
        x = range(len(plot_df))
        width = 0.35
        plt.bar(x, plot_df['noc_rows'], width, label='Rows')
        plt.bar([i + width for i in x], plot_df['noc_cols'], width, label='Columns')
        plt.xlabel('Mapping Strategy')
        plt.ylabel('Grid Dimension')
        plt.title('NoC Grid Dimensions by Mapping Strategy')
        plt.xticks([i + width/2 for i in x], plot_df['mapping_strategy'])
        plt.legend()
        
        # Bar plot of total grid size
        plt.subplot(2, 2, 2)
        plt.bar(plot_df['mapping_strategy'], plot_df['grid_size'])
        plt.xlabel('Mapping Strategy')
        plt.ylabel('Total PEs in Grid')
        plt.title('NoC Grid Size by Mapping Strategy')
        
        # Bar plot of PE usage breakdown
        plt.subplot(2, 2, 3)
        x = range(len(plot_df))
        width = 0.3
        plt.bar(x, plot_df['computation_pes'], width, label='Computation PEs')
        plt.bar([i + width for i in x], plot_df['aggregation_pes'], width, label='Aggregation PEs')
        plt.bar([i + 2*width for i in x], plot_df['total_used_pes'], width, label='Total Used PEs')
        plt.xlabel('Mapping Strategy')
        plt.ylabel('Number of PEs')
        plt.title('PE Usage Breakdown by Mapping Strategy')
        plt.xticks([i + width for i in x], plot_df['mapping_strategy'])
        plt.legend()
        
        # Bar plot of utilization percentage
        plt.subplot(2, 2, 4)
        plt.bar(plot_df['mapping_strategy'], plot_df['utilization_pct'])
        plt.xlabel('Mapping Strategy')
        plt.ylabel('Utilization (%)')
        plt.title('NoC Utilization by Mapping Strategy')
        plt.ylim(0, 100)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(plot_filename, dpi=300)
            print(f"Plot saved to {plot_filename}")
        
        plt.show()
    else:
        print("No valid results to plot")
    
    # Also save results to CSV for further analysis
    csv_filename = plot_filename.replace('.png', '.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    return df


def run_all_analyses(output_dir="analysis_results", use_effective_dimensions=True):
    """
    Run all analysis functions and save results to a directory.
    
    Args:
        output_dir: Directory to save results to
        use_effective_dimensions: Whether to use effective dimensions based on actual PE coordinates
    """
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Run PE memory size analysis
    memory_df = analyze_pe_memory_impact(
        input_dim=12,
        hidden_dim=12*3,
        output_dim=12,
        mapping_strategy="grid_wise",
        split_strategy="hybrid_split",
        plot_filename=os.path.join(output_dir, "pe_memory_analysis.png"),
        use_effective_dimensions=use_effective_dimensions
    )
    
    # # Run split strategy analysis
    # split_df = analyze_split_strategies(
    #     input_dim=64,
    #     hidden_dim=128,
    #     output_dim=64,
    #     mapping_strategy="grid_wise",
    #     plot_filename=os.path.join(output_dir, "split_strategy_analysis.png"),
    #     use_effective_dimensions=use_effective_dimensions
    # )
    
    # # Run mapping strategy analysis
    # mapping_df = analyze_mapping_strategies(
    #     input_dim=64,
    #     hidden_dim=128,
    #     output_dim=64,
    #     split_strategy="hybrid_split",
    #     plot_filename=os.path.join(output_dir, "mapping_strategy_analysis.png"),
    #     use_effective_dimensions=use_effective_dimensions
    # )
    
    # # Run network dimension analysis
    # dim_df = analyze_network_dimensions(
    #     base_dim=16,
    #     mapping_strategy="grid_wise",
    #     split_strategy="hybrid_split",
    #     plot_filename=os.path.join(output_dir, "network_dimension_analysis.png"),
    #     use_effective_dimensions=use_effective_dimensions
    # )
    
    print(f"All analyses completed. Results saved to {output_dir}/")
    print(f"Using {'effective' if use_effective_dimensions else 'allocated'} grid dimensions")
    
    return {
        "memory_analysis": memory_df,
        # "split_strategy_analysis": split_df,
        # "mapping_strategy_analysis": mapping_df,
        # "dimension_analysis": dim_df
    }

# Update the if __name__ == "__main__" block to include the new functions
if __name__ == "__main__":
    # Run all analyses and save results to a directory using effective dimensions
    #run_all_analyses(use_effective_dimensions=True)
    
    run_example()
