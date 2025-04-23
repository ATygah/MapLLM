import torch
import pandas as pd
import math
from typing import List, Tuple, Dict, Optional
from .pe_noc import NoCTopology, PE
from data_structs import dtype_size

class FCLayerMapper:
    """Maps fully connected layers to NoC topology."""
    def __init__(self, 
                 noc: NoCTopology, 
                 input_dim: int, 
                 output_dims: List[int],
                 seq_len: int,
                 mapping_strategy: str = "column_wise",
                 split_strategy: str = "column_split",
                 data_type: str = "float16",
                 neural_network=None,
                 allow_wrapping: bool = False,
                 seq_split_factor: int = 1):
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
                           - "sequence_split": Split across sequence dimension (replicates weights)
            data_type: The data type of the weights
            neural_network: Reference to the parent neural network
            allow_wrapping: Whether to allow wrapping around the edges of the NoC (default: False)
            seq_split_factor: Number of sequence chunks to process in parallel (default: 1)
        """
        self.noc = noc
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.seq_len = seq_len
        self.mapping_strategy = mapping_strategy
        self.split_strategy = split_strategy
        self.layer_count = len(output_dims)
        self.data_type = data_type
        self.neural_network = neural_network  # Store reference to the parent neural network
        self.allow_wrapping = allow_wrapping
        self.seq_split_factor = seq_split_factor
        
        # Import data_type_bytes from external module to avoid circular imports
        self.data_type_bytes = dtype_size(self.data_type)
        
        # Maps (layer_id, pe_index) to PE coordinates
        self.layer_pe_map = {}
        # Maps PE coordinates to (layer_id, pe_index)
        self.pe_layer_map = {}
        
        # For sequence splitting, keep track of which part of the sequence each PE handles
        # Maps (layer_id, pe_index) to sequence range (start, end)
        self.pe_seq_ranges = {}
        
        # Validate and apply mapping strategy
        self._map_layers_to_pes(allow_wrapping=self.allow_wrapping)
    
    def _map_layers_to_pes(self, allow_wrapping=False):
        """
        Map layers to PEs based on splitting and mapping strategies.
        
        Args:
            allow_wrapping: Whether to allow wrapping around the edges of the NoC (default: False)
        """
        # Calculate PE requirements for each layer based on split strategy
        layer_pe_requirements = self._calculate_pe_requirements()
        
        # Apply mapping strategy to place PEs on the NoC
        if self.mapping_strategy == "column_wise":
            self._map_to_columns(layer_pe_requirements)
        elif self.mapping_strategy == "row_wise":
            self._map_to_rows(layer_pe_requirements)
        elif self.mapping_strategy == "grid_wise":
            self._map_to_grid(layer_pe_requirements, allow_wrapping=allow_wrapping)
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
            
            # Calculate weights per PE without activation memory first
            full_memory_weights_per_pe = self.noc.pe_memory_size // self.data_type_bytes
            
            # Calculate effective sequence length per PE if we're using sequence splitting
            effective_seq_len = math.ceil(self.seq_len / self.seq_split_factor)
            
            if self.split_strategy == "sequence_split":
                # For sequence splitting, we replicate the weights and distribute the sequence
                # Each PE handles the full weight matrix but only part of the sequence
                
                # Calculate memory required for weights
                weight_memory = weight_matrix_rows * weight_matrix_cols * self.data_type_bytes
                
                # Calculate memory required for the partial sequence
                input_activation_memory = weight_matrix_rows * effective_seq_len * self.data_type_bytes
                output_activation_memory = weight_matrix_cols * effective_seq_len * self.data_type_bytes
                activation_memory = max(input_activation_memory, output_activation_memory)
                
                # Check if we can fit the full weight matrix plus activations in a PE
                if weight_memory + activation_memory > self.noc.pe_memory_size:
                    raise ValueError(f"PE memory ({self.noc.pe_memory_size} bytes) is insufficient to store "
                                    f"the full weight matrix ({weight_memory} bytes) plus activations "
                                    f"({activation_memory} bytes) for layer {layer_id}, even with sequence splitting. "
                                    f"Consider using a different split strategy or larger PEs.")
                
                # For sequence splitting, we need seq_split_factor PEs per layer
                # Each PE handles the full weight matrix but a different part of the sequence
                split_ranges = []
                for seq_idx in range(self.seq_split_factor):
                    seq_start = seq_idx * effective_seq_len
                    seq_end = min((seq_idx + 1) * effective_seq_len, self.seq_len)
                    
                    # Split dimension 3 indicates sequence splitting
                    # Full weight matrix for each PE
                    split_ranges.append((3, 0, weight_matrix_rows, 0, weight_matrix_cols, seq_start, seq_end))
                
            elif self.split_strategy == "column_split":
                # For column-wise split, each PE handles:
                # - Same input dimensions (full rows)
                # - Subset of output dimensions (partial columns)
                
                # First determine how many columns can fit per PE if we only consider weights
                max_cols_per_pe_without_activations = max(1, full_memory_weights_per_pe // weight_matrix_rows)
                
                # For each potential columns_per_pe value, check if there's enough memory for activations
                cols_per_pe = max_cols_per_pe_without_activations
                while cols_per_pe > 0:
                    # Weight memory for this split
                    weight_memory = weight_matrix_rows * cols_per_pe * self.data_type_bytes
                    
                    # Each PE receives the full input rows, but produces only cols_per_pe outputs
                    input_activation_memory = weight_matrix_rows * effective_seq_len * self.data_type_bytes
                    output_activation_memory = cols_per_pe * effective_seq_len * self.data_type_bytes
                    
                    # Total memory required
                    total_memory_required = weight_memory + max(input_activation_memory, output_activation_memory)
                    
                    if total_memory_required <= self.noc.pe_memory_size:
                        break
                    
                    cols_per_pe -= 1
                
                if cols_per_pe == 0:
                    raise ValueError(f"PE memory ({self.noc.pe_memory_size} bytes) is insufficient to store "
                                    f"weights and activations for layer {layer_id}. Even a single column split "
                                    f"would require more memory than available.")
                
                pes_needed = math.ceil(weight_matrix_cols / cols_per_pe)
                
                # Calculate split ranges for each PE
                split_ranges = []
                
                # If we're doing sequence splitting on top of column splitting
                if self.seq_split_factor > 1:
                    for seq_idx in range(self.seq_split_factor):
                        seq_start = seq_idx * effective_seq_len
                        seq_end = min((seq_idx + 1) * effective_seq_len, self.seq_len)
                        
                        for pe_idx in range(pes_needed):
                            col_start = pe_idx * cols_per_pe
                            col_end = min((pe_idx + 1) * cols_per_pe, weight_matrix_cols)
                            # Split dimension 1 with sequence information
                            split_ranges.append((1, 0, weight_matrix_rows, col_start, col_end, seq_start, seq_end))
                else:
                    # Standard column splitting without sequence splitting
                    for pe_idx in range(pes_needed):
                        col_start = pe_idx * cols_per_pe
                        col_end = min((pe_idx + 1) * cols_per_pe, weight_matrix_cols)
                        # split_dim=1 means splitting by columns
                        # Use full rows (0 to weight_matrix_rows) and specific columns
                        split_ranges.append((1, 0, weight_matrix_rows, col_start, col_end))
            
            elif self.split_strategy == "row_split":
                # For row-wise split, each PE handles:
                # - Subset of input dimensions (partial rows)
                # - Full output dimensions (all columns)
                
                # First determine how many rows can fit per PE if we only consider weights
                max_rows_per_pe_without_activations = max(1, full_memory_weights_per_pe // weight_matrix_cols)
                
                # For each potential rows_per_pe value, check if there's enough memory for activations
                rows_per_pe = max_rows_per_pe_without_activations
                while rows_per_pe > 0:
                    # Weight memory for this split
                    weight_memory = rows_per_pe * weight_matrix_cols * self.data_type_bytes
                    
                    # Each PE receives only rows_per_pe inputs, but must produce full outputs (which will be aggregated)
                    input_activation_memory = rows_per_pe * effective_seq_len * self.data_type_bytes
                    output_activation_memory = weight_matrix_cols * effective_seq_len * self.data_type_bytes
                    
                    # Total memory required
                    total_memory_required = weight_memory + max(input_activation_memory, output_activation_memory)
                    
                    if total_memory_required <= self.noc.pe_memory_size:
                        break
                    
                    rows_per_pe -= 1
                
                if rows_per_pe == 0:
                    raise ValueError(f"PE memory ({self.noc.pe_memory_size} bytes) is insufficient to store "
                                    f"weights and activations for layer {layer_id}. Even a single row split "
                                    f"would require more memory than available.")
                
                pes_needed = math.ceil(weight_matrix_rows / rows_per_pe)
                
                # Calculate split ranges for each PE
                split_ranges = []
                
                # If we're doing sequence splitting on top of row splitting
                if self.seq_split_factor > 1:
                    for seq_idx in range(self.seq_split_factor):
                        seq_start = seq_idx * effective_seq_len
                        seq_end = min((seq_idx + 1) * effective_seq_len, self.seq_len)
                        
                        for pe_idx in range(pes_needed):
                            row_start = pe_idx * rows_per_pe
                            row_end = min((pe_idx + 1) * rows_per_pe, weight_matrix_rows)
                            # Split dimension 0 with sequence information
                            split_ranges.append((0, row_start, row_end, 0, weight_matrix_cols, seq_start, seq_end))
                else:
                    # Standard row splitting without sequence splitting
                    for pe_idx in range(pes_needed):
                        row_start = pe_idx * rows_per_pe
                        row_end = min((pe_idx + 1) * rows_per_pe, weight_matrix_rows)
                        # split_dim=0 means splitting by rows
                        # Use specific rows and full columns (0 to weight_matrix_cols)
                        split_ranges.append((0, row_start, row_end, 0, weight_matrix_cols))
                
            elif self.split_strategy == "hybrid_split":
                # For hybrid split, try to find an optimal balance
                best_row_pes = None
                best_col_pes = None
                best_total_pes = float('inf')
                best_rows_per_pe = 0
                best_cols_per_pe = 0
                
                # Try different combinations of row and column divisions
                for row_div in range(1, weight_matrix_rows + 1):
                    rows_per_pe = math.ceil(weight_matrix_rows / row_div)
                    
                    for col_div in range(1, weight_matrix_cols + 1):
                        cols_per_pe = math.ceil(weight_matrix_cols / col_div)
                        
                        # Calculate memory requirements for this split
                        weight_memory = rows_per_pe * cols_per_pe * self.data_type_bytes
                        input_activation_memory = rows_per_pe * effective_seq_len * self.data_type_bytes
                        output_activation_memory = cols_per_pe * effective_seq_len * self.data_type_bytes
                        
                        total_memory_required = weight_memory + max(input_activation_memory, output_activation_memory)
                        
                        # Check if this split fits in PE memory
                        if total_memory_required <= self.noc.pe_memory_size:
                            total_pes = row_div * col_div
                            
                            # Keep track of the best (minimum) number of PEs needed
                            if total_pes < best_total_pes:
                                best_row_pes = row_div
                                best_col_pes = col_div
                                best_total_pes = total_pes
                                best_rows_per_pe = rows_per_pe
                                best_cols_per_pe = cols_per_pe
                
                if best_row_pes is None or best_col_pes is None:
                    raise ValueError(f"Could not find a viable hybrid split for layer {layer_id} that fits "
                                    f"within PE memory ({self.noc.pe_memory_size} bytes) accounting for both "
                                    f"weights and activations. Consider increasing PE memory size.")
                
                # Create the grid of PEs with both row and column splits
                split_ranges = []
                
                # If we're doing sequence splitting on top of hybrid splitting
                if self.seq_split_factor > 1:
                    for seq_idx in range(self.seq_split_factor):
                        seq_start = seq_idx * effective_seq_len
                        seq_end = min((seq_idx + 1) * effective_seq_len, self.seq_len)
                        
                        for row_idx in range(best_row_pes):
                            for col_idx in range(best_col_pes):
                                row_start = row_idx * best_rows_per_pe
                                row_end = min((row_idx + 1) * best_rows_per_pe, weight_matrix_rows)
                                
                                col_start = col_idx * best_cols_per_pe
                                col_end = min((col_idx + 1) * best_cols_per_pe, weight_matrix_cols)
                                
                                # For hybrid split with sequence splitting
                                split_ranges.append((2, row_start, row_end, col_start, col_end, seq_start, seq_end))
                else:
                    # Standard hybrid splitting without sequence splitting
                    for row_idx in range(best_row_pes):
                        for col_idx in range(best_col_pes):
                            row_start = row_idx * best_rows_per_pe
                            row_end = min((row_idx + 1) * best_rows_per_pe, weight_matrix_rows)
                            
                            col_start = col_idx * best_cols_per_pe
                            col_end = min((col_idx + 1) * best_cols_per_pe, weight_matrix_cols)
                            
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
        # Get already used PEs from the parent LLM if available
        already_used_pes = set()
        if hasattr(self, 'neural_network') and hasattr(self.neural_network, 'llm') and self.neural_network.llm is not None:
            already_used_pes = set(self.neural_network.llm.used_pes)
        
        for layer_id, split_ranges in layer_pe_requirements.items():
            # Start with the preferred column (layer_id)
            col = layer_id
            valid_column_found = False
            
            # Try columns starting from layer_id until we find one with enough available rows
            for col_offset in range(self.noc.cols):
                # Calculate actual column to try (wrap around if needed)
                col = (layer_id + col_offset) % self.noc.cols
                
                # Check if we have enough rows in this column, considering already used PEs
                available_rows = [row for row in range(self.noc.rows) if (col, row) not in already_used_pes]
                if len(split_ranges) <= len(available_rows):
                    valid_column_found = True
                    break
            
            if not valid_column_found:
                raise ValueError(f"Layer {layer_id} requires {len(split_ranges)} PEs, "
                                 f"but couldn't find any column with enough available rows. "
                                 f"Consider increasing NoC dimensions.")
            
            # For hybrid split, organize PEs in a grid
            if self.split_strategy == "hybrid_split":
                # Group PEs by their column split (each group handles a subset of output neurons)
                col_groups = {}
                for pe_idx, split_info in enumerate(split_ranges):
                    # Handle both with and without sequence splitting
                    if len(split_info) >= 7:  # With sequence splitting
                        split_type, row_start, row_end, col_start, col_end, seq_start, seq_end = split_info
                    else:  # Without sequence splitting
                        split_type, row_start, row_end, col_start, col_end = split_info
                        seq_start = 0
                        seq_end = self.seq_len
                        
                    col_group = (col_start, col_end)
                    if col_group not in col_groups:
                        col_groups[col_group] = []
                    col_groups[col_group].append((pe_idx, split_info))
                
                # Place PEs in a more structured way
                pe_idx_map = {}  # Maps PE index to actual coordinates
                current_row_idx = 0
                
                # For each column group, place all its PEs one after another
                for col_group, pe_list in col_groups.items():
                    for pe_idx, _ in pe_list:
                        # Get the next available row from our filtered list
                        if current_row_idx >= len(available_rows):
                            raise ValueError(f"Not enough available PEs in column {col} for layer {layer_id}")
                        
                        row = available_rows[current_row_idx]
                        pe_coords = (col, row)
                        pe_idx_map[pe_idx] = pe_coords
                        already_used_pes.add(pe_coords)  # Mark this PE as used
                        current_row_idx += 1
            
            # Map each PE for this layer to the corresponding row in this column
            for pe_idx, split_info in enumerate(split_ranges):
                if self.split_strategy == "hybrid_split":
                    # For hybrid, use the pre-calculated coordinates
                    pe_coords = pe_idx_map[pe_idx]
                else:
                    # For non-hybrid, get the next available row
                    if pe_idx >= len(available_rows):
                        raise ValueError(f"Not enough available rows in column {col} for layer {layer_id}")
                    
                    row = available_rows[pe_idx]
                    pe_coords = (col, row)
                    already_used_pes.add(pe_coords)  # Mark this PE as used
                
                # Get the PE at these coordinates
                pe = self.noc.get_pe(*pe_coords)
                
                # Handle both with and without sequence splitting
                if len(split_info) >= 7:  # With sequence splitting
                    split_dim, row_start, row_end, col_start, col_end, seq_start, seq_end = split_info
                    
                    # Store sequence range for this PE
                    self.pe_seq_ranges[(layer_id, pe_idx)] = (seq_start, seq_end)
                else:  # Without sequence splitting
                    split_dim, row_start, row_end, col_start, col_end = split_info
                    seq_start = 0
                    seq_end = self.seq_len
                
                # Create weights tensor
                pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                
                # Set weights along with sequence information
                if split_dim == 3:  # Sequence split
                    pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                    pe.seq_start = seq_start
                    pe.seq_end = seq_end
                else:
                    pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                    pe.seq_start = seq_start
                    pe.seq_end = seq_end
                
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
        # Get already used PEs from the parent LLM if available
        already_used_pes = set()
        if hasattr(self, 'neural_network') and hasattr(self.neural_network, 'llm') and self.neural_network.llm is not None:
            already_used_pes = set(self.neural_network.llm.used_pes)
            
        for layer_id, split_ranges in layer_pe_requirements.items():
            # Start with the preferred row (layer_id)
            row = layer_id
            valid_row_found = False
            
            # Try rows starting from layer_id until we find one with enough available columns
            for row_offset in range(self.noc.rows):
                # Calculate actual row to try (wrap around if needed)
                row = (layer_id + row_offset) % self.noc.rows
                
                # Check if we have enough columns in this row, considering already used PEs
                available_cols = [col for col in range(self.noc.cols) if (col, row) not in already_used_pes]
                if len(split_ranges) <= len(available_cols):
                    valid_row_found = True
                    break
            
            if not valid_row_found:
                raise ValueError(f"Layer {layer_id} requires {len(split_ranges)} PEs, "
                                 f"but couldn't find any row with enough available columns. "
                                 f"Consider increasing NoC dimensions.")
            
            # For hybrid split, organize PEs in a grid
            if self.split_strategy == "hybrid_split":
                # Group PEs by their column split (each group handles a subset of output neurons)
                col_groups = {}
                for pe_idx, split_info in enumerate(split_ranges):
                    # Handle both with and without sequence splitting
                    if len(split_info) >= 7:  # With sequence splitting
                        split_type, row_start, row_end, col_start, col_end, seq_start, seq_end = split_info
                    else:  # Without sequence splitting
                        split_type, row_start, row_end, col_start, col_end = split_info
                        seq_start = 0
                        seq_end = self.seq_len
                        
                    col_group = (col_start, col_end)
                    if col_group not in col_groups:
                        col_groups[col_group] = []
                    col_groups[col_group].append((pe_idx, split_info))
                
                # Place PEs in a more structured way
                pe_idx_map = {}  # Maps PE index to actual coordinates
                current_col_idx = 0
                
                # For each column group, place all its PEs one after another
                for col_group, pe_list in col_groups.items():
                    for pe_idx, _ in pe_list:
                        # Get the next available column from our filtered list
                        if current_col_idx >= len(available_cols):
                            raise ValueError(f"Not enough available PEs in row {row} for layer {layer_id}")
                        
                        col = available_cols[current_col_idx]
                        pe_coords = (col, row)
                        pe_idx_map[pe_idx] = pe_coords
                        already_used_pes.add(pe_coords)  # Mark this PE as used
                        current_col_idx += 1
            
            # Map each PE for this layer to the corresponding column in this row
            for pe_idx, split_info in enumerate(split_ranges):
                if self.split_strategy == "hybrid_split":
                    # For hybrid, use the pre-calculated coordinates
                    pe_coords = pe_idx_map[pe_idx]
                else:
                    # For non-hybrid, get the next available column
                    if pe_idx >= len(available_cols):
                        raise ValueError(f"Not enough available columns in row {row} for layer {layer_id}")
                    
                    col = available_cols[pe_idx]
                    pe_coords = (col, row)
                    already_used_pes.add(pe_coords)  # Mark this PE as used
                
                # Get the PE at these coordinates
                pe = self.noc.get_pe(*pe_coords)
                
                # Handle both with and without sequence splitting
                if len(split_info) >= 7:  # With sequence splitting
                    split_dim, row_start, row_end, col_start, col_end, seq_start, seq_end = split_info
                    
                    # Store sequence range for this PE
                    self.pe_seq_ranges[(layer_id, pe_idx)] = (seq_start, seq_end)
                else:  # Without sequence splitting
                    split_dim, row_start, row_end, col_start, col_end = split_info
                    seq_start = 0
                    seq_end = self.seq_len
                
                # Create weights tensor
                pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                
                # Set weights along with sequence information
                if split_dim == 3:  # Sequence split
                    pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                    pe.seq_start = seq_start
                    pe.seq_end = seq_end
                else:
                    pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                    pe.seq_start = seq_start
                    pe.seq_end = seq_end
                
                # Update maps
                self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
    
    def _map_to_grid(self, layer_pe_requirements, allow_wrapping=False):
        """
        Map layers to 2D grids in the NoC.
        Each layer is mapped to a rectangular region, attempting to minimize total NoC dimensions.
        
        Args:
            layer_pe_requirements: Dictionary mapping layer_id to a list of split information.
            allow_wrapping: Whether to allow wrapping around the edges of the NoC (default: False)
        """
        # Get already used PEs from the parent LLM if available
        already_used_pes = set()
        if hasattr(self, 'neural_network') and hasattr(self.neural_network, 'llm') and self.neural_network.llm is not None:
            already_used_pes = set(self.neural_network.llm.used_pes)
            
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
                
                for pe_idx, split_info in enumerate(split_ranges):
                    # Handle both with and without sequence splitting
                    if len(split_info) >= 7:  # With sequence splitting
                        split_type, row_start, row_end, col_start, col_end, seq_start, seq_end = split_info
                    else:  # Without sequence splitting
                        split_type, row_start, row_end, col_start, col_end = split_info
                        seq_start = 0
                        seq_end = self.seq_len
                    
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
                
                # First priority: Try to place each column group in a single column for efficient mapping
                valid_placement_found = False
                
                # Try a column-aligned placement first (keeping column groups in the same column)
                if num_col_groups <= self.noc.cols:  # Only try this if we have enough columns
                    # Sort column groups for consistent placement
                    sorted_col_groups = sorted(col_groups.items())
                    
                    # Try different starting positions
                    max_row_range = self.noc.rows - num_row_groups if not allow_wrapping else self.noc.rows
                    max_col_range = self.noc.cols - num_col_groups if not allow_wrapping else self.noc.cols
                    
                    for start_y in range(max_row_range):
                        # Try to find consecutive columns that have enough free rows
                        for start_x in range(max_col_range):
                            region_is_free = True
                            
                            # Check each column group
                            for col_idx, (col_group, _) in enumerate(sorted_col_groups):
                                col_x = start_x + col_idx
                                if allow_wrapping:
                                    col_x = col_x % self.noc.cols
                                
                                # Check if this column has enough available rows
                                available_rows = []
                                for row_offset in range(num_row_groups):
                                    row_y = start_y + row_offset
                                    if allow_wrapping:
                                        row_y = row_y % self.noc.rows
                                    
                                    pe_coord = (col_x, row_y)
                                    if pe_coord in already_used_pes:
                                        region_is_free = False
                                        break
                                    available_rows.append(row_y)
                                
                                if not region_is_free or len(available_rows) < num_row_groups:
                                    region_is_free = False
                                    break
                            
                            if region_is_free:
                                # We found a valid column-aligned placement
                                valid_placement_found = True
                                
                                # Sort row groups for consistent placement
                                sorted_row_groups = sorted(row_groups.items())
                                
                                # Keep track of PE positions in the grid
                                pe_positions = {}  # Maps pe_idx to (col_x, row_y) coordinates
                                
                                # Assign positions for each PE
                                for row_idx, (row_group, row_pe_indices) in enumerate(sorted_row_groups):
                                    for col_idx, (col_group, col_pe_indices) in enumerate(sorted_col_groups):
                                        # Find PEs that belong to both this row and column group
                                        for intersect_pe_idx in set(row_pe_indices) & set(col_pe_indices):
                                            col_x = start_x + col_idx
                                            row_y = start_y + row_idx
                                            if allow_wrapping:
                                                col_x = col_x % self.noc.cols
                                                row_y = row_y % self.noc.rows
                                            
                                            pe_positions[intersect_pe_idx] = (col_x, row_y)
                                
                                # Map each PE to its position in the NoC
                                for pe_idx, split_info in enumerate(split_ranges):
                                    if pe_idx not in pe_positions:
                                        # This shouldn't happen, but skip if not in the mapping
                                        continue
                                        
                                    pe_coords = pe_positions[pe_idx]
                                    
                                    # Final check to ensure the PE is not already used
                                    if pe_coords in already_used_pes:
                                        raise ValueError(f"PE at coordinates {pe_coords} is already in use by another network")
                                    
                                    # Mark this PE as used
                                    already_used_pes.add(pe_coords)
                                    
                                    pe = self.noc.get_pe(*pe_coords)
                                    
                                    # Handle both with and without sequence splitting
                                    if len(split_info) >= 7:  # With sequence splitting
                                        split_dim, row_start, row_end, col_start, col_end, seq_start, seq_end = split_info
                                        
                                        # Store sequence range for this PE
                                        self.pe_seq_ranges[(layer_id, pe_idx)] = (seq_start, seq_end)
                                    else:  # Without sequence splitting
                                        split_dim, row_start, row_end, col_start, col_end = split_info
                                        seq_start = 0
                                        seq_end = self.seq_len
                                    
                                    # Create weights tensor
                                    pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                                    pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                                    
                                    # Update maps
                                    self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                                    self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
                                
                                break  # Exit the start_x loop
                        
                        if valid_placement_found:
                            break  # Exit the start_y loop
                
                # Second priority: If column-aligned placement failed, try a rectangular grid placement
                if not valid_placement_found:
                    # Try different starting positions
                    max_row_range = self.noc.rows - num_row_groups if not allow_wrapping else self.noc.rows
                    max_col_range = self.noc.cols - num_col_groups if not allow_wrapping else self.noc.cols
                    
                    for start_y in range(max_row_range):
                        for start_x in range(max_col_range):
                            # Check if this region is free
                            region_is_free = True
                            for y_offset in range(num_row_groups):
                                for x_offset in range(num_col_groups):
                                    # Calculate position
                                    noc_x = start_x + x_offset
                                    noc_y = start_y + y_offset
                                    if allow_wrapping:
                                        noc_x = noc_x % self.noc.cols
                                        noc_y = noc_y % self.noc.rows
                                    
                                    pe_coord = (noc_x, noc_y)
                                    if pe_coord in already_used_pes:
                                        region_is_free = False
                                        break
                                if not region_is_free:
                                    break
                            
                            if region_is_free:
                                valid_placement_found = True
                                
                                # Sort row and column groups for consistent mapping
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
                                for pe_idx, split_info in enumerate(split_ranges):
                                    if pe_idx not in pe_positions:
                                        # This shouldn't happen, but skip if not in the mapping
                                        continue
                                        
                                    row_idx, col_idx = pe_positions[pe_idx]
                                    
                                    # Calculate absolute NoC coordinates
                                    noc_x = start_x + col_idx
                                    noc_y = start_y + row_idx
                                    if allow_wrapping:
                                        noc_x = noc_x % self.noc.cols
                                        noc_y = noc_y % self.noc.rows
                                    
                                    # Final check to ensure the PE is not already used
                                    pe_coords = (noc_x, noc_y)
                                    if pe_coords in already_used_pes:
                                        raise ValueError(f"PE at coordinates {pe_coords} is already in use by another network")
                                    
                                    # Mark this PE as used
                                    already_used_pes.add(pe_coords)
                                    
                                    pe = self.noc.get_pe(*pe_coords)
                                    
                                    # Handle both with and without sequence splitting
                                    if len(split_info) >= 7:  # With sequence splitting
                                        split_dim, row_start, row_end, col_start, col_end, seq_start, seq_end = split_info
                                        
                                        # Store sequence range for this PE
                                        self.pe_seq_ranges[(layer_id, pe_idx)] = (seq_start, seq_end)
                                    else:  # Without sequence splitting
                                        split_dim, row_start, row_end, col_start, col_end = split_info
                                        seq_start = 0
                                        seq_end = self.seq_len
                                    
                                    # Create weights tensor
                                    pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                                    pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                                    
                                    # Update maps
                                    self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                                    self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
                                
                                break  # Exit the start_x loop
                        
                        if valid_placement_found:
                            break  # Exit the start_y loop
                
                # Last resort: If no suitable placement found, try any available positions
                if not valid_placement_found and allow_wrapping:
                    # Create a list of available PEs
                    available_pes = []
                    for x in range(self.noc.cols):
                        for y in range(self.noc.rows):
                            if (x, y) not in already_used_pes:
                                available_pes.append((x, y))
                    
                    # Check if we have enough PEs
                    total_pes_needed = num_row_groups * num_col_groups
                    if len(available_pes) >= total_pes_needed:
                        valid_placement_found = True
                        
                        # Sort row and column groups for consistent mapping
                        sorted_row_groups = sorted(row_groups.items())
                        sorted_col_groups = sorted(col_groups.items())
                        
                        # Keep track of PE positions in the grid
                        pe_positions = {}  # Maps pe_idx to PE coordinates
                        
                        # Assign positions using available PEs
                        pe_idx_counter = 0
                        for row_idx, (row_range, row_pe_indices) in enumerate(sorted_row_groups):
                            for col_idx, (col_range, col_pe_indices) in enumerate(sorted_col_groups):
                                # Find PEs that belong to both this row and column group
                                for intersect_pe_idx in set(row_pe_indices) & set(col_pe_indices):
                                    if pe_idx_counter < len(available_pes):
                                        pe_coords = available_pes[pe_idx_counter]
                                        pe_positions[intersect_pe_idx] = pe_coords
                                        already_used_pes.add(pe_coords)
                                        pe_idx_counter += 1
                        
                        # Map each PE to its position in the NoC
                        for pe_idx, split_info in enumerate(split_ranges):
                            if pe_idx not in pe_positions:
                                # This shouldn't happen, but skip if not in the mapping
                                continue
                                
                            pe_coords = pe_positions[pe_idx]
                            pe = self.noc.get_pe(*pe_coords)
                            
                            # Handle both with and without sequence splitting
                            if len(split_info) >= 7:  # With sequence splitting
                                split_dim, row_start, row_end, col_start, col_end, seq_start, seq_end = split_info
                                
                                # Store sequence range for this PE
                                self.pe_seq_ranges[(layer_id, pe_idx)] = (seq_start, seq_end)
                            else:  # Without sequence splitting
                                split_dim, row_start, row_end, col_start, col_end = split_info
                                seq_start = 0
                                seq_end = self.seq_len
                            
                            # Create weights tensor
                            pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                            pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                            
                            # Update maps
                            self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                            self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
                
                if not valid_placement_found:
                    raise ValueError(f"Could not find space in NoC for layer {layer_id} requiring " 
                                   f"{num_row_groups}x{num_col_groups} PEs. NoC dimensions: {self.noc.rows}x{self.noc.cols}. "
                                   f"Consider increasing NoC dimensions or setting allow_wrapping=True.")
            else:
                # For non-hybrid splits, place PEs in a simple square-ish grid
                num_pes = len(split_ranges)
                
                # Calculate an efficient grid shape for this layer
                # Try to keep it relatively square
                grid_height = int(math.sqrt(num_pes * 1.5))
                grid_width = math.ceil(num_pes / grid_height)
                
                # Find a place in the NoC grid with enough consecutive available PEs
                valid_placement_found = False
                
                # Try different starting positions
                max_row_range = self.noc.rows - grid_height if not allow_wrapping else self.noc.rows
                max_col_range = self.noc.cols - grid_width if not allow_wrapping else self.noc.cols
                
                if max_row_range > 0 and max_col_range > 0:
                    for start_y in range(max_row_range):
                        for start_x in range(max_col_range):
                            # Check if this region is free
                            region_is_free = True
                            for y_offset in range(grid_height):
                                for x_offset in range(grid_width):
                                    # Only check positions that would actually be used by this layer
                                    pe_idx = y_offset * grid_width + x_offset
                                    if pe_idx >= num_pes:
                                        continue
                                    
                                    # Calculate position
                                    noc_x = start_x + x_offset
                                    noc_y = start_y + y_offset
                                    if allow_wrapping:
                                        noc_x = noc_x % self.noc.cols
                                        noc_y = noc_y % self.noc.rows
                                    
                                    pe_coord = (noc_x, noc_y)
                                    if pe_coord in already_used_pes:
                                        region_is_free = False
                                        break
                                if not region_is_free:
                                    break
                            
                            if region_is_free:
                                valid_placement_found = True
                                
                                # Map each PE
                                for pe_idx, split_info in enumerate(split_ranges):
                                    # Calculate position within the layer's grid
                                    local_x = pe_idx % grid_width
                                    local_y = pe_idx // grid_width
                                    
                                    # Calculate absolute NoC coordinates
                                    noc_x = start_x + local_x
                                    noc_y = start_y + local_y
                                    if allow_wrapping:
                                        noc_x = noc_x % self.noc.cols
                                        noc_y = noc_y % self.noc.rows
                                    
                                    # Check if the PE is not already used
                                    pe_coords = (noc_x, noc_y)
                                    if pe_coords in already_used_pes:
                                        raise ValueError(f"PE at coordinates {pe_coords} is already in use by another network")
                                    
                                    # Mark this PE as used
                                    already_used_pes.add(pe_coords)
                                    
                                    pe = self.noc.get_pe(*pe_coords)
                                    
                                    # Handle both with and without sequence splitting
                                    if len(split_info) >= 7:  # With sequence splitting
                                        split_dim, row_start, row_end, col_start, col_end, seq_start, seq_end = split_info
                                        
                                        # Store sequence range for this PE
                                        self.pe_seq_ranges[(layer_id, pe_idx)] = (seq_start, seq_end)
                                    else:  # Without sequence splitting
                                        split_dim, row_start, row_end, col_start, col_end = split_info
                                        seq_start = 0
                                        seq_end = self.seq_len
                                    
                                    # Create weights tensor
                                    pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                                    pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                                    
                                    # Update maps
                                    self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                                    self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
                                
                                break  # Exit the start_x loop
                        
                        if valid_placement_found:
                            break  # Exit the start_y loop
                
                # Last resort: If no suitable placement found, try any available positions
                if not valid_placement_found and allow_wrapping:
                    # Create a list of available PEs
                    available_pes = []
                    for x in range(self.noc.cols):
                        for y in range(self.noc.rows):
                            if (x, y) not in already_used_pes:
                                available_pes.append((x, y))
                    
                    # Check if we have enough PEs
                    if len(available_pes) >= num_pes:
                        valid_placement_found = True
                        
                        # Map each PE to an available position
                        for pe_idx, split_info in enumerate(split_ranges):
                            if pe_idx < len(available_pes):
                                pe_coords = available_pes[pe_idx]
                                already_used_pes.add(pe_coords)
                                
                                pe = self.noc.get_pe(*pe_coords)
                                
                                # Handle both with and without sequence splitting
                                if len(split_info) >= 7:  # With sequence splitting
                                    split_dim, row_start, row_end, col_start, col_end, seq_start, seq_end = split_info
                                    
                                    # Store sequence range for this PE
                                    self.pe_seq_ranges[(layer_id, pe_idx)] = (seq_start, seq_end)
                                else:  # Without sequence splitting
                                    split_dim, row_start, row_end, col_start, col_end = split_info
                                    seq_start = 0
                                    seq_end = self.seq_len
                                
                                # Create weights tensor
                                pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                                pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                                
                                # Update maps
                                self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                                self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
                
                if not valid_placement_found:
                    raise ValueError(f"Could not find space in NoC for layer {layer_id} requiring " 
                                   f"{grid_height}x{grid_width} PEs. NoC dimensions: {self.noc.rows}x{self.noc.cols}. "
                                   f"Consider increasing NoC dimensions or setting allow_wrapping=True.")
    
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
        # Include PEs that are already used by other neural networks in the LLM
        all_used_pes = set(all_computation_pes)
        
        # Check if this neural network has a reference to an LLM
        # This happens when the neural network is part of a larger LLM structure
        if hasattr(neural_network, 'llm') and neural_network.llm is not None:
            # Include PEs used by the LLM
            all_used_pes.update(neural_network.llm.used_pes)
            print("all_used_pes-------", all_used_pes)
        
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
                    # Only proceed with column aggregation if enabled
                    if hasattr(neural_network, 'column_aggregation_enabled') and neural_network.column_aggregation_enabled:
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
                                        print(f"Column aggregation PE: {col_agg_pe}")
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