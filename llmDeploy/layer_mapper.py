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
                 allow_wrapping: bool = False):
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
            neural_network: Reference to the parent neural network
            allow_wrapping: Whether to allow wrapping around the edges of the NoC (default: False)
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
        
        # Import data_type_bytes from external module to avoid circular imports
        self.data_type_bytes = dtype_size(self.data_type)
        
        # Maps (layer_id, pe_index) to PE coordinates
        self.layer_pe_map = {}
        # Maps PE coordinates to (layer_id, pe_index)
        self.pe_layer_map = {}
        
        # Validate and apply mapping strategy
        self._map_layers_to_pes(allow_wrapping=self.allow_wrapping)
    
    def _map_layers_to_pes(self, allow_wrapping=False):
        """
        Map layers to PEs based on the selected mapping strategy.
        
        Args:
            allow_wrapping: Whether to allow wrapping around the edges of the NoC (default: False)
        """
        # Calculate PE requirements for each layer
        layer_pe_requirements = self._calculate_pe_requirements()
        
        # Clear existing mappings
        self.layer_pe_map = {}
        self.pe_layer_map = {}
        
        # Map layers based on the selected strategy
        if self.mapping_strategy == "column_wise":
            self._map_to_columns(layer_pe_requirements)
        elif self.mapping_strategy == "row_wise":
            self._map_to_rows(layer_pe_requirements)
        elif self.mapping_strategy == "grid_wise":
            self._map_to_grid(layer_pe_requirements, allow_wrapping)
        elif self.mapping_strategy == "compact":
            self._map_to_compact(layer_pe_requirements, allow_wrapping)
        elif self.mapping_strategy == "proximity":
            # For proximity mapping, we'll use the center of the NoC as the default target
            self._map_to_proximity(layer_pe_requirements, allow_wrapping=allow_wrapping)
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
                for pe_idx, (split_type, row_start, row_end, col_start, col_end) in enumerate(split_ranges):
                    col_group = (col_start, col_end)
                    if col_group not in col_groups:
                        col_groups[col_group] = []
                    col_groups[col_group].append((pe_idx, (split_type, row_start, row_end, col_start, col_end)))
                
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
                
                # Unpack split information
                split_dim, row_start, row_end, col_start, col_end = split_info
                
                # Create weights tensor
                pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                
                # Update maps
                self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
    
    # def _map_to_rows(self, layer_pe_requirements):
    #     """
    #     Map layers to rows in the NoC.
    #     Each layer is mapped to a row, and multiple PEs in that row handle splits.
    #     For hybrid split, we arrange PEs in a 2D grid for each layer.
        
    #     Args:
    #         layer_pe_requirements: Dictionary mapping layer_id to a list of split information.
    #     """
    #     # Get already used PEs from the parent LLM if available
    #     already_used_pes = set()
    #     if hasattr(self, 'neural_network') and hasattr(self.neural_network, 'llm') and self.neural_network.llm is not None:
    #         already_used_pes = set(self.neural_network.llm.used_pes)
            
    #     for layer_id, split_ranges in layer_pe_requirements.items():
    #         # Start with the preferred row (layer_id)
    #         row = layer_id
    #         valid_row_found = False
            
    #         # Try rows starting from layer_id until we find one with enough available columns
    #         for row_offset in range(self.noc.rows):
    #             # Calculate actual row to try (wrap around if needed)
    #             row = (layer_id + row_offset) % self.noc.rows
                
    #             # Check if we have enough columns in this row, considering already used PEs
    #             available_cols = [col for col in range(self.noc.cols) if (col, row) not in already_used_pes]
    #             if len(split_ranges) <= len(available_cols):
    #                 valid_row_found = True
    #                 break
            
    #         if not valid_row_found:
    #             raise ValueError(f"Layer {layer_id} requires {len(split_ranges)} PEs, "
    #                              f"but couldn't find any row with enough available columns. "
    #                              f"Consider increasing NoC dimensions.")
            
    #         # For hybrid split, organize PEs in a grid
    #         if self.split_strategy == "hybrid_split":
    #             # Group PEs by their column split (each group handles a subset of output neurons)
    #             col_groups = {}
    #             for pe_idx, (split_type, row_start, row_end, col_start, col_end) in enumerate(split_ranges):
    #                 col_group = (col_start, col_end)
    #                 if col_group not in col_groups:
    #                     col_groups[col_group] = []
    #                 col_groups[col_group].append((pe_idx, (split_type, row_start, row_end, col_start, col_end)))
                
    #             # Place PEs in a more structured way
    #             pe_idx_map = {}  # Maps PE index to actual coordinates
    #             current_col_idx = 0
                
    #             # For each column group, place all its PEs one after another
    #             for col_group, pe_list in col_groups.items():
    #                 for pe_idx, _ in pe_list:
    #                     # Get the next available column from our filtered list
    #                     if current_col_idx >= len(available_cols):
    #                         raise ValueError(f"Not enough available PEs in row {row} for layer {layer_id}")
                        
    #                     col = available_cols[current_col_idx]
    #                     pe_coords = (col, row)
    #                     pe_idx_map[pe_idx] = pe_coords
    #                     already_used_pes.add(pe_coords)  # Mark this PE as used
    #                     current_col_idx += 1
            
    #         # Map each PE for this layer to the corresponding column in this row
    #         for pe_idx, split_info in enumerate(split_ranges):
    #             if self.split_strategy == "hybrid_split":
    #                 # For hybrid, use the pre-calculated coordinates
    #                 pe_coords = pe_idx_map[pe_idx]
    #             else:
    #                 # For non-hybrid, get the next available column
    #                 if pe_idx >= len(available_cols):
    #                     raise ValueError(f"Not enough available columns in row {row} for layer {layer_id}")
                    
    #                 col = available_cols[pe_idx]
    #                 pe_coords = (col, row)
    #                 already_used_pes.add(pe_coords)  # Mark this PE as used
                
    #             # Get the PE at these coordinates
    #             pe = self.noc.get_pe(*pe_coords)
                
    #             # Weight matrix dimensions
    #             if layer_id == 0:
    #                 input_dim = self.input_dim
    #             else:
    #                 input_dim = self.output_dims[layer_id - 1]
    #             output_dim = self.output_dims[layer_id]
                
    #             # Unpack split information
    #             split_dim, row_start, row_end, col_start, col_end = split_info
                
    #             # Create weights tensor
    #             pe_weights = torch.empty((row_end - row_start, col_end - col_start))
    #             pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                
    #             # Update maps
    #             self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
    #             self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
    
    def _map_to_rows(self, layer_pe_requirements):
        """
        Map layers to rows in the NoC.
        Each layer starts mapping to its corresponding row, and continues to subsequent rows if needed.
        PEs are assigned sequentially as needed without requiring the entire row to be available.
        
        Args:
            layer_pe_requirements: Dictionary mapping layer_id to a list of split information.
        """
        # Get already used PEs from the parent LLM if available
        already_used_pes = set()
        if hasattr(self, 'neural_network') and hasattr(self.neural_network, 'llm') and self.neural_network.llm is not None:
            already_used_pes = set(self.neural_network.llm.used_pes)
            
        for layer_id, split_ranges in layer_pe_requirements.items():
            # Start with the preferred row (layer_id)
            current_row = layer_id % self.noc.rows
            current_col = 0
            assigned_pes = 0
            pe_idx_map = {}  # Maps PE index to actual coordinates
            
            # Iterate through all split ranges that need to be assigned
            for pe_idx, split_info in enumerate(split_ranges):
                # Find the next available PE, potentially crossing rows
                while assigned_pes < len(split_ranges):
                    # Check if current position is available
                    if (current_col, current_row) not in already_used_pes:
                        # Found an available PE, mark it and use it
                        pe_coords = (current_col, current_row)
                        pe_idx_map[pe_idx] = pe_coords
                        already_used_pes.add(pe_coords)
                        assigned_pes += 1
                        break
                    
                    # Move to next position
                    current_col += 1
                    if current_col >= self.noc.cols:
                        current_col = 0
                        current_row = (current_row + 1) % self.noc.rows
                        
                if assigned_pes == len(split_ranges):
                    break
                    
            if assigned_pes < len(split_ranges):
                raise ValueError(f"Not enough available PEs in the NoC to accommodate layer {layer_id}. "
                            f"Required: {len(split_ranges)}, Found: {assigned_pes}")
                
            # Process the PE assignments based on the mapping strategy
            for pe_idx, split_info in enumerate(split_ranges):
                pe_coords = pe_idx_map[pe_idx]
                
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

    def _map_to_compact(self, layer_pe_requirements, allow_wrapping=False):
        """
        Map layers to compact regions in the NoC.
        Each layer is mapped to the most compact possible configuration,
        prioritizing minimizing the total area used.
        
        Args:
            layer_pe_requirements: Dictionary mapping layer_id to a list of split information.
            allow_wrapping: Whether to allow wrapping around the edges of the NoC (default: False)
        """
        # Get already used PEs from the parent LLM if available
        already_used_pes = set()
        if hasattr(self, 'neural_network') and hasattr(self.neural_network, 'llm') and self.neural_network.llm is not None:
            already_used_pes = set(self.neural_network.llm.used_pes)
        
        # Calculate center of already used PEs if any exist
        target_x, target_y = 0, 0  # Default to (0,0)
        if already_used_pes:
            # Calculate center of used PEs
            x_coords = [x for x, y in already_used_pes]
            y_coords = [y for x, y in already_used_pes]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            # Round to nearest integer since we need discrete coordinates
            target_x = int(round(center_x))
            target_y = int(round(center_y))
        
        # Create an availability map for the entire NoC
        availability_map = {}
        for x in range(self.noc.cols):
            for y in range(self.noc.rows):
                availability_map[(x, y)] = (x, y) not in already_used_pes
        
        # Process layers in sequence
        for layer_id, split_ranges in layer_pe_requirements.items():
            num_pes = len(split_ranges)
            
            # Find the most compact rectangular shape that can fit all PEs
            best_shape = self._find_compact_shape(num_pes, self.noc.rows, self.noc.cols)
            grid_height, grid_width = best_shape
            
            # Find the best position to place this shape in the NoC, starting from the target
            best_position = self._find_best_position(
                grid_height, 
                grid_width, 
                availability_map, 
                allow_wrapping,
                target_x,
                target_y
            )
            
            if best_position is None:
                raise ValueError(f"Could not find space in NoC for layer {layer_id} requiring " 
                              f"{grid_height}x{grid_width} PEs. NoC dimensions: {self.noc.rows}x{self.noc.cols}. "
                              f"Consider increasing NoC dimensions or setting allow_wrapping=True.")
            
            start_y, start_x = best_position
            
            # Map each PE for this layer
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
                availability_map[pe_coords] = False
                
                pe = self.noc.get_pe(*pe_coords)
                
                # Unpack split information
                split_dim, row_start, row_end, col_start, col_end = split_info
                
                # Create weights tensor
                pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                
                # Update maps
                self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
    
    def _find_compact_shape(self, num_pes, max_rows, max_cols):
        """
        Find the most compact rectangular shape that can fit all PEs.
        
        Args:
            num_pes: Number of PEs to fit
            max_rows: Maximum number of rows available
            max_cols: Maximum number of columns available
            
        Returns:
            Tuple of (height, width) for the most compact shape
        """
        # Start with a square-ish shape
        best_perimeter = float('inf')
        best_shape = (1, num_pes)  # Default to a 1-row configuration
        
        # Try different heights and calculate corresponding widths
        for height in range(1, min(num_pes, max_rows) + 1):
            width = math.ceil(num_pes / height)
            
            # Skip if width exceeds the maximum columns
            if width > max_cols:
                continue
                
            # Calculate perimeter (lower is more compact)
            perimeter = 2 * (height + width)
            
            # Update best shape if this one has a smaller perimeter
            if perimeter < best_perimeter:
                best_perimeter = perimeter
                best_shape = (height, width)
        
        return best_shape
    
    def _find_best_position(self, height, width, availability_map, allow_wrapping, target_x=0, target_y=0):
        """
        Find the best position to place a rectangular shape in the NoC.
        
        Args:
            height: Height of the shape
            width: Width of the shape
            availability_map: Dictionary mapping PE coordinates to availability
            allow_wrapping: Whether to allow wrapping around the edges
            target_x: Target x-coordinate to aim for (default: 0)
            target_y: Target y-coordinate to aim for (default: 0)
            
        Returns:
            Tuple of (start_y, start_x) for the best position, or None if no position found
        """
        best_score = float('inf')
        best_position = None
        
        # Adjust the search range based on wrapping
        max_row_range = self.noc.rows if allow_wrapping else self.noc.rows - height + 1
        max_col_range = self.noc.cols if allow_wrapping else self.noc.cols - width + 1
        
        # Create a list of all possible starting positions sorted by distance from target
        starting_positions = []
        for start_y in range(max_row_range):
            for start_x in range(max_col_range):
                # Calculate distance to target
                if allow_wrapping:
                    # When wrapping is allowed, consider the shortest path around edges
                    dx = min(abs(start_x - target_x), self.noc.cols - abs(start_x - target_x))
                    dy = min(abs(start_y - target_y), self.noc.rows - abs(start_y - target_y))
                else:
                    dx = abs(start_x - target_x)
                    dy = abs(start_y - target_y)
                distance = dx + dy
                starting_positions.append((distance, start_y, start_x))
        
        # Sort by distance from target (closest first)
        starting_positions.sort()
        
        # Try positions in order of increasing distance from target
        for _, start_y, start_x in starting_positions:
                # Check if all required positions are available
                all_available = True
                positions = []
                
                for y_offset in range(height):
                    for x_offset in range(width):
                        # Only check positions that would actually be used
                        pe_idx = y_offset * width + x_offset
                        if pe_idx >= height * width:
                            continue
                        
                        # Calculate position with wraparound if needed
                        noc_x = (start_x + x_offset) % self.noc.cols if allow_wrapping else start_x + x_offset
                        noc_y = (start_y + y_offset) % self.noc.rows if allow_wrapping else start_y + y_offset
                        
                        # Skip this position if it's not available
                        if not availability_map.get((noc_x, noc_y), False):
                            all_available = False
                            break
                        
                        positions.append((noc_x, noc_y))
                    
                    if not all_available:
                        break
                
                if all_available:
                    # Calculate the score for this position
                    # Score is now primarily based on distance from target
                    # Lower score is better
                    x_values = [x for x, y in positions]
                    y_values = [y for x, y in positions]
                    
                    # Calculate distance from target to center of the placement
                    center_x = sum(x_values) / len(x_values)
                    center_y = sum(y_values) / len(y_values)
                
                    if allow_wrapping:
                        dx = min(abs(center_x - target_x), self.noc.cols - abs(center_x - target_x))
                        dy = min(abs(center_y - target_y), self.noc.rows - abs(center_y - target_y))
                    else:
                        dx = abs(center_x - target_x)
                        dy = abs(center_y - target_y)
                
                    distance_from_target = dx + dy
                
                    # Calculate the area of the bounding box as a secondary factor
                    x_span = max(x_values) - min(x_values) + 1
                    y_span = max(y_values) - min(y_values) + 1
                    area = x_span * y_span
                    
                    # Final score prioritizes proximity to target, then compactness
                    score = distance_from_target * 1000 + area
                        
                    if score < best_score:
                        best_score = score
                        best_position = (start_y, start_x)
                    
                    # If we find a perfect score (directly at target), return immediately
                    if distance_from_target == 0:
                        return best_position
        
        return best_position
    
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

    def get_effective_noc_dimensions(self):
        """
        Calculate the effective NoC dimensions based on actual PE usage of the associated neural network.
        
        Returns:
            Tuple of (effective_rows, effective_cols, effective_grid_size)
        """
        # Get all active PEs from the associated neural network
        if not hasattr(self, 'neural_network') or self.neural_network is None:
            print("Warning: No neural network associated with this mapper")
            return 0, 0, 0
            
        active_pes = self.neural_network.active_pes
        
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

    def get_network_bounds(self, active_pes=None):
        """
        Calculate the network boundaries based on active PEs.
        
        Args:
            active_pes: Optional set of active PEs. If None, uses the neural network's active PEs.
        
        Returns:
            Dictionary containing:
                x_range: Tuple of (min_x, max_x)
                y_range: Tuple of (min_y, max_y)
                width: Width of the bounding box
                height: Height of the bounding box
                area: Area of the bounding box
                coordinates: Tuple of (x_min, x_max, y_min, y_max)
                
        If no active PEs are found, returns None.
        """
        # Use provided active_pes or get from neural network
        if active_pes is None:
            if not hasattr(self, 'neural_network') or self.neural_network is None:
                print("Warning: No neural network associated with this mapper")
                return None
                
            active_pes = self.neural_network.active_pes
        
        if not active_pes:
            return None
            
        # Extract x and y coordinates
        x_coords = [x for x, y in active_pes]
        y_coords = [y for x, y in active_pes]
        
        # Calculate min/max coordinates
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Calculate dimensions
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        area = width * height
        
        return {
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'width': width,
            'height': height,
            'area': area,
            'coordinates': (x_min, x_max, y_min, y_max)
        } 
    
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
                
                # Try both orientations: normal and rotated
                orientations = [
                    ("normal", num_row_groups, num_col_groups),
                    ("rotated", num_col_groups, num_row_groups)
                ]
                
                valid_placement_found = False
                
                # Try orientation options
                for orientation, grid_height, grid_width in orientations:
                    if valid_placement_found:
                        break
                        
                    # First priority: Try column-aligned placement
                    if (orientation == "normal" and grid_width <= self.noc.cols) or \
                    (orientation == "rotated" and grid_height <= self.noc.cols):
                        # Try different starting positions - search the entire NoC
                        # Allow wrapping checks all possible starting positions
                        max_row_range = self.noc.rows if allow_wrapping else self.noc.rows - grid_height + 1
                        max_col_range = self.noc.cols if allow_wrapping else self.noc.cols - grid_width + 1
                        
                        # Sort column groups for consistent placement
                        sorted_col_groups = sorted(col_groups.items())
                        sorted_row_groups = sorted(row_groups.items())
                        
                        # Try all possible starting positions
                        for start_y in range(max_row_range):
                            for start_x in range(max_col_range):
                                region_is_free = True
                                
                                # Check if this region is free
                                for y_offset in range(grid_height):
                                    for x_offset in range(grid_width):
                                        # Calculate position with potential wrapping
                                        noc_x = (start_x + x_offset) % self.noc.cols if allow_wrapping else start_x + x_offset
                                        noc_y = (start_y + y_offset) % self.noc.rows if allow_wrapping else start_y + y_offset
                                        
                                        # Skip if outside bounds (when not wrapping)
                                        if not allow_wrapping and (noc_x >= self.noc.cols or noc_y >= self.noc.rows):
                                            region_is_free = False
                                            break
                                        
                                        pe_coord = (noc_x, noc_y)
                                        if pe_coord in already_used_pes:
                                            region_is_free = False
                                            break
                                    
                                    if not region_is_free:
                                        break
                                
                                if region_is_free:
                                    # We found a valid placement
                                    valid_placement_found = True
                                    
                                    # Keep track of PE positions in the grid
                                    pe_positions = {}  # Maps pe_idx to (col_x, row_y) coordinates
                                    
                                    # Assign positions based on orientation
                                    if orientation == "normal":
                                        # Map each row group to a row and each column group to a column
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
                                    else:  # rotated
                                        # Swap row and column roles
                                        for row_idx, (row_group, row_pe_indices) in enumerate(sorted_row_groups):
                                            for col_idx, (col_group, col_pe_indices) in enumerate(sorted_col_groups):
                                                # Find PEs that belong to both this row and column group
                                                for intersect_pe_idx in set(row_pe_indices) & set(col_pe_indices):
                                                    col_x = start_x + row_idx  # Swap row/col for rotation
                                                    row_y = start_y + col_idx  # Swap row/col for rotation
                                                    if allow_wrapping:
                                                        col_x = col_x % self.noc.cols
                                                        row_y = row_y % self.noc.rows
                                                    
                                                    pe_positions[intersect_pe_idx] = (col_x, row_y)
                                    
                                    # Map each PE to its position in the NoC
                                    for pe_idx, (split_type, row_start, row_end, col_start, col_end) in enumerate(split_ranges):
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
                                        
                                        # Create weights tensor
                                        pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                                        pe.set_weights(pe_weights, layer_id, split_type, row_start, row_end, col_start, col_end)
                                        
                                        # Update maps
                                        self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                                        self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
                                    
                                    break  # Exit the start_x loop
                            
                            if valid_placement_found:
                                break  # Exit the start_y loop
                
                # Last resort: If no suitable placement found, try any available positions
                # Removed dependence on allow_wrapping to make this more aggressive
                if not valid_placement_found:
                    # Create a list of available PEs
                    available_pes = []
                    for x in range(self.noc.cols):
                        for y in range(self.noc.rows):
                            if (x, y) not in already_used_pes:
                                available_pes.append((x, y))
                    
                    # Check if we have enough PEs
                    total_pes_needed = len(split_ranges)
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
                        for pe_idx, (split_type, row_start, row_end, col_start, col_end) in enumerate(split_ranges):
                            if pe_idx not in pe_positions:
                                # This shouldn't happen, but skip if not in the mapping
                                continue
                                
                            pe_coords = pe_positions[pe_idx]
                            pe = self.noc.get_pe(*pe_coords)
                            
                            # Create weights tensor
                            pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                            pe.set_weights(pe_weights, layer_id, split_type, row_start, row_end, col_start, col_end)
                            
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
                
                # Try multiple aspect ratios for better fitting
                aspect_ratios = [
                    (int(math.sqrt(num_pes)), math.ceil(num_pes / int(math.sqrt(num_pes)))),  # Square-ish
                    (int(math.sqrt(num_pes * 0.5)), math.ceil(num_pes / int(math.sqrt(num_pes * 0.5)))),  # Wide rectangle
                    (int(math.sqrt(num_pes * 2)), math.ceil(num_pes / int(math.sqrt(num_pes * 2)))),  # Tall rectangle
                    (1, num_pes),  # Single row
                    (num_pes, 1),  # Single column
                ]
                
                # Add more aspect ratios if num_pes is large
                if num_pes > 16:
                    for divisor in range(2, min(int(math.sqrt(num_pes)) + 1, 10)):
                        if num_pes % divisor == 0:
                            aspect_ratios.append((divisor, num_pes // divisor))
                            aspect_ratios.append((num_pes // divisor, divisor))
                
                # Remove duplicates and sort by total area (h*w)
                aspect_ratios = sorted(set(aspect_ratios), key=lambda x: x[0] * x[1])
                
                valid_placement_found = False
                
                # Try each aspect ratio in turn
                for grid_height, grid_width in aspect_ratios:
                    if valid_placement_found:
                        break
                    
                    # Try rotated version too
                    for is_rotated in [False, True]:
                        if valid_placement_found:
                            break
                        
                        # Apply rotation if needed
                        h, w = (grid_width, grid_height) if is_rotated else (grid_height, grid_width)
                        
                        # Allow wrapping checks all possible starting positions
                        max_row_range = self.noc.rows if allow_wrapping else self.noc.rows - h + 1
                        max_col_range = self.noc.cols if allow_wrapping else self.noc.cols - w + 1
                        
                        # Skip if this configuration can't fit in the NoC
                        if max_row_range <= 0 or max_col_range <= 0:
                            continue
                        
                        # Try all possible starting positions
                        # Use different starting positions to increase chances of finding a spot
                        start_positions = []
                        
                        # Add corner positions first (more likely to result in efficient packing)
                        corners = [(0, 0), (0, max_row_range-1), (max_col_range-1, 0), (max_col_range-1, max_row_range-1)]
                        for x, y in corners:
                            if 0 <= x < max_col_range and 0 <= y < max_row_range:
                                start_positions.append((x, y))
                        
                        # Then add all other positions
                        for y in range(max_row_range):
                            for x in range(max_col_range):
                                if (x, y) not in start_positions:
                                    start_positions.append((x, y))
                        
                        for start_x, start_y in start_positions:
                            # Check if this region is free
                            region_is_free = True
                            for y_offset in range(h):
                                for x_offset in range(w):
                                    # Only check positions that would actually be used by this layer
                                    pe_idx = y_offset * w + x_offset
                                    if pe_idx >= num_pes:
                                        continue
                                    
                                    # Calculate position with potential wrapping
                                    noc_x = (start_x + x_offset) % self.noc.cols if allow_wrapping else start_x + x_offset
                                    noc_y = (start_y + y_offset) % self.noc.rows if allow_wrapping else start_y + y_offset
                                    
                                    # Skip if outside bounds (when not wrapping)
                                    if not allow_wrapping and (noc_x >= self.noc.cols or noc_y >= self.noc.rows):
                                        region_is_free = False
                                        break
                                    
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
                                    local_x = pe_idx % w
                                    local_y = pe_idx // w
                                    
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
                                    
                                    # Unpack split information
                                    split_dim, row_start, row_end, col_start, col_end = split_info
                                    
                                    # Create weights tensor
                                    pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                                    pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                                    
                                    # Update maps
                                    self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                                    self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
                                
                                break  # Exit the start position loop
                
                # Last resort: If no suitable placement found, try any available positions
                # Removed dependence on allow_wrapping to make this more aggressive
                if not valid_placement_found:
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
                                
                                # Unpack split information
                                split_dim, row_start, row_end, col_start, col_end = split_info
                                
                                # Create weights tensor
                                pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                                pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                                
                                # Update maps
                                self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                                self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
                
                if not valid_placement_found:
                    raise ValueError(f"Could not find space in NoC for layer {layer_id} requiring {num_pes} PEs. " 
                                f"NoC dimensions: {self.noc.rows}x{self.noc.cols}. "
                                f"Consider increasing NoC dimensions or setting allow_wrapping=True.")

    def _map_to_proximity(self, layer_pe_requirements, target_x=None, target_y=None, allow_wrapping=False):
        """
        Map layers to PEs based on proximity to a target location.
        Places PEs as close as possible to the target without regard for organization.
        
        Args:
            layer_pe_requirements: Dictionary mapping layer_id to a list of split information.
            target_x: Target x-coordinate to aim for (default: center of used PEs or (0,0))
            target_y: Target y-coordinate to aim for (default: center of used PEs or (0,0))
            allow_wrapping: Whether to allow wrapping around the edges of the NoC (default: False)
        """
        # Get already used PEs from the parent LLM if available
        already_used_pes = set()
        if hasattr(self, 'neural_network') and hasattr(self.neural_network, 'llm') and self.neural_network.llm is not None:
            already_used_pes = set(self.neural_network.llm.used_pes)
        
        # Set default target to center of already used PEs or (0,0) if no PEs are used
        if target_x is None or target_y is None:
            if already_used_pes:
                # Calculate center of used PEs
                x_coords = [x for x, y in already_used_pes]
                y_coords = [y for x, y in already_used_pes]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                # Round to nearest integer since we need discrete coordinates
                target_x = int(round(center_x)) if target_x is None else target_x
                target_y = int(round(center_y)) if target_y is None else target_y
            else:
                # Default to (0,0) if no PEs are used
                target_x = 0 if target_x is None else target_x
                target_y = 0 if target_y is None else target_y
        
        # Create a list of all available PEs with their distances to target
        available_pes = []
        for x in range(self.noc.cols):
            for y in range(self.noc.rows):
                if (x, y) not in already_used_pes:
                    # Calculate Manhattan distance to target
                    if allow_wrapping:
                        # When wrapping is allowed, consider the shortest path around edges
                        dx = min(abs(x - target_x), self.noc.cols - abs(x - target_x))
                        dy = min(abs(y - target_y), self.noc.rows - abs(y - target_y))
                    else:
                        dx = abs(x - target_x)
                        dy = abs(y - target_y)
                    distance = dx + dy
                    available_pes.append((distance, (x, y)))
        
        # Sort available PEs by distance to target
        available_pes.sort(key=lambda x: x[0])
        
        # Process layers in sequence
        for layer_id, split_ranges in layer_pe_requirements.items():
            num_pes = len(split_ranges)
            
            # Check if we have enough available PEs
            if len(available_pes) < num_pes:
                raise ValueError(f"Not enough available PEs for layer {layer_id}. "
                              f"Required: {num_pes}, Available: {len(available_pes)}")
            
            # Map each PE to the closest available position
            for pe_idx, split_info in enumerate(split_ranges):
                # Get the closest available PE
                _, pe_coords = available_pes[pe_idx]
                
                # Mark this PE as used
                already_used_pes.add(pe_coords)
                
                pe = self.noc.get_pe(*pe_coords)
                
                # Unpack split information
                split_dim, row_start, row_end, col_start, col_end = split_info
                
                # Create weights tensor
                pe_weights = torch.empty((row_end - row_start, col_end - col_start))
                pe.set_weights(pe_weights, layer_id, split_dim, row_start, row_end, col_start, col_end)
                
                # Update maps
                self.layer_pe_map[(layer_id, pe_idx)] = pe_coords
                self.pe_layer_map[pe_coords] = (layer_id, pe_idx)
            
            # Remove used PEs from available list
            available_pes = available_pes[num_pes:]