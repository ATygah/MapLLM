"""
LLM implementation that manages NoC resources and neural network components.
"""
#TODO: Add support for multiplying
#TODO: Add support for sending output from multiple head attention mechanism to the MLP.
#TODO: Add support for sending the output from the FC network to many heads based on their allocated portions of the output dimension.

from typing import List, Tuple, Dict, Optional, Set, Union, Any
import torch
import numpy as np
import math
import pandas as pd

from .pe_noc import NoCTopology, PE
from .neural_network import FCNeuralNetwork, ArithmeticNetwork
from data_structs import dtype_size

class LLM:
    def __init__(self, 
                 seq_len: int = 1,
                 pe_memory_size: int = 64 * 1024,
                 mapping_strategy: str = "column_wise",
                 split_strategy: str = "column_split",
                 reuse_pe_for_aggregation: bool = True,
                 row_aggregation_enabled: bool = True,
                 column_aggregation_enabled: bool = False,
                 data_type: str = "float16",
                 channel_bandwidth: float = 32.0,
                 noc_rows: int = None,
                 noc_cols: int = None,
                 allow_wrapping: bool = False):
        """
        Initialize an LLM with a sequence of layers on a shared NoC.
        
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden_dim1, hidden_dim2, ..., output_dim]
            seq_len: Sequence length
            pe_memory_size: Memory size of each PE in bytes
            mapping_strategy: How to map layers to PEs (column_wise, row_wise, grid)
            split_strategy: How to split computation (column_split, row_split, hybrid_split)
            reuse_pe_for_aggregation: Whether to reuse PEs for aggregation
            row_aggregation_enabled: Whether to aggregate partial results (True) or pass unaggregated results to downstream networks (False)
            column_aggregation_enabled: Whether to perform column aggregation in hybrid_split mode (True)
                                       or pass column-wise partial results directly (False)
            data_type: Data type for computations (float16, float32)
            channel_bandwidth: Bandwidth of NoC channels in Bytes/cycle
            noc_rows: Number of rows in the NoC (if None, calculated automatically)
            noc_cols: Number of columns in the NoC (if None, calculated automatically)
            allow_wrapping: Whether to allow PEs to wrap around the grid edges
        """
            
        self.seq_len = seq_len
        self.pe_memory_size = pe_memory_size
        self.mapping_strategy = mapping_strategy
        self.split_strategy = split_strategy
        self.reuse_pe_for_aggregation = reuse_pe_for_aggregation
        self.row_aggregation_enabled = row_aggregation_enabled
        self.column_aggregation_enabled = column_aggregation_enabled
        self.data_type = data_type
        self.channel_bandwidth = channel_bandwidth
        self.allow_wrapping = allow_wrapping
        
        # Data type size in bytes using the proper function
        self.data_type_bytes = dtype_size(data_type)
        
        # Calculate or use provided NoC dimensions
        if noc_rows is None or noc_cols is None:
            noc_rows, noc_cols = self._calculate_noc_dimensions(
                pe_memory_size=pe_memory_size,
                mapping_strategy=mapping_strategy,
                split_strategy=split_strategy,
                reuse_pe_for_aggregation=reuse_pe_for_aggregation,
                allow_wrapping=allow_wrapping
            )
        
        self.noc_rows = noc_rows
        self.noc_cols = noc_cols
        
        # Create a single shared NoC topology
        self.noc = NoCTopology(
            rows=noc_rows,
            cols=noc_cols,
            pe_memory_size=pe_memory_size,
            channel_bandwidth=channel_bandwidth,
            data_type_bytes=self.data_type_bytes
        )
        
        # Track used PEs across all networks
        self.used_pes: Set[Tuple[int, int]] = set()
        
        # Register the external PE at (0,0) as reserved to prevent conflicts
        self.external_pe = (0, 0)
        self.used_pes.add(self.external_pe)
        
        # Dictionary to store networks by name for easy reference
        self.networks = {}
        
        # List of execution steps, each step is a set of networks that run in parallel
        self.execution_steps = []
        
        # Network connections for data flow 
        self.connections = {}
    
    def _calculate_noc_dimensions(self, pe_memory_size, mapping_strategy, split_strategy, reuse_pe_for_aggregation, allow_wrapping):
        """Calculate dimensions of the NoC based on strategies."""
        # Calculate PE requirements for each layer
        layer_pe_counts = {}
        row_pe_counts = {}  # For hybrid strategy: PEs needed along input dimension
        col_pe_counts = {}  # For hybrid strategy: PEs needed along output dimension
        current_dim = self.layer_dims[0]  # Input dimension
        
        for layer_id in range(len(self.layer_dims) - 1):
            input_dim = self.layer_dims[layer_id]
            output_dim = self.layer_dims[layer_id + 1]
            
            # Calculate weight matrix dimensions
            weight_matrix_rows = input_dim
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
            # Use a larger buffer factor if wrapping is not allowed
            buffer_factor = 3.0 if not allow_wrapping else 1.5
            if not reuse_pe_for_aggregation:
                buffer_factor *= 1.5  # Additional buffer for aggregation PEs
            
            if split_strategy == "hybrid_split":
                # Calculate total area needed by all layers
                total_area = 0
                layer_areas = {}
                
                for layer_id in range(len(self.layer_dims) - 1):
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
                
                # Add buffer space - use larger buffer if wrapping is not allowed
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
                for layer_id in range(len(self.layer_dims) - 1):
                    if layer_id in row_pe_counts and layer_id in col_pe_counts:
                        layer_rows = row_pe_counts[layer_id] * col_pe_counts[layer_id]
                        if not reuse_pe_for_aggregation:
                            layer_rows += col_pe_counts[layer_id] + 1  # Add aggregation PEs
                        max_layer_rows = max(max_layer_rows, layer_rows)
                
                noc_cols = len(self.layer_dims) - 1  # Number of layers
                noc_rows = max_layer_rows
            else:
                # Standard column mapping
                noc_cols = len(self.layer_dims) - 1  # Number of layers
                noc_rows = max(layer_pe_counts.values()) if layer_pe_counts else 8
                
                # Add buffer for non-reusable aggregation
                if not reuse_pe_for_aggregation and split_strategy == "row_split":
                    noc_rows += len(self.layer_dims) - 1  # Add one row per layer for aggregation
            
            return noc_rows, noc_cols
        
        # For row_wise mapping
        elif mapping_strategy == "row_wise":
            if split_strategy == "hybrid_split":
                # For hybrid with row mapping, we need a 2D grid for each layer
                # Each layer gets a row, but we need multiple columns per layer
                max_layer_cols = 0
                for layer_id in range(len(self.layer_dims) - 1):
                    if layer_id in row_pe_counts and layer_id in col_pe_counts:
                        layer_cols = row_pe_counts[layer_id] * col_pe_counts[layer_id]
                        if not reuse_pe_for_aggregation:
                            layer_cols += col_pe_counts[layer_id] + 1  # Add aggregation PEs
                        max_layer_cols = max(max_layer_cols, layer_cols)
                
                noc_rows = len(self.layer_dims) - 1  # Number of layers
                noc_cols = max_layer_cols
            else:
                # Standard row mapping
                noc_rows = len(self.layer_dims) - 1  # Number of layers
                noc_cols = max(layer_pe_counts.values()) if layer_pe_counts else 8
                
                # Add buffer for non-reusable aggregation
                if not reuse_pe_for_aggregation and split_strategy == "row_split":
                    noc_cols += len(self.layer_dims) - 1  # Add one column per layer for aggregation
            
            return noc_rows, noc_cols
        
        # If we somehow get here (shouldn't happen), return a safe default
        return max(8, len(self.layer_dims) - 1), max(8, sum(layer_pe_counts.values()) // max(1, len(layer_pe_counts)))
    
    def _update_used_pes(self, network):
        """Update the set of used PEs with PEs from this network."""
        # Get all active PEs from the network
        if hasattr(network, 'active_pes'):
            for pe_coord in network.active_pes:
                self.used_pes.add(pe_coord)
    
    def _get_all_pes(self) -> List[Tuple[int, int]]:
        """Get all PEs used by all networks."""
        return list(self.used_pes)
    
    def get_pe_utilization(self) -> float:
        """
        Calculate overall PE utilization.
        
        Returns:
            Percentage of PEs in use
        """
        if not self.networks:
            return 0.0
        
        total_pes = self.noc.rows * self.noc.cols
        used_pes = len(self.used_pes)
        return used_pes / total_pes * 100 if total_pes > 0 else 0
    
    def get_network_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the network structure.
        
        Returns:
            Dict with network details
        """
        layer_details = []
        network_types = []
        
        for network in self.networks:
            if isinstance(network, FCNeuralNetwork):
                layer_details.append((network.input_dim, network.layer_dims[-1]))
                network_types.append("FC")
            elif isinstance(network, ArithmeticNetwork):
                layer_details.append((network.seq_len, network.seq_len))
                if hasattr(network, 'operation'):
                    network_types.append(network.operation)
                else:
                    network_types.append("arithmetic")
            
        return {
            "total_layers": len(self.networks),
            "layer_dims": layer_details,
            "network_types": network_types,
            "pe_utilization": self.get_pe_utilization(),
            "total_used_pes": len(self.used_pes),
            "noc_dimensions": (self.noc.rows, self.noc.cols),
            "networks": self.networks
        }
    
    def print_network_structure(self):
        """Print the structure of the network."""
        print(f"LLM with {len(self.networks)} layers")
        print(f"Sequence length: {self.seq_len}")
        print(f"Data type: {self.data_type} ({self.data_type_bytes} bytes)")
        print(f"Mapping strategy: {self.mapping_strategy}")
        print(f"Split strategy: {self.split_strategy}")
        print(f"NoC dimensions: {self.noc.rows}x{self.noc.cols}")
        print(f"PE utilization: {self.get_pe_utilization():.2f}%")
        print(f"Used PEs: {len(self.used_pes)}")
        print("\nLayer Structure:")
        
        for i, network in enumerate(self.networks):
            if isinstance(network, FCNeuralNetwork):
                print(f"  Layer {i+1}: FC({network.input_dim} -> {network.layer_dims[-1]})")
            elif isinstance(network, ArithmeticNetwork):
                operation_type = network.operation if hasattr(network, 'operation') else 'arithmetic'
                print(f"  Layer {i+1}: {operation_type}({network.d_model} -> {network.d_model})")
    
    def get_pe_mapping_details(self) -> pd.DataFrame:
        """
        Get a DataFrame containing the mapping details of all PEs across all networks.
        
        Returns:
            pd.DataFrame: A DataFrame with PE mapping information
        """
        # Initialize an empty list to collect PE details
        all_pe_details = []
        
        # Iterate through all networks to collect PE details
        for network_idx, (name, network_obj) in enumerate(self.networks.items()):
            if isinstance(network_obj, FCNeuralNetwork):
                if hasattr(network_obj, 'mapper') and hasattr(network_obj.mapper, 'get_pe_details'):
                    pe_details = network_obj.mapper.get_pe_details()
                    pe_details['network_idx'] = network_idx
                    pe_details['network_name'] = name
                    pe_details['network_type'] = 'FC'
                    all_pe_details.append(pe_details)
            elif isinstance(network_obj, ArithmeticNetwork):
                if hasattr(network_obj, 'mapper') and hasattr(network_obj.mapper, 'get_pe_details'):
                    pe_details = network_obj.mapper.get_pe_details()
                    pe_details['network_idx'] = network_idx
                    pe_details['network_name'] = name
                    pe_details['network_type'] = network_obj.operation if hasattr(network_obj, 'operation') else 'arithmetic'
                all_pe_details.append(pe_details)
        
        # If we have collected any details
        if all_pe_details:
            # Concatenate all DataFrames
            combined_details = pd.concat(all_pe_details, ignore_index=True)
            
            # Reorder columns to have network_idx and network_type first
            cols = ['network_idx', 'network_name', 'network_type'] + [col for col in combined_details.columns 
                   if col not in ['network_idx', 'network_name', 'network_type']]
            combined_details = combined_details[cols]
            
            return combined_details
        else:
            # Return an empty DataFrame with appropriate columns
            return pd.DataFrame(columns=['network_idx', 'network_name', 'network_type', 'pe_coords', 'layer_id', 
                                      'pe_idx', 'split_dim', 'weight_tensor', 'weight_shape'])

    def create_fc_network(self, 
                       name: str,
                       input_dim: int, 
                       output_dim: int,
                       seq_len: int = None,
                       mapping_strategy: str = None,
                       split_strategy: str = None,
                       data_type: str = None,
                       reuse_pe_for_aggregation: bool = None,
                       row_aggregation_enabled: bool = None,
                       column_aggregation_enabled: bool = None,
                       allow_wrapping: bool = None) -> FCNeuralNetwork:
        """
        Create a fully connected neural network and add it to the LLM.
        
        Args:
            name: Unique name for the network
            input_dim: Input dimension
            output_dim: Output dimension
            seq_len: Sequence length (defaults to LLM's seq_len if None)
            mapping_strategy: How to map layers to PEs (defaults to LLM's mapping_strategy if None)
            split_strategy: How to split computation (defaults to LLM's split_strategy if None)
            data_type: Data type for computations (defaults to LLM's data_type if None)
            reuse_pe_for_aggregation: Whether to reuse PEs for aggregation (defaults to LLM's value if None)
            row_aggregation_enabled: Whether to aggregate partial results (defaults to LLM's value if None)
            column_aggregation_enabled: Whether to perform column aggregation in hybrid_split mode (defaults to LLM's value if None)
            allow_wrapping: Whether to allow PEs to wrap around the grid edges (defaults to LLM's value if None)
            
        Returns:
            FCNeuralNetwork: The created neural network
        """
        # Check for duplicate name
        if name in self.networks:
            raise ValueError(f"Network with name '{name}' already exists")
        
        # Use provided values or default to LLM settings
        seq_len = seq_len if seq_len is not None else self.seq_len
        mapping_strategy = mapping_strategy if mapping_strategy is not None else self.mapping_strategy
        split_strategy = split_strategy if split_strategy is not None else self.split_strategy
        data_type = data_type if data_type is not None else self.data_type
        reuse_pe_for_aggregation = reuse_pe_for_aggregation if reuse_pe_for_aggregation is not None else self.reuse_pe_for_aggregation
        row_aggregation_enabled = row_aggregation_enabled if row_aggregation_enabled is not None else self.row_aggregation_enabled
        column_aggregation_enabled = column_aggregation_enabled if column_aggregation_enabled is not None else self.column_aggregation_enabled
        allow_wrapping = allow_wrapping if allow_wrapping is not None else self.allow_wrapping
        
        nn = FCNeuralNetwork(
            noc=self.noc,  # Pass the shared NoC
            input_dim=input_dim,
            layer_dims=[output_dim],  # Single layer with output_dim
            seq_len=seq_len,
            mapping_strategy=mapping_strategy,
            split_strategy=split_strategy,
            data_type=data_type,
            reuse_pe_for_aggregation=reuse_pe_for_aggregation,
            row_aggregation_enabled=row_aggregation_enabled,
            column_aggregation_enabled=column_aggregation_enabled,
            llm=self,  # Pass reference to this LLM instance
            allow_wrapping=allow_wrapping
        )
        
        # Store the network
        self.networks[name] = nn
        
        # Track used PEs
        self._update_used_pes(nn)
        
        return nn

    def create_arithmetic_network(self, 
                                name: str,
                                seq_len: int = None,
                                d_model: int = None,
                                operation: str = "matmul",
                                mapping_strategy: str = None,
                                split_strategy: str = None,
                                data_type: str = None,
                                reuse_pe_for_aggregation: bool = None,
                                row_aggregation_enabled: bool = None,
                                column_aggregation_enabled: bool = None,
                                allow_wrapping: bool = None) -> ArithmeticNetwork:
        """
        Create an arithmetic network for matrix multiplication or element-wise operations.
        
        Args:
            name: Unique name for the network
            seq_len: Sequence length (defaults to LLM's seq_len if None)
            d_model: Model dimension (e.g. embedding dimension)
            operation: Operation type ("matmul" or "element_wise")
            mapping_strategy: Mapping strategy (defaults to LLM's mapping_strategy if None)
            split_strategy: Split strategy (defaults to LLM's split_strategy if None)
            data_type: Data type for computations (defaults to LLM's data_type if None)
            reuse_pe_for_aggregation: Whether to reuse PEs for aggregation (defaults to LLM's value if None)
            row_aggregation_enabled: Whether to aggregate partial results (defaults to LLM's value if None)
            column_aggregation_enabled: Whether to perform column aggregation in hybrid_split mode (defaults to LLM's value if None)
            allow_wrapping: Whether to allow PEs to wrap around the grid edges (defaults to LLM's value if None)
            
        Returns:
            ArithmeticNetwork: The created arithmetic network
        """
        # Check for duplicate name
        if name in self.networks:
            raise ValueError(f"Network with name '{name}' already exists")
        
        # Use provided values or default to LLM settings
        seq_len = seq_len if seq_len is not None else self.seq_len
        mapping_strategy = mapping_strategy if mapping_strategy is not None else self.mapping_strategy
        split_strategy = split_strategy if split_strategy is not None else self.split_strategy
        data_type = data_type if data_type is not None else self.data_type
        reuse_pe_for_aggregation = reuse_pe_for_aggregation if reuse_pe_for_aggregation is not None else self.reuse_pe_for_aggregation
        row_aggregation_enabled = row_aggregation_enabled if row_aggregation_enabled is not None else self.row_aggregation_enabled
        column_aggregation_enabled = column_aggregation_enabled if column_aggregation_enabled is not None else self.column_aggregation_enabled
        allow_wrapping = allow_wrapping if allow_wrapping is not None else self.allow_wrapping
        
        network = ArithmeticNetwork(
            noc=self.noc,
            seq_len=seq_len,
            d_model=d_model,
            operation=operation,
            mapping_strategy=mapping_strategy,
            split_strategy=split_strategy,
            data_type=data_type,
            reuse_pe_for_aggregation=reuse_pe_for_aggregation,
            row_aggregation_enabled=row_aggregation_enabled,
            column_aggregation_enabled=column_aggregation_enabled,
            llm=self,
            allow_wrapping=allow_wrapping
        )
        
        # Store the network
        self.networks[name] = network
        
        # Track used PEs
        self._update_used_pes(network)
        
        return network
    
    def set_execution_order(self, execution_steps: List[List[str]]) -> None:
        """
        Set the execution order for networks, defining which networks run in parallel at each step.
        
        Args:
            execution_steps: List of lists, where each inner list contains the names of networks 
                            to be executed in parallel at that step
                            
        Example:
            set_execution_order([
                ["Q_proj", "K_proj", "V_proj"],  # Step 1: Run Q, K, V projections in parallel
                ["QK_attn"],                     # Step 2: Compute attention scores
                ["attn_out"],                    # Step 3: Apply attention to values
                ["output_proj"]                  # Step 4: Final projection
            ])
        """
        # Validate that all network names exist
        all_networks = []
        for step in execution_steps:
            for network_name in step:
                if network_name not in self.networks:
                    raise ValueError(f"Network '{network_name}' does not exist")
                all_networks.append(network_name)
        
        # Check if all networks are included in the execution order
        if set(all_networks) != set(self.networks.keys()):
            missing = set(self.networks.keys()) - set(all_networks)
            extra = set(all_networks) - set(self.networks.keys())
            message = ""
            if missing:
                message += f"Missing networks: {missing}. "
            if extra:
                message += f"Extra networks: {extra}. "
            if len(all_networks) != len(set(all_networks)):
                message += "Some networks appear multiple times in the execution order."
            
            raise ValueError(f"Execution order doesn't match defined networks. {message}")
        
        # Set the execution steps
        self.execution_steps = execution_steps

    def connect_networks(self, 
                       source_network: str, 
                       dest_network: str, 
                       connection_type: str = "default",
                       source_range: Tuple[int, int] = None,
                       dest_range: Tuple[int, int] = None) -> None:
        """
        Create a connection between networks to define data flow.
        
        Args:
            source_network: Name of the source network
            dest_network: Name of the destination network
            connection_type: Type of connection (default, matmul_a, matmul_b)
            source_range: Optional tuple (start, end) specifying column range of the source tensor to use
            dest_range: Optional tuple (start, end) specifying column range of the destination tensor where source data will go
        """
        # Validate networks exist
        if source_network not in self.networks:
            raise ValueError(f"Source network '{source_network}' does not exist")
        if dest_network not in self.networks:
            raise ValueError(f"Destination network '{dest_network}' does not exist")
        
        # Store the connection with range information
        self.connections[(source_network, dest_network)] = {
            "type": connection_type,
            "source_range": source_range,
            "dest_range": dest_range
        }

    def run_inference(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run inference through the entire network following the execution steps.
        
        Args:
            inputs: Dictionary mapping network names to input tensors
            
        Returns:
            Dictionary mapping network names to output tensors
        """
        if not self.networks:
            raise ValueError("No networks defined in LLM")
        
        if not self.execution_steps:
            raise ValueError("No execution steps defined, use set_execution_order() to define the execution order")
        
        # Store results from each network
        network_outputs = {}
        
        # Iterate through execution steps
        for step_idx, step_networks in enumerate(self.execution_steps):
            step_results = {}
            
            # Execute all networks in this step in parallel
            for network_name in step_networks:
                network = self.networks[network_name]
                
                # Check for direct inputs first
                network_input = None
                if network_name in inputs:
                    network_input = inputs[network_name]
                else:
                    # Handle outputs from previous networks with potential destination ranges
                    network_input = None
                    
                    # Check if this network is expecting inputs to specific ranges
                    # Create a dict to collect inputs by range
                    range_inputs = {}
                    
                    # Look for inputs from connected networks
                    for (src, dst), details in self.connections.items():
                        if dst == network_name and src in network_outputs:
                            connection_type = details.get("type", "default")
                            source_range = details.get("source_range", None)
                            dest_range = details.get("dest_range", None)
                            
                            # Get source tensor
                            src_tensor = network_outputs[src]
                            
                            # Apply source range if specified
                            if source_range is not None and isinstance(src_tensor, torch.Tensor):
                                start, end = source_range
                                src_tensor = src_tensor[:, start:end]
                            
                            # If destination range is specified, we need to combine inputs differently
                            if dest_range is not None and isinstance(src_tensor, torch.Tensor):
                                # Store the tensor with its target destination range
                                if network_input is None:
                                    # Initialize with zeros if this is the first input with a destination range
                                    # We need to figure out the full shape
                                    input_type = self.networks.get(dst)
                                    if isinstance(input_type, FCNeuralNetwork):
                                        input_dim = input_type.input_dim
                                        seq_len = src_tensor.shape[0]  # Use same sequence length as source
                                        network_input = torch.zeros(seq_len, input_dim, dtype=src_tensor.dtype, device=src_tensor.device)
                                
                                # Now copy the data to the appropriate destination range
                                start, end = dest_range
                                if network_input is not None:
                                    network_input[:, start:end] = src_tensor
                            elif isinstance(src_tensor, dict) and dest_range is not None:
                                # If we have a dictionary of PE outputs with tensor, range, task_id tuples
                                # and a destination range, we need to handle this differently
                                
                                # Initialize network input dictionary if needed
                                if network_input is None:
                                    network_input = {}
                                
                                start, end = dest_range
                                
                                # Process each PE output and adjust its range for the destination
                                for pe_coord, (tensor, pe_range, task_id) in src_tensor.items():
                                    # Create a new adjusted range that reflects the destination position
                                    if isinstance(pe_range, tuple) and len(pe_range) == 2:
                                        if isinstance(pe_range[0], tuple):
                                            # Handle nested format ((row_start, row_end), (col_start, col_end))
                                            row_range, col_range = pe_range
                                            if col_range != (-1, -1) and None not in col_range:
                                                # Adjust column range to destination position
                                                col_start, col_end = col_range
                                                # Ensure we don't exceed the destination range
                                                new_col_start = col_start + start
                                                new_col_end = min(col_end + start, end)
                                                # Create the adjusted range
                                                adjusted_range = (row_range, (new_col_start, new_col_end))
                                            else:
                                                # Special ranges remain unchanged
                                                adjusted_range = pe_range
                                        else:
                                            # Handle simple format (start, end)
                                            if None not in pe_range:
                                                # Adjust range to destination position
                                                range_start, range_end = pe_range
                                                # Ensure we don't exceed the destination range
                                                adjusted_start = range_start + start
                                                adjusted_end = min(range_end + start, end)
                                                adjusted_range = (adjusted_start, adjusted_end)
                                            else:
                                                # Special ranges remain unchanged
                                                adjusted_range = pe_range
                                    else:
                                        # Unknown range format, keep as is
                                        adjusted_range = pe_range
                                    
                                    # Use the original PE coordinate
                                    if pe_coord in network_input:
                                        # If PE is already in use, make a note but continue
                                        # This is a simplification - ideally we would merge tensors properly
                                        print(f"Warning: PE {pe_coord} already has output assigned for network {dst}")
                                    
                                    # Store the tensor with adjusted range
                                    network_input[pe_coord] = (tensor, adjusted_range, task_id)
                            else:
                                # If no destination range or the source is a dict but no dest_range,
                                # use the input directly - this handles the typical case
                                network_input = src_tensor
                                break
                
                # Ensure we have an input
                if network_input is None:
                    raise ValueError(f"No input provided for network '{network_name}' in step {step_idx}")
                
                # Call the appropriate method based on network type and operation
                if isinstance(network, FCNeuralNetwork):
                    # FCNeuralNetwork now accepts both tensor and dictionary inputs
                    if isinstance(network_input, dict):
                        print(f"Running FCNeuralNetwork '{network_name}' with dictionary input")
                    else:
                        print(f"Running FCNeuralNetwork '{network_name}' with input shape: {network_input.shape}")
                    result = network.run_inference(network_input)
                    step_results[network_name] = result
                
                elif isinstance(network, ArithmeticNetwork):
                    operation = network.operation if hasattr(network, 'operation') else "matmul"
                    
                    # 1. Matrix multiplication - needs two inputs (a and b)
                    if operation == "matmul":
                        input_a = None
                        input_b = None
                        
                        # Check for direct inputs
                        if f"{network_name}_a" in inputs:
                            input_a = inputs[f"{network_name}_a"]
                        if f"{network_name}_b" in inputs:
                            input_b = inputs[f"{network_name}_b"]
                            transpose_b = False
                        elif f"{network_name}_b_transpose" in inputs:
                            input_b = inputs[f"{network_name}_b_transpose"]
                            transpose_b = True
                            
                        # Check for inputs from connections
                        for (src, dst), details in self.connections.items():
                            if dst == network_name and src in network_outputs:
                                if details["type"] == "matmul_a":
                                    input_a = network_outputs[src]
                                elif details["type"] == "matmul_b":
                                    transpose_b = False
                                    input_b = network_outputs[src]
                                elif details["type"] == "matmul_b_transpose":
                                    transpose_b = True
                                    input_b = network_outputs[src]
                                elif details["type"] == "default" and input_a is None:
                                    # If no specific mapping, use as input_a by default
                                    input_a = network_outputs[src]
                        
                        # Ensure we have both inputs
                        if input_a is None or input_b is None:
                            raise ValueError(f"Missing inputs for matrix multiplication in network '{network_name}' in step {step_idx}")
                        
                        # Run matrix multiplication
                        result = network.matrix_multiply(input_a, input_b, transpose_b=transpose_b)
                        step_results[network_name] = result
                    
                    # 1.5. Attention computation - needs three inputs (Q, K, V)
                    elif operation == "attention":
                        Q = None
                        K = None
                        V = None
                        
                        # Check for direct inputs
                        if f"{network_name}_q" in inputs:
                            Q = inputs[f"{network_name}_q"]
                        if f"{network_name}_k" in inputs:
                            K = inputs[f"{network_name}_k"]
                        if f"{network_name}_v" in inputs:
                            V = inputs[f"{network_name}_v"]
                            
                        # Check for inputs from connections
                        for (src, dst), details in self.connections.items():
                            if dst == network_name and src in network_outputs:
                                connection_type = details["type"]
                                source_range = details.get("source_range", None)
                                
                                # Get the source tensor from the previous network
                                src_tensor = network_outputs[src]
                                
                                # Handle dictionary outputs differently from tensor outputs
                                if isinstance(src_tensor, dict) and source_range is not None:
                                    # Create a new dictionary that only includes PEs whose range 
                                    # overlaps with the source_range
                                    filtered_dict = {}
                                    start, end = source_range
                                    
                                    for pe_coord, (tensor, pe_range, task_id) in src_tensor.items():
                                        # Extract the column range based on the pe_range format
                                        # If the range is a nested tuple ((row_start, row_end), (col_start, col_end))
                                        if isinstance(pe_range, tuple) and len(pe_range) == 2 and isinstance(pe_range[0], tuple):
                                            _, col_range = pe_range
                                            # Check if this PE's output overlaps with the desired range
                                            if col_range != (-1, -1) and None not in col_range:
                                                col_start, col_end = col_range
                                                # Include if there's overlap with our target range
                                                if col_end > start and col_start < end:
                                                    # Calculate overlap between the PE's range and our target range
                                                    overlap_start = max(col_start, start)
                                                    overlap_end = min(col_end, end)
                                                    
                                                    # Calculate relative positions in the original tensor
                                                    # (tensor columns represent col_start to col_end)
                                                    tensor_start_idx = overlap_start - col_start
                                                    tensor_end_idx = overlap_end - col_start
                                                    
                                                    # Slice the correct portion from the tensor
                                                    sliced_tensor = tensor[:, tensor_start_idx:tensor_end_idx]
                                                    
                                                    # Convert to coordinates local to the target head
                                                    local_start = overlap_start - start
                                                    local_end = overlap_end - start
                                                    
                                                    # Create new range within the head's local coordinate system
                                                    new_range = (pe_range[0], (local_start, local_end))
                                                    filtered_dict[pe_coord] = (sliced_tensor, new_range, task_id)
                                        elif isinstance(pe_range, tuple) and len(pe_range) == 2:
                                            # For simple ranges (start_col, end_col)
                                            if None not in pe_range:
                                                range_start, range_end = pe_range
                                                # Include if there's overlap with our target range
                                                if range_end > start and range_start < end:
                                                    # Calculate overlap between the PE's range and our target range
                                                    overlap_start = max(range_start, start)
                                                    overlap_end = min(range_end, end)
                                                    
                                                    # Calculate relative positions in the original tensor
                                                    tensor_start_idx = overlap_start - range_start
                                                    tensor_end_idx = overlap_end - range_start
                                                    
                                                    # Slice the correct portion from the tensor
                                                    sliced_tensor = tensor[:, tensor_start_idx:tensor_end_idx]
                                                    
                                                    # Convert to coordinates local to the target head
                                                    local_start = overlap_start - start
                                                    local_end = overlap_end - start
                                                    
                                                    # Create new range within the head's local coordinate system
                                                    filtered_dict[pe_coord] = (sliced_tensor, (local_start, local_end), task_id)
                                    
                                    # If we found any relevant PEs, use the filtered dictionary
                                    if filtered_dict:
                                        src_tensor = filtered_dict
                                elif isinstance(src_tensor, torch.Tensor) and source_range is not None:
                                    # For tensor inputs, simply slice the tensor along the feature dimension
                                    start, end = source_range
                                    src_tensor = src_tensor[:, start:end]
                                
                                # Assign to the appropriate input based on connection type
                                if connection_type == "attention_q":
                                    Q = src_tensor
                                elif connection_type == "attention_k":
                                    K = src_tensor
                                elif connection_type == "attention_v":
                                    V = src_tensor
                                elif connection_type == "default" and Q is None:
                                    # If no specific mapping and Q is not set, use as Q by default
                                    Q = src_tensor
                                    
                        # Ensure we have all three inputs
                        if Q is None or K is None or V is None:
                            raise ValueError(f"Missing inputs for attention computation in network '{network_name}' in step {step_idx}")
                        
                        # Run attention computation (Q·K^T·V)
                        result = network.attention_computation(Q, K, V)
                        step_results[network_name] = result
                    
                    # 2. Element-wise operations - needs two inputs
                    elif operation in ["element_wise", "add", "subtract", "multiply", "divide"]:
                        input_a = None
                        input_b = None
                        element_op = operation if operation != "element_wise" else "add"  # Default to add if generic
                        
                        # Check for direct inputs
                        if f"{network_name}_a" in inputs:
                            input_a = inputs[f"{network_name}_a"]
                        if f"{network_name}_b" in inputs:
                            input_b = inputs[f"{network_name}_b"]
                            
                        # Check for inputs from connections
                        connected_inputs = []
                        for (src, dst), details in self.connections.items():
                            if dst == network_name and src in network_outputs:
                                connection_type = details.get("type", "default")
                                if connection_type == "element_a":
                                    input_a = network_outputs[src]
                                elif connection_type == "element_b":
                                    input_b = network_outputs[src]
                                else:
                                    connected_inputs.append((src, network_outputs[src]))
                        
                        # If we still don't have inputs, use connected inputs in order
                        if input_a is None and len(connected_inputs) > 0:
                            input_a = connected_inputs[0][1]
                        if input_b is None and len(connected_inputs) > 1:
                            input_b = connected_inputs[1][1]
                        
                        # Ensure we have both inputs
                        if input_a is None or input_b is None:
                            raise ValueError(f"Missing inputs for element-wise operation in network '{network_name}' in step {step_idx}")
                        
                        # Run element-wise operation
                        result = network.element_wise(input_a, input_b, operation=element_op)
                        step_results[network_name] = result
                    
                    # 3. Reduction operations - needs one input
                    elif operation in ["reduce", "sum", "mean", "max", "min"]:
                        reduce_op = operation if operation != "reduce" else "sum"  # Default to sum if generic
                        axis = None  # Default to global reduction
                        
                        # Determine input for reduce operation
                        if network_name in inputs:
                            network_input = inputs[network_name]
                        else:
                            # Look for inputs from previous networks based on connections
                            connected_inputs = []
                            
                            for (src, dst), details in self.connections.items():
                                if dst == network_name and src in network_outputs:
                                    # Check if axis is specified in connection details
                                    if "axis" in details:
                                        axis = details["axis"]
                                    connected_inputs.append((src, network_outputs[src]))
                            
                            # Use the connected input if available
                            if connected_inputs:
                                network_input = connected_inputs[0][1]
                            else:
                                raise ValueError(f"No input provided for reduce operation in network '{network_name}' in step {step_idx}")
                        
                        # Run reduce operation
                        result = network.reduce(network_input, operation=reduce_op, axis=axis)
                        step_results[network_name] = result
                    # 4. Default case (unknown operation)
                    else:
                        raise ValueError(f"Unknown arithmetic operation '{operation}' for network '{network_name}'")
                
                # Unknown network type
                else:
                    raise ValueError(f"Unknown network type for network '{network_name}'")
            
            # Update network outputs with results from this step
            network_outputs.update(step_results)
        
        # Extract and format final outputs
        final_outputs = {}
        for name, output in network_outputs.items():
            if isinstance(output, dict):
                # For outputs that are dictionaries of PE results, we need to properly 
                # combine the tensors based on their ranges
                
                # First, determine the full output dimension by examining the ranges
                max_dim = 0
                seq_len = None
                
                for pe_coord, (tensor, range_info, _) in output.items():
                    # Set sequence length from first tensor
                    if seq_len is None and tensor is not None:
                        seq_len = tensor.shape[0]
                    
                    # Extract column range information
                    if isinstance(range_info, tuple) and len(range_info) == 2:
                        if isinstance(range_info[0], tuple):
                            # Handle hybrid format: ((row_start, row_end), (col_start, col_end))
                            _, col_range = range_info
                            if col_range != (-1, -1) and None not in col_range:
                                max_dim = max(max_dim, col_range[1])
                        else:
                            # Handle simple format: (start, end)
                            if None not in range_info:
                                max_dim = max(max_dim, range_info[1])
                
                # If we found valid dimensions
                if max_dim > 0 and seq_len is not None:
                    # Determine output data type from first tensor
                    sample_tensor = next(iter(output.values()))[0]
                    dtype = sample_tensor.dtype
                    device = sample_tensor.device
                    
                    # Create a zero tensor of the appropriate size
                    combined_tensor = torch.zeros(seq_len, max_dim, dtype=dtype, device=device)
                    
                    # Fill in the tensor with data from each PE output
                    for pe_coord, (tensor, range_info, _) in output.items():
                        if tensor is None:
                            continue
                            
                        # Extract column range information
                        if isinstance(range_info, tuple) and len(range_info) == 2:
                            if isinstance(range_info[0], tuple):
                                # Handle hybrid format: ((row_start, row_end), (col_start, col_end))
                                _, col_range = range_info
                                if col_range != (-1, -1) and None not in col_range:
                                    col_start, col_end = col_range
                                    # Place this tensor's data in the correct position
                                    combined_tensor[:, col_start:col_end] = tensor
                            else:
                                # Handle simple format: (start, end)
                                if None not in range_info:
                                    start, end = range_info
                                    # Place this tensor's data in the correct position
                                    combined_tensor[:, start:end] = tensor
                    
                    final_outputs[name] = combined_tensor
                else:
                    # Fallback to old behavior if we can't determine proper ranges
                    output_tensors = []
                    for pe_coord, (tensor, _, _) in sorted(output.items()):
                        output_tensors.append(tensor)
                    
                    if output_tensors:
                        final_outputs[name] = torch.cat(output_tensors, dim=-1)
            else:
                # Direct tensor output
                final_outputs[name] = output
        
        return final_outputs
    
