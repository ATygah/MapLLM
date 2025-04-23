import torch
import pandas as pd
import math
from typing import List, Tuple, Dict, Optional, Set, Union
from .pe_noc import NoCTopology
from .layer_mapper import FCLayerMapper

class FCNeuralNetwork:
    """Fully Connected Neural Network on NoC."""
    def __init__(self, 
                 noc: NoCTopology,
                 input_dim: int,
                 layer_dims: List[int],
                 seq_len: int = 1,
                 mapping_strategy: str = "column_wise",
                 split_strategy: str = "column_split",
                 data_type: str = "float16",
                 reuse_pe_for_aggregation: bool = True,
                 row_aggregation_enabled: bool = True,
                 column_aggregation_enabled: bool = False,
                 llm=None,
                 allow_wrapping: bool = False):
        """
        Initialize the FC Neural Network.
        
        Args:
            noc: NoC topology to map computations onto
            input_dim: Input dimension
            layer_dims: List of output dimensions for each layer in the network
            seq_len: Sequence length for inference
            mapping_strategy: How to map layers onto the NoC
            split_strategy: How to split large weight matrices
            data_type: Data type for weights and activations
            reuse_pe_for_aggregation: Whether to reuse computation PEs for aggregation
            row_aggregation_enabled: Whether to aggregate partial results at each layer (True)
                                    or pass unaggregated results to the next network (False)
            column_aggregation_enabled: Whether to perform column aggregation in hybrid_split mode (True)
                                       or pass column-wise partial results directly (False)
            llm: Reference to parent LLM if part of a larger model
            allow_wrapping: Whether to allow wrapping around the edges of the NoC (default: False)
        """
        self.noc = noc
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.seq_len = seq_len
        self.layer_count = len(layer_dims)
        self.mapping_strategy = mapping_strategy
        self.split_strategy = split_strategy
        self.data_type = data_type
        self.reuse_pe_for_aggregation = reuse_pe_for_aggregation
        self.row_aggregation_enabled = row_aggregation_enabled
        self.column_aggregation_enabled = column_aggregation_enabled
        self.llm = llm
        self.allow_wrapping = allow_wrapping
        
        # Map layers to PEs
        self.mapper = FCLayerMapper(
            self.noc, input_dim, layer_dims, seq_len, mapping_strategy, split_strategy, data_type, 
            neural_network=self, allow_wrapping=self.allow_wrapping)
        
        # Set to track active PEs (used for computation or aggregation)
        self.active_pes = set()
        
        # Track aggregation data structures
        self.row_aggregation_pes = {}  # Maps (layer_id, col_group) to aggregation PE
        self.column_aggregation_pes = {}  # Maps layer_id to final aggregation PE
        
        # Track dedicated aggregation PEs
        self.aggregation_pes = {}
        if (not self.reuse_pe_for_aggregation and 
             (split_strategy == "row_split" or split_strategy == "hybrid_split")):
            self.aggregation_pes, self.row_aggregation_pes, self.column_aggregation_pes = self.mapper.assign_aggregation_pes(self)

        # Register active PEs after mapping is complete
        self._register_active_pes()
    
    def _register_active_pes(self):
        """Register all active PEs (computation and aggregation) in the active_pes set."""
        # Track all used PEs
        # Add computation PEs
        for layer_id in range(len(self.layer_dims)):
            layer_pes = self.mapper.get_layer_pes(layer_id)
            self.active_pes.update(layer_pes)
        
        # Add aggregation PEs
        used_aggregation_pes = 0
        
        if not self.reuse_pe_for_aggregation:
            if self.split_strategy == "row_split":
                # For row_split, we have one aggregation PE per layer
                used_aggregation_pes += len(self.aggregation_pes)
                self.active_pes.update(self.aggregation_pes.values())
            
            elif self.split_strategy == "hybrid_split":
                # For hybrid_split, we have row aggregation PEs for each column group
                # and potentially a final column aggregation PE
                used_aggregation_pes += len(self.row_aggregation_pes)
                self.active_pes.update(self.row_aggregation_pes.values())
                
                # Add column aggregation PEs if they exist
                if self.column_aggregation_pes:
                    used_aggregation_pes += len(self.column_aggregation_pes)
                    self.active_pes.update(self.column_aggregation_pes.values())
            
            else:  # column_split
                # For column_split, we have one aggregation PE per layer
                used_aggregation_pes += len(self.aggregation_pes)
                self.active_pes.update(self.aggregation_pes.values())
        
        print(f"Active PEs registered: {self.active_pes}")
    
    def run_inference(self, 
                       input_tensor: Union[torch.Tensor, Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]],
                       source_pe=None,
                       source_range=None,
                       source_task_ids=None) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]:
        """
        Run inference on the network and generate traffic table.
        
        Args:
            input_tensor: Input tensor for inference or dictionary output from a previous network
            source_pe: Source PE(s) providing input to this network (only used if input_tensor is a tensor)
                       Can be a single tuple (x,y), a list of tuples, or None (uses default virtual PE)
            source_range: Ranges of the source PE's tensor (only used if input_tensor is a tensor)
                        Can be a single tuple of tuples, a list of such tuples (one per source_pe), or None (full range)
            source_task_ids: Task IDs to wait for before processing input (only used if input_tensor is a tensor)
                           Can be a single string, a list of strings, or None (no dependencies)
                           
        Returns:
            Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples.
            - output_tensor: The actual output tensor for this PE
            - output_range: A tuple ((row_start, row_end), (col_start, col_end)) defining boundaries of the output
            - task_id: The task ID for this PE's final computation
        """
        # Handle dictionary input from previous network
        if isinstance(input_tensor, dict):
            if not input_tensor:
                raise ValueError("Empty dictionary provided for input_tensor")
            
            # Instead of just taking the first PE, collect all PEs, ranges, and task IDs
            source_pes = []
            source_ranges = []
            source_task_ids_list = []
            representative_tensor = None
            
            # For dimension validation
            total_feature_dim = 0
            unique_input_ranges = set()
            
            # Process all PEs in the dictionary
            for pe_coords, (tensor, range_info, task_id) in input_tensor.items():
                source_pes.append(pe_coords)
                source_ranges.append(range_info)
                source_task_ids_list.append(task_id)
                
                # Track the unique column ranges for dimension validation
                if isinstance(range_info, tuple) and len(range_info) == 2:
                    # For standard column ranges: (start, end)
                    if isinstance(range_info[0], int) and isinstance(range_info[1], int):
                        col_start, col_end = range_info
                        if col_start is not None and col_end is not None:
                            unique_input_ranges.add((col_start, col_end))
                            total_feature_dim += (col_end - col_start)
                    # For hybrid_split format: ((row_start, row_end), (col_start, col_end))
                    elif isinstance(range_info[0], tuple) and isinstance(range_info[1], tuple):
                        row_range, col_range = range_info
                        # Only use column range for dimension calculation
                        if None not in col_range and col_range != (-1, -1):
                            col_start, col_end = col_range
                            unique_input_ranges.add((col_start, col_end))
                            total_feature_dim += (col_end - col_start)
                
                # Use the first tensor as a representative for shape checking
                if representative_tensor is None:
                    representative_tensor = tensor
            
            # Update source parameters with lists
            source_pe = source_pes
            source_range = source_ranges
            source_task_ids = source_task_ids_list
            
            # Use the representative tensor for input
            input_tensor = representative_tensor
            
            # Validate total feature dimensions
            if len(source_pes) > 1 and total_feature_dim > 0 and total_feature_dim != self.input_dim:
                print(f"Total feature dimension from all PEs ({total_feature_dim}) doesn't match expected input_dim ({self.input_dim})")
            
        if isinstance(input_tensor, torch.Tensor):
            # Check input dimensions properly
            if isinstance(source_pe, list):
                # For multiple PEs, dimension check already done in dictionary case above
                pass
            elif input_tensor.shape[1] != self.input_dim:
                # Validate for single source cases
                raise ValueError(f"Input tensor must have {self.input_dim} features, got {input_tensor.shape[1]}")
            
            # Determine source PE if none provided
            if source_pe is None:
                # Use shared external PE (0,0) for all mapping strategies
                source_pe = (0, 0)  # External PE
            
            # Select appropriate inference method based on split strategy
            if self.split_strategy == "column_split":
                return self._run_column_split_inference(input_tensor, source_pe, source_range, source_task_ids)
            elif self.split_strategy == "row_split":
                return self._run_row_split_inference(input_tensor, source_pe, source_range, source_task_ids)
            elif self.split_strategy == "hybrid_split":
                return self._run_hybrid_split_inference(input_tensor, source_pe, source_range, source_task_ids)
            else:
                raise ValueError(f"Unknown split strategy: {self.split_strategy}")
        else:
            raise ValueError("Unsupported input type. Expected torch.Tensor or dictionary.")

    # Import the inference implementation methods from separate files to avoid making this file too long
    from .inference_methods import (_run_column_split_inference, _run_row_split_inference, 
                                  _run_hybrid_split_inference, _distribute_input_to_first_layer,
                                  _run_matrix_multiply, _distribute_arithmetic)
    
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
        
        if not self.allow_wrapping:
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
            # Use nominal NoC dimensions
            total_pes = self.noc.rows * self.noc.cols
            # Avoid division by zero
            computation_utilization = (used_computation_pes / total_pes) * 100 if total_pes > 0 else 0
            total_utilization = (total_used_pes / total_pes) * 100 if total_pes > 0 else 0
            
            # Create result dictionary
            result = {
                "total_pes": total_pes,
                "used_computation_pes": used_computation_pes,
                "used_aggregation_pes": used_aggregation_pes,
                "total_used_pes": total_used_pes,
                "computation_utilization": computation_utilization,
                "total_utilization": total_utilization,
                "rows": self.noc.rows,
                "cols": self.noc.cols
            }
        
        return result 

class ArithmeticNetwork:
    """Network for performing distributed arithmetic operations on NoC."""
    def __init__(self, 
                 noc: NoCTopology,
                 seq_len: int = 1,
                 d_model: int = None,
                 operation: str = "matmul",
                 mapping_strategy: str = "column_wise",
                 split_strategy: str = "column_split",
                 data_type: str = "float16",
                 reuse_pe_for_aggregation: bool = True,
                 row_aggregation_enabled: bool = True,
                 column_aggregation_enabled: bool = False,
                 llm=None,
                 allow_wrapping: bool = False):
        """
        Initialize the Arithmetic Network for distributed computation.
        
        Args:
            noc: NoC topology to map computations onto
            seq_len: Sequence length for operations
            d_model: Model dimension (e.g. embedding dimension)
            operation: Operation type ("matmul" or "element_wise")
            mapping_strategy: How to map operations onto the NoC
            split_strategy: How to split large matrices
            data_type: Data type for computations
            reuse_pe_for_aggregation: Whether to reuse computation PEs for aggregation
            row_aggregation_enabled: Whether to aggregate partial results
            column_aggregation_enabled: Whether to perform column aggregation in hybrid_split mode
            llm: Reference to parent LLM if part of a larger model
            allow_wrapping: Whether to allow wrapping around the edges of the NoC
        """
        self.noc = noc
        self.seq_len = seq_len
        self.d_model = d_model
        self.operation = operation  # Operation type (matmul, element_wise, etc.)
        self.mapping_strategy = mapping_strategy
        self.split_strategy = split_strategy
        self.data_type = data_type
        self.reuse_pe_for_aggregation = reuse_pe_for_aggregation
        self.row_aggregation_enabled = row_aggregation_enabled
        self.column_aggregation_enabled = column_aggregation_enabled
        self.llm = llm
        self.allow_wrapping = allow_wrapping
        
        # Set to track active PEs (used for computation or aggregation)
        self.active_pes = set()
        
        # Track aggregation data structures
        self.row_aggregation_pes = {}  # Maps (op_id, col_group) to aggregation PE
        self.column_aggregation_pes = {}  # Maps op_id to final aggregation PE
        self.aggregation_pes = {}
        
        # Import data_type_bytes from external module
        from data_structs import dtype_size
        self.data_type_bytes = dtype_size(self.data_type)
        
        # Create mapper for PE assignments
        # Since we are only considering the momory due to the attention score matrix, we put the input and output dimensions as seq_len
        self.mapper = FCLayerMapper(
            self.noc, seq_len, [seq_len], seq_len, mapping_strategy, split_strategy, data_type,
            neural_network=self, allow_wrapping=self.allow_wrapping
        )
        
        # Register active PEs after mapping is complete
        self._register_active_pes()
    
    def _register_active_pes(self):
        """Register all active PEs (computation and aggregation) in the active_pes set."""
        # Add computation PEs
        layer_pes = self.mapper.get_layer_pes(0)  # We only have one "layer" for matrix multiply
        self.active_pes.update(layer_pes)
        
        # Add aggregation PEs if not reusing computation PEs
        if not self.reuse_pe_for_aggregation:
            if self.split_strategy == "row_split":
                self.active_pes.update(self.aggregation_pes.values())
            elif self.split_strategy == "hybrid_split":
                self.active_pes.update(self.row_aggregation_pes.values())
                if self.column_aggregation_enabled:
                    self.active_pes.update(self.column_aggregation_pes.values())
    
    def register_active_pes(self):
        """
        Register the active PEs for this network in the NoC.
        """
        # Clear any existing active PEs
        self.active_pes = set()
        
        # Add all PEs from all layers
        for layer_id in range(len(self.layer_dims)):
            layer_pes = self.mapper.get_layer_pes(layer_id)
            self.active_pes.update(layer_pes)
            
        # Add any aggregation PEs if applicable
        if hasattr(self, 'aggregation_pes'):
            self.active_pes.update(self.aggregation_pes.values())
            
        # Add any row aggregation PEs if applicable
        if hasattr(self, 'row_aggregation_pes'):
            self.active_pes.update(self.row_aggregation_pes.values())
            
        # Add any column aggregation PEs if applicable
        if hasattr(self, 'column_aggregation_pes'):
            self.active_pes.update(self.column_aggregation_pes.values())
      
    # Note: The following methods are imported from inference_methods.py:
    # - _filter_tensor_for_region
    # - _distribute_arithmetic_solo
    # - _run_matrix_multiply
    # - _run_attention_computation
     
    def matrix_multiply(self, 
                       input_a: Union[torch.Tensor, Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]],
                       input_b: Union[torch.Tensor, Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]],
                       transpose_b: bool = False,
                       source_pe_a=None,
                       source_pe_b=None) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]:
        """
        Perform distributed matrix multiplication A @ B or A @ B^T.
        
        Args:
            input_a: First input tensor [seq_len × d_model] or dictionary of PE outputs from an FC network
            input_b: Second input tensor [seq_len × d_model] or dictionary of PE outputs from an FC network
            transpose_b: Whether to transpose the second matrix (True for Q @ K^T)
            source_pe_a: Source PE for first input (only used if input_a is a tensor)
            source_pe_b: Source PE for second input (only used if input_b is a tensor)
            
        Returns:
            Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples
        """
        # Initialize default values for source parameters
        source_range_a = None
        source_range_b = None
        source_task_ids_a = None
        source_task_ids_b = None
        
        # Handle input_a in FC network output format
        if isinstance(input_a, dict):
            # Get the first PE from the dictionary for simplicity
            if not input_a:
                raise ValueError("Empty dictionary provided for input_a")
            
            pe_a = next(iter(input_a.keys()))
            tensor_a, range_a, task_id_a = input_a[pe_a]
            
            # Update source parameters
            source_pe_a = pe_a
            source_range_a = range_a
            source_task_ids_a = task_id_a
            
            # Use the tensor from the first PE
            input_a = tensor_a
        
        # Handle input_b in FC network output format
        if isinstance(input_b, dict):
            # Get the first PE from the dictionary for simplicity
            if not input_b:
                raise ValueError("Empty dictionary provided for input_b")
            
            pe_b = next(iter(input_b.keys()))
            tensor_b, range_b, task_id_b = input_b[pe_b]
            
            # Update source parameters
            source_pe_b = pe_b
            source_range_b = range_b
            source_task_ids_b = task_id_b
            
            # Use the tensor from the first PE
            input_b = tensor_b
        
        # Set default source PEs if not provided
        if source_pe_a is None:
            source_pe_a = (0, 0)

        if source_pe_b is None:
            source_pe_b = (0, 0)
            
        if transpose_b:
            if input_a.shape[1] != input_b.shape[1]:
                raise ValueError("Column dimension of input_a must be equal to column dimension of input_b when transpose_b is True")
        else:
            if input_a.shape[1] != input_b.shape[0]:
                raise ValueError("Column dimension of input_a must be equal to row dimension of input_b when transpose_b is False")

        # Use the consolidated matrix multiplication function with appropriate strategy
        return self._run_matrix_multiply(
            input_a, input_b, transpose_b, source_pe_a, source_pe_b,
            strategy=self.split_strategy, 
            source_range_a=source_range_a,
            source_range_b=source_range_b,
            source_task_ids_a=source_task_ids_a,
            source_task_ids_b=source_task_ids_b
        )
    
    def element_wise(self, input_a: torch.Tensor, input_b: torch.Tensor, operation: str = "add", source_pe_a=None, source_pe_b=None) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]:
        """
        Perform distributed element-wise operations.
        
        Args:
            input_a: First input tensor
            input_b: Second input tensor
            operation: Operation to perform ("add", "subtract", "multiply", "divide")
            source_pe_a: Source PE(s) for first input
            source_pe_b: Source PE(s) for second input
            
        Returns:
            Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples
        """
        # Element-wise operations can use similar distribution strategies as matrix multiply
        if self.split_strategy == "column_split":
            return self._run_column_split_elementwise(input_a, input_b, operation, source_pe_a, source_pe_b)
        elif self.split_strategy == "row_split":
            return self._run_row_split_elementwise(input_a, input_b, operation, source_pe_a, source_pe_b)
        elif self.split_strategy == "hybrid_split":
            return self._run_hybrid_split_elementwise(input_a, input_b, operation, source_pe_a, source_pe_b)
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
    
    def reduce(self, input_tensor: torch.Tensor, operation: str = "sum", axis: Optional[int] = None, source_pe=None) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]:
        """
        Perform distributed reduction operations.
        
        Args:
            input_tensor: Input tensor
            operation: Reduction operation ("sum", "max", "min", "mean")
            axis: Axis along which to reduce (None for global reduction)
            source_pe: Source PE(s) for input
            
        Returns:
            Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples
        """
        if self.split_strategy == "column_split":
            return self._run_column_split_reduce(input_tensor, operation, axis, source_pe)
        elif self.split_strategy == "row_split":
            return self._run_row_split_reduce(input_tensor, operation, axis, source_pe)
        elif self.split_strategy == "hybrid_split":
            return self._run_hybrid_split_reduce(input_tensor, operation, axis, source_pe)
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
    
    def get_traffic_table(self) -> pd.DataFrame:
        """Get the traffic table."""
        return self.noc.scheduler.get_traffic_table()
    
    def print_pe_outputs(self, pe_outputs: Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]):
        """Print the outputs of each PE."""
        for pe_coords, (pe_output, output_range, computation_task_id) in pe_outputs.items():
            # Format the output range in a compact way
            if isinstance(output_range, tuple) and len(output_range) == 2 and isinstance(output_range[0], int):
                # Format for column_split: (start, end)
                start = output_range[0] if output_range[0] is not None else "None"
                end = output_range[1] if output_range[1] is not None else "None"
                range_str = f"(:, {start}:{end})"
            elif isinstance(output_range, tuple) and len(output_range) == 2 and isinstance(output_range[0], tuple):
                # Format for hybrid_split: ((row_start, row_end), (col_start, col_end))
                row_range, col_range = output_range
                row_part = f"{row_range[0]}:{row_range[1]}" if None not in row_range else "None:None"
                col_part = f"{col_range[0]}:{col_range[1]}" if None not in col_range else "None:None"
                range_str = f"({row_part}, {col_part})"
            else:
                range_str = str(output_range)
            
            print(f"PE{pe_coords} output: {pe_output.shape}, tensor slice: {range_str}, task_id: {computation_task_id}")
    
    def get_pe_utilization(self, use_effective_dimensions=False) -> dict:
        """
        Calculate the utilization of PEs in the NoC grid.
        
        Args:
            use_effective_dimensions: Whether to use effective dimensions based on actual PE coordinates
            
        Returns:
            Dictionary containing utilization statistics
        """
        # Count PEs used for computation
        used_computation_pes = len(self.active_pes)
        
        # Count PEs used for aggregation
        used_aggregation_pes = 0
        if not self.reuse_pe_for_aggregation:
            if self.split_strategy == "row_split":
                used_aggregation_pes = len(self.aggregation_pes)
            elif self.split_strategy == "hybrid_split":
                used_aggregation_pes = len(self.row_aggregation_pes) + len(self.column_aggregation_pes)
                # Remove duplicates if same PE used for both
                row_agg_pes = set(self.row_aggregation_pes.values())
                col_agg_pes = set(self.column_aggregation_pes.values())
                used_aggregation_pes = len(row_agg_pes.union(col_agg_pes))
        
        # Total used PEs
        total_used_pes = used_computation_pes + used_aggregation_pes
        
        if use_effective_dimensions:
            # Get effective dimensions based on actual PE coordinates
            effective_rows = len(set(y for _, y in self.active_pes))
            effective_cols = len(set(x for x, _ in self.active_pes))
            total_pes = effective_rows * effective_cols
            
            result = {
                "total_pes": total_pes,
                "used_computation_pes": used_computation_pes,
                "used_aggregation_pes": used_aggregation_pes,
                "total_used_pes": total_used_pes,
                "computation_utilization": (used_computation_pes / total_pes * 100) if total_pes > 0 else 0,
                "total_utilization": (total_used_pes / total_pes * 100) if total_pes > 0 else 0,
                "effective_rows": effective_rows,
                "effective_cols": effective_cols
            }
        else:
            # Use nominal NoC dimensions
            total_pes = self.noc.rows * self.noc.cols
            
            result = {
                "total_pes": total_pes,
                "used_computation_pes": used_computation_pes,
                "used_aggregation_pes": used_aggregation_pes,
                "total_used_pes": total_used_pes,
                "computation_utilization": (used_computation_pes / total_pes * 100) if total_pes > 0 else 0,
                "total_utilization": (total_used_pes / total_pes * 100) if total_pes > 0 else 0,
                "rows": self.noc.rows,
                "cols": self.noc.cols
            }
        
        return result
        
    # Import methods from inference_methods.py
    from .inference_methods import (_run_matrix_multiply, _distribute_arithmetic, _distribute_arithmetic_solo, 
                                   _run_attention_computation, attention_computation)