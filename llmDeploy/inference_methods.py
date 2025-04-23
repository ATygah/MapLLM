import torch
from typing import Dict, Tuple, Optional, List, Any, Union

def _distribute_input_to_first_layer(self, 
                                    input_tensor: torch.Tensor, 
                                    source_pe: Union[Tuple[int, int], List[Tuple[int, int]]],
                                    source_range=None,
                                    source_task_ids=None) -> Dict[Tuple[int, int], List[str]]:
    """
    Intelligently distribute input data to the first layer PEs based on their splitting strategy.
    
    This function analyzes how the first layer PEs are split (row, column, or hybrid) and
    ensures each PE receives exactly the portion of the input tensor it needs for computation.
    
    Args:
        input_tensor: Input tensor for inference of shape [seq_len, input_dim]
        source_pe: Coordinates of the virtual source PE or a list of source PEs
        source_range: Ranges of the source PE's tensor in the format ((row_start, row_end), (col_start, col_end))
                     or a list of such tuples, one per source_pe
        source_task_ids: Task IDs to wait for before processing input (dependencies from previous network)
    
    Returns:
        Dictionary mapping PE coordinates to a list of task IDs that send data to it
    """
    seq_len, input_dim = input_tensor.shape
    
    # Get network_id if this network is part of an LLM
    network_id = None
    if hasattr(self, 'llm') and self.llm is not None:
        # Find this network's index in the LLM
        for idx, network in enumerate(self.llm.networks):
            if network is self:
                network_id = idx
                break
    
    # Get first layer PEs - filter for active PEs only
    first_layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(0)
                      if pe_coords in self.active_pes]
    
    if not first_layer_pes:
        raise ValueError("No active PEs found for the first layer")
    
    # Initialize dependency map for all first layer PEs
    pe_to_dependencies_map = {pe_coords: [] for pe_coords in first_layer_pes}
    
    # Determine which portions of input each PE needs based on the split strategy
    pe_input_slices = {}  # Maps PE coordinates to (row_start, row_end) slices
    
    # Analyze first layer PEs to determine what input slices they need
    for pe_coords in first_layer_pes:
        pe = self.noc.get_pe(*pe_coords)
        
        if self.split_strategy == "row_split":
            # In row-split, each PE handles specific rows (input features)
            if not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or pe.row_start is None or pe.row_end is None:
                continue
            
            # Row-split: PE needs specific rows of input
            row_start, row_end = pe.row_start, pe.row_end
            pe_input_slices[pe_coords] = (row_start, row_end)
            
        elif self.split_strategy == "column_split":
            # In column-split, each PE actually only needs the input dimensions
            # that correspond to the weight matrix rows it will use
            if not hasattr(pe, 'weight_shape') or pe.weight_shape is None:
                continue
                
            # For column split, check if we have row information
            if hasattr(pe, 'row_start') and hasattr(pe, 'row_end') and pe.row_start is not None and pe.row_end is not None:
                # Use specified row range if available
                row_start, row_end = pe.row_start, pe.row_end
            else:
                # Fallback to full input (should be rare)
                row_start, row_end = 0, input_dim
                
            pe_input_slices[pe_coords] = (row_start, row_end)
            
        elif self.split_strategy == "hybrid_split":
            # In hybrid-split, each PE handles specific rows
            if not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or pe.row_start is None or pe.row_end is None:
                continue
            
            # Hybrid-split: PE needs specific rows of input
            row_start, row_end = pe.row_start, pe.row_end
            pe_input_slices[pe_coords] = (row_start, row_end)
    
    # Handle input distribution from source PE(s)
    if isinstance(source_pe, list):
        # Multiple source PEs (from previous network)
        if source_task_ids is None:
            source_task_ids = [None] * len(source_pe)
        elif not isinstance(source_task_ids, list):
            source_task_ids = [source_task_ids]
        
        # Normalize source_range to list format, one range per source_pe
        source_ranges = []
        if source_range is None:
            # Default: each source PE covers the full input dimension
            source_ranges = [((0, seq_len), (0, input_dim))] * len(source_pe)
        elif isinstance(source_range, tuple) and len(source_range) == 2:
            # Check if this is a nested tuple ((row_start, row_end), (col_start, col_end))
            if isinstance(source_range[0], tuple) and isinstance(source_range[1], tuple):
                # Single range for all source PEs
                source_ranges = [source_range] * len(source_pe)
            else:
                # Convert legacy format (col_start, col_end) to full format
                source_ranges = [((0, seq_len), source_range)] * len(source_pe)
        elif isinstance(source_range, list):
            # List of ranges, one per source PE
            source_ranges = []
            for sr in source_range:
                if isinstance(sr, tuple) and len(sr) == 2:
                    if isinstance(sr[0], tuple) and isinstance(sr[1], tuple):
                        # Already in correct format ((row_start, row_end), (col_start, col_end))
                        source_ranges.append(sr)
                    else:
                        # Convert legacy format (col_start, col_end) to full format
                        source_ranges.append(((0, seq_len), sr))
                else:
                    # Invalid format, default to full range
                    source_ranges.append(((0, seq_len), (0, input_dim)))
            
            # Pad if necessary
            if len(source_ranges) < len(source_pe):
                source_ranges.extend([((0, seq_len), (0, input_dim))] * (len(source_pe) - len(source_ranges)))
        else:
            # Invalid format, default to full range
            source_ranges = [((0, seq_len), (0, input_dim))] * len(source_pe)
            
        # For each destination PE, determine which source PE has the data it needs
        for dest_pe_coords, (dest_row_start, dest_row_end) in pe_input_slices.items():
            # Check each source PE for relevant data
            for src_idx, src in enumerate(source_pe):
                if src_idx >= len(source_ranges):
                    continue  # Skip if no range defined for this source PE
                    
                # Get source range for this PE
                src_row_range, src_col_range = source_ranges[src_idx]
                src_col_start, src_col_end = src_col_range
                
                # For matrix multiplication, source's columns must overlap with destination's rows
                overlap_start = max(dest_row_start, src_col_start)
                overlap_end = min(dest_row_end, src_col_end)
                
                # Only create a task if there's a meaningful overlap
                if overlap_end <= overlap_start:
                    # No overlap, so this source PE has no data to send to this destination PE
                    continue
                
                # Get wait ID for this source PE if available
                wait_id = source_task_ids[src_idx] if src_idx < len(source_task_ids) else None
                wait_ids = [wait_id] if wait_id else []
                
                # Create task to send only the overlapping portion
                task_id = self.noc.scheduler.create_task(
                    src_pe=src,
                    dest_pe=dest_pe_coords,
                    tensor_shape=(seq_len, overlap_end - overlap_start),
                    wait_ids=wait_ids,
                    description=f"Network {network_id} input  (0:{seq_len},{overlap_start}:{overlap_end}) from previous network PE{src} to PE{dest_pe_coords}",
                    network_id=network_id
                )
                pe_to_dependencies_map[dest_pe_coords].append(task_id)
    else:
        # Single source PE
        # Normalize source_range to the new format
        src_row_range = (0, seq_len)  # Default row range
        src_col_range = (0, input_dim)  # Default column range
        
        if source_range is not None:
            if isinstance(source_range, tuple) and len(source_range) == 2:
                if isinstance(source_range[0], tuple) and isinstance(source_range[1], tuple):
                    # Already in correct format ((row_start, row_end), (col_start, col_end))
                    src_row_range, src_col_range = source_range
                else:
                    # Legacy format (col_start, col_end), convert to new format
                    src_col_range = source_range
        
        src_col_start, src_col_end = src_col_range
        
        for dest_pe_coords, (row_start, row_end) in pe_input_slices.items():
            # Prepare wait IDs from previous network
            wait_ids = []
            if source_task_ids:
                if isinstance(source_task_ids, list):
                    wait_ids.extend([task_id for task_id in source_task_ids if task_id])
                else:
                    wait_ids.append(source_task_ids)
            
            # For matrix multiplication, source columns must overlap with destination rows
            overlap_start = max(row_start, src_col_start)
            overlap_end = min(row_end, src_col_end)
            
            # Only create a task if there's a meaningful overlap
            if overlap_end <= overlap_start:
                # No overlap, so no data to send
                continue
            
            # Create task to send required input slice to destination PE
            input_slice_shape = (seq_len, overlap_end - overlap_start)
            task_id = self.noc.scheduler.create_task(
                src_pe=source_pe,
                dest_pe=dest_pe_coords,
                tensor_shape=input_slice_shape,
                wait_ids=wait_ids,
                description=f"Input distribution (rows {overlap_start}:{overlap_end}) to PE{dest_pe_coords}",
                network_id=network_id
            )
            pe_to_dependencies_map[dest_pe_coords].append(task_id)
    
    return pe_to_dependencies_map

def _run_column_split_inference(self, input_tensor: torch.Tensor, source_pe: Union[Tuple[int, int], List[Tuple[int, int]]], source_range=None, source_task_ids=None) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]:
    """
    Run inference with column-split strategy.
    Weight matrices are split by columns (output neurons).
    No aggregation needed since each PE computes complete outputs for its subset of neurons.
    
    Args:
        input_tensor: Input tensor for inference
        source_pe: Coordinates of the virtual source PE or a list of source PEs
        source_range: Ranges of the source PE's tensor in the format ((row_start, row_end), (col_start, col_end))
                     or a list of such tuples, one per source_pe
        source_task_ids: Task IDs to wait for before processing input (dependencies from previous network)
    
    Returns:
        Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples.
        - output_tensor: The actual output tensor for this PE
        - output_range: A tuple ((row_start, row_end), (col_start, col_end)) defining boundaries of the output
        - task_id: The task ID for this PE's final computation
    """
    seq_len, input_dim = input_tensor.shape
    
    # Get network_id if this network is part of an LLM
    network_id = None
    if hasattr(self, 'llm') and self.llm is not None:
        # Find this network's index in the LLM
        for idx, network in enumerate(self.llm.networks):
            if network is self:
                network_id = idx
                break
    
    # Use the new distribution function to handle input distribution to first layer PEs
    pe_to_dependencies_map = self._distribute_input_to_first_layer(
        input_tensor, source_pe, source_range, source_task_ids
    )
    
    # Keep track of computation tasks for each PE across all layers
    pe_computation_tasks = {}
    
    # First layer PEs - filter for active PEs only
    first_layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(0)
                      if pe_coords in self.active_pes]
    
    if not first_layer_pes:
        raise ValueError("No active PEs found for the first layer")
    
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
            
            # Get the row and column ranges for this PE's weight matrix
            row_range = None
            if hasattr(pe, 'row_start') and hasattr(pe, 'row_end') and pe.row_start is not None and pe.row_end is not None:
                row_range = (pe.row_start, pe.row_end)
            
            col_range = None
            if hasattr(pe, 'col_start') and hasattr(pe, 'col_end') and pe.col_start is not None and pe.col_end is not None:
                col_range = (pe.col_start, pe.col_end)
            
            # This PE should wait for all tasks that sent data to it
            wait_ids = pe_to_dependencies_map.get(pe_coords, [])
            
            # PE computes its portion of the output based on its weight matrix
            compute_task_id = self.noc.scheduler.create_task(
                src_pe=pe_coords,
                dest_pe=pe_coords,  # Computing within the same PE
                tensor_shape=(seq_len, pe.weight_shape[1]),
                wait_ids=wait_ids,  # Wait for all input transfers
                description=(f"Layer {layer_id} PE{pe_coords} computation " + 
                           (f"(input rows {row_range[0]}:{row_range[1]}, " if row_range else "") + 
                           (f"output cols {col_range[0]}:{col_range[1]})" if col_range else "")),
                network_id=network_id
            )
            
            # Store the computation task ID for this PE
            pe_computation_tasks[pe_coords] = compute_task_id
            
            # If not the last layer, distribute outputs to next layer
            if layer_id < len(self.layer_dims) - 1:
                # Get next layer's PEs that need input from this PE
                next_layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(layer_id + 1)
                                 if pe_coords in self.active_pes]
                
                # Determine which portions of this PE's output are needed by each PE in the next layer
                for next_pe_coords in next_layer_pes:
                    next_pe = self.noc.get_pe(*next_pe_coords)
                    
                    # Skip next PEs without proper row information
                    if not hasattr(next_pe, 'row_start') or not hasattr(next_pe, 'row_end') or next_pe.row_start is None or next_pe.row_end is None:
                        # If row info is missing, send entire output
                        transfer_task_id = self.noc.scheduler.create_task(
                            src_pe=pe_coords,
                            dest_pe=next_pe_coords,
                            tensor_shape=(seq_len, pe.weight_shape[1]),
                            wait_ids=[compute_task_id],  # Wait for this PE's computation
                            description=f"Layer {layer_id} PE{pe_coords} -> Layer {layer_id+1} PE{next_pe_coords} (full output)",
                            network_id=network_id
                        )
                        
                        # Add this transfer as a dependency for the next layer's PE
                        next_pe_to_dependencies_map[next_pe_coords].append(transfer_task_id)
                    else:
                        # Next layer PE needs specific columns from this PE's output if they match its row inputs
                        # Check if this PE's output columns overlap with next PE's input rows
                        if col_range is not None and col_range[1] > next_pe.row_start and col_range[0] < next_pe.row_end:
                            # Calculate the overlap
                            overlap_start = max(col_range[0], next_pe.row_start)
                            overlap_end = min(col_range[1], next_pe.row_end)
                            
                            if overlap_end > overlap_start:
                                # Send only the overlapping portion to the next PE
                                transfer_task_id = self.noc.scheduler.create_task(
                                    src_pe=pe_coords,
                                    dest_pe=next_pe_coords,
                                    tensor_shape=(seq_len, overlap_end - overlap_start),
                                    wait_ids=[compute_task_id],  # Wait for this PE's computation
                                    description=f"Layer {layer_id} PE{pe_coords} (cols {overlap_start}:{overlap_end}) -> Layer {layer_id+1} PE{next_pe_coords}",
                                    network_id=network_id
                                )
                                
                                # Add this transfer as a dependency for the next layer's PE
                                next_pe_to_dependencies_map[next_pe_coords].append(transfer_task_id)
                        else:
                            # In column-split, we may need to send the full output to each next layer PE
                            # if there's no clear mapping between this PE's outputs and next PEs' inputs
                            transfer_task_id = self.noc.scheduler.create_task(
                                src_pe=pe_coords,
                                dest_pe=next_pe_coords,
                                tensor_shape=(seq_len, pe.weight_shape[1]),
                                wait_ids=[compute_task_id],  # Wait for this PE's computation
                                description=f"Layer {layer_id} PE{pe_coords} -> Layer {layer_id+1} PE{next_pe_coords}",
                                network_id=network_id
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
            ((0, seq_len), col_range),  # Range of output neurons this PE computed
            pe_computation_tasks.get(pe_coords)
        )
    
    return pe_outputs

def _run_row_split_inference(self, input_tensor: torch.Tensor, source_pe: Union[Tuple[int, int], List[Tuple[int, int]]], source_range=None, source_task_ids=None) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]:
    """
    Run inference with row-split strategy.
    Weight matrices are split by rows (input neurons).
    Requires aggregation since each PE computes a partial output.
    
    Args:
        input_tensor: Input tensor for inference
        source_pe: Coordinates of the virtual source PE or a list of source PEs
        source_range: Ranges of the source PE's tensor in the format ((row_start, row_end), (col_start, col_end))
                     or a list of such tuples, one per source_pe
        source_task_ids: Task IDs to wait for before processing input (dependencies from previous network)
    
    Returns:
        Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples.
        - output_tensor: The actual output tensor for this PE
        - output_range: A tuple ((row_start, row_end), (col_start, col_end)) defining boundaries of the output
        - task_id: The task ID for this PE's final computation
    """
    seq_len, input_dim = input_tensor.shape
    
    # Get network_id if this network is part of an LLM
    network_id = None
    if hasattr(self, 'llm') and self.llm is not None:
        # Find this network's index in the LLM
        for idx, network in enumerate(self.llm.networks):
            if network is self:
                network_id = idx
                break
    
    # Use the new distribution function to handle input distribution to first layer PEs
    pe_to_dependencies_map = self._distribute_input_to_first_layer(
        input_tensor, source_pe, source_range, source_task_ids
    )
    
    # Keep track of computation tasks for each PE across all layers
    pe_computation_tasks = {}
    
    # Get PEs for this layer - filter for active PEs only
    layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(0)
                if pe_coords in self.active_pes]
    
    if not layer_pes:
        raise ValueError("No active PEs found for the layer")
    
    # For row split, we need the input range of each PE
    pe_input_ranges = {}
    pe_weight_shapes = {}
    
    for pe_coords in layer_pes:
        pe = self.noc.get_pe(*pe_coords)
        
        # Skip PEs without proper splits defined
        if not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or pe.row_start is None or pe.row_end is None:
            continue
        
        # Get the input range for this PE
        row_start, row_end = pe.row_start, pe.row_end
        
        # Store PE input range
        pe_input_ranges[pe_coords] = (row_start, row_end)
        pe_weight_shapes[pe_coords] = pe.weight_shape
    
    # Compute outputs for all PEs first, with dependencies only on input distribution
    for pe_coords in layer_pes:
        if pe_coords not in pe_input_ranges:
            continue
        
        # Get row range for this PE
        row_range = pe_input_ranges[pe_coords]
        row_start, row_end = row_range
        
        # This PE should wait for all tasks that sent data to it from the input distribution
        wait_ids = pe_to_dependencies_map.get(pe_coords, [])
        
        # PE computes its partial output based on its input slice
        weight_shape = pe_weight_shapes[pe_coords]
        compute_task_id = self.noc.scheduler.create_task(
            src_pe=pe_coords,
            dest_pe=pe_coords,  # Computing within the same PE
            tensor_shape=(seq_len, weight_shape[1]),
            wait_ids=wait_ids,  # Wait only for input distribution
            description=f"PE{pe_coords} partial output computation (input rows {row_start}:{row_end})",
            network_id=network_id
        )
        
        # Store the computation task ID for this PE
        pe_computation_tasks[pe_coords] = compute_task_id
    
    # Track the outputs from final layer
    final_layer_id = len(self.layer_dims) - 1
    
    # Create output dictionary for PE outputs
    pe_outputs = {}
    
    # Check if row aggregation is enabled
    if self.row_aggregation_enabled:
        # For row split, determine the aggregation PE
        aggregation_pe = None
        
        if self.reuse_pe_for_aggregation:
            # Reuse the first PE for aggregation
            aggregation_pe = next(iter(layer_pes)) if layer_pes else None
        else:
            # Check if we have a dedicated aggregation PE from the mapper
            if hasattr(self, 'aggregation_pes') and 0 in self.aggregation_pes:
                aggregation_pe = self.aggregation_pes[0]
        
        if aggregation_pe is None:
            raise ValueError("No aggregation PE available for row-split strategy")
        
        # Collect partial results and send them to the aggregation PE
        # Create a separate list to track aggregation dependencies
        aggregation_dependencies = []
        
        # First add the aggregation PE's own computation as a dependency
        agg_pe_computation_task = pe_computation_tasks.get(aggregation_pe)
        if agg_pe_computation_task:
            aggregation_dependencies.append(agg_pe_computation_task)
        
        # Then add transfers from other PEs 
        for pe_coords in layer_pes:
            if pe_coords == aggregation_pe:
                # Skip sending to self
                continue
            
            # This transfer should wait for the computation at this PE
            wait_id = pe_computation_tasks.get(pe_coords)
            if wait_id is None:
                continue
            
            # Get row range for this PE for the description
            row_range = pe_input_ranges.get(pe_coords, (None, None))
            row_start, row_end = row_range
            
            # Send partial output to the aggregation PE
            transfer_task_id = self.noc.scheduler.create_task(
                src_pe=pe_coords,
                dest_pe=aggregation_pe,
                tensor_shape=(seq_len, self.layer_dims[0]),
                wait_ids=[wait_id],  # Wait for this PE's computation ONLY
                description=f"Partial output from PE{pe_coords} (input rows {row_start}:{row_end}) -> aggregation PE{aggregation_pe}",
                network_id=network_id
            )
            
            # Add this task as a dependency for the aggregation task
            aggregation_dependencies.append(transfer_task_id)
        
        # Final aggregation task - sum all partial outputs
        # Use a set to remove any duplicate dependencies
        aggregation_wait_ids = list(set(aggregation_dependencies))
        
        # Create the aggregation task
        aggregation_task_id = self.noc.scheduler.create_task(
            src_pe=aggregation_pe,
            dest_pe=aggregation_pe,
            tensor_shape=(seq_len, self.layer_dims[0]),
            wait_ids=aggregation_wait_ids,
            description=f"Final output aggregation at PE{aggregation_pe}",
            network_id=network_id
        )
        
        # Get only active PEs for the final layer
        final_layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(final_layer_id) 
                          if pe_coords in self.active_pes]
        final_aggregation_pe = aggregation_pe
        
        # Include the final aggregated output from the aggregation PE
        # This contains the complete output for all neurons in the layer
        aggregated_output = torch.zeros((seq_len, self.layer_dims[final_layer_id]))
        pe_outputs[final_aggregation_pe] = (
            aggregated_output,
            ((0, seq_len), (0, self.layer_dims[final_layer_id])),  # Full range of output
            aggregation_task_id
        )
    else:
        # If row aggregation is disabled, include all partial outputs directly
        # Each PE produces partial computations for the entire output dimension
        for pe_coords, compute_task_id in pe_computation_tasks.items():
            if pe_coords not in pe_input_ranges:
                continue
                
            # Get input range for this PE for output description
            row_range = pe_input_ranges.get(pe_coords, (None, None))
            
            # Create placeholder tensor for this PE's partial output
            # Note: Each PE computes a partial output for ALL output dimensions
            partial_output = torch.zeros((seq_len, self.layer_dims[final_layer_id]))
            
            pe_outputs[pe_coords] = (
                partial_output,
                ((0, seq_len), (0, self.layer_dims[final_layer_id])),  # Full range but partial values
                compute_task_id
            )
    
    return pe_outputs

def _run_hybrid_split_inference(self, input_tensor: torch.Tensor, source_pe: Union[Tuple[int, int], List[Tuple[int, int]]], source_range=None, source_task_ids=None) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]:
    """
    Run inference with hybrid-split strategy.
    Weight matrices are split by both rows and columns.
    Each PE handles a subset of inputs and outputs, computing a partial result.
    
    Args:
        input_tensor: Input tensor for inference
        source_pe: Coordinates of the virtual source PE or a list of source PEs
        source_range: Ranges of the source PE's tensor in the format ((row_start, row_end), (col_start, col_end))
                     or a list of such tuples, one per source_pe
        source_task_ids: Task IDs to wait for before processing input (dependencies from previous network)
    
    Returns:
        Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples.
        - output_tensor: The actual output tensor for this PE
        - output_range: A tuple ((row_start, row_end), (col_start, col_end)) defining boundaries of the output
        - task_id: The task ID for this PE's final computation
    """
    seq_len, input_dim = input_tensor.shape
    
    # Get network_id if this network is part of an LLM
    network_id = None
    if hasattr(self, 'llm') and self.llm is not None:
        # Find this network's index in the LLM
        for idx, network in enumerate(self.llm.networks):
            if network is self:
                network_id = idx
                break
    
    # Use the new distribution function to handle input distribution to first layer PEs
    pe_wait_ids = self._distribute_input_to_first_layer(
        input_tensor, source_pe, source_range, source_task_ids
    )
    
    # For hybrid split, we need to know which column group each PE belongs to
    pe_col_groups = {}  # Maps pe_coords -> (col_start, col_end)
    pe_row_groups = {}  # Maps pe_coords -> (row_start, row_end)
    
    # Track computation tasks for each PE
    pe_computation_tasks = {}
    
    # Group PEs by row and column groups
    layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(0)
                if pe_coords in self.active_pes]
    
    if not layer_pes:
        raise ValueError("No active PEs found for the layer")
    
    # First identify row and column groups
    for pe_coords in layer_pes:
        pe = self.noc.get_pe(*pe_coords)
        
        # Skip PEs without proper splits defined
        if (not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or 
            not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or
            pe.row_start is None or pe.row_end is None or 
            pe.col_start is None or pe.col_end is None):
            continue
        
        # Store PE's row and column ranges
        pe_row_groups[pe_coords] = (pe.row_start, pe.row_end)
        pe_col_groups[pe_coords] = (pe.col_start, pe.col_end)
    
    # Collect unique column groups
    col_groups = {}  # Maps (col_start, col_end) -> list of PEs in this column group
    
    for pe_coords, col_range in pe_col_groups.items():
        if col_range not in col_groups:
            col_groups[col_range] = []
        col_groups[col_range].append(pe_coords)

    # Check if we have actual row splitting
    has_row_splitting = False
    full_row_range = (0, self.input_dim)
    
    # Check if any PE has a row range that's not the full input dimension
    for row_range in pe_row_groups.values():
        if row_range != full_row_range:
            has_row_splitting = True
            break
    
    # Step 2: Each PE computes its portion of the output based only on input dependencies
    for pe_coords in layer_pes:
        if pe_coords not in pe_row_groups or pe_coords not in pe_col_groups:
            continue
        
        row_range = pe_row_groups[pe_coords]
        col_range = pe_col_groups[pe_coords]
        
        # This PE should wait for all tasks that sent data to it
        wait_ids = pe_wait_ids[pe_coords]
        
        # PE computes its partial output (slice of both input and output)
        pe = self.noc.get_pe(*pe_coords)
        
        # Skip PEs without proper weight shape
        if not hasattr(pe, 'weight_shape') or pe.weight_shape is None:
            continue
        
        # For hybrid split, always create the computation task because tasks are needed for dependency tracking
        # But mark in the description if it's a full computation (not partial)
        is_full_computation = row_range == full_row_range
        description = (
            f"PE{pe_coords} {'full' if is_full_computation else 'partial'} computation "
            f"(input:{row_range}, output:{col_range})"
        )
        
        compute_task_id = self.noc.scheduler.create_task(
            src_pe=pe_coords,
            dest_pe=pe_coords,  # Computing within the same PE
            tensor_shape=(seq_len, col_range[1] - col_range[0]),  # Output slice shape
            wait_ids=wait_ids,  # Wait for input distribution only
            description=description,
            network_id=network_id
        )
        
        # Store the computation task ID for this PE
        pe_computation_tasks[pe_coords] = compute_task_id
    
    # Track which PEs are aggregation PEs for each column group
    aggregation_pes = set()
    if has_row_splitting and self.row_aggregation_enabled:
        for col_range, col_pes in col_groups.items():
            if len(col_pes) > 0:
                if self.reuse_pe_for_aggregation:
                    # Use first PE in column group as aggregation PE
                    aggregation_pes.add(col_pes[0])
                else:
                    # Use assigned aggregation PE from layer_mapper
                    if hasattr(self, 'row_aggregation_pes') and (0, col_range) in self.row_aggregation_pes:
                        aggregation_pes.add(self.row_aggregation_pes[(0, col_range)])

    # Step 3: Row aggregation within each column group - only if we have actual row splitting
    if has_row_splitting and self.row_aggregation_enabled:
        for col_range, col_pes in col_groups.items():
            # Only perform row aggregation if we have multiple PEs in the column group
            if len(col_pes) < 2:
                continue
            
            # Find the PE that will perform aggregation for this column group
            if self.reuse_pe_for_aggregation:
                agg_pe = col_pes[0]  # Use first PE in column group
            else:
                # Use assigned aggregation PE from layer_mapper
                if not (hasattr(self, 'row_aggregation_pes') and (0, col_range) in self.row_aggregation_pes):
                    continue  # Skip if no aggregation PE assigned
                agg_pe = self.row_aggregation_pes[(0, col_range)]
            
            # Create transfer tasks to send partial results to aggregation PE
            transfer_wait_ids = []
            for pe in col_pes:
                if pe == agg_pe and self.reuse_pe_for_aggregation:
                    # Skip sending to self only if reusing computation PE
                    continue
                    
                if pe in pe_computation_tasks:
                    # Create transfer task from this PE to aggregation PE
                    transfer_task_id = self.noc.scheduler.create_task(
                        src_pe=pe,
                        dest_pe=agg_pe,
                        tensor_shape=(seq_len, col_range[1] - col_range[0]),
                        wait_ids=[pe_computation_tasks[pe]],  # Wait for computation to complete
                        description=f"Send partial results from PE{pe} to aggregation PE{agg_pe}",
                        network_id=network_id
                    )
                    transfer_wait_ids.append(transfer_task_id)
            
            # Create aggregation task that depends on all compute and transfer tasks
            agg_wait_ids = []
            # Add compute task from aggregation PE itself if it exists and we're reusing
            if self.reuse_pe_for_aggregation and agg_pe in pe_computation_tasks:
                agg_wait_ids.append(pe_computation_tasks[agg_pe])
            # Add all transfer tasks
            agg_wait_ids.extend(transfer_wait_ids)
            
            if agg_wait_ids:
                agg_task_id = self.noc.scheduler.create_task(
                    src_pe=agg_pe,
                    dest_pe=agg_pe,  # Aggregation happens at the same PE
                    tensor_shape=(seq_len, col_range[1] - col_range[0]),  # Same shape as compute output
                    wait_ids=agg_wait_ids,  # Wait for all compute and transfer tasks
                    description=f"Row aggregation for column range {col_range} at PE{agg_pe}",
                    network_id=network_id
                )
                
                # Add aggregation task to the aggregator PE's task list
                pe_computation_tasks[agg_pe] = agg_task_id
    
    # Create output dictionary for PE outputs
    outputs = {}
    
    # Skip row and column aggregation if disabled
    if not self.row_aggregation_enabled or not has_row_splitting:
        # If aggregation is disabled or not needed, include all partial outputs directly
        for pe_coords in layer_pes:
            if pe_coords not in pe_row_groups or pe_coords not in pe_col_groups:
                continue
            
            if pe_coords not in pe_computation_tasks:
                continue
            
            row_range = pe_row_groups[pe_coords]
            col_range = pe_col_groups[pe_coords]
            
            # Create placeholder tensor for this PE's output
            partial_output = torch.zeros((seq_len, col_range[1] - col_range[0]))
            
            outputs[pe_coords] = (
                partial_output,
                ((0, seq_len), col_range),  # Row and column ranges
                pe_computation_tasks[pe_coords]
            )
            
        return outputs
    
    # Step 4: Column aggregation - Concatenate results from all column groups
    # The rest of the function remains the same
    
    # Skip column aggregation if disabled
    if not hasattr(self, 'column_aggregation_enabled') or self.column_aggregation_enabled:
        # Determine which PE to use for final column aggregation
        col_aggregation_pe = None
        
        # Check if we have a dedicated column aggregation PE
        if hasattr(self, 'column_aggregation_pes') and 0 in self.column_aggregation_pes:
            col_aggregation_pe = self.column_aggregation_pes[0]
        elif self.reuse_pe_for_aggregation and pe_computation_tasks:
            # Reuse the first PE with a computation task
            first_pe = next(iter(pe_computation_tasks.keys()))
            col_aggregation_pe = first_pe
        
        if col_aggregation_pe is None or col_aggregation_pe not in self.active_pes:
            # Can't do column aggregation without a valid PE
            # Just return the row-aggregated results
            outputs = {}
            
            for pe_coords in layer_pes:
                if pe_coords not in pe_row_groups or pe_coords not in pe_col_groups:
                    continue
                
                if pe_coords not in pe_computation_tasks:
                    continue
                
                row_range = pe_row_groups[pe_coords]
                col_range = pe_col_groups[pe_coords]
                
                # Create placeholder tensor for this PE's output
                partial_output = torch.zeros((seq_len, col_range[1] - col_range[0]))
                
                outputs[pe_coords] = (
                    partial_output,
                    ((0, seq_len), col_range),  # Row and column ranges
                    pe_computation_tasks[pe_coords]
                )
            
            return outputs
        
        # Each PE (except the column aggregation PE if reusing)
        # sends its result to the column aggregation PE
        col_agg_dependencies = []  # Track dependencies specifically for column aggregation
        
        # If reusing a PE for column aggregation, include its own task as a dependency first
        if pe_computation_tasks.get(col_aggregation_pe):
            col_agg_dependencies.append(pe_computation_tasks[col_aggregation_pe])
        
        # Now create the transfer tasks from other PEs
        transfer_task_ids = []  # Store the transfer task IDs separately
        for pe_coords, task_id in pe_computation_tasks.items():
            if pe_coords == col_aggregation_pe:
                # Skip if we're using the same PE (already handled above)
                continue
            
            col_range = pe_col_groups.get(pe_coords)
            if not col_range:
                continue
                
            # Send results for this column group to the column aggregation PE
            transfer_task_id = self.noc.scheduler.create_task(
                src_pe=pe_coords,
                dest_pe=col_aggregation_pe,
                tensor_shape=(seq_len, col_range[1] - col_range[0]),
                wait_ids=[task_id],  # Wait for computation to complete
                description=f"Send results for columns {col_range[0]}:{col_range[1]} from PE{pe_coords} to column aggregation PE{col_aggregation_pe}",
                network_id=network_id
            )
            
            transfer_task_ids.append(transfer_task_id)
        
        # Add the transfer tasks to the column aggregation dependencies
        col_agg_dependencies.extend(transfer_task_ids)
        
        # Final column concatenation task
        col_agg_task_id = None
        if col_agg_dependencies:
            # Use a set to remove any duplicate dependencies
            col_agg_dependencies = list(set(col_agg_dependencies))
            
            col_agg_task_id = self.noc.scheduler.create_task(
                src_pe=col_aggregation_pe,
                dest_pe=col_aggregation_pe,
                tensor_shape=(seq_len, self.layer_dims[0]),  # Full output dimension
                wait_ids=col_agg_dependencies,
                description=f"Column aggregation (final output) at PE{col_aggregation_pe}",
                network_id=network_id
            )
        
        # Include outputs from each computation PE
        for pe_coords in layer_pes:
            if pe_coords not in pe_row_groups or pe_coords not in pe_col_groups:
                continue
            
            if pe_coords not in pe_computation_tasks:
                continue
            
            row_range = pe_row_groups[pe_coords]
            col_range = pe_col_groups[pe_coords]
            
            # Create placeholder tensor for this PE's output
            partial_output = torch.zeros((seq_len, col_range[1] - col_range[0]))
            
            outputs[pe_coords] = (
                partial_output,
                ((0, seq_len), col_range),  # Row and column ranges
                pe_computation_tasks[pe_coords]
            )
        
        # Include the final aggregated output
        if col_agg_task_id is not None:
            # Create placeholder tensor for the final output
            final_output = torch.zeros((seq_len, self.layer_dims[0]))
            
            outputs[col_aggregation_pe] = (
                final_output,
                ((0, seq_len), (0, self.layer_dims[0])),  # Full output range
                col_agg_task_id
            )
    else:
        # If column aggregation is disabled, include row-aggregated results directly
        outputs = {}
        
        # Iterate over aggregation PEs directly
        for pe_coords in aggregation_pes:
            pe = self.noc.get_pe(*pe_coords)
            
            # Get ranges directly from the PE
            if not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or pe.col_start is None or pe.col_end is None:
                continue
                
            col_range = (pe.col_start, pe.col_end)
            
            # Create placeholder tensor for this PE's output
            partial_output = torch.zeros((seq_len, col_range[1] - col_range[0]))
            
            outputs[pe_coords] = (
                partial_output,
                ((0, seq_len), col_range),  # Row and column ranges
                pe_computation_tasks.get(pe_coords)  # Use get() since it might not exist
            )
            
    return outputs

def _run_matrix_multiply(self, input_a, input_b, transpose_b, source_pe_a, source_pe_b, 
                     strategy="column_split", source_range_a=None, source_range_b=None, 
                     source_task_ids_a=None, source_task_ids_b=None):
    """
    Run matrix multiplication with the specified strategy.
    
    Args:
        input_a: First input tensor (e.g., Q matrix)
        input_b: Second input tensor (e.g., K matrix)
        transpose_b: Whether to transpose the second input (True for Q @ K^T)
        source_pe_a: Source PE(s) for first input, can be a single PE or a list of PEs
        source_pe_b: Source PE(s) for second input, can be a single PE or a list of PEs
        strategy: Strategy for matrix multiplication ("column_split", "row_split", or "hybrid_split")
        source_range_a: Range of first input in source PE(s), can be a single range or a list of ranges
        source_range_b: Range of second input in source PE(s), can be a single range or a list of ranges
        source_task_ids_a: Task IDs to wait for first input, can be a single ID or a list of IDs
        source_task_ids_b: Task IDs to wait for second input, can be a single ID or a list of IDs
    
    Returns:
        Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples
    """
    # Get PE assignments from the mapper
    layer_pes = [pe for pe in self.mapper.get_layer_pes(0) if pe in self.active_pes]
    
    # Ensure that source_pe_a and source_pe_b are lists
    if not isinstance(source_pe_a, list):
        source_pe_a = [source_pe_a]
    
    if not isinstance(source_pe_b, list):
        source_pe_b = [source_pe_b]
    
    # Ensure that source_range_a and source_range_b are lists matching the source_pe lists
    if source_range_a is None:
        source_range_a = [None] * len(source_pe_a)
    elif not isinstance(source_range_a, list):
        source_range_a = [source_range_a]
    
    # Pad source_range_a if needed
    if len(source_range_a) < len(source_pe_a):
        source_range_a.extend([None] * (len(source_pe_a) - len(source_range_a)))
    
    if source_range_b is None:
        source_range_b = [None] * len(source_pe_b)
    elif not isinstance(source_range_b, list):
        source_range_b = [source_range_b]
    
    # Pad source_range_b if needed
    if len(source_range_b) < len(source_pe_b):
        source_range_b.extend([None] * (len(source_pe_b) - len(source_range_b)))
    
    # Ensure that source_task_ids_a and source_task_ids_b are lists matching the source_pe lists
    if source_task_ids_a is None:
        source_task_ids_a = [None] * len(source_pe_a)
    elif not isinstance(source_task_ids_a, list):
        source_task_ids_a = [source_task_ids_a]
    
    # Pad source_task_ids_a if needed
    if len(source_task_ids_a) < len(source_pe_a):
        source_task_ids_a.extend([None] * (len(source_pe_a) - len(source_task_ids_a)))
    
    if source_task_ids_b is None:
        source_task_ids_b = [None] * len(source_pe_b)
    elif not isinstance(source_task_ids_b, list):
        source_task_ids_b = [source_task_ids_b]
    
    # Pad source_task_ids_b if needed
    if len(source_task_ids_b) < len(source_pe_b):
        source_task_ids_b.extend([None] * (len(source_pe_b) - len(source_task_ids_b)))
    
    # Distribute both inputs in a single call
    pe_to_dependencies_map = {}
    
    # First distribute input_a (row-wise) from all source PEs
    row_deps = {}
    for idx, src_pe in enumerate(source_pe_a):
        src_range = source_range_a[idx] if idx < len(source_range_a) else None
        src_task_id = source_task_ids_a[idx] if idx < len(source_task_ids_a) else None
        
        # Distribute from this source PE
        deps = self._distribute_arithmetic_solo(
            input=input_a,
            source_pe=src_pe,
            transpose=False,
            source_range=src_range,
            source_task_ids=src_task_id,
            dimension="row"
        )
        
        # Merge with existing row dependencies
        for pe_coords, task_ids in deps.items():
            if pe_coords not in row_deps:
                row_deps[pe_coords] = []
            row_deps[pe_coords].extend(task_ids)
    
    # Then distribute input_b (column-wise) from all source PEs
    col_deps = {}
    for idx, src_pe in enumerate(source_pe_b):
        src_range = source_range_b[idx] if idx < len(source_range_b) else None
        src_task_id = source_task_ids_b[idx] if idx < len(source_task_ids_b) else None
        
        # Distribute from this source PE
        deps = self._distribute_arithmetic_solo(
            input=input_b,
            source_pe=src_pe,
            transpose=transpose_b,
            source_range=src_range,
            source_task_ids=src_task_id,
            dimension="column"
        )
        
        # Merge with existing column dependencies
        for pe_coords, task_ids in deps.items():
            if pe_coords not in col_deps:
                col_deps[pe_coords] = []
            col_deps[pe_coords].extend(task_ids)
    
    # Merge the two dependency maps
    for pe_coords in set(row_deps.keys()).union(col_deps.keys()):
        pe_to_dependencies_map[pe_coords] = []
        if pe_coords in row_deps:
            pe_to_dependencies_map[pe_coords].extend(row_deps[pe_coords])
        if pe_coords in col_deps:
            pe_to_dependencies_map[pe_coords].extend(col_deps[pe_coords])
    
    # Create computation tasks that wait for both inputs
    pe_computation_tasks = {}
    
    for pe_coords in layer_pes:
        pe = self.noc.get_pe(*pe_coords)
        
        # Skip PEs without proper splits defined
        if strategy == "column_split":
            if not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or pe.col_start is None or pe.col_end is None:
                continue
                
            # Output tensor shape for column-split
            tensor_shape = (self.seq_len, pe.col_end - pe.col_start)
            description = f"Matrix multiply at PE{pe_coords} (output cols {pe.col_start}:{pe.col_end})"
            
        elif strategy == "row_split":
            if not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or pe.row_start is None or pe.row_end is None:
                continue
                
            # Output tensor shape for row-split
            tensor_shape = (pe.row_end - pe.row_start, self.seq_len)
            description = f"Matrix multiply at PE{pe_coords} (output rows {pe.row_start}:{pe.row_end})"
            
        elif strategy == "hybrid_split":
            if (not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or 
                not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or
                pe.row_start is None or pe.row_end is None or
                pe.col_start is None or pe.col_end is None):
                continue
                
            # Output tensor shape for hybrid-split
            tensor_shape = (pe.row_end - pe.row_start, pe.col_end - pe.col_start)
            description = f"Matrix multiply at PE{pe_coords} (output block {pe.row_start}:{pe.row_end}, {pe.col_start}:{pe.col_end})"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # This PE should wait for both input distributions
        wait_ids = pe_to_dependencies_map.get(pe_coords, [])
        
        # Create computation task
        compute_task_id = self.noc.scheduler.create_task(
            src_pe=pe_coords,
            dest_pe=pe_coords,
            tensor_shape=tensor_shape,
            wait_ids=wait_ids,
            description=description,
            network_id=None
        )
        
        pe_computation_tasks[pe_coords] = compute_task_id
    
    # Create output dictionary
    outputs = {}
    
    for pe_coords in layer_pes:
        pe = self.noc.get_pe(*pe_coords)
        
        if strategy == "column_split":
            if not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or pe.col_start is None or pe.col_end is None:
                continue
                
            # Create placeholder tensor for this PE's portion of output
            output = torch.zeros((self.seq_len, pe.col_end - pe.col_start))
            output_range = ((0, self.seq_len), (pe.col_start, pe.col_end))
            
        elif strategy == "row_split":
            if not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or pe.row_start is None or pe.row_end is None:
                continue
                
            # Create placeholder tensor for this PE's portion of output
            output = torch.zeros((pe.row_end - pe.row_start, self.seq_len))
            output_range = ((pe.row_start, pe.row_end), (0, self.seq_len))
            
        elif strategy == "hybrid_split":
            if (not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or 
                not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or
                pe.row_start is None or pe.row_end is None or
                pe.col_start is None or pe.col_end is None):
                continue
                
            # Create placeholder tensor for this PE's portion of output
            output = torch.zeros((pe.row_end - pe.row_start, pe.col_end - pe.col_start))
            output_range = ((pe.row_start, pe.row_end), (pe.col_start, pe.col_end))
        else:
            continue
        
        outputs[pe_coords] = (
            output,
            output_range,
            pe_computation_tasks.get(pe_coords)
        )
    
    return outputs

def _distribute_arithmetic(self, 
                          input_a: torch.Tensor,
                          input_b: torch.Tensor,
                          source_pe_a: Union[Tuple[int, int], List[Tuple[int, int]]],
                          source_pe_b: Union[Tuple[int, int], List[Tuple[int, int]]],
                          transpose_b: bool = True,
                          source_range_a=None,
                          source_range_b=None,
                          source_task_ids_a=None,
                          source_task_ids_b=None) -> Dict[Tuple[int, int], List[str]]:
    """
    Distribute two input matrices to PEs for arithmetic operations.
    Handles row/column matching logic for both inputs based on PE assignments.
    
    Args:
        input_a: First input tensor (e.g., Q matrix)
        input_b: Second input tensor (e.g., K matrix)
        source_pe_a: Source PE(s) for first input
        source_pe_b: Source PE(s) for second input
        transpose_b: Whether to transpose the second input (default True for Q @ K^T)
        source_range_a: Range of first input in source PE(s) in format ((row_start, row_end), (col_start, col_end))
        source_range_b: Range of second input in source PE(s) in format ((row_start, row_end), (col_start, col_end))
        source_task_ids_a: Task IDs to wait for first input
        source_task_ids_b: Task IDs to wait for second input
    
    Returns:
        Dictionary mapping PE coordinates to list of task IDs for both inputs
    """
    # Get network_id if this network is part of an LLM
    network_id = None
    if hasattr(self, 'llm') and self.llm is not None:
        for idx, network in enumerate(self.llm.networks.values()):
            if network is self:
                network_id = idx
                break
    
    # Get active PEs for this layer
    layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(0)
                if pe_coords in self.active_pes]
    
    if not layer_pes:
        raise ValueError("No active PEs found for the layer")
    
    # Initialize dependency map for all PEs
    pe_to_dependencies_map = {pe_coords: [] for pe_coords in layer_pes}
    
    # Set default source ranges if not provided
    if source_range_a is None:
        source_range_a = ((0, self.seq_len), (0, self.d_model))
    elif isinstance(source_range_a, tuple) and len(source_range_a) == 2 and not isinstance(source_range_a[0], tuple):
        # Convert (col_start, col_end) format to ((row_start, row_end), (col_start, col_end)) format
        source_range_a = ((0, self.seq_len), source_range_a)
    
    if source_range_b is None:
        source_range_b = ((0, self.seq_len), (0, self.d_model))
    elif isinstance(source_range_b, tuple) and len(source_range_b) == 2 and not isinstance(source_range_b[0], tuple):
        # Convert (col_start, col_end) format to ((row_start, row_end), (col_start, col_end)) format
        source_range_b = ((0, self.seq_len), source_range_b)
    
    # Ensure source_task_ids_a and source_task_ids_b are lists or None
    if source_task_ids_a is not None and not isinstance(source_task_ids_a, list):
        source_task_ids_a = [source_task_ids_a]
    if source_task_ids_b is not None and not isinstance(source_task_ids_b, list):
        source_task_ids_b = [source_task_ids_b]
    
    # Unpack source ranges
    (src_row_start_a, src_row_end_a), (src_col_start_a, src_col_end_a) = source_range_a
    (src_row_start_b, src_row_end_b), (src_col_start_b, src_col_end_b) = source_range_b
    
    # Handle first input (input_a) distribution
    # Each PE needs rows of input_a that match its row assignment
    for pe_coords in layer_pes:
        pe = self.noc.get_pe(*pe_coords)
        # Skip PEs without proper row information
        if not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or pe.row_start is None or pe.row_end is None:
            continue
        
        # Calculate overlap between PE's row range and source's available rows
        overlap_row_start = max(pe.row_start, src_row_start_a)
        overlap_row_end = min(pe.row_end, src_row_end_a)
        
        # The source might only have a portion of columns in the d_model dimension
        col_width = src_col_end_a - src_col_start_a

        if overlap_row_end > overlap_row_start:
            # Create task to send overlapping portion of input_a
            task_id = self.noc.scheduler.create_task(
                src_pe=source_pe_a,
                dest_pe=pe_coords,
                tensor_shape=(overlap_row_end - overlap_row_start, col_width),
                wait_ids=source_task_ids_a if source_task_ids_a else [],
                description=f"Send rows {overlap_row_start}:{overlap_row_end} of input_a to PE{pe_coords}",
                network_id=network_id
            )
            pe_to_dependencies_map[pe_coords].append(task_id)
    
    # Handle second input (input_b) distribution
    # Distribution depends on transpose_b flag
    for pe_coords in layer_pes:
        pe = self.noc.get_pe(*pe_coords)
        
        # Skip PEs without proper column information
        if not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or pe.col_start is None or pe.col_end is None:
            continue
        
        if transpose_b:
            # When transposed, we need rows of input_b that match PE's column assignment
            # These rows will become columns after transpose
            # Compare PE's column range with source's available rows
            overlap_row_start = max(pe.col_start, src_row_start_b)
            overlap_row_end = min(pe.col_end, src_row_end_b)
            
            # The source might only have a portion of columns in the d_model dimension
            col_width = src_col_end_b - src_col_start_b
            
            if overlap_row_end > overlap_row_start:
                # Create task to send overlapping rows of input_b
                task_id = self.noc.scheduler.create_task(
                    src_pe=source_pe_b,
                    dest_pe=pe_coords,
                    tensor_shape=(overlap_row_end - overlap_row_start, col_width),
                    wait_ids=source_task_ids_b if source_task_ids_b else [],
                    description=f"Send rows {overlap_row_start}:{overlap_row_end} of input_b (for columns after transpose) to PE{pe_coords}",
                    network_id=network_id
                )
                pe_to_dependencies_map[pe_coords].append(task_id)
        else:
            # When not transposed, we need columns of input_b that match PE's column assignment
            # Compare PE's column range with source's available columns
            overlap_col_start = max(pe.col_start, src_col_start_b)
            overlap_col_end = min(pe.col_end, src_col_end_b)
            
            # The source might only have a portion of rows in the seq_len dimension
            row_height = src_row_end_b - src_row_start_b
            
            if overlap_col_end > overlap_col_start:
                # Create task to send overlapping columns of input_b
                task_id = self.noc.scheduler.create_task(
                    src_pe=source_pe_b,
                    dest_pe=pe_coords,
                    tensor_shape=(row_height, overlap_col_end - overlap_col_start),
                    wait_ids=source_task_ids_b if source_task_ids_b else [],
                    description=f"Send columns {overlap_col_start}:{overlap_col_end} of input_b to PE{pe_coords}",
                    network_id=network_id
                )
                pe_to_dependencies_map[pe_coords].append(task_id)
    
    return pe_to_dependencies_map 

def _distribute_arithmetic_solo(self, 
                          input: torch.Tensor,
                          source_pe: Union[Tuple[int, int], List[Tuple[int, int]]],
                          transpose: bool = True,
                          source_range=None,
                          source_task_ids=None,
                          dimension="row") -> Dict[Tuple[int, int], List[str]]:
    """
    Distribute input tensor to PEs for arithmetic operations.
    Handles row/column distribution logic based on PE assignments.
    
    Args:
        input: Input tensor to distribute
        source_pe: Source PE(s) for the input, can be a single PE or a list of PEs
        transpose: Whether to transpose the input (affects distribution logic)
        source_range: Range of input in source PE(s) in format ((row_start, row_end), (col_start, col_end))
                     or a list of such tuples, one per source_pe
        source_task_ids: Task IDs to wait for input
        dimension: The dimension to distribute the input over ("row" or "column")
    
    Returns:
        Dictionary mapping PE coordinates to list of task IDs for input
    """
    # Get network_id if this network is part of an LLM
    network_id = None
    if hasattr(self, 'llm') and self.llm is not None:
        for idx, network in enumerate(self.llm.networks.values()):
            if network is self:
                network_id = idx
                break
    
    # Get active PEs for this layer
    layer_pes = [pe_coords for pe_coords in self.mapper.get_layer_pes(0)
                if pe_coords in self.active_pes]
    
    if not layer_pes:
        raise ValueError("No active PEs found for the layer")
    
    # Initialize dependency map for all PEs
    pe_to_dependencies_map = {pe_coords: [] for pe_coords in layer_pes}
    
    # Ensure source_pe is a tuple (not a list)
    if isinstance(source_pe, list) and len(source_pe) > 0:
        source_pe = source_pe[0]  # Take the first PE from the list
    
    # Handle source_range - ensure it's a single range tuple, not a list
    if isinstance(source_range, list) and len(source_range) > 0:
        source_range = source_range[0]  # Take the first range from the list
    
    # Set default source ranges if not provided
    if source_range is None:
        source_range = ((0, self.seq_len), (0, self.d_model))
    elif isinstance(source_range, tuple) and len(source_range) == 2 and not isinstance(source_range[0], tuple):
        # Convert (col_start, col_end) format to ((row_start, row_end), (col_start, col_end)) format
        source_range = ((0, self.seq_len), source_range)
    
    # Ensure source_task_ids is a scalar (not a list)
    if isinstance(source_task_ids, list) and len(source_task_ids) > 0:
        source_task_ids = source_task_ids[0]  # Take the first task ID from the list
    
    # Unpack source ranges
    (src_row_start, src_row_end), (src_col_start, src_col_end) = source_range
    
    for pe_coords in layer_pes:
        pe = self.noc.get_pe(*pe_coords)
        
        if dimension == "row":
            # Skip PEs without proper row information
            if not hasattr(pe, 'row_start') or not hasattr(pe, 'row_end') or pe.row_start is None or pe.row_end is None:
                continue
            
            overlap_row_start = max(pe.row_start, src_row_start)
            overlap_row_end = min(pe.row_end, src_row_end)
            col_width = src_col_end - src_col_start
            
            if transpose:
                overlap_row_start = max(pe.row_start, src_col_start)
                overlap_row_end = min(pe.row_end, src_col_end)
                
                # The source might only have a portion of columns in the d_model dimension
                col_width = src_row_end - src_row_start
            
            if overlap_row_end > overlap_row_start:
                # Create task to send overlapping rows of input
                task_id = self.noc.scheduler.create_task(
                    src_pe=source_pe,
                    dest_pe=pe_coords,
                    tensor_shape=(overlap_row_end - overlap_row_start, col_width),
                    wait_ids=[source_task_ids] if source_task_ids else [],
                    description=f"{source_pe} sends ({overlap_row_start}:{overlap_row_end}, {src_col_start}:{src_col_end}) of input to PE{pe_coords}",
                    network_id=network_id
                )
                pe_to_dependencies_map[pe_coords].append(task_id)
        
        else:  # dimension == "column"
            # Skip PEs without proper column information
            if not hasattr(pe, 'col_start') or not hasattr(pe, 'col_end') or pe.col_start is None or pe.col_end is None:
                continue
            if not transpose:
                overlap_col_start = max(pe.col_start, src_col_start)
                overlap_col_end = min(pe.col_end, src_col_end)
            
                # The source might only have a portion of rows in the seq_len dimension
                row_height = src_row_end - src_row_start

                if overlap_col_end > overlap_col_start:
                # Create task to send overlapping columns of input
                    task_id = self.noc.scheduler.create_task(
                        src_pe=source_pe,
                        dest_pe=pe_coords,
                        tensor_shape=(row_height, overlap_col_end - overlap_col_start),
                        wait_ids=[source_task_ids] if source_task_ids else [],
                        description=f"{source_pe} sends ({src_row_start}:{src_row_end}, {overlap_col_start}:{overlap_col_end}) of input to PE{pe_coords}",
                        network_id=network_id
                    )
                    pe_to_dependencies_map[pe_coords].append(task_id)
        
            else:
                overlap_col_start = max(pe.col_start, src_row_start)
                overlap_col_end = min(pe.col_end, src_row_end)
                
                # The source might only have a portion of columns in the d_model dimension
                row_height = src_col_end - src_col_start
                
                if overlap_col_end > overlap_col_start:
                    # Create task to send overlapping columns of input
                    task_id = self.noc.scheduler.create_task(
                        src_pe=source_pe,
                        dest_pe=pe_coords,
                        tensor_shape=(row_height, overlap_col_end - overlap_col_start),
                        wait_ids=[source_task_ids] if source_task_ids else [],
                        description=f"{source_pe} sends ({src_col_start}:{src_col_end}, {overlap_col_start}:{overlap_col_end}) of input Transposed to PE{pe_coords}",
                        network_id=network_id
                    )
                    pe_to_dependencies_map[pe_coords].append(task_id)
        
    return pe_to_dependencies_map

def _run_attention_computation(self, Q, K, V, 
                              source_pe_q, source_pe_k, source_pe_v,
                              source_range_q=None, source_range_k=None, source_range_v=None,
                              source_task_ids_q=None, source_task_ids_k=None, source_task_ids_v=None):
    """
    Run the complete attention computation: (Q·K^T)·V
    
    This method performs the attention computation in two steps:
    1. First calculate attention scores: Q·K^T (seq×seq matrix)
    2. Then calculate context vectors: (Q·K^T)·V (seq×d_model matrix)
    
    Args:
        Q: Query tensor (seq×d_model)
        K: Key tensor (seq×d_model)
        V: Value tensor (seq×d_model)
        source_pe_q: Source PE(s) for Q input, single PE or list of PEs
        source_pe_k: Source PE(s) for K input, single PE or list of PEs
        source_pe_v: Source PE(s) for V input, single PE or list of PEs
        source_range_q: Range of Q input in source PE(s), single range or list of ranges
        source_range_k: Range of K input in source PE(s), single range or list of ranges
        source_range_v: Range of V input in source PE(s), single range or list of ranges
        source_task_ids_q: Task IDs to wait for Q input, single ID or list of IDs
        source_task_ids_k: Task IDs to wait for K input, single ID or list of IDs
        source_task_ids_v: Task IDs to wait for V input, single ID or list of IDs
    
    Returns:
        Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples
        for the final context vectors
    """
    # Ensure that source_pe_*, source_range_*, and source_task_ids_* are all lists or None
    # This ensures we handle both single PE and multiple PE cases consistently
    
    # Handle source_pe_q
    if source_pe_q is not None and not isinstance(source_pe_q, list):
        source_pe_q = [source_pe_q]
    
    # Handle source_pe_k
    if source_pe_k is not None and not isinstance(source_pe_k, list):
        source_pe_k = [source_pe_k]
    
    # Handle source_pe_v
    if source_pe_v is not None and not isinstance(source_pe_v, list):
        source_pe_v = [source_pe_v]
    
    # Handle source_range_q
    if source_range_q is not None and not isinstance(source_range_q, list):
        source_range_q = [source_range_q]
    
    # Handle source_range_k
    if source_range_k is not None and not isinstance(source_range_k, list):
        source_range_k = [source_range_k]
    
    # Handle source_range_v
    if source_range_v is not None and not isinstance(source_range_v, list):
        source_range_v = [source_range_v]
    
    # Handle source_task_ids_q
    if source_task_ids_q is not None and not isinstance(source_task_ids_q, list):
        source_task_ids_q = [source_task_ids_q]
    
    # Handle source_task_ids_k
    if source_task_ids_k is not None and not isinstance(source_task_ids_k, list):
        source_task_ids_k = [source_task_ids_k]
    
    # Handle source_task_ids_v
    if source_task_ids_v is not None and not isinstance(source_task_ids_v, list):
        source_task_ids_v = [source_task_ids_v]
    
    # Step 1: Calculate attention scores (Q·K^T)
    # This produces a seq×seq matrix
    attention_scores = self._run_matrix_multiply(
        input_a=Q, 
        input_b=K, 
        transpose_b=True,  # Transpose K to calculate Q·K^T
        source_pe_a=source_pe_q,
        source_pe_b=source_pe_k,
        strategy=self.split_strategy,  # Use hybrid split for the large seq×seq matrix
        source_range_a=source_range_q,
        source_range_b=source_range_k,
        source_task_ids_a=source_task_ids_q,
        source_task_ids_b=source_task_ids_k
    )
    
    # Step 2: Calculate context vectors by multiplying attention scores with V
    # Get active PEs for this layer
    layer_pes = [pe for pe in self.mapper.get_layer_pes(0) if pe in self.active_pes]
    
    # Initialize output dictionary
    context_vector_outputs = {}
    
    
    network_id = None
    if hasattr(self, 'llm') and self.llm is not None:
        for idx, network in enumerate(self.llm.networks.values()):
            if network is self:
                network_id = idx
                break
    # Simplified approach: For each PE with attention scores, 
    # send the relevant rows of V that match the columns of the attention matrix
    for pe_coords, (attn_score_tensor, attn_score_range, attn_score_task_id) in attention_scores.items():
        # Skip if this PE doesn't have attention scores
        if attn_score_task_id is None:
            continue
        
        # Get the row range for this PE's attention scores
        (attn_row_start, attn_row_end), (attn_col_start, attn_col_end) = attn_score_range
        
        # For attention computation, we need V rows that match the columns of the attention matrix
        # The columns of attention matrix (attn_col_start:attn_col_end) correspond to rows of V
        v_row_start, v_row_end = attn_col_start, attn_col_end
        
        # Send the relevant rows of V to this PE
        v_task_ids = []
        
        # Track the columns of V that are actually sent to this PE
        v_col_ranges = []
        
        # Process all source PEs for V
        for v_src_idx, v_src_pe in enumerate(source_pe_v):
            v_src_range = source_range_v[v_src_idx] if v_src_idx < len(source_range_v) else None
            v_src_task_id = source_task_ids_v[v_src_idx] if v_src_idx < len(source_task_ids_v) else None
            
            # Default full range if not specified
            if v_src_range is None:
                v_src_range = ((0, V.shape[0]), (0, V.shape[1]))
            elif isinstance(v_src_range, tuple) and len(v_src_range) == 2 and not isinstance(v_src_range[0], tuple):
                # Convert (col_start, col_end) format to ((row_start, row_end), (col_start, col_end)) format
                v_src_range = ((0, V.shape[0]), v_src_range)
            
            # Extract source range
            (v_src_row_start, v_src_row_end), (v_src_col_start, v_src_col_end) = v_src_range
            
            # Calculate overlap between needed V rows and available rows from this source
            overlap_row_start = max(v_row_start, v_src_row_start)
            overlap_row_end = min(v_row_end, v_src_row_end)
            
            # Only distribute if there's an overlap
            if overlap_row_end > overlap_row_start:
                # Send all columns of V for the overlapping rows
                v_cols = v_src_col_end - v_src_col_start
                
                # Create task to send the overlapping rows of V
                task_id = self.noc.scheduler.create_task(
                    src_pe=v_src_pe,
                    dest_pe=pe_coords,
                    tensor_shape=(overlap_row_end - overlap_row_start, v_cols),
                    wait_ids=[v_src_task_id] if v_src_task_id else [],
                    description=f"Send V ({overlap_row_start}:{overlap_row_end}, {v_src_col_start}:{v_src_col_end}) from PE{v_src_pe} to PE{pe_coords}",
                    network_id=network_id
                )
                v_task_ids.append(task_id)
                
                # Track this column range that was actually sent to the PE
                v_col_ranges.append((v_src_col_start, v_src_col_end))
        
        # If we didn't send any V tensors, set a default range
        if not v_col_ranges:
            total_v_cols = 0
            v_total_col_start = 0
            v_total_col_end = 0
        else:
            # Find the full column range that was sent to this PE by merging all ranges
            # First sort ranges by their start position
            v_col_ranges.sort()
            
            # Initialize with the first range
            v_total_col_start, v_total_col_end = v_col_ranges[0]
            total_v_cols = v_total_col_end - v_total_col_start
            
            # Merge with subsequent ranges
            for col_start, col_end in v_col_ranges[1:]:
                # If this range overlaps or is adjacent to the current total range,
                # extend the total range
                if col_start <= v_total_col_end:
                    v_total_col_end = max(v_total_col_end, col_end)
                    total_v_cols = v_total_col_end - v_total_col_start
                else:
                    # Handle non-contiguous ranges - this is a simplification
                    # In a real implementation, we'd need to handle multiple separate ranges
                    # For now, we'll just extend the range to include everything
                    total_v_cols += (col_end - col_start)
                    v_total_col_end = max(v_total_col_end, col_end)
        
        # Dependencies for context vector computation: attention scores and V rows
        dependencies = [attn_score_task_id] + v_task_ids
        
        # Create computation task for final attention output
        # Output shape is [attn_rows × V_cols] where V_cols is derived from the actual column ranges sent
        context_tensor_shape = (attn_row_end - attn_row_start, total_v_cols)
        
        context_task_id = self.noc.scheduler.create_task(
            src_pe=pe_coords,
            dest_pe=pe_coords,
            tensor_shape=context_tensor_shape,
            wait_ids=dependencies,
            description=f"Attention computation ((Q·K^T)·V) at PE{pe_coords} ({attn_row_start}:{attn_row_end}, {v_total_col_start}:{v_total_col_end})",
            network_id=network_id
        )
        
        # Create placeholder tensor for this PE's portion of context vectors
        context_tensor = torch.zeros(context_tensor_shape)
        
        # Store in output dictionary with the correct column range
        context_vector_outputs[pe_coords] = (
            context_tensor,
            ((attn_row_start, attn_row_end), (v_total_col_start, v_total_col_end)),
            context_task_id
        )
    
    return context_vector_outputs

def attention_computation(self, 
                       Q: Union[torch.Tensor, Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]],
                       K: Union[torch.Tensor, Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]],
                       V: Union[torch.Tensor, Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]],
                       source_pe_q=None,
                       source_pe_k=None,
                       source_pe_v=None) -> Dict[Tuple[int, int], Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int]], Optional[str]]]:
    """
    Perform distributed attention computation: (Q·K^T)·V.
    This calculates attention scores and then applies them to the value matrix.
    
    Args:
        Q: Query tensor [seq_len × d_model] or dictionary of PE outputs from an FC network
        K: Key tensor [seq_len × d_model] or dictionary of PE outputs from an FC network
        V: Value tensor [seq_len × d_model] or dictionary of PE outputs from an FC network
        source_pe_q: Source PE for query input (only used if Q is a tensor)
        source_pe_k: Source PE for key input (only used if K is a tensor)
        source_pe_v: Source PE for value input (only used if V is a tensor)
        
    Returns:
        Dictionary mapping PE coordinates to (output_tensor, output_range, task_id) tuples
    """
    # Initialize default values for source parameters
    source_range_q = None
    source_range_k = None
    source_range_v = None
    source_task_ids_q = None
    source_task_ids_k = None
    source_task_ids_v = None
    
    # Handle Q in FC network output format
    if isinstance(Q, dict):
        # Create lists to hold all source PEs, ranges, and task IDs
        q_pes = []
        q_ranges = []
        q_task_ids = []
        q_tensor = None
        
        # Process all PEs in the dictionary
        for pe_q, (tensor_q, range_q, task_id_q) in Q.items():
            q_pes.append(pe_q)
            q_ranges.append(range_q)
            q_task_ids.append(task_id_q)
            
            # For now, use the first tensor to determine shape
            # The full reconstruction will be handled by _run_attention_computation
            if q_tensor is None:
                q_tensor = tensor_q
        
        # Update source parameters with lists of all PEs, ranges, and task IDs
        source_pe_q = q_pes
        source_range_q = q_ranges
        source_task_ids_q = q_task_ids
        
        # Use the tensor from the first PE for shape checking
        Q = q_tensor
    
    # Handle K in FC network output format
    if isinstance(K, dict):
        # Create lists to hold all source PEs, ranges, and task IDs
        k_pes = []
        k_ranges = []
        k_task_ids = []
        k_tensor = None
        
        # Process all PEs in the dictionary
        for pe_k, (tensor_k, range_k, task_id_k) in K.items():
            k_pes.append(pe_k)
            k_ranges.append(range_k)
            k_task_ids.append(task_id_k)
            
            # For now, use the first tensor to determine shape
            if k_tensor is None:
                k_tensor = tensor_k
        
        # Update source parameters with lists of all PEs, ranges, and task IDs
        source_pe_k = k_pes
        source_range_k = k_ranges
        source_task_ids_k = k_task_ids
        
        # Use the tensor from the first PE for shape checking
        K = k_tensor
    
    # Handle V in FC network output format
    if isinstance(V, dict):
        # Create lists to hold all source PEs, ranges, and task IDs
        v_pes = []
        v_ranges = []
        v_task_ids = []
        v_tensor = None
        
        # Process all PEs in the dictionary
        for pe_v, (tensor_v, range_v, task_id_v) in V.items():
            v_pes.append(pe_v)
            v_ranges.append(range_v)
            v_task_ids.append(task_id_v)
            
            # For now, use the first tensor to determine shape
            if v_tensor is None:
                v_tensor = tensor_v
        
        # Update source parameters with lists of all PEs, ranges, and task IDs
        source_pe_v = v_pes
        source_range_v = v_ranges
        source_task_ids_v = v_task_ids
        
        # Use the tensor from the first PE for shape checking
        V = v_tensor
    
    # Set default source PEs if not provided
    if source_pe_q is None:
        # Use standardized external PE (0,0) for Q input
        source_pe_q = (0, 0)  # External PE
            
    if source_pe_k is None:
        # Use standardized external PE (0,0) for K input
        source_pe_k = (0, 0)  # External PE
    
    if source_pe_v is None:
        # Use standardized external PE (0,0) for V input
        source_pe_v = (0, 0)  # External PE
    
    # Verify tensor dimensions
    # Q and K should have the same dimension for their columns
    if Q.shape[1] != K.shape[1]:
        raise ValueError("Column dimension of Q must be equal to column dimension of K")
    
    # K and V should have the same sequence length
    if K.shape[0] != V.shape[0]:
        raise ValueError("Row dimension of K must be equal to row dimension of V")
    
    # Compute attention using the internal implementation
    return self._run_attention_computation(
        Q=Q, K=K, V=V,
        source_pe_q=source_pe_q,
        source_pe_k=source_pe_k,
        source_pe_v=source_pe_v,
        source_range_q=source_range_q,
        source_range_k=source_range_k,
        source_range_v=source_range_v,
        source_task_ids_q=source_task_ids_q,
        source_task_ids_k=source_task_ids_k,
        source_task_ids_v=source_task_ids_v
    ) 