import math
import matplotlib.pyplot as plt
import numpy as np
import os

def minimize_pe_count(model_params, hardware_constraints, layer_type="fc", strategy="optimal", execution_mode="parallel"):
    """
    Find splitting strategy that minimizes the number of PEs required for different layer types.
    
    Args:
        model_params: Dictionary containing model parameters
            - batch_size: Batch size (b)
            - seq_length: Sequence length (s)
            - input_dim: Input dimension (d_in)
            - output_dim: Output dimension (d_out)
            - head_dim: Dimension of attention heads (for attention layers)
            - num_heads: Number of attention heads (for attention layers)
            - bytes_per_param: Memory size of each parameter (typically 2 or 4 bytes)
        
        hardware_constraints: Dictionary containing hardware constraints
            - pe_memory: Available memory per PE (bytes)
            - min_dim_size: Minimum dimension size after splitting (for alignment/efficiency)
        
        layer_type: Type of layer to analyze
            - "fc": Fully connected layer
            - "attn": Attention layer (only attention score and context vector computation)
            
        strategy: Splitting strategy to use
            - FC layer strategies:
                - "row_split": Row-major splitting. Tries all possible row splits first, then increases sequence splits if needed.
                - "column_split": Column-major splitting. Tries all possible column splits first, then increases sequence splits if needed.
                - "hybrid_split": Split along both input and output dimensions
                - "sequence_split": Sequence-major splitting. Tries all possible sequence splits first, then adds column and row if needed.
                - "combined": Use row, column and sequence splitting
                - "optimal": Search for optimal configuration (default)
            - Attention layer strategies:
                - "query_sequence_split": Split along query sequence dimension
                - "key_sequence_split": Split along key sequence dimension
                - "head_split": Split along attention heads dimension
                - "combined": Use query, key sequence and head splitting
                - "optimal": Search for optimal configuration (default)
                
        execution_mode: Memory usage calculation mode
            - "parallel": All intermediates coexist in memory (conservative estimate)
            - "pipelined": Only peak memory usage during each stage (realistic for pipelined hardware)
    
    Returns:
        Dictionary containing the optimal splitting strategy
    """
    # Extract model parameters
    b = model_params['batch_size']
    s = model_params['seq_length']
    d_in = model_params['input_dim']
    d_out = model_params['output_dim']
    bytes_per_param = model_params['bytes_per_param']
    
    # For attention layers
    h = model_params.get('num_heads', 1)
    d_h = model_params.get('head_dim', d_out // h if h > 0 else d_out)
    
    # Extract hardware constraints
    pe_memory = hardware_constraints['pe_memory']
    min_dim_size = hardware_constraints.get('min_dim_size', 8)  # Default minimum dimension size
    
    # Memory overhead factor (for workspace, system overhead, etc.)
    memory_overhead_factor = 1.1  # 10% overhead
    
    if layer_type == "fc":
        # Calculate total sizes for FC layer
        weight_size = d_in * d_out * bytes_per_param
        input_act_size = b * s * d_in * bytes_per_param
        output_act_size = b * s * d_out * bytes_per_param
        
        # Function to check if a configuration fits in memory for FC layer
        def fits_in_memory_fc(row_splits, col_splits, seq_splits):
            # Partitioned sizes
            part_weight_size = (d_in / row_splits) * (d_out / col_splits) * bytes_per_param
            part_input_size = b * (s / seq_splits) * (d_in / row_splits) * bytes_per_param
            part_output_size = b * (s / seq_splits) * (d_out / col_splits) * bytes_per_param
            
            # Total memory required per PE
            if execution_mode == "parallel":
                # All intermediate results coexist in memory
                total_memory = (part_weight_size + part_input_size + part_output_size) * memory_overhead_factor
            else:  # pipelined execution
                # Only consider peak memory across pipeline stages
                # Stage 1: Input activations + weights (compute partial sums)
                stage1_memory = part_weight_size + part_input_size
                # Stage 2: Weights + output activations (accumulate partial sums)
                stage2_memory = part_weight_size + part_output_size
                # Use the maximum memory required across stages
                total_memory = max(stage1_memory, stage2_memory) * memory_overhead_factor
            
            # Check if it fits
            return total_memory <= pe_memory and \
                   (d_in / row_splits) >= min_dim_size and \
                   (d_out / col_splits) >= min_dim_size and \
                   (s / seq_splits) >= 1
        
        # Start with smallest possible configuration
        best_config = None
        min_pe_count = float('inf')
        
        # Implement different strategies based on the table
        if strategy == "row_split":
            # Row-major split: Incrementally increase sequence splits and check all row splits for each seq_split value
            for seq_splits in range(1, s + 1):  # Start with seq_splits = 1, increase if needed
                col_splits = 1  # Fixed at 1 for row_split strategy
                found_valid_config = False
                
                # Try all possible row splits for current seq_splits
                for row_splits in range(1, d_in + 1):
                    if fits_in_memory_fc(row_splits, col_splits, seq_splits):
                        pe_count = row_splits * seq_splits
                        
                        if pe_count < min_pe_count:
                            min_pe_count = pe_count
                            best_config = {'row_splits': row_splits, 'column_splits': col_splits, 'sequence_splits': seq_splits, 'pe_count': pe_count}
                            found_valid_config = True
                
                # If we found at least one valid config for this seq_splits value, no need to increase seq_splits further
                if found_valid_config:
                    break
                        
        elif strategy == "column_split":
            # Column-major split: Incrementally increase sequence splits and check all column splits for each seq_split value
            for seq_splits in range(1, s + 1):  # Start with seq_splits = 1, increase if needed
                row_splits = 1  # Fixed at 1 for column_split strategy
                found_valid_config = False
                
                # Try all possible column splits for current seq_splits
                for col_splits in range(1, d_out + 1):
                    if fits_in_memory_fc(row_splits, col_splits, seq_splits):
                        pe_count = col_splits * seq_splits
                        
                        if pe_count < min_pe_count:
                            min_pe_count = pe_count
                            best_config = {'row_splits': row_splits, 'column_splits': col_splits, 'sequence_splits': seq_splits, 'pe_count': pe_count}
                            found_valid_config = True
                
                # If we found at least one valid config for this seq_splits value, no need to increase seq_splits further
                if found_valid_config:
                    break
                        
        elif strategy == "hybrid_split":
            # Hybrid-Split: split along both input and output dimensions (r, c)
            seq_splits = 1  # Fixed at 1
            
            # Search for valid combinations of row_splits and col_splits
            for row_splits in range(1, d_in + 1):
                for col_splits in range(1, d_out + 1):
                    if fits_in_memory_fc(row_splits, col_splits, seq_splits):
                        pe_count = row_splits * col_splits  # PE_hybrid = r × c
                        
                        if pe_count < min_pe_count:
                            min_pe_count = pe_count
                            best_config = {'row_splits': row_splits, 'column_splits': col_splits, 'sequence_splits': seq_splits, 'pe_count': pe_count}
                            
        elif strategy == "sequence_split":
            # Sequence-major split with fallbacks to column and row
            # First try with just increasing sequence splits
            found_valid_config = False
            
            # Phase 1: Try sequence-only splits
            for seq_splits in range(1, s + 1):
                row_splits = 1  # Fixed at 1 
                col_splits = 1   # Fixed at 1
                
                if fits_in_memory_fc(row_splits, col_splits, seq_splits):
                    pe_count = seq_splits
                    
                    if pe_count < min_pe_count:
                        min_pe_count = pe_count
                        best_config = {'row_splits': row_splits, 'column_splits': col_splits, 'sequence_splits': seq_splits, 'pe_count': pe_count}
                        found_valid_config = True
            
            # Phase 2: If no success with sequence-only, try sequence-column splits
            if not found_valid_config:
                for col_splits in range(2, d_out + 1):  # Start from 2 since we already tried with 1
                    row_splits = 1  # Fixed at 1
                    found_in_this_col = False
                    
                    for seq_splits in range(1, s + 1):
                        if fits_in_memory_fc(row_splits, col_splits, seq_splits):
                            pe_count = seq_splits * col_splits
                            
                            if pe_count < min_pe_count:
                                min_pe_count = pe_count
                                best_config = {'row_splits': row_splits, 'column_splits': col_splits, 'sequence_splits': seq_splits, 'pe_count': pe_count}
                                found_valid_config = True
                                found_in_this_col = True
                    
                    # If we found a valid config for this col_splits value, we exit the outer loop
                    if found_in_this_col:
                        break
                        
            # Phase 3: If still no success, try sequence-column-row splits
            if not found_valid_config:
                for row_splits in range(2, d_in + 1):  # Start from 2 since we already tried with 1
                    found_in_this_row = False
                    
                    for col_splits in range(1, d_out + 1):
                        for seq_splits in range(1, s + 1):
                            if fits_in_memory_fc(row_splits, col_splits, seq_splits):
                                pe_count = seq_splits * col_splits * row_splits
                                
                                if pe_count < min_pe_count:
                                    min_pe_count = pe_count
                                    best_config = {'row_splits': row_splits, 'column_splits': col_splits, 'sequence_splits': seq_splits, 'pe_count': pe_count}
                                    found_valid_config = True
                                    found_in_this_row = True
                                    break
                        
                        if found_in_this_row:
                            break
                    
                    if found_in_this_row:
                        break
                        
        elif strategy == "combined" or strategy == "optimal":
            # Combined: use row, column and sequence splitting (r, c, s)
            max_row_splits = min(d_in, 64)  # Limit search space
            max_col_splits = min(d_out, 64)  # Limit search space
            max_seq_splits = min(s, 64)      # Limit search space
            
            # Search for valid combinations
            for row_splits in range(1, max_row_splits + 1):
                for col_splits in range(1, max_col_splits + 1):
                    for seq_splits in range(1, max_seq_splits + 1):
                        if fits_in_memory_fc(row_splits, col_splits, seq_splits):
                            pe_count = row_splits * col_splits * seq_splits  # PE_total = r × c × s
                            
                            if pe_count < min_pe_count:
                                min_pe_count = pe_count
                                best_config = {'row_splits': row_splits, 'column_splits': col_splits, 'sequence_splits': seq_splits, 'pe_count': pe_count}
                                
                                # For optimal search, once we find a valid config, we can stop the inner loop
                                if strategy == "optimal":
                                    break
                    if strategy == "optimal" and best_config is not None:
                        break
                if strategy == "optimal" and best_config is not None:
                    break
                    
        else:
            return {"error": f"Unknown strategy '{strategy}' for FC layer"}
        
        if best_config is None:
            return {"error": "No viable configuration found with given constraints",
                   "layer_type": layer_type,
                   "strategy": strategy,
                   "execution_mode": execution_mode,
                   "pe_count": float('inf')}
                   
        # Calculate memory utilization for the best configuration
        row_splits = best_config['row_splits']
        col_splits = best_config['column_splits']
        seq_splits = best_config['sequence_splits']
        
        part_weight_size = (d_in / row_splits) * (d_out / col_splits) * bytes_per_param
        part_input_size = b * (s / seq_splits) * (d_in / row_splits) * bytes_per_param
        part_output_size = b * (s / seq_splits) * (d_out / col_splits) * bytes_per_param
        
        if execution_mode == "parallel":
            total_memory_used = part_weight_size + part_input_size + part_output_size
            peak_memory_stage = "all-at-once"
        else:  # pipelined
            stage1_memory = part_weight_size + part_input_size
            stage2_memory = part_weight_size + part_output_size
            if stage1_memory > stage2_memory:
                total_memory_used = stage1_memory
                peak_memory_stage = "input-weight"
            else:
                total_memory_used = stage2_memory
                peak_memory_stage = "weight-output"
        
        best_config.update({
            'layer_type': layer_type,
            'strategy': strategy,
            'execution_mode': execution_mode,
            'memory_utilization_per_pe': total_memory_used,
            'memory_utilization_percentage': (total_memory_used / pe_memory) * 100,
            'total_memory_all_pes': total_memory_used * best_config['pe_count'],
            'weight_memory': part_weight_size,
            'input_memory': part_input_size,
            'output_memory': part_output_size,
            'peak_memory_stage': peak_memory_stage
        })
        
    elif layer_type == "attn":
        # Calculate sizes for attention score and context vector computation
        # Q, K, V activations (already computed, but needed as inputs)
        q_size = b * s * d_h * h * bytes_per_param
        k_size = b * s * d_h * h * bytes_per_param
        v_size = b * s * d_h * h * bytes_per_param
        
        # Attention scores (Q·K^T)
        attn_scores_size = b * h * s * s * bytes_per_param
        
        # Softmax(Q·K^T)·V output
        context_vectors_size = b * s * d_h * h * bytes_per_param
        
        # Function to check if a configuration fits in memory for attention computation
        def fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
            # Partitioned sizes for attention components
            # Q, K, V inputs (partitioned)
            part_q = b * (s / query_seq_splits) * d_h * (h / head_splits) * bytes_per_param
            part_k = b * (s / key_seq_splits) * d_h * (h / head_splits) * bytes_per_param
            part_v = b * (s / key_seq_splits) * d_h * (h / head_splits) * bytes_per_param
            
            # Attention scores (partitioned Q·K^T)
            part_attn_scores = b * (h / head_splits) * (s / query_seq_splits) * (s / key_seq_splits) * bytes_per_param
            
            # Context vectors (partitioned Softmax(Q·K^T)·V)
            part_context = b * (s / query_seq_splits) * d_h * (h / head_splits) * bytes_per_param
            
            # Total memory required per PE
            if execution_mode == "parallel":
                # All intermediate results coexist in memory (conservative)
                total_memory = (part_q + part_k + part_v + part_attn_scores + part_context) * memory_overhead_factor
            else:  # pipelined execution
                # Stage 1: Q and K to compute attention scores
                stage1_memory = part_q + part_k
                # Stage 2: Attention scores and V to compute context
                stage2_memory = part_attn_scores + part_v
                # Stage 3: Context output
                stage3_memory = part_context
                # Use the maximum memory required across stages
                total_memory = max(stage1_memory, stage2_memory, stage3_memory) * memory_overhead_factor
            
            # Check if it fits
            return total_memory <= pe_memory and \
                   (s / query_seq_splits) >= 1 and \
                   (s / key_seq_splits) >= 1 and \
                   (h / head_splits) >= 1
        
        # Start with smallest possible configuration
        best_config = None
        min_pe_count = float('inf')
        
        # Implement different strategies based on the table
        if strategy == "query_sequence_split":
            # Query-Sequence-Split: only split along query sequence dimension (sq)
            key_seq_splits = 1    # Fixed at 1
            head_splits = 1       # Fixed at 1
            
            # Phase 1: Try query-sequence-only splits first
            found_valid_config = False
            
            # Search for valid query_seq_splits only
            for query_seq_splits in range(1, s + 1):
                if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                    pe_count = query_seq_splits  # PE_q,seq = sq
                    
                    if pe_count < min_pe_count:
                        min_pe_count = pe_count
                        best_config = {'query_seq_splits': query_seq_splits, 
                                     'key_seq_splits': key_seq_splits,
                                     'head_splits': head_splits,
                                     'pe_count': pe_count}
                        found_valid_config = True
            
            # Phase 2: If no valid query-sequence-only config is found, fall back to query-sequence + head splits
            if not found_valid_config:
                for head_splits in range(2, h + 1):  # Start from 2 since we already tried with 1
                    found_in_this_head = False
                    
                    for query_seq_splits in range(1, s + 1):
                        if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                            pe_count = query_seq_splits * head_splits
                            
                            if pe_count < min_pe_count:
                                min_pe_count = pe_count
                                best_config = {'query_seq_splits': query_seq_splits, 
                                             'key_seq_splits': key_seq_splits,
                                             'head_splits': head_splits,
                                             'pe_count': pe_count}
                                found_valid_config = True
                                found_in_this_head = True
                    
                    # If we found a valid config with this head_splits value, we can exit the outer loop
                    if found_in_this_head:
                        break
                        
        elif strategy == "key_sequence_split":
            # Key-Sequence-Split: only split along key sequence dimension (sk)
            query_seq_splits = 1  # Fixed at 1
            head_splits = 1       # Fixed at 1
            
            # Phase 1: Try key-sequence-only splits first
            found_valid_config = False
            
            # Search for valid key_seq_splits only
            for key_seq_splits in range(1, s + 1):
                if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                    pe_count = key_seq_splits  # PE_k,seq = sk
                    
                    if pe_count < min_pe_count:
                        min_pe_count = pe_count
                        best_config = {'query_seq_splits': query_seq_splits, 
                                     'key_seq_splits': key_seq_splits,
                                     'head_splits': head_splits,
                                     'pe_count': pe_count}
                        found_valid_config = True
            
            # Phase 2: If no valid key-sequence-only config is found, fall back to key-sequence + head splits
            if not found_valid_config:
                for head_splits in range(2, h + 1):  # Start from 2 since we already tried with 1
                    found_in_this_head = False
                    
                    for key_seq_splits in range(1, s + 1):
                        if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                            pe_count = key_seq_splits * head_splits
                            
                            if pe_count < min_pe_count:
                                min_pe_count = pe_count
                                best_config = {'query_seq_splits': query_seq_splits, 
                                             'key_seq_splits': key_seq_splits,
                                             'head_splits': head_splits,
                                             'pe_count': pe_count}
                                found_valid_config = True
                                found_in_this_head = True
                    
                    # If we found a valid config with this head_splits value, we can exit the outer loop
                    if found_in_this_head:
                        break
        
        elif strategy == "head_split":
            # Head-Split: only split along attention heads dimension (h)
            query_seq_splits = 1  # Fixed at 1
            key_seq_splits = 1    # Fixed at 1
            
            # Phase 1: Try head-only splits first
            found_valid_config = False
            
            # Search for valid head_splits only
            for head_splits in range(1, h + 1):
                if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                    pe_count = head_splits  # PE_head = h
                    
                    if pe_count < min_pe_count:
                        min_pe_count = pe_count
                        best_config = {'query_seq_splits': query_seq_splits, 
                                     'key_seq_splits': key_seq_splits,
                                     'head_splits': head_splits,
                                     'pe_count': pe_count}
                        found_valid_config = True
            
            # Phase 2: If no valid head-only config is found, fall back to query-sequence + head splits
            # But with query_seq_splits in the outer loop and head_splits in the inner loop (opposite of query_sequence_split)
            if not found_valid_config:
                for query_seq_splits in range(2, s + 1):  # Start from 2 since we already tried with 1
                    found_in_this_query = False
                    
                    for head_splits in range(1, h + 1):
                        if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                            pe_count = query_seq_splits * head_splits
                            
                            if pe_count < min_pe_count:
                                min_pe_count = pe_count
                                best_config = {'query_seq_splits': query_seq_splits, 
                                             'key_seq_splits': key_seq_splits,
                                             'head_splits': head_splits,
                                             'pe_count': pe_count}
                                found_valid_config = True
                                found_in_this_query = True
                    
                    # If we found a valid config with this query_seq_splits value, we can exit the outer loop
                    if found_in_this_query:
                        break
        
        elif strategy == "hybrid_split":
            # Hybrid-Split: split along both key sequence and head dimensions (sk, h)
            query_seq_splits = 1  # Fixed at 1
            
            # Search for valid combinations of key_seq_splits and head_splits
            for key_seq_splits in range(1, s + 1):
                for head_splits in range(1, h + 1):
                    if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                        pe_count = key_seq_splits * head_splits  # PE_hybrid = sk × h
                        
                        if pe_count < min_pe_count:
                            min_pe_count = pe_count
                            best_config = {'query_seq_splits': query_seq_splits, 
                                         'key_seq_splits': key_seq_splits,
                                         'head_splits': head_splits,
                                         'pe_count': pe_count}
                            
        elif strategy == "combined" or strategy == "optimal":
            # Combined: use query sequence, key sequence and head splitting (sq, sk, h)
            max_query_seq_splits = min(s, 64)     # Limit search space
            max_key_seq_splits = min(s, 64)       # Limit search space
            max_head_splits = min(h, 64)          # Limit search space
            
            # Search for valid combinations
            for query_seq_splits in range(1, max_query_seq_splits + 1):
                for key_seq_splits in range(1, max_key_seq_splits + 1):
                    for head_splits in range(1, max_head_splits + 1):
                        if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                            pe_count = query_seq_splits * key_seq_splits * head_splits  # PE_attn = sq × sk × h
                            
                            if pe_count < min_pe_count:
                                min_pe_count = pe_count
                                best_config = {'query_seq_splits': query_seq_splits, 
                                             'key_seq_splits': key_seq_splits,
                                             'head_splits': head_splits,
                                             'pe_count': pe_count}
                                
                                # For optimal search, once we find a valid config, we can stop the inner loop
                                if strategy == "optimal":
                                    break
                    if strategy == "optimal" and best_config is not None:
                        break
                if strategy == "optimal" and best_config is not None:
                    break
                    
        else:
            return {"error": f"Unknown strategy '{strategy}' for attention layer"}
        
        if best_config is None:
            return {"error": "No viable configuration found with given constraints",
                   "layer_type": layer_type,
                   "strategy": strategy,
                   "execution_mode": execution_mode,
                   "pe_count": float('inf')}
                   
        # Calculate memory utilization for the best configuration
        query_seq_splits = best_config['query_seq_splits']
        key_seq_splits = best_config['key_seq_splits']
        head_splits = best_config['head_splits']
        
        # Partitioned memories
        part_q = b * (s / query_seq_splits) * d_h * (h / head_splits) * bytes_per_param
        part_k = b * (s / key_seq_splits) * d_h * (h / head_splits) * bytes_per_param
        part_v = b * (s / key_seq_splits) * d_h * (h / head_splits) * bytes_per_param
        part_attn_scores = b * (h / head_splits) * (s / query_seq_splits) * (s / key_seq_splits) * bytes_per_param
        part_context = b * (s / query_seq_splits) * d_h * (h / head_splits) * bytes_per_param
        
        if execution_mode == "parallel":
            total_memory_used = part_q + part_k + part_v + part_attn_scores + part_context
            peak_memory_stage = "all-at-once"
        else:  # pipelined
            stage1_memory = part_q + part_k
            stage2_memory = part_attn_scores + part_v
            stage3_memory = part_context
            
            # Determine which stage has peak memory
            if stage1_memory >= stage2_memory and stage1_memory >= stage3_memory:
                total_memory_used = stage1_memory
                peak_memory_stage = "query-key"
            elif stage2_memory >= stage1_memory and stage2_memory >= stage3_memory:
                total_memory_used = stage2_memory
                peak_memory_stage = "attn-value"
            else:
                total_memory_used = stage3_memory
                peak_memory_stage = "context"
        
        best_config.update({
            'layer_type': layer_type,
            'strategy': strategy,
            'execution_mode': execution_mode,
            'memory_utilization_per_pe': total_memory_used,
            'memory_utilization_percentage': (total_memory_used / pe_memory) * 100,
            'total_memory_all_pes': total_memory_used * best_config['pe_count'],
            'query_memory': part_q,
            'key_memory': part_k,
            'value_memory': part_v,
            'attn_scores_memory': part_attn_scores,
            'context_memory': part_context,
            'peak_memory_stage': peak_memory_stage
        })
    
    else:
        return {"error": f"Unknown layer type '{layer_type}'"}
    
    return best_config


def compare_execution_modes(model_params, hardware_constraints, layer_type="fc", strategy="optimal"):
    """
    Compare parallel and pipelined execution modes to show the difference in required PEs.
    
    Args:
        Same as minimize_pe_count function
        
    Returns:
        Dictionary comparing results from both execution modes
    """
    parallel_result = minimize_pe_count(model_params, hardware_constraints, layer_type, strategy, "parallel")
    pipelined_result = minimize_pe_count(model_params, hardware_constraints, layer_type, strategy, "pipelined")
    
    # Calculate the reduction in PEs
    if isinstance(parallel_result, dict) and isinstance(pipelined_result, dict) and \
       'pe_count' in parallel_result and 'pe_count' in pipelined_result:
        pe_reduction = parallel_result['pe_count'] - pipelined_result['pe_count']
        pe_reduction_percent = (pe_reduction / parallel_result['pe_count']) * 100 if parallel_result['pe_count'] > 0 else 0
    else:
        pe_reduction = "Error"
        pe_reduction_percent = "Error"
    
    return {
        'parallel_execution': parallel_result,
        'pipelined_execution': pipelined_result,
        'pe_reduction': pe_reduction,
        'pe_reduction_percent': pe_reduction_percent,
        'layer_type': layer_type,
        'strategy': strategy
    }

# Write a function that compares different strategies for a given layer for both parallel and pipelined execution modes and 
# plots them on a bar graph with x-axis as the strategies and y-axis as the PEs required, each strategy have 2 bars, one for parallel and one for pipelined
def plot_pe_comparison(model_params, hardware_constraints, layer_type="fc"):
    """
    Compare different strategies for a given layer type across both parallel and pipelined execution modes.
    Creates a bar graph visualization showing PE requirements for each strategy.
    
    Args:
        model_params: Dictionary containing model parameters (batch_size, seq_length, etc.)
        hardware_constraints: Dictionary containing hardware constraints (pe_memory, etc.)
        layer_type: Type of layer to analyze ("fc" or "attn")
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    # Import required colormaps - try to import seaborn for better color palettes if available
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False
    
    # Define strategies based on layer type
    if layer_type == "fc":
        strategies = ["row_split", "column_split", "hybrid_split", "sequence_split", "combined"]
        strategy_names = ["Input-Embedding", "Output-Neurons", "Hybrid", "Sequence", "Combined"]
    elif layer_type == "attn":
        strategies = ["head_split", "key_sequence_split", "hybrid_split", "query_sequence_split", "combined"]
        strategy_names = ["Head-embedding", "Key-Sequence", "Hybrid", "Query-Sequence", "Combined"]
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    # Collect PE requirements for each strategy in both execution modes
    parallel_pe_counts = []
    pipelined_pe_counts = []
    valid_flags = [False] * len(strategies)  # Track validity for each strategy
    
    for i, strategy in enumerate(strategies):
        # Get results for both execution modes
        parallel_result = minimize_pe_count(model_params, hardware_constraints, layer_type, strategy, "parallel")
        pipelined_result = minimize_pe_count(model_params, hardware_constraints, layer_type, strategy, "pipelined")
        
        # Check if valid configurations were found
        if isinstance(parallel_result, dict) and 'error' not in parallel_result and \
           isinstance(pipelined_result, dict) and 'error' not in pipelined_result:
            # Add to our data for plotting
            parallel_pe_counts.append(parallel_result['pe_count'])
            pipelined_pe_counts.append(pipelined_result['pe_count'])
            valid_flags[i] = True
        else:
            # Add placeholder for invalid config
            parallel_pe_counts.append(None)
            pipelined_pe_counts.append(None)
            print(f"Skipping strategy '{strategy}' due to error finding valid configuration")
    
    # Setup plot with larger figure size
    fig, ax = plt.subplots(figsize=(16, 10))
    bar_width = 0.3  # Reduced bar width
    x_positions = np.arange(len(strategies))
    
    # Define color palettes - use Set2 palette similar to plot_split_distribution
    if has_seaborn:
        colors = sns.color_palette("Set2", 2)  # Just need 2 colors for parallel and pipelined
        parallel_color = colors[0]
        pipelined_color = colors[1]
    else:
        # Fallback colors if seaborn is not available
        parallel_color = plt.cm.Set2(0)
        pipelined_color = plt.cm.Set2(1)
    
    # Create bars for each strategy
    has_valid_strategy = False
    legend_handles = []
    legend_labels = []
    
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 24,  # Increased from 20
        'axes.labelsize': 24,  # Increased from 20
        'axes.titlesize': 28,  # Increased from 20
        'xtick.labelsize': 20,  # Increased from 14
        'ytick.labelsize': 20,  # Increased from 14
        'legend.fontsize': 22  # Increased from 16
    })
    
    for i in range(len(strategies)):
        if valid_flags[i]:
            has_valid_strategy = True
            # Plot actual PE counts for valid strategies
            bar1 = ax.bar(x_positions[i] - bar_width/2, parallel_pe_counts[i], bar_width,
                          color=parallel_color)
            bar2 = ax.bar(x_positions[i] + bar_width/2, pipelined_pe_counts[i], bar_width,
                          color=pipelined_color)
            
            # Keep track of the first valid strategy for legend
            if len(legend_handles) == 0:
                legend_handles = [bar1, bar2]
                legend_labels = ['Parallel Execution', 'Pipelined Execution']
            
            # PE reduction % with larger font
            reduction = parallel_pe_counts[i] - pipelined_pe_counts[i]
            if parallel_pe_counts[i] > 0 and reduction > 0:
                pct = (reduction / parallel_pe_counts[i]) * 100
                ax.annotate(f'-{pct:.1f}%', xy=(x_positions[i], pipelined_pe_counts[i]),
                            xytext=(0, 20), textcoords="offset points",
                            ha='center', va='bottom', color='green', fontsize=20, fontweight='bold')
        else:
            # Add transparent bars with a hatch for missing configs
            ax.bar(x_positions[i] - bar_width/2, 0.1, bar_width,
                   color='lightgrey', edgecolor='black', hatch='//')
            ax.bar(x_positions[i] + bar_width/2, 0.1, bar_width,
                   color='lightgrey', edgecolor='black', hatch='//')
            ax.text(x_positions[i], 1, "N/A", ha='center', va='bottom', color='gray', fontstyle='italic', fontsize=20)
    
    # Ensure we have legend entries even if no valid strategies were found
    if not has_valid_strategy:
        bar1 = ax.bar([0], [0], bar_width, color=parallel_color)
        bar2 = ax.bar([0], [0], bar_width, color=pipelined_color)
        legend_handles = [bar1, bar2]
        legend_labels = ['Parallel Execution', 'Pipelined Execution']
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('Strategy', fontsize=24, fontweight='bold')
    ax.set_ylabel('Number of PEs Required', fontsize=24, fontweight='bold')
    title = f"PE Requirements by Strategy for {layer_type.upper()} Layer"
    ax.set_title(title, fontsize=28, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(strategy_names, rotation=15, fontsize=20, fontweight='bold')
    
    # Add the legend using our explicit handles and labels
    ax.legend(legend_handles, legend_labels, fontsize=22)
    
    # Calculate y-axis limits
    valid_pe_counts = [pe for pe in parallel_pe_counts + pipelined_pe_counts if pe is not None]
    max_y = max(valid_pe_counts, default=10)
    ax.set_ylim(0, max_y * 1.2)  # Reduced padding at top
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to make it more compact
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

import matplotlib.pyplot as plt
import numpy as np

def plot_memory_breakdown(model_params, hardware_constraints, layer_type="fc", execution_mode="pipelined"):
    """
    Create a stacked bar chart showing memory breakdown for each strategy.
    
    Args:
        model_params: Dictionary containing model parameters
        hardware_constraints: Dictionary containing hardware constraints
        layer_type: Type of layer to analyze ("fc" or "attn")
        execution_mode: Execution mode ("parallel" or "pipelined")
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False
    
    # Define strategies based on layer type
    if layer_type == "fc":
        strategies = ["row_split", "column_split", "hybrid_split", "sequence_split", "combined"]
        strategy_names = ["Input-Embedding", "Output-Neurons", "Hybrid", "Sequence", "Combined"]
        # Define all possible memory components
        if execution_mode == "parallel":
            # For parallel, we always show all components
            memory_components = ["weight_memory", "input_memory", "output_memory"]
            component_labels = ["Weights", "Input Activations", "Output Activations"]
        else:
            # For pipelined, we'll select relevant components per strategy below
            memory_components = ["weight_memory", "input_memory", "output_memory"]
            component_labels = ["Weights", "Input Activations", "Output Activations"]
    elif layer_type == "attn":
        strategies = ["head_split", "key_sequence_split", "hybrid_split", "query_sequence_split", "combined"]
        strategy_names = ["Head-embedding", "Key-Sequence", "Hybrid", "Query-Sequence", "Combined"]
        # Define all possible memory components
        if execution_mode == "parallel":
            # For parallel, we always show all components
            memory_components = ["query_memory", "key_memory", "value_memory", "attn_scores_memory", "context_memory"]
            component_labels = ["Query", "Key", "Value", "Attention Scores", "Context Vectors"]
        else:
            # For pipelined, we'll select relevant components per strategy below
            memory_components = ["query_memory", "key_memory", "value_memory", "attn_scores_memory", "context_memory"]
            component_labels = ["Query", "Key", "Value", "Attention Scores", "Context Vectors"]
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    # Collect memory breakdown for each strategy
    valid_flags = [False] * len(strategies)
    pe_counts = []
    memory_utilization = []
    peak_memory_stages = []
    
    # For pipelined mode, we create strategy-specific memory data
    if execution_mode == "pipelined":
        # Create strategy-specific memory data structures to hold different peak components
        memory_data = []
        active_components = []
        
        for i, strategy in enumerate(strategies):
            # Get results for the specified execution mode
            result = minimize_pe_count(model_params, hardware_constraints, layer_type, strategy, execution_mode)
            
            # Check if valid configuration was found
            if isinstance(result, dict) and 'error' not in result:
                peak_stage = result.get('peak_memory_stage', '')
                peak_memory_stages.append(peak_stage)
                
                # Create a list with zeros for all components
                strategy_memory = [0] * len(memory_components)
                
                if layer_type == "fc":
                    # Set values only for the components that matter for peak memory
                    weight_idx = memory_components.index("weight_memory")
                    strategy_memory[weight_idx] = result.get('weight_memory', 0)
                    
                    if peak_stage == "input-weight":
                        input_idx = memory_components.index("input_memory")
                        strategy_memory[input_idx] = result.get('input_memory', 0)
                        # Create a list of active components for this strategy
                        active_comps = [weight_idx, input_idx]
                    else:  # weight-output
                        output_idx = memory_components.index("output_memory")
                        strategy_memory[output_idx] = result.get('output_memory', 0)
                        # Create a list of active components for this strategy
                        active_comps = [weight_idx, output_idx]
                else:  # attention layer
                    if peak_stage == "query-key":
                        query_idx = memory_components.index("query_memory")
                        key_idx = memory_components.index("key_memory")
                        strategy_memory[query_idx] = result.get('query_memory', 0)
                        strategy_memory[key_idx] = result.get('key_memory', 0)
                        active_comps = [query_idx, key_idx]
                    elif peak_stage == "attn-value":
                        attn_idx = memory_components.index("attn_scores_memory")
                        value_idx = memory_components.index("value_memory")
                        strategy_memory[attn_idx] = result.get('attn_scores_memory', 0)
                        strategy_memory[value_idx] = result.get('value_memory', 0)
                        active_comps = [attn_idx, value_idx]
                    else:  # context
                        context_idx = memory_components.index("context_memory")
                        strategy_memory[context_idx] = result.get('context_memory', 0)
                        active_comps = [context_idx]
                
                memory_data.append(strategy_memory)
                active_components.append(active_comps)
                valid_flags[i] = True
                pe_counts.append(result['pe_count'])
                memory_utilization.append(result['memory_utilization_percentage'])
            else:
                # Add placeholder for invalid config
                memory_data.append([0] * len(memory_components))
                active_components.append([])
                peak_memory_stages.append("")
                valid_flags[i] = False
                pe_counts.append(0)
                memory_utilization.append(0)
                print(f"Skipping strategy '{strategy}' due to error finding valid configuration")
    else:
        # For parallel mode, just extract all components for each strategy
        memory_data = []
        for i, strategy in enumerate(strategies):
            result = minimize_pe_count(model_params, hardware_constraints, layer_type, strategy, execution_mode)
            if isinstance(result, dict) and 'error' not in result:
                strategy_memory = [result.get(comp, 0) for comp in memory_components]
                memory_data.append(strategy_memory)
                valid_flags[i] = True
                pe_counts.append(result['pe_count'])
                memory_utilization.append(result['memory_utilization_percentage'])
                peak_memory_stages.append("")
            else:
                memory_data.append([0] * len(memory_components))
                valid_flags[i] = False
                pe_counts.append(0)
                memory_utilization.append(0)
                peak_memory_stages.append("")
                print(f"Skipping strategy '{strategy}' due to error finding valid configuration")
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.6
    x_positions = np.arange(len(strategies))
    
    # Increase font size and weight for better visibility
    plt.rcParams.update({
        'font.size': 24,  # Increased from 12
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 28,  # Increased from 14
        'xtick.labelsize': 20,  # Increased from 12
        'ytick.labelsize': 20,  # Increased from 12
        'legend.fontsize': 22  # Increased from 12
    })
    
    # Define color palette
    if has_seaborn:
        colors = sns.color_palette("viridis", len(memory_components))
    else:
        # Fallback colors if seaborn is not available
        colors = plt.cm.viridis(np.linspace(0, 1, len(memory_components)))
    
    # Create stacked bars for each strategy
    bottoms = np.zeros(len(strategies))
    legend_elements = []
    
    # Track which components are used in each strategy for pipelined mode
    used_components = set()
    
    for i, component_idx in enumerate(range(len(memory_components))):
        values = []
        
        for j, strategy_idx in enumerate(range(len(strategies))):
            if valid_flags[j]:
                # For pipelined mode, only include values for active components
                if execution_mode == "pipelined":
                    if component_idx in active_components[j]:
                        value = memory_data[j][component_idx]
                        used_components.add(component_idx)
                    else:
                        value = 0
                else:
                    value = memory_data[j][component_idx]
            else:
                value = 0
            values.append(value)
        
        # Convert to MB for better readability
        values_mb = [v / (1024*1024) for v in values]
        
        # Only create bars for components that are used
        if any(values):
            bars = ax.bar(x_positions, values_mb, bar_width, bottom=bottoms, 
                         label=component_labels[i], color=colors[i])
            bottoms += values_mb
            legend_elements.append(bars)
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('Strategy', fontsize=24, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=24, fontweight='bold')
    title = f"Memory Breakdown by Strategy for {layer_type.upper()} Layer ({execution_mode.capitalize()} Mode)"
    ax.set_title(title, fontsize=28, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(strategy_names, rotation=15, fontweight='bold', fontsize=20)
    
    # Add the memory utilization as a secondary axis
    ax2 = ax.twinx()
    utilization_color = 'red'
    util_line = ax2.plot(x_positions, memory_utilization, 'o-', color=utilization_color, 
                         linewidth=2, markersize=8, label='Memory Utilization %')
    ax2.set_ylabel('Memory Utilization (%)', color=utilization_color, fontsize=24, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=utilization_color, labelsize=20)
    
    # Calculate y-axis limits and add some padding at the top
    y_max = max(memory_utilization) if any(memory_utilization) else 100
    ax2.set_ylim(0, y_max * 1.2)  # Add 20% padding at the top
    
    # Add utilization percentage labels directly on the line points
    for i, util_pct in enumerate(memory_utilization):
        if valid_flags[i]:
            # Position labels at the point
            ax2.text(x_positions[i], util_pct, f'{util_pct:.1f}%', 
                   ha='center', va='bottom', color=utilization_color, fontsize=20, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
    
    # Add peak memory stage annotations for pipelined mode
    if execution_mode == "pipelined":
        for i, stage in enumerate(peak_memory_stages):
            if valid_flags[i] and stage:
                # Format stage name for display
                stage_display = stage.replace('-', '+').title()
                ax.text(x_positions[i], bottoms[i]/2, f"{stage_display}", 
                       ha='center', va='center', color='white', fontsize=20, fontweight='bold')
    
    # Create legend only for used components
    if execution_mode == "pipelined":
        # Only include the components that are used in at least one strategy
        used_legend_elements = [legend_elements[i] for i in range(len(legend_elements)) if i in used_components]
        used_component_labels = [component_labels[i] for i in range(len(component_labels)) if i in used_components]
        legend1 = ax.legend(used_legend_elements, used_component_labels, loc='upper left', fontsize=22)
    else:
        # For parallel mode, include all components
        legend1 = ax.legend(legend_elements, component_labels, loc='upper left', fontsize=22)
    
    ax.add_artist(legend1)
    ax2.legend(loc='upper right', fontsize=22)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to ensure everything fits
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

def plot_split_distribution(model_params, hardware_constraints, layer_type="fc", execution_mode="pipelined"):
    """
    Create a visualization of the splits chosen for each strategy.
    
    Args:
        model_params: Dictionary containing model parameters
        hardware_constraints: Dictionary containing hardware constraints
        layer_type: Type of layer to analyze ("fc" or "attn")
        execution_mode: Execution mode ("parallel" or "pipelined")
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False
    
    # Define strategies based on layer type
    if layer_type == "fc":
        strategies = ["row_split", "column_split", "hybrid_split", "sequence_split", "combined"]
        strategy_names = ["Input-Embedding", "Output-Neurons", "Hybrid", "Sequence", "Combined"]
        # Split dimensions for FC layer
        split_dims = ["row_splits", "column_splits", "sequence_splits"]
        dim_labels = ["Input Dimension Splits", "Output Dimension Splits", "Sequence Splits"]
        dim_totals = [model_params['input_dim'], model_params['output_dim'], model_params['seq_length']]
    elif layer_type == "attn":
        strategies = ["head_split", "key_sequence_split", "hybrid_split", "query_sequence_split", "combined"]
        strategy_names = ["Head-embedding", "Key-Sequence", "Hybrid", "Query-Sequence", "Combined"]
        # Split dimensions for attention layer
        split_dims = ["query_seq_splits", "key_seq_splits", "head_splits"]
        dim_labels = ["Query Sequence Splits", "Key Sequence Splits", "Head Splits"]
        dim_totals = [model_params['seq_length'], model_params['seq_length'], model_params['num_heads']]
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    # Collect split values for each strategy
    split_data = []
    valid_flags = [False] * len(strategies)
    pe_counts = []
    
    for i, strategy in enumerate(strategies):
        # Get results for the specified execution mode
        result = minimize_pe_count(model_params, hardware_constraints, layer_type, strategy, execution_mode)
        
        # Check if valid configuration was found
        if isinstance(result, dict) and 'error' not in result:
            # Extract split values
            strategy_splits = [result.get(dim, 1) for dim in split_dims]
            split_data.append(strategy_splits)
            valid_flags[i] = True
            pe_counts.append(result['pe_count'])
        else:
            # Add placeholder for invalid config
            split_data.append([0] * len(split_dims))
            pe_counts.append(0)
            print(f"Skipping strategy '{strategy}' due to error finding valid configuration")
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create positions for grouped bars
    x_positions = np.arange(len(strategies))
    bar_width = 0.2
    offsets = np.linspace(-(len(split_dims)-1)*bar_width/2, (len(split_dims)-1)*bar_width/2, len(split_dims))
    
    # Define color palette
    if has_seaborn:
        colors = sns.color_palette("Set2", len(split_dims))
    else:
        # Fallback colors if seaborn is not available
        colors = plt.cm.Set2(np.linspace(0, 1, len(split_dims)))
    
    # Calculate portions left after splitting
    portion_data = []
    for s_idx, splits in enumerate(split_data):
        if valid_flags[s_idx]:
            # Calculate portion sizes after splits
            portions = [dim_totals[i] / splits[i] if splits[i] > 0 else 0 for i in range(len(splits))]
            portion_data.append(portions)
        else:
            portion_data.append([0] * len(split_dims))
    
    # Create split factor bars (the number of partitions)
    legend_elements = []
    
    for i, dim in enumerate(split_dims):
        split_values = [split_data[j][i] if valid_flags[j] else 0 for j in range(len(strategies))]
        bars = ax.bar(x_positions + offsets[i], split_values, bar_width, label=dim_labels[i], color=colors[i])
        
        # Keep reference for legend
        if any(split_values):
            legend_elements.append(bars)
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('Strategy', fontsize=24, fontweight='bold')
    ax.set_ylabel('Number of Splits', fontsize=24, fontweight='bold')
    title = f"Split Distribution by Strategy for {layer_type.upper()} Layer ({execution_mode.capitalize()} Mode)"
    ax.set_title(title, fontsize=28, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(strategy_names, rotation=15, fontsize=20, fontweight='bold')
    
    # Add a secondary axis for portion sizes
    ax2 = ax.twinx()
    ax2.set_ylabel('Size After Split (Dimension / Splits)', fontsize=24, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred', labelsize=20)
    
    # Plot portion sizes as markers
    marker_styles = ['o', 's', 'd']  # Different marker for each dimension
    marker_legend_elements = []
    
    for i, dim in enumerate(split_dims):
        portion_values = [portion_data[j][i] if valid_flags[j] else 0 for j in range(len(strategies))]
        valid_positions = [pos for idx, pos in enumerate(x_positions) if valid_flags[idx]]
        valid_portions = [val for idx, val in enumerate(portion_values) if valid_flags[idx]]
        
        if valid_portions:
            line = ax2.plot(valid_positions, valid_portions, marker=marker_styles[i], linestyle='--', 
                       color=colors[i], alpha=1.0, markersize=8, linewidth=2)  # Changed alpha from 0.7 to 1.0 and added linewidth=2
            
            # Add value labels for portions
            for j, (pos, val) in enumerate(zip(valid_positions, valid_portions)):
                if val > 0:
                    ax2.text(pos, val, f'{val:.1f}', ha='center', va='bottom', fontsize=16, color='darkred')
            
            marker_legend_elements.append(line[0])
    
    # Combine legends
    ax.legend(legend_elements, dim_labels, loc='upper left', fontsize=22)
    if marker_legend_elements:
        ax2.legend(marker_legend_elements, [f"{dim} Size After Split" for dim in dim_labels], 
                  loc='upper right', fontsize=22)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to ensure everything fits
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig

# Function to generate all plots for a given layer type
def generate_all_plots(model_params, hardware_constraints, layer_type="fc"):
    """
    Generate all visualization plots for a given layer type.
    
    Args:
        model_params: Dictionary containing model parameters
        hardware_constraints: Dictionary containing hardware constraints
        layer_type: Type of layer to analyze ("fc" or "attn")
        
    Returns:
        List of figure objects
    """
    figures = []
    
    # 1. PE count comparison (parallel vs pipelined)
    pe_fig = plot_pe_comparison(model_params, hardware_constraints, layer_type)
    figures.append(("pe_comparison", pe_fig))
    
    # 2. Memory breakdown for pipelined execution
    mem_fig_pipelined = plot_memory_breakdown(model_params, hardware_constraints, layer_type, "pipelined")
    figures.append(("memory_breakdown_pipelined", mem_fig_pipelined))
    
    # 3. Memory breakdown for parallel execution
    mem_fig_parallel = plot_memory_breakdown(model_params, hardware_constraints, layer_type, "parallel")
    figures.append(("memory_breakdown_parallel", mem_fig_parallel))
    
    # 4. Split distribution for pipelined execution
    split_fig_pipelined = plot_split_distribution(model_params, hardware_constraints, layer_type, "pipelined")
    figures.append(("split_distribution_pipelined", split_fig_pipelined))
    
    # 5. Split distribution for parallel execution
    split_fig_parallel = plot_split_distribution(model_params, hardware_constraints, layer_type, "parallel")
    figures.append(("split_distribution_parallel", split_fig_parallel))
    
    return figures

# Example usage
if __name__ == "__main__":
    model_params_test = {
        'batch_size': 1,
        'seq_length': 1,
        'input_dim': 1024,
        'output_dim': 1024, 
        'num_heads': 16,
        'head_dim': 64,
        'bytes_per_param': 2
    }
    
    model_params = {
        'batch_size': 1,
        'seq_length': 2048,
        'input_dim': 14336,
        'output_dim': 14336, 
        'num_heads': 112,
        'head_dim': 128,
        'bytes_per_param': 2
    }

    model_mlp1_params = {
        'batch_size': 1,
        'seq_length': 2048,
        'input_dim': 14336,
        'output_dim': 4*14336, 
        'num_heads': 112,
        'head_dim': 128,
        'bytes_per_param': 2
    }

    model_mlp2_params = {
        'batch_size': 1,
        'seq_length': 2048,
        'input_dim': 4*14336,
        'output_dim': 14336, 
        'num_heads': 112,
        'head_dim': 128,
        'bytes_per_param': 2
    }
    
    hardware_constraints = {
        'pe_memory': 10000000,  # 10 MB memory per PE
        #'pe_memory': 1024*1024,  # 10 MB memory per PE
        'min_dim_size': 8
    }
    
    # Create pecount directory if it doesn't exist
    pecount_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pecount")
    os.makedirs(pecount_dir, exist_ok=True)
    
    # Generate all plots for FC layer
    fc_figures = generate_all_plots(model_params, hardware_constraints, "fc")
    for name, fig in fc_figures:
        fig.savefig(os.path.join(pecount_dir, f"fc_{name}.png"), dpi=300, bbox_inches='tight')
    
    # Generate all plots for Attention layer
    # attn_figures = generate_all_plots(model_params, hardware_constraints, "attn")
    # for name, fig in attn_figures:
    #    fig.savefig(os.path.join(pecount_dir, f"attn_{name}.png"), dpi=300, bbox_inches='tight')

    # # Generate all plots for MLP1 layer
    # mlp1_figures = generate_all_plots(model_mlp1_params, hardware_constraints, "fc")
    # for name, fig in mlp1_figures:
    #     fig.savefig(os.path.join(pecount_dir, f"mlp1_{name}.png"), dpi=300, bbox_inches='tight')
    
    # # Generate all plots for MLP2 layer
    # mlp2_figures = generate_all_plots(model_mlp2_params, hardware_constraints, "fc")
    # for name, fig in mlp2_figures:
    #     fig.savefig(os.path.join(pecount_dir, f"mlp2_{name}.png"), dpi=300, bbox_inches='tight')


#Example usage (uncomment to run):
# if __name__ == "__main__":
#     model_params = {
#         'batch_size': 1,
#         'seq_length': 2048,
#         'input_dim': 768,
#         'output_dim': 768, 
#         'num_heads': 12,
#         'head_dim': 64,
#         'bytes_per_param': 2
#     }
    
#     hardware_constraints = {
#         'pe_memory': 1000000,  # 1 MB memory per PE
#         'min_dim_size': 8
#     }
    
#     # Plot FC layer comparison
#     fc_fig = plot_pe_comparison(model_params, hardware_constraints, "fc")
#     fc_fig.savefig("fc_strategy_comparison.png", dpi=300, bbox_inches='tight')
    
#     # Plot Attention layer comparison
#     attn_fig = plot_pe_comparison(model_params, hardware_constraints, "attn")
#     attn_fig.savefig("attn_strategy_comparison.png", dpi=300, bbox_inches='tight')
