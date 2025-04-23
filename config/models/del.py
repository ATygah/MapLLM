import math

def minimize_pe_count(model_params, hardware_constraints, layer_type="fc", strategy="optimal"):
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
                - "row_split": Only split along input dimension
                - "column_split": Only split along output dimension
                - "hybrid_split": Split along both input and output dimensions
                - "sequence_split": Only split along sequence dimension
                - "combined": Use row, column and sequence splitting
                - "optimal": Search for optimal configuration (default)
            - Attention layer strategies:
                - "query_sequence_split": Split along query sequence dimension
                - "key_sequence_split": Split along key sequence dimension
                - "head_split": Split along attention heads dimension
                - "combined": Use query, key sequence and head splitting
                - "optimal": Search for optimal configuration (default)
    
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
            total_memory = (part_weight_size + part_input_size + part_output_size) * memory_overhead_factor
            
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
            # Row-Split: only split along input dimension (r)
            col_splits = 1  # Fixed at 1
            seq_splits = 1  # Fixed at 1
            
            # Search for valid row_splits
            for row_splits in range(1, d_in + 1):
                if fits_in_memory_fc(row_splits, col_splits, seq_splits):
                    pe_count = row_splits  # PE_row = r
                    
                    if pe_count < min_pe_count:
                        min_pe_count = pe_count
                        best_config = {'row_splits': row_splits, 'column_splits': col_splits, 'sequence_splits': seq_splits, 'pe_count': pe_count}
                        
        elif strategy == "column_split":
            # Column-Split: only split along output dimension (c)
            row_splits = 1  # Fixed at 1
            seq_splits = 1  # Fixed at 1
            
            # Search for valid col_splits
            for col_splits in range(1, d_out + 1):
                if fits_in_memory_fc(row_splits, col_splits, seq_splits):
                    pe_count = col_splits  # PE_col = c
                    
                    if pe_count < min_pe_count:
                        min_pe_count = pe_count
                        best_config = {'row_splits': row_splits, 'column_splits': col_splits, 'sequence_splits': seq_splits, 'pe_count': pe_count}
                        
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
            # Sequence-Split: only split along sequence dimension (s)
            row_splits = 1  # Fixed at 1
            col_splits = 1  # Fixed at 1
            
            # Search for valid seq_splits
            for seq_splits in range(1, s + 1):
                if fits_in_memory_fc(row_splits, col_splits, seq_splits):
                    pe_count = seq_splits  # PE_seq = s
                    
                    if pe_count < min_pe_count:
                        min_pe_count = pe_count
                        best_config = {'row_splits': row_splits, 'column_splits': col_splits, 'sequence_splits': seq_splits, 'pe_count': pe_count}
                        
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
                   "pe_count": float('inf')}
                   
        # Calculate memory utilization for the best configuration
        row_splits = best_config['row_splits']
        col_splits = best_config['column_splits']
        seq_splits = best_config['sequence_splits']
        
        part_weight_size = (d_in / row_splits) * (d_out / col_splits) * bytes_per_param
        part_input_size = b * (s / seq_splits) * (d_in / row_splits) * bytes_per_param
        part_output_size = b * (s / seq_splits) * (d_out / col_splits) * bytes_per_param
        total_memory_used = part_weight_size + part_input_size + part_output_size
        
        best_config.update({
            'layer_type': layer_type,
            'strategy': strategy,
            'memory_utilization_per_pe': total_memory_used,
            'memory_utilization_percentage': (total_memory_used / pe_memory) * 100,
            'total_memory_all_pes': total_memory_used * best_config['pe_count'],
            'weight_memory': part_weight_size,
            'input_memory': part_input_size,
            'output_memory': part_output_size
        })
        
    elif layer_type == "attn":
        # MODIFIED: Now only focusing on attention score computation and context vector calculation
        # We assume Q, K, V are already computed and available as inputs
        
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
            total_memory = (part_q + part_k + part_v + part_attn_scores + part_context) * memory_overhead_factor
            
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
            
            # Search for valid query_seq_splits
            for query_seq_splits in range(1, s + 1):
                if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                    pe_count = query_seq_splits  # PE_q,seq = sq
                    
                    if pe_count < min_pe_count:
                        min_pe_count = pe_count
                        best_config = {'query_seq_splits': query_seq_splits, 
                                     'key_seq_splits': key_seq_splits,
                                     'head_splits': head_splits,
                                     'pe_count': pe_count}
                        
        elif strategy == "key_sequence_split":
            # Key-Sequence-Split: only split along key sequence dimension (sk)
            query_seq_splits = 1  # Fixed at 1
            head_splits = 1       # Fixed at 1
            
            # Search for valid key_seq_splits
            for key_seq_splits in range(1, s + 1):
                if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                    pe_count = key_seq_splits  # PE_k,seq = sk
                    
                    if pe_count < min_pe_count:
                        min_pe_count = pe_count
                        best_config = {'query_seq_splits': query_seq_splits, 
                                     'key_seq_splits': key_seq_splits,
                                     'head_splits': head_splits,
                                     'pe_count': pe_count}
        
        elif strategy == "head_split":
            # Head-Split: only split along attention heads dimension (h)
            query_seq_splits = 1  # Fixed at 1
            key_seq_splits = 1    # Fixed at 1
            
            # Search for valid head_splits
            for head_splits in range(1, h + 1):
                if fits_in_memory_attn(query_seq_splits, key_seq_splits, head_splits):
                    pe_count = head_splits  # PE_head = h
                    
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
        
        total_memory_used = part_q + part_k + part_v + part_attn_scores + part_context
        
        best_config.update({
            'layer_type': layer_type,
            'strategy': strategy,
            'memory_utilization_per_pe': total_memory_used,
            'memory_utilization_percentage': (total_memory_used / pe_memory) * 100,
            'total_memory_all_pes': total_memory_used * best_config['pe_count'],
            'query_memory': part_q,
            'key_memory': part_k,
            'value_memory': part_v,
            'attn_scores_memory': part_attn_scores,
            'context_memory': part_context
        })
    
    else:
        return {"error": f"Unknown layer type '{layer_type}'"}
    
    return best_config