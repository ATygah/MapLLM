import math
import matplotlib.pyplot as plt
import numpy as np

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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_pe_comparison(model_params, hardware_constraints, layer_type="fc"):
    sns.set_style("whitegrid")
    
    if layer_type == "fc":
        strategies = ["row_split", "column_split", "hybrid_split", "sequence_split", "combined"]
        strategy_names = ["Row-Split", "Column-Split", "Hybrid-Split", "Sequence-Split", "Combined"]
    elif layer_type == "attn":
        strategies = ["query_sequence_split", "key_sequence_split", "head_split", "combined"]
        strategy_names = ["Query-Sequence-Split", "Key-Sequence-Split", "Embedding(Head)-Split", "Combined"]
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    parallel_pe_counts = []
    pipelined_pe_counts = []
    valid_flags = []
    
    for i, strategy in enumerate(strategies):
        parallel_result = minimize_pe_count(model_params, hardware_constraints, layer_type, strategy, "parallel")
        pipelined_result = minimize_pe_count(model_params, hardware_constraints, layer_type, strategy, "pipelined")
        
        is_valid = isinstance(parallel_result, dict) and 'error' not in parallel_result and \
                   isinstance(pipelined_result, dict) and 'error' not in pipelined_result
        
        if is_valid:
            parallel_pe_counts.append(parallel_result['pe_count'])
            pipelined_pe_counts.append(pipelined_result['pe_count'])
        else:
            parallel_pe_counts.append(None)
            pipelined_pe_counts.append(None)
        
        valid_flags.append(is_valid)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.35
    x_positions = np.arange(len(strategies))

    parallel_colors = sns.color_palette("Blues_d", len(strategies))
    pipelined_colors = sns.color_palette("Reds_d", len(strategies))

    for i in range(len(strategies)):
        if valid_flags[i]:
            bar1 = ax.bar(x_positions[i] - bar_width/2, parallel_pe_counts[i], bar_width,
                          color=parallel_colors[i], label='Parallel Execution' if i == 0 else "")
            bar2 = ax.bar(x_positions[i] + bar_width/2, pipelined_pe_counts[i], bar_width,
                          color=pipelined_colors[i], label='Pipelined Execution' if i == 0 else "")
            
            # Value labels
            ax.text(x_positions[i] - bar_width/2, parallel_pe_counts[i] + 1, f'{parallel_pe_counts[i]}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(x_positions[i] + bar_width/2, pipelined_pe_counts[i] + 1, f'{pipelined_pe_counts[i]}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # PE reduction %
            reduction = parallel_pe_counts[i] - pipelined_pe_counts[i]
            if parallel_pe_counts[i] > 0 and reduction > 0:
                pct = (reduction / parallel_pe_counts[i]) * 100
                ax.annotate(f'-{pct:.1f}%', xy=(x_positions[i], pipelined_pe_counts[i]),
                            xytext=(0, -15), textcoords="offset points",
                            ha='center', va='top', color='green', fontweight='bold')
        else:
            # Add transparent bars with a hatch for missing configs
            ax.bar(x_positions[i] - bar_width/2, 0.1, bar_width,
                   color='lightgrey', edgecolor='black', hatch='//')
            ax.bar(x_positions[i] + bar_width/2, 0.1, bar_width,
                   color='lightgrey', edgecolor='black', hatch='//')
            ax.text(x_positions[i], 1, "N/A", ha='center', va='bottom', color='gray', fontstyle='italic')

    ax.set_xlabel('Strategy', fontsize=13)
    ax.set_ylabel('Number of PEs Required', fontsize=13)
    ax.set_title(f"PE Requirements by Strategy for {layer_type.upper()} Layer", fontsize=15, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(strategy_names, rotation=15)
    ax.legend()

    max_y = max([val for val in parallel_pe_counts + pipelined_pe_counts if val is not None], default=10)
    ax.set_ylim(0, max_y * 1.3)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    

    fig.text(0.5, 0.01, 
             f"Strategies compared for {layer_type.upper()} layers. Pipelined execution often reduces PE count via temporal reuse.",
             ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


#Example usage (uncomment to run):
if __name__ == "__main__":
    model_params = {
        'batch_size': 1,
        'seq_length': 512,
        'input_dim': 768,
        'output_dim': 768, 
        'num_heads': 12,
        'head_dim': 64,
        'bytes_per_param': 2
    }
    
    hardware_constraints = {
        'pe_memory': 1000000,  # 1 MB memory per PE
        'min_dim_size': 8
    }
    
    # Plot FC layer comparison
    fc_fig = plot_pe_comparison(model_params, hardware_constraints, "fc")
    fc_fig.savefig("fc_strategy_comparison.png", dpi=300, bbox_inches='tight')
    
    # Plot Attention layer comparison
    attn_fig = plot_pe_comparison(model_params, hardware_constraints, "attn")
    attn_fig.savefig("attn_strategy_comparison.png", dpi=300, bbox_inches='tight')
