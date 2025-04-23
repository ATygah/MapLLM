import math
import matplotlib.pyplot as plt
import numpy as np

# Helper function to format time values
def format_time(seconds):
    """Format time in appropriate units (ns, μs, ms, s)"""
    if seconds < 1e-9:
        return f"{seconds*1e12:.1f} ps"
    elif seconds < 1e-6:
        return f"{seconds*1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds*1e6:.1f} μs"
    elif seconds < 1:
        return f"{seconds*1e3:.1f} ms"
    else:
        return f"{seconds:.3f} s"

def calculate_streaming_time(model_params, hardware_constraints, layer_type="fc", strategy="optimal"):
    """
    Calculate input and output streaming time for different layer types across various splitting strategies.
    
    Args:
        model_params: Dictionary containing model parameters
            - batch_size: Batch size (b)
            - seq_length: Sequence length (s)
            - input_dim: Input dimension (d_in or I)
            - output_dim: Output dimension (d_out or O)
            - head_dim: Dimension of attention heads (for attention layers)
            - num_heads: Number of attention heads (for attention layers)
            - bytes_per_param: Memory size of each parameter (typically 2 or 4 bytes)
        
        hardware_constraints: Dictionary containing hardware constraints
            - bandwidth: Available bandwidth for data transfer (bytes/s)
            - min_dim_size: Minimum dimension size after splitting (for alignment/efficiency)
        
        layer_type: Type of layer to analyze
            - "fc": Fully connected layer
            - "attn": Attention layer
            
        strategy: Splitting strategy to use
            - FC layer strategies:
                - "row_split": Only split along input dimension (r)
                - "column_split": Only split along output dimension (c)
                - "hybrid_split": Split along both input and output dimensions (r, c)
                - "sequence_split": Only split along sequence dimension (s_splits)
                - "combined": Use row, column and sequence splitting
                - "optimal": Search for optimal configuration (default)
            - Attention layer strategies:
                - "embedding_split": Split along embedding dimension (r)
                - "query_sequence_split": Split along query sequence dimension (q_s)
                - "key_sequence_split": Split along key sequence dimension (k_s)
                - "combined_split": Use embeddings, query and key sequence splitting
                - "optimal": Search for optimal configuration (default)
    
    Returns:
        Dictionary containing streaming time calculations for the strategy
    """
    # Extract model parameters
    b = model_params['batch_size']
    s = model_params['seq_length']
    I = model_params['input_dim']
    O = model_params['output_dim']
    bytes_per_param = model_params['bytes_per_param']
    
    # For attention layers
    h = model_params.get('num_heads', 1)
    E = model_params.get('head_dim', O // h if h > 0 else O)
    
    # Extract hardware constraints
    BW = hardware_constraints['bandwidth']  # Bandwidth in bytes/second
    
    
    results = {
        'layer_type': layer_type,
        'strategy': strategy,
        'input_streaming_time': 0,
        'output_streaming_time': 0,
        'total_streaming_time': 0
    }
    
    if layer_type == "fc":
        # Calculate streaming times for different FC layer splitting strategies
        if strategy == "row_split":
            # Row-Split: Only split along input dimension (r)
            r = model_params.get('row_splits', 1)
            # Input streaming time: b×s×(I/r)/BW
            input_time = (b * s * (I / r) * bytes_per_param) / BW
            # Output streaming time: b×s×O/BW
            output_time = (b * s * O * bytes_per_param) / BW
            
        elif strategy == "column_split":
            # Column-Split: Only split along output dimension (c)
            c = model_params.get('column_splits', 1)
            # Input streaming time: b×s×I/BW
            input_time = (b * s * I * bytes_per_param) / BW
            # Output streaming time: b×s×(O/c)/BW
            output_time = (b * s * (O / c) * bytes_per_param) / BW
            
        elif strategy == "hybrid_split":
            # Hybrid-Split: Split along both input and output dimensions (r, c)
            r = model_params.get('row_splits', 1)
            c = model_params.get('column_splits', 1)
            # Input streaming time: b×s×(I/r)/BW
            input_time = (b * s * (I / r) * bytes_per_param) / BW
            # Output streaming time: b×s×(O/c)/BW
            output_time = (b * s * (O / c) * bytes_per_param) / BW
            
        elif strategy == "sequence_split":
            # Sequence-Split: Only split along sequence dimension (s_splits)
            s_splits = model_params.get('sequence_splits', 1)
            # Input streaming time: b×(s/s_splits)×I/BW
            input_time = (b * (s / s_splits) * I * bytes_per_param) / BW
            # Output streaming time: b×(s/s_splits)×O/BW
            output_time = (b * (s / s_splits) * O * bytes_per_param) / BW
            
        elif strategy == "combined" or strategy == "optimal":
            # Combined: Use row, column and sequence splitting (r, c, s_splits)
            r = model_params.get('row_splits', 1)
            c = model_params.get('column_splits', 1)
            s_splits = model_params.get('sequence_splits', 1)
            
            # Input streaming time: b×(s/s_splits)×(I/r)/BW
            input_time = (b * (s / s_splits) * (I / r) * bytes_per_param) / BW
            # Output streaming time: b×(s/s_splits)×(O/c)/BW
            output_time = (b * (s / s_splits) * (O / c) * bytes_per_param) / BW
            
        else:
            return {"error": f"Unknown strategy '{strategy}' for FC layer"}
        
    elif layer_type == "attn":
        # Get sequence lengths for query and key/value (usually the same but can be different)
        Sq = model_params.get('query_seq_length', s)
        Sk = model_params.get('key_seq_length', s)
        
        # Calculate streaming times for different attention mechanism splitting strategies
        if strategy == "embedding_split":
            # Embedding-Split: Split along embedding dimension (r)
            r = model_params.get('embedding_splits', 1)
            
            # Input streaming times
            # Query input: b×Sq×(E/r)/BW
            input_time_Q = (b * Sq * (E/r) * bytes_per_param) / BW
            # Key input: b×Sk×(E/r)/BW
            input_time_K = (b * Sk * (E/r) * bytes_per_param) / BW
            # Value input: b×Sk×(E/r)/BW
            input_time_V = (b * Sk * (E/r) * bytes_per_param) / BW
            
            # Output streaming time (context vector): b×Sq×(E/r)/BW
            output_time = (b * Sq * (E/r) * bytes_per_param) / BW
            
            # Total input streaming time
            input_time = input_time_Q + input_time_K + input_time_V
            
        elif strategy == "query_sequence_split":
            # Query-Sequence-Split: Split along query sequence dimension (q_s)
            q_s = model_params.get('query_seq_splits', 1)
            
            # Input streaming times
            # Query input: b×(Sq/q_s)×E/BW
            input_time_Q = (b * (Sq/q_s) * E * bytes_per_param) / BW
            # Key input: b×Sk×E/BW
            input_time_K = (b * Sk * E * bytes_per_param) / BW
            # Value input: b×Sk×E/BW
            input_time_V = (b * Sk * E * bytes_per_param) / BW
            
            # Output streaming time (context vector): b×(Sq/q_s)×E/BW
            output_time = (b * (Sq/q_s) * E * bytes_per_param) / BW
            
            # Total input streaming time
            input_time = input_time_Q + input_time_K + input_time_V
            
        elif strategy == "key_sequence_split":
            # Key-Sequence-Split: Split along key sequence dimension (k_s)
            k_s = model_params.get('key_seq_splits', 1)
            
            # Input streaming times
            # Query input: b×Sq×E/BW
            input_time_Q = (b * Sq * E * bytes_per_param) / BW
            # Key input: b×(Sk/k_s)×E/BW
            input_time_K = (b * (Sk/k_s) * E * bytes_per_param) / BW
            # Value input: b×(Sk/k_s)×E/BW
            input_time_V = (b * (Sk/k_s) * E * bytes_per_param) / BW
            
            # Output streaming time (context vector): b×Sq×E/BW
            output_time = (b * Sq * E * bytes_per_param) / BW
            
            # Total input streaming time
            input_time = input_time_Q + input_time_K + input_time_V
            
        elif strategy == "combined_split" or strategy == "optimal":
            # Combined Splitting: Split along embedding, query and key sequence dimensions (r, q_s, k_s)
            r = model_params.get('embedding_splits', 1)
            q_s = model_params.get('query_seq_splits', 1)
            k_s = model_params.get('key_seq_splits', 1)
            
            # Input streaming times
            # Query input: b×(Sq/q_s)×(E/r)/BW
            input_time_Q = (b * (Sq/q_s) * (E/r) * bytes_per_param) / BW
            # Key input: b×(Sk/k_s)×(E/r)/BW
            input_time_K = (b * (Sk/k_s) * (E/r) * bytes_per_param) / BW
            # Value input: b×(Sk/k_s)×(E/r)/BW
            input_time_V = (b * (Sk/k_s) * (E/r) * bytes_per_param) / BW
            
            # Output streaming time (context vector): b×(Sq/q_s)×(E/r)/BW
            output_time = (b * (Sq/q_s) * (E/r) * bytes_per_param) / BW
            
            # Total input streaming time
            input_time = input_time_Q + input_time_K + input_time_V
            
        else:
            return {"error": f"Unknown strategy '{strategy}' for attention layer"}
    
    else:
        return {"error": f"Unknown layer type '{layer_type}'"}
    
    # Populate results
    results['input_streaming_time'] = input_time
    results['output_streaming_time'] = output_time
    results['total_streaming_time'] = input_time + output_time
    
    # Add specific layer details
    if layer_type == "fc":
        results.update({
            'input_transfer_formula': f"b×{'(s/s_splits)' if 'sequence_splits' in model_params else 's'}×{'(I/r)' if 'row_splits' in model_params else 'I'}/BW",
            'output_transfer_formula': f"b×{'(s/s_splits)' if 'sequence_splits' in model_params else 's'}×{'(O/c)' if 'column_splits' in model_params else 'O'}/BW",
        })
    elif layer_type == "attn":
        results.update({
            'query_transfer_time': input_time_Q,
            'key_transfer_time': input_time_K,
            'value_transfer_time': input_time_V,
            'context_transfer_time': output_time,
            'query_transfer_formula': f"b×{'(Sq/q_s)' if 'query_seq_splits' in model_params else 'Sq'}×{'(E/r)' if 'embedding_splits' in model_params else 'E'}/BW",
            'key_transfer_formula': f"b×{'(Sk/k_s)' if 'key_seq_splits' in model_params else 'Sk'}×{'(E/r)' if 'embedding_splits' in model_params else 'E'}/BW",
            'value_transfer_formula': f"b×{'(Sk/k_s)' if 'key_seq_splits' in model_params else 'Sk'}×{'(E/r)' if 'embedding_splits' in model_params else 'E'}/BW",
            'context_transfer_formula': f"b×{'(Sq/q_s)' if 'query_seq_splits' in model_params else 'Sq'}×{'(E/r)' if 'embedding_splits' in model_params else 'E'}/BW"
        })
    
    return results



def optimize_streaming_parameters(model_params, hardware_constraints, layer_type="fc"):
    """
    Find optimal splitting strategy and parameters that minimize streaming time.
    
    Args:
        model_params: Dictionary containing model parameters
        hardware_constraints: Dictionary containing hardware constraints
        layer_type: Type of layer to optimize ("fc" or "attn")
        
    Returns:
        Dictionary with optimal strategy, parameters, and resulting streaming time
    """
    # Define available strategies based on layer type
    if layer_type == "fc":
        strategies = ["row_split", "column_split", "hybrid_split", "sequence_split", "combined"]
    elif layer_type == "attn":
        strategies = ["embedding_split", "query_sequence_split", "key_sequence_split", "combined_split"]
    else:
        return {"error": f"Unknown layer type '{layer_type}'"}
    
    # Track best strategy and parameters
    best_strategy = None
    best_params = {}
    best_streaming_time = float('inf')
    best_result = None
    
    # For each strategy, find its optimal parameters
    for strategy in strategies:
        # Set up search ranges based on layer type and strategy
        if layer_type == "fc":
            max_row_splits = min(model_params['input_dim'], 64)  # Limit search space
            max_col_splits = min(model_params['output_dim'], 64)  # Limit search space
            max_seq_splits = min(model_params['seq_length'], 64)  # Limit search space

            strategy_best_params = {'row_splits': 1, 'column_splits': 1, 'sequence_splits': 1}
            strategy_best_time = float('inf')
            
            # Strategy-specific parameter search
            if strategy == "row_split":
                for r in range(1, max_row_splits + 1):
                    search_params = model_params.copy()
                    search_params['row_splits'] = r
                    
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < strategy_best_time:
                        strategy_best_time = result['total_streaming_time']
                        strategy_best_params['row_splits'] = r
                        strategy_best_result = result
                        
            elif strategy == "column_split":
                for c in range(1, max_col_splits + 1):
                    search_params = model_params.copy()
                    search_params['column_splits'] = c
                    
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < strategy_best_time:
                        strategy_best_time = result['total_streaming_time']
                        strategy_best_params['column_splits'] = c
                        strategy_best_result = result
                        
            elif strategy == "hybrid_split":
                for r in range(1, max_row_splits + 1):
                    for c in range(1, max_col_splits + 1):
                        search_params = model_params.copy()
                        search_params['row_splits'] = r
                        search_params['column_splits'] = c
                        
                        result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                        if result['total_streaming_time'] < strategy_best_time:
                            strategy_best_time = result['total_streaming_time']
                            strategy_best_params['row_splits'] = r
                            strategy_best_params['column_splits'] = c
                            strategy_best_result = result
                        
            elif strategy == "sequence_split":
                for s_splits in range(1, max_seq_splits + 1):
                    search_params = model_params.copy()
                    search_params['sequence_splits'] = s_splits
                    
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < strategy_best_time:
                        strategy_best_time = result['total_streaming_time']
                        strategy_best_params['sequence_splits'] = s_splits
                        strategy_best_result = result
                        
            elif strategy == "combined":
                # Use step sizes to manage search space
                step_r = max(1, max_row_splits // 8)
                step_c = max(1, max_col_splits // 8)
                step_s = max(1, max_seq_splits // 8)
                
                for r in range(1, max_row_splits + 1, step_r):
                    for c in range(1, max_col_splits + 1, step_c):
                        for s_splits in range(1, max_seq_splits + 1, step_s):
                            search_params = model_params.copy()
                            search_params['row_splits'] = r
                            search_params['column_splits'] = c
                            search_params['sequence_splits'] = s_splits
                            
                            result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                            if result['total_streaming_time'] < strategy_best_time:
                                strategy_best_time = result['total_streaming_time']
                                strategy_best_params['row_splits'] = r
                                strategy_best_params['column_splits'] = c
                                strategy_best_params['sequence_splits'] = s_splits
                                strategy_best_result = result
                            
        elif layer_type == "attn":
            max_embed_splits = min(model_params.get('head_dim', model_params['output_dim']), 64)
            max_query_splits = min(model_params.get('query_seq_length', model_params['seq_length']), 64)
            max_key_splits = min(model_params.get('key_seq_length', model_params['seq_length']), 64)
            
            strategy_best_params = {'embedding_splits': 1, 'query_seq_splits': 1, 'key_seq_splits': 1}
            strategy_best_time = float('inf')
            
            # Strategy-specific parameter search
            if strategy == "embedding_split":
                for r in range(1, max_embed_splits + 1):
                    search_params = model_params.copy()
                    search_params['embedding_splits'] = r
                    
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < strategy_best_time:
                        strategy_best_time = result['total_streaming_time']
                        strategy_best_params['embedding_splits'] = r
                        strategy_best_result = result
                        
            elif strategy == "query_sequence_split":
                for q_s in range(1, max_query_splits + 1):
                    search_params = model_params.copy()
                    search_params['query_seq_splits'] = q_s
                    
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < strategy_best_time:
                        strategy_best_time = result['total_streaming_time']
                        strategy_best_params['query_seq_splits'] = q_s
                        strategy_best_result = result
                        
            elif strategy == "key_sequence_split":
                for k_s in range(1, max_key_splits + 1):
                    search_params = model_params.copy()
                    search_params['key_seq_splits'] = k_s
                    
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < strategy_best_time:
                        strategy_best_time = result['total_streaming_time']
                        strategy_best_params['key_seq_splits'] = k_s
                        strategy_best_result = result
                        
            elif strategy == "combined_split":
                # Use step sizes to manage search space
                step_r = max(1, max_embed_splits // 8)
                step_q = max(1, max_query_splits // 8)
                step_k = max(1, max_key_splits // 8)
                
                for r in range(1, max_embed_splits + 1, step_r):
                    for q_s in range(1, max_query_splits + 1, step_q):
                        for k_s in range(1, max_key_splits + 1, step_k):
                            search_params = model_params.copy()
                            search_params['embedding_splits'] = r
                            search_params['query_seq_splits'] = q_s
                            search_params['key_seq_splits'] = k_s
                            
                            result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                            if result['total_streaming_time'] < strategy_best_time:
                                strategy_best_time = result['total_streaming_time']
                                strategy_best_params['embedding_splits'] = r
                                strategy_best_params['query_seq_splits'] = q_s
                                strategy_best_params['key_seq_splits'] = k_s
                                strategy_best_result = result
        
        # Check if this strategy is the best so far
        if strategy_best_time < best_streaming_time:
            best_streaming_time = strategy_best_time
            best_strategy = strategy
            best_params = strategy_best_params
            best_result = strategy_best_result
    
    # Return best strategy with its optimal parameters
    return {
        'optimal_strategy': best_strategy,
        'optimal_parameters': best_params,
        'streaming_time': best_streaming_time,
        'input_streaming_time': best_result['input_streaming_time'],
        'output_streaming_time': best_result['output_streaming_time'],
        'details': best_result
    }

def plot_transfer_time_comparison(model_params, hardware_constraints, layer_type="fc"):
    """
    Compare transfer times across different strategies for a given layer type.
    Creates a stacked bar visualization showing input and output transfer times,
    along with the optimal splitting configuration for each strategy.
    
    Args:
        model_params: Dictionary containing model parameters
        hardware_constraints: Dictionary containing hardware constraints
        layer_type: Type of layer to analyze ("fc" or "attn")
        
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
        strategy_names = ["Row-Split", "Column-Split", "Hybrid-Split", "Sequence-Split", "Combined"]
        param_keys = ["row_splits", "column_splits", "sequence_splits"]
        param_labels = ["Row Splits", "Column Splits", "Sequence Splits"]
    elif layer_type == "attn":
        strategies = ["embedding_split", "query_sequence_split", "key_sequence_split", "combined_split"]
        strategy_names = ["Embedding-Split", "Query-Sequence-Split", "Key-Sequence-Split", "Combined-Split"]
        param_keys = ["embedding_splits", "query_seq_splits", "key_seq_splits"]
        param_labels = ["Embedding Splits", "Query Seq Splits", "Key Seq Splits"]
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    # Get transfer times and optimal parameters for each strategy
    transfer_times = []
    input_times = []
    output_times = []
    optimal_params = []
    
    for strategy in strategies:
        # Optimize each strategy individually
        best_params = {}
        best_time = float('inf')
        best_result = None
        
        # For FC layer
        if layer_type == "fc":
            if strategy == "row_split":
                max_splits = min(model_params['input_dim'], 64)
                for r in range(1, max_splits + 1):
                    search_params = model_params.copy()
                    search_params['row_splits'] = r
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < best_time:
                        best_time = result['total_streaming_time']
                        best_params = {'row_splits': r, 'column_splits': 1, 'sequence_splits': 1}
                        best_result = result
                        
            elif strategy == "column_split":
                max_splits = min(model_params['output_dim'], 64)
                for c in range(1, max_splits + 1):
                    search_params = model_params.copy()
                    search_params['column_splits'] = c
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < best_time:
                        best_time = result['total_streaming_time']
                        best_params = {'row_splits': 1, 'column_splits': c, 'sequence_splits': 1}
                        best_result = result
                        
            elif strategy == "hybrid_split":
                max_row_splits = min(model_params['input_dim'], 16)
                max_col_splits = min(model_params['output_dim'], 16)
                for r in range(1, max_row_splits + 1):
                    for c in range(1, max_col_splits + 1):
                        search_params = model_params.copy()
                        search_params['row_splits'] = r
                        search_params['column_splits'] = c
                        result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                        if result['total_streaming_time'] < best_time:
                            best_time = result['total_streaming_time']
                            best_params = {'row_splits': r, 'column_splits': c, 'sequence_splits': 1}
                            best_result = result
                            
            elif strategy == "sequence_split":
                max_splits = min(model_params['seq_length'], 64)
                for s_splits in range(1, max_splits + 1):
                    search_params = model_params.copy()
                    search_params['sequence_splits'] = s_splits
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < best_time:
                        best_time = result['total_streaming_time']
                        best_params = {'row_splits': 1, 'column_splits': 1, 'sequence_splits': s_splits}
                        best_result = result
                        
            elif strategy == "combined":
                # Use limited search to manage complexity
                max_row_splits = min(model_params['input_dim'], 8)
                max_col_splits = min(model_params['output_dim'], 8)
                max_seq_splits = min(model_params['seq_length'], 8)
                
                for r in range(1, max_row_splits + 1):
                    for c in range(1, max_col_splits + 1):
                        for s in range(1, max_seq_splits + 1):
                            search_params = model_params.copy()
                            search_params['row_splits'] = r
                            search_params['column_splits'] = c
                            search_params['sequence_splits'] = s
                            result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                            if result['total_streaming_time'] < best_time:
                                best_time = result['total_streaming_time']
                                best_params = {'row_splits': r, 'column_splits': c, 'sequence_splits': s}
                                best_result = result
        
        # For attention layer
        elif layer_type == "attn":
            if strategy == "embedding_split":
                max_splits = min(model_params.get('head_dim', model_params['output_dim']), 64)
                for r in range(1, max_splits + 1):
                    search_params = model_params.copy()
                    search_params['embedding_splits'] = r
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < best_time:
                        best_time = result['total_streaming_time']
                        best_params = {'embedding_splits': r, 'query_seq_splits': 1, 'key_seq_splits': 1}
                        best_result = result
                        
            elif strategy == "query_sequence_split":
                max_splits = min(model_params['seq_length'], 64)
                for q_s in range(1, max_splits + 1):
                    search_params = model_params.copy()
                    search_params['query_seq_splits'] = q_s
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < best_time:
                        best_time = result['total_streaming_time']
                        best_params = {'embedding_splits': 1, 'query_seq_splits': q_s, 'key_seq_splits': 1}
                        best_result = result
                        
            elif strategy == "key_sequence_split":
                max_splits = min(model_params['seq_length'], 64)
                for k_s in range(1, max_splits + 1):
                    search_params = model_params.copy()
                    search_params['key_seq_splits'] = k_s
                    result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                    if result['total_streaming_time'] < best_time:
                        best_time = result['total_streaming_time']
                        best_params = {'embedding_splits': 1, 'query_seq_splits': 1, 'key_seq_splits': k_s}
                        best_result = result
                        
            elif strategy == "combined_split":
                # Use limited search to manage complexity
                max_embed_splits = min(model_params.get('head_dim', model_params['output_dim']), 8)
                max_query_splits = min(model_params['seq_length'], 8)
                max_key_splits = min(model_params['seq_length'], 8)
                
                for r in range(1, max_embed_splits + 1):
                    for q_s in range(1, max_query_splits + 1):
                        for k_s in range(1, max_key_splits + 1):
                            search_params = model_params.copy()
                            search_params['embedding_splits'] = r
                            search_params['query_seq_splits'] = q_s
                            search_params['key_seq_splits'] = k_s
                            result = calculate_streaming_time(search_params, hardware_constraints, layer_type, strategy)
                            if result['total_streaming_time'] < best_time:
                                best_time = result['total_streaming_time']
                                best_params = {'embedding_splits': r, 'query_seq_splits': q_s, 'key_seq_splits': k_s}
                                best_result = result
        
        # Store results for this strategy
        if best_result:
            transfer_times.append(best_time)
            input_times.append(best_result['input_streaming_time'])
            output_times.append(best_result['output_streaming_time'])
            optimal_params.append(best_params)
        else:
            transfer_times.append(0)
            input_times.append(0)
            output_times.append(0)
            optimal_params.append({})
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Increase font size and weight for better visibility
    plt.rcParams.update({
        'font.size': 12,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    # Create stacked bar chart
    bar_width = 0.6
    x_positions = np.arange(len(strategies))
    
    # Define color palette
    if has_seaborn:
        colors = sns.color_palette("Set2", 2)  # Just need 2 colors for input and output
    else:
        colors = [plt.cm.Set2(0), plt.cm.Set2(1)]  # Fallback colors
    
    # Create stacked bars
    bars1 = ax.bar(x_positions, input_times, bar_width, label='Input Transfer Time', color=colors[0])
    bars2 = ax.bar(x_positions, output_times, bar_width, bottom=input_times, label='Output Transfer Time', color=colors[1])
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('Strategy', fontsize=13, fontweight='bold')
    ax.set_ylabel('Transfer Time', fontsize=13, fontweight='bold')
    title = f"Transfer Time Comparison by Strategy for {layer_type.upper()} Layer"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(strategy_names, rotation=15, fontweight='bold')
    
    # Add optimization parameter labels above each bar
    for i, params in enumerate(optimal_params):
        param_strs = []
        for key, label in zip(param_keys, param_labels):
            if key in params and params[key] > 1:
                param_strs.append(f"{label}={params[key]}")
        
        param_text = "\n".join(param_strs) if param_strs else "No Splits"
        total_height = input_times[i] + output_times[i]
        
        # Use smaller background padding and smaller font
        ax.text(x_positions[i], total_height + (max(transfer_times) * 0.03), 
                param_text, ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, pad=1.5, boxstyle='round,pad=0.3'))
    
    # Add value labels on the bars
    for i in range(len(strategies)):
        total_time = input_times[i] + output_times[i]
        
        # Only add labels if total_time is greater than zero
        if total_time > 0:
            # Format time value in appropriate units
            time_text = format_time(total_time)
            
            # Position text in the middle of the bar
            ax.text(x_positions[i], total_time / 2, time_text, ha='center', va='center', 
                    color='white', fontsize=11, fontweight='bold')
    
    # Add grid and legend
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    # Format y-axis ticks as appropriate time units
    ax.set_ylim(0, max(transfer_times) * 1.15)  # Add some padding at the top
    
    # Set up custom y-axis formatter
    from matplotlib.ticker import FuncFormatter
    
    def time_formatter(x, pos):
        return format_time(x)
    
    ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))
    
    # Add explanatory caption
    layer_desc = "fully connected" if layer_type == "fc" else "attention"
    fig.text(0.5, 0.01, 
             f"Transfer time comparison for {layer_desc} layer strategies with bandwidth of 32 bytes/clock at 2GHz.",
             ha='center', fontsize=11, fontweight='bold')
    
    # Adjust layout to ensure everything fits
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    return fig

# Example usage
if __name__ == "__main__":
    # Example model parameters for a transformer layer
    model_params = {
        'batch_size': 1,             # Batch size (b)
        'seq_length': 2048,           # Sequence length (s)
        'input_dim': 14336,           # Input dimension (I) 
        'output_dim': 14336,          # Output dimension (O)
        'num_heads': 112,             # Number of attention heads (h)
        'head_dim': 128,              # Dimension per head (E = 64)
        'bytes_per_param': 2         # 2 bytes per parameter (FP16)
    }

    # Calculate bandwidth based on 32 bytes/clock at 2GHz
    bytes_per_clock = 32
    clock_frequency = 2e9  # 2 GHz in Hz
    bandwidth = bytes_per_clock * clock_frequency  # bytes/second
    
    # Example hardware constraints with the specified bandwidth
    hardware_constraints = {
        'bandwidth': bandwidth,      # 32 bytes/clock at 2 GHz
        'min_dim_size': 8            # Minimum dimension size after splitting
    }
    
    # Create plots for both layer types
    import os
    
    # Create time directory if it doesn't exist
    time_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "time")
    os.makedirs(time_dir, exist_ok=True)
    
    # Plot FC layer transfer time comparison
    fc_fig = plot_transfer_time_comparison(model_params, hardware_constraints, "fc")
    fc_fig.savefig(os.path.join(time_dir, "fc_transfer_time_comparison.png"), dpi=300, bbox_inches='tight')
    
    # Plot Attention layer transfer time comparison
    attn_fig = plot_transfer_time_comparison(model_params, hardware_constraints, "attn")
    attn_fig.savefig(os.path.join(time_dir, "attn_transfer_time_comparison.png"), dpi=300, bbox_inches='tight')
    
    # Find optimal parameters for the best FC strategy
    optimal_fc = optimize_streaming_parameters(model_params, hardware_constraints, "fc")
    print(f"Optimal FC parameters: {optimal_fc['optimal_parameters']}")
    print(f"Optimal FC streaming time: {format_time(optimal_fc['streaming_time'])}")
    
    # Find optimal parameters for the best Attention strategy
    optimal_attn = optimize_streaming_parameters(model_params, hardware_constraints, "attn")
    print(f"Optimal Attention parameters: {optimal_attn['optimal_parameters']}")
    print(f"Optimal Attention streaming time: {format_time(optimal_attn['streaming_time'])}")