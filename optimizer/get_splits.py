from optimizer.peCount import minimize_pe_count

def print_splits():
    model_params = {
        'batch_size': 1,
        'seq_length': 2048,
        'input_dim': 14336,
        'output_dim': 14336,
        'num_heads': 112,
        'head_dim': 128,
        'bytes_per_param': 2
    }
    
    hardware_constraints = {
        'pe_memory': 10000000,  # 10 MB memory per PE
        'min_dim_size': 8
    }
    
    print("\n=== FC LAYER SPLITS ===\n")
    
    # FC layer strategies
    fc_strategies = ["row_split", "column_split", "hybrid_split", "sequence_split", "combined"]
    
    print("FC Layer - Parallel Execution:")
    print("-----------------------------")
    for strategy in fc_strategies:
        result = minimize_pe_count(model_params, hardware_constraints, "fc", strategy, "parallel")
        row = result.get('row_splits', 1)
        col = result.get('column_splits', 1)
        seq = result.get('sequence_splits', 1)
        pe = result.get('pe_count', 0)
        
        print(f"  {strategy:15} - rows: {row:4}, cols: {col:4}, seq: {seq:4} - PE count: {pe}")
    
    print("\nFC Layer - Pipelined Execution:")
    print("------------------------------")
    for strategy in fc_strategies:
        result = minimize_pe_count(model_params, hardware_constraints, "fc", strategy, "pipelined")
        row = result.get('row_splits', 1)
        col = result.get('column_splits', 1)
        seq = result.get('sequence_splits', 1)
        pe = result.get('pe_count', 0)
        
        print(f"  {strategy:15} - rows: {row:4}, cols: {col:4}, seq: {seq:4} - PE count: {pe}")
    
    print("\n=== ATTENTION LAYER SPLITS ===\n")
    
    # Attention layer strategies
    attn_strategies = ["query_sequence_split", "key_sequence_split", "head_split", "combined"]
    
    print("Attention Layer - Parallel Execution:")
    print("----------------------------------")
    for strategy in attn_strategies:
        result = minimize_pe_count(model_params, hardware_constraints, "attn", strategy, "parallel")
        q_seq = result.get('query_seq_splits', 1)
        k_seq = result.get('key_seq_splits', 1)
        head = result.get('head_splits', 1)
        pe = result.get('pe_count', 0)
        
        print(f"  {strategy:20} - query seq: {q_seq:4}, key seq: {k_seq:4}, heads: {head:4} - PE count: {pe}")
    
    print("\nAttention Layer - Pipelined Execution:")
    print("-----------------------------------")
    for strategy in attn_strategies:
        result = minimize_pe_count(model_params, hardware_constraints, "attn", strategy, "pipelined")
        q_seq = result.get('query_seq_splits', 1)
        k_seq = result.get('key_seq_splits', 1)
        head = result.get('head_splits', 1)
        pe = result.get('pe_count', 0)
        
        print(f"  {strategy:20} - query seq: {q_seq:4}, key seq: {k_seq:4}, heads: {head:4} - PE count: {pe}")
    
    # Print split values in a format suitable for direct use in code
    print("\n=== SPLIT VALUES FOR CODE USE ===\n")
    
    print("# FC Layer Splits - Parallel")
    for strategy in fc_strategies:
        result = minimize_pe_count(model_params, hardware_constraints, "fc", strategy, "parallel")
        row = result.get('row_splits', 1)
        col = result.get('column_splits', 1)
        seq = result.get('sequence_splits', 1)
        print(f"fc_parallel_{strategy} = [{row}, {col}, {seq}]  # [row_splits, column_splits, sequence_splits]")
    
    print("\n# FC Layer Splits - Pipelined")
    for strategy in fc_strategies:
        result = minimize_pe_count(model_params, hardware_constraints, "fc", strategy, "pipelined")
        row = result.get('row_splits', 1)
        col = result.get('column_splits', 1)
        seq = result.get('sequence_splits', 1)
        print(f"fc_pipelined_{strategy} = [{row}, {col}, {seq}]  # [row_splits, column_splits, sequence_splits]")
    
    print("\n# Attention Layer Splits - Parallel")
    for strategy in attn_strategies:
        result = minimize_pe_count(model_params, hardware_constraints, "attn", strategy, "parallel")
        q_seq = result.get('query_seq_splits', 1)
        k_seq = result.get('key_seq_splits', 1)
        head = result.get('head_splits', 1)
        print(f"attn_parallel_{strategy} = [{q_seq}, {k_seq}, {head}]  # [query_seq_splits, key_seq_splits, head_splits]")
    
    print("\n# Attention Layer Splits - Pipelined")
    for strategy in attn_strategies:
        result = minimize_pe_count(model_params, hardware_constraints, "attn", strategy, "pipelined")
        q_seq = result.get('query_seq_splits', 1)
        k_seq = result.get('key_seq_splits', 1)
        head = result.get('head_splits', 1)
        print(f"attn_pipelined_{strategy} = [{q_seq}, {k_seq}, {head}]  # [query_seq_splits, key_seq_splits, head_splits]")

if __name__ == "__main__":
    print_splits() 