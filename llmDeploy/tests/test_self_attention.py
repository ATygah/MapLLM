import torch
import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.llm import LLM

def test_self_attention():
    """
    Test a single transformer layer with self-attention mechanism.
    This includes:
    1. FC layers for Q, K, V projections
    2. Arithmetic network for Q @ K^T (attention scores)
    3. Arithmetic network for attention weights @ V
    4. FC layer for MLP (final layer)
    """
    print("\n===== Testing Self-Attention Transformer Layer =====\n")
    
    # Hyper-parameters
    seq_len = 8          # Sequence length
    d_model = 64         # Model dimension
    d_hidden = 32        # Hidden dimension (what the FC networks actually output)
    
    # Create an LLM instance
    llm = LLM(
        seq_len=seq_len,
        pe_memory_size=4096,  # 4KB per PE
        noc_rows=10,
        noc_cols=10,
        mapping_strategy="column_wise",
        split_strategy="column_split",
        data_type="float16"
    )
    
    # Create the Q, K, V projection networks
    q_proj = llm.create_fc_network(
        name="q_proj",
        input_dim=d_model,
        output_dim=d_hidden,  # Output is actually 32 features
        seq_len=seq_len,
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    k_proj = llm.create_fc_network(
        name="k_proj",
        input_dim=d_model,
        output_dim=d_hidden,  # Output is actually 32 features
        seq_len=seq_len,
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    v_proj = llm.create_fc_network(
        name="v_proj",
        input_dim=d_model,
        output_dim=d_hidden,  # Output is actually 32 features
        seq_len=seq_len,
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    # Create arithmetic network for Q @ K^T (attention scores)
    # The output will be a [seq_len, seq_len] matrix
    attn_scores = llm.create_arithmetic_network(
        name="attn_scores",
        seq_len=seq_len,
        d_model=seq_len,  # Use seq_len as d_model for attention scores output
        operation="matmul",
        mapping_strategy="grid_wise",
        split_strategy="column_split"
    )
    
    # Create arithmetic network for attention weights @ V
    # Input will be [seq_len, seq_len] attention weights and [seq_len, d_hidden] value matrix
    # Output will be [seq_len, d_hidden]
    attn_output = llm.create_arithmetic_network(
        name="attn_output",
        seq_len=seq_len,
        d_model=d_hidden,  # This should match v_proj's output dimension
        operation="matmul",
        mapping_strategy="grid_wise",
        split_strategy="column_split"
    )
    
    # Create the final projection layer (like an MLP output layer)
    out_proj = llm.create_fc_network(
        name="out_proj",
        input_dim=d_hidden,  # This should match the output dimension of attn_output
        output_dim=d_model,
        seq_len=seq_len,
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    # Print network configurations
    print("\nNetwork Configurations:")
    print(f"q_proj: input_dim={q_proj.input_dim}, output_dim={q_proj.layer_dims[0]}, seq_len={q_proj.seq_len}")
    print(f"k_proj: input_dim={k_proj.input_dim}, output_dim={k_proj.layer_dims[0]}, seq_len={k_proj.seq_len}")
    print(f"v_proj: input_dim={v_proj.input_dim}, output_dim={v_proj.layer_dims[0]}, seq_len={v_proj.seq_len}")
    print(f"attn_scores: seq_len={seq_len}, d_model={seq_len}")
    print(f"attn_output: seq_len={seq_len}, d_model={d_hidden}")
    print(f"out_proj: input_dim={out_proj.input_dim}, output_dim={out_proj.layer_dims[0]}, seq_len={out_proj.seq_len}")
    
    # Set execution order
    print("\nSetting execution order...")
    llm.set_execution_order([
        ["q_proj", "k_proj", "v_proj"],  # Run Q, K, V projections in parallel
        ["attn_scores"],                 # Q @ K^T
        ["attn_output"],                 # attention weights @ V
        ["out_proj"]                     # Final output projection
    ])
    
    # Connect networks
    print("\nConnecting networks...")
    # Q projection to attention scores (as input A)
    llm.connect_networks("q_proj", "attn_scores", connection_type="matmul_a")
    # K projection to attention scores (as input B)
    llm.connect_networks("k_proj", "attn_scores", connection_type="matmul_b")
    # Attention scores to attention output (as input A)
    llm.connect_networks("attn_scores", "attn_output", connection_type="matmul_a")
    # V projection to attention output (as input B)
    llm.connect_networks("v_proj", "attn_output", connection_type="matmul_b_no_transpose")
    # Attention output to final projection
    llm.connect_networks("attn_output", "out_proj")
    
    # Create a mapping of PE coordinates to network names
    pe_to_network = {}
    for name, network in llm.networks.items():
        for pe_coord in network.active_pes:
            pe_to_network[pe_coord] = name
    
    print("\nCreating test input...")
    # Create test input for q_proj, k_proj, v_proj (typically the same input)
    input_tensor = torch.randn(seq_len, d_model)  # Input shape: [seq_len, d_model]
    inputs = {
        "q_proj": input_tensor,
        "k_proj": input_tensor,
        "v_proj": input_tensor
    }
    
    print("Running inference through LLM...")
    # Debug print shapes of inputs
    for input_name, tensor in inputs.items():
        print(f"Input tensor shape for {input_name}: {tensor.shape}")
    
    # Run inference through the LLM
    outputs = llm.run_inference(inputs)
    
    # Print active PEs for each network
    print("\nActive Processing Elements:")
    print(f"q_proj: {q_proj.active_pes}")
    print(f"k_proj: {k_proj.active_pes}")
    print(f"v_proj: {v_proj.active_pes}")
    print(f"attn_scores: {attn_scores.active_pes}")
    print(f"attn_output: {attn_output.active_pes}")
    print(f"out_proj: {out_proj.active_pes}")
    
    # Get PE allocation table by network
    print("\nPE Allocation by Network:")
    all_pes = set()
    network_pe_count = {}
    
    for name, network in llm.networks.items():
        network_pe_count[name] = len(network.active_pes)
        all_pes.update(network.active_pes)
    
    total_pes = len(all_pes)
    print(f"Total unique PEs used: {total_pes}")
    
    print("\nPE Count by Network:")
    for name, count in network_pe_count.items():
        utilization = (count / (llm.noc_rows * llm.noc_cols)) * 100
        print(f"{name}: {count} PEs ({utilization:.2f}% of NoC)")
    
    # Print output shapes
    print("\nOutput shapes:")
    for name, tensor in outputs.items():
        print(f"{name}: {tensor.shape}")
    
    # Get and print traffic statistics
    traffic_table = llm.noc.scheduler.get_traffic_table()
    
    # Enhance the traffic table with network information
    if not traffic_table.empty:
        # Create new columns for source and destination network names
        src_networks = []
        dest_networks = []
        
        for _, row in traffic_table.iterrows():
            # Extract PE coordinates from string like "(0, 0)"
            src_pe_str = row['src_pe'].strip('()')
            if src_pe_str and ',' in src_pe_str:
                src_x, src_y = map(int, src_pe_str.split(','))
                src_pe = (src_x, src_y)
                src_network = pe_to_network.get(src_pe, "external")
            else:
                src_network = "external"
            src_networks.append(src_network)
            
            dest_pe_str = row['dest_pe'].strip('()') if row['dest_pe'] != "None" else ""
            if dest_pe_str and ',' in dest_pe_str:
                dest_x, dest_y = map(int, dest_pe_str.split(','))
                dest_pe = (dest_x, dest_y)
                dest_network = pe_to_network.get(dest_pe, "external")
            else:
                dest_network = "external"
            dest_networks.append(dest_network)
        
        # Add the network names to the traffic table
        traffic_table['src_network'] = src_networks
        traffic_table['dest_network'] = dest_networks
        
        # Create enhanced source and destination PE columns that include network names
        traffic_table['src_pe_with_network'] = traffic_table.apply(
            lambda row: f"{row['src_pe']} ({row['src_network']})", axis=1
        )
        traffic_table['dest_pe_with_network'] = traffic_table.apply(
            lambda row: f"{row['dest_pe']} ({row['dest_network']})", axis=1
        )
    
    print("\nTraffic Statistics:")
    if not traffic_table.empty:
        # Print full combined traffic table with wait IDs
        print("\nCombined Traffic Table (First 10 rows):")
        pd.set_option('display.max_rows', 10)
        pd.set_option('display.width', 200)
        display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'cycles', 'wait_ids', 'description']
        print(traffic_table[display_cols].head(10).to_string())
        
        # Group by source and destination networks
        network_traffic = traffic_table.groupby(['src_network', 'dest_network']).agg({
            'bytes': 'sum',
            'task_id': 'count'
        }).reset_index()
        
        print("\nTraffic Between Networks:")
        print(network_traffic.to_string(index=False))
        
        # Print overall statistics
        total_bytes = traffic_table['bytes'].sum()
        total_tasks = len(traffic_table)
        avg_bytes_per_task = total_bytes / total_tasks if total_tasks > 0 else 0
        
        print(f"\nOverall Statistics:")
        print(f"Total bytes transferred: {total_bytes:,}")
        print(f"Total communication tasks: {total_tasks}")
        print(f"Average bytes per task: {avg_bytes_per_task:.2f}")
    else:
        print("No traffic recorded")
    
    # Calculate PE utilization for each network
    utilization_data = []
    for name, network in llm.networks.items():
        if hasattr(network, 'get_pe_utilization'):
            utilization = network.get_pe_utilization(use_effective_dimensions=True)
            utilization_data.append({
                'Network': name,
                'Total PEs': utilization['total_pes'],
                'Used PEs': utilization['used_computation_pes'],
                'Computation Utilization (%)': utilization['computation_utilization'],
                'Total Utilization (%)': utilization['total_utilization']
            })
    
    if utilization_data:
        print("\nPE Utilization by Network:")
        utilization_df = pd.DataFrame(utilization_data)
        print(utilization_df.to_string(index=False))
    
    # Save results to log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"self_attention_test_{timestamp}.log")
    
    with open(log_file, "w") as f:
        f.write("===== Self-Attention Transformer Layer Test =====\n\n")
        f.write(f"Sequence length: {seq_len}\n")
        f.write(f"Model dimension: {d_model}\n")
        f.write(f"Hidden dimension: {d_hidden}\n\n")
        
        f.write("Network Configurations:\n")
        f.write(f"q_proj: input_dim={q_proj.input_dim}, output_dim={q_proj.layer_dims[0]}, seq_len={q_proj.seq_len}\n")
        f.write(f"k_proj: input_dim={k_proj.input_dim}, output_dim={k_proj.layer_dims[0]}, seq_len={k_proj.seq_len}\n")
        f.write(f"v_proj: input_dim={v_proj.input_dim}, output_dim={v_proj.layer_dims[0]}, seq_len={v_proj.seq_len}\n")
        f.write(f"attn_scores: seq_len={seq_len}, d_model={seq_len}\n")
        f.write(f"attn_output: seq_len={seq_len}, d_model={d_hidden}\n")
        f.write(f"out_proj: input_dim={out_proj.input_dim}, output_dim={out_proj.layer_dims[0]}, seq_len={out_proj.seq_len}\n\n")
        
        f.write("Active PEs:\n")
        f.write(f"q_proj: {q_proj.active_pes}\n")
        f.write(f"k_proj: {k_proj.active_pes}\n")
        f.write(f"v_proj: {v_proj.active_pes}\n")
        f.write(f"attn_scores: {attn_scores.active_pes}\n")
        f.write(f"attn_output: {attn_output.active_pes}\n")
        f.write(f"out_proj: {out_proj.active_pes}\n\n")
        
        f.write("PE Allocation by Network:\n")
        for name, count in network_pe_count.items():
            utilization = (count / (llm.noc_rows * llm.noc_cols)) * 100
            f.write(f"{name}: {count} PEs ({utilization:.2f}% of NoC)\n")
        f.write(f"Total unique PEs used: {total_pes}\n\n")
        
        f.write("Output shapes:\n")
        for name, tensor in outputs.items():
            f.write(f"{name}: {tensor.shape}\n")
        
        # Write detailed traffic statistics to log file
        if not traffic_table.empty:
            f.write("\nDetailed Traffic Statistics:\n")
            f.write("=" * 80 + "\n\n")
            
            # Log traffic between networks
            f.write("Traffic Between Networks:\n")
            f.write(network_traffic.to_string(index=False) + "\n\n")
            
            # Log overall statistics
            f.write(f"Overall Traffic Statistics:\n")
            f.write(f"Total bytes transferred: {total_bytes:,}\n")
            f.write(f"Total communication tasks: {total_tasks}\n")
            f.write(f"Average bytes per task: {avg_bytes_per_task:.2f}\n\n")
            
            # Log detailed traffic information
            f.write("Detailed Traffic Table (Top 20 rows):\n")
            display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'cycles', 'wait_ids', 'description']
            f.write(traffic_table[display_cols].head(20).to_string() + "\n\n")
        
        # Log PE utilization
        if utilization_data:
            f.write("\nPE Utilization by Network:\n")
            f.write(utilization_df.to_string(index=False) + "\n")
    
    print(f"\nLog saved to: {log_file}")
    print("\nSelf-attention transformer layer test completed successfully!")
    return outputs

if __name__ == "__main__":
    test_self_attention() 