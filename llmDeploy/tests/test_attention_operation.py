import torch
import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.llm import LLM
from llmDeploy.run_utils import export_traffic_table_to_file

def test_attention_operation():
    """
    Test the new attention operation in the LLM framework with multihead attention.
    This includes:
    1. FC layers for Q, K, V projections
    2. Multiple arithmetic networks with 'attention' operation, one per head
    3. FC layer for output projection
    
    This demonstrates multihead attention where each head processes different parts of the
    Q, K, V projections independently, then results are combined.
    """
    # Initialize the log file before running the test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"attention_operation_test_00.log")
    
    with open(log_file, "w") as f:
        f.write("===== Testing Multihead Attention Operation =====\n\n")
    
        # Hyper-parameters - even smaller dimensions to fit on the NoC grid
        seq_len = 4          # Sequence length (reduced)
        d_model = 3         # Model dimension (embedding size)
        n_heads = 3          # Number of attention heads
        head_dim = d_model // n_heads  # Dimension per head (24/3 = 8)
        d_hidden = 2 * d_model  # Hidden dimension for MLP (reduced multiplier)
        
        f.write(f"Sequence length: {seq_len}\n")
        f.write(f"Model dimension (d_model): {d_model}\n")
        f.write(f"Number of attention heads: {n_heads}\n")
        f.write(f"Dimension per head: {head_dim}\n")
        f.write(f"Hidden dimension: {d_hidden}\n\n")
        
        # Create an LLM instance with allow_wrapping enabled
        llm = LLM(
            seq_len=seq_len,
            pe_memory_size=6,  # 4KB per PE
            noc_rows=100,       # Reduced grid size
            noc_cols=100,       # Reduced grid size
            mapping_strategy="grid_wise",
            split_strategy="hybrid_split",
            data_type="float16",
            allow_wrapping=True  # Enable wrapping
        )
        
        # Create the Q, K, V projection networks
        # Each projection outputs d_model-sized tensor which will be split across heads
        q_proj = llm.create_fc_network(
            name="q_proj",
            input_dim=d_model,
            output_dim=d_model,  # Output same size as input, will be split across heads
            seq_len=seq_len,
            mapping_strategy="grid_wise",
            split_strategy="hybrid_split"
        )
        
        k_proj = llm.create_fc_network(
            name="k_proj",
            input_dim=d_model,
            output_dim=d_model,  # Output same size as input, will be split across heads
            seq_len=seq_len,
            mapping_strategy="grid_wise",
            split_strategy="hybrid_split"
        )
        
        v_proj = llm.create_fc_network(
            name="v_proj",
            input_dim=d_model,
            output_dim=d_model,  # Output same size as input, will be split across heads
            seq_len=seq_len,
            mapping_strategy="grid_wise",
            split_strategy="hybrid_split"
        )
        
        # Create an attention network for each head
        attention_heads = []
        for i in range(n_heads):
            head = llm.create_arithmetic_network(
                name=f"attention_head_{i}",
                seq_len=seq_len,
                d_model=head_dim,  # Each head processes a portion of the model dimension
                operation="attention",
                mapping_strategy="grid_wise",
                split_strategy="hybrid_split"
            )
            attention_heads.append(head)

        # Create output projection to combine all attention heads' outputs
        output_proj = llm.create_fc_network(
            name="output_proj",
            input_dim=d_model,  # Combined output from all heads (n_heads * head_dim)
            output_dim=d_model,
            seq_len=seq_len,
            mapping_strategy="grid_wise",
            split_strategy="hybrid_split"
        )
        
        # MLP networks after attention
        mlp1 = llm.create_fc_network(
            name="mlp1",
            input_dim=d_model,
            output_dim=d_hidden,
            seq_len=seq_len,
            mapping_strategy="grid_wise",
            split_strategy="hybrid_split"
        )

        mlp2 = llm.create_fc_network(
            name="mlp2",
            input_dim=d_hidden,
            output_dim=d_model,
            seq_len=seq_len,
            mapping_strategy="grid_wise",
            split_strategy="hybrid_split"
        )
        
        # Write network configurations to log
        f.write("\nNetwork Configurations:\n")
        f.write(f"q_proj: input_dim={q_proj.input_dim}, output_dim={q_proj.layer_dims[0]}, seq_len={q_proj.seq_len}\n")
        f.write(f"k_proj: input_dim={k_proj.input_dim}, output_dim={k_proj.layer_dims[0]}, seq_len={k_proj.seq_len}\n")
        f.write(f"v_proj: input_dim={v_proj.input_dim}, output_dim={v_proj.layer_dims[0]}, seq_len={v_proj.seq_len}\n")
        
        for i, head in enumerate(attention_heads):
            f.write(f"attention_head_{i}: seq_len={seq_len}, d_model={head_dim}, operation=attention\n")
        
        f.write(f"output_proj: input_dim={output_proj.input_dim}, output_dim={output_proj.layer_dims[0]}, seq_len={output_proj.seq_len}\n")
        f.write(f"mlp1: input_dim={mlp1.input_dim}, output_dim={mlp1.layer_dims[0]}, seq_len={mlp1.seq_len}\n")
        f.write(f"mlp2: input_dim={mlp2.input_dim}, output_dim={mlp2.layer_dims[0]}, seq_len={mlp2.seq_len}\n\n")

        # Set execution order
        f.write("\nSetting execution order...\n")
        
        execution_order = [
            ["q_proj", "k_proj", "v_proj"],  # Run Q, K, V projections in parallel
        ]
        
        # Add all attention heads to execute in parallel
        attention_head_names = [f"attention_head_{i}" for i in range(n_heads)]
        execution_order.append(attention_head_names)
        
        # Add output projection and MLP layers
        execution_order.extend([
            ["output_proj"], 
            ["mlp1"], 
            ["mlp2"]
        ])
        
        llm.set_execution_order(execution_order)
        
        f.write(f"Execution order: {execution_order}\n\n")
        
        # Connect networks
        f.write("\nConnecting networks...\n")
        
        # Connect Q, K, V projections to each attention head with the appropriate slice
        for i, head in enumerate(attention_heads):
            head_name = f"attention_head_{i}"
            # Define which slice of the output should go to this head
            slice_start = i * head_dim
            slice_end = (i + 1) * head_dim
            
            # Connect Q projection to this attention head (slicing the output)
            llm.connect_networks(
                "q_proj", head_name, 
                connection_type="attention_q",
                source_range=(slice_start, slice_end)  # Specify the column range for this head
            )
            
            # Connect K projection to this attention head (slicing the output)
            llm.connect_networks(
                "k_proj", head_name, 
                connection_type="attention_k",
                source_range=(slice_start, slice_end)  # Specify the column range for this head
            )
            
            # Connect V projection to this attention head (slicing the output)
            llm.connect_networks(
                "v_proj", head_name, 
                connection_type="attention_v",
                source_range=(slice_start, slice_end)  # Specify the column range for this head
            )
            
            # Connect this attention head's output to the output projection
            llm.connect_networks(
                head_name, "output_proj",
                dest_range=(slice_start, slice_end)  # Specify where this head's output goes
            )
            
            f.write(f"q_proj[{slice_start}:{slice_end}] -> {head_name} (connection_type=attention_q)\n")
            f.write(f"k_proj[{slice_start}:{slice_end}] -> {head_name} (connection_type=attention_k)\n")
            f.write(f"v_proj[{slice_start}:{slice_end}] -> {head_name} (connection_type=attention_v)\n")
            f.write(f"{head_name} -> output_proj[{slice_start}:{slice_end}]\n")
        
        # Connect output projection to MLP
        llm.connect_networks("output_proj", "mlp1")
        llm.connect_networks("mlp1", "mlp2")
        
        f.write("output_proj -> mlp1\n")
        f.write("mlp1 -> mlp2\n\n")
        
        # Create a mapping of PE coordinates to network names
        pe_to_network = {}
        for name, network in llm.networks.items():
            for pe_coord in network.active_pes:
                pe_to_network[pe_coord] = name
        
        f.write("\nCreating test input...\n")
        # Create test input for the networks
        input_tensor = torch.randn(seq_len, d_model)  # Input shape: [seq_len, d_model]
        inputs = {
            "q_proj": input_tensor,
            "k_proj": input_tensor,
            "v_proj": input_tensor
        }
        
        f.write("Running inference through LLM...\n")
        # Debug print shapes of inputs
        for input_name, tensor in inputs.items():
            f.write(f"Input tensor shape for {input_name}: {tensor.shape}\n")
        
        # Run inference through the LLM
        outputs = llm.run_inference(inputs)
        
        # Write active PEs for each network to log
        f.write("\nActive Processing Elements:\n")
        f.write(f"q_proj: {q_proj.active_pes}\n")
        f.write(f"k_proj: {k_proj.active_pes}\n")
        f.write(f"v_proj: {v_proj.active_pes}\n")
        
        for i, head in enumerate(attention_heads):
            f.write(f"attention_head_{i}: {head.active_pes}\n")
            
        f.write(f"output_proj: {output_proj.active_pes}\n")
        f.write(f"mlp1: {mlp1.active_pes}\n")
        f.write(f"mlp2: {mlp2.active_pes}\n\n")
        
        # Get PE allocation table by network
        f.write("\nPE Allocation by Network:\n")
        all_pes = set()
        network_pe_count = {}
        
        for name, network in llm.networks.items():
            network_pe_count[name] = len(network.active_pes)
            all_pes.update(network.active_pes)
        
        total_pes = len(all_pes)
        f.write(f"Total unique PEs used: {total_pes}\n")
        
        # # Add weight tensor distribution tables
        # f.write("\nWeight Tensor Distribution for FC Layers:\n")
        # f.write("-" * 80 + "\n")
        # f.write(f"{'Network':15} {'PE':12} {'Weight Matrix Portion':50}\n")
        # f.write("-" * 80 + "\n")
        
        # # Log FC layer weight distributions
        # for name, network in llm.networks.items():
        #     if hasattr(network, 'layer_dims') and len(network.active_pes) > 0:  # This is an FC network
        #         for pe_idx, pe in enumerate(network.active_pes):
        #             # Calculate the weight matrix portion for this PE based on mapping strategy
        #             if network.split_strategy == "column_split":
        #                 cols_per_pe = network.layer_dims[0] // len(network.active_pes)
        #                 start_col = pe_idx * cols_per_pe
        #                 end_col = start_col + cols_per_pe if pe_idx < len(network.active_pes) - 1 else network.layer_dims[0]
        #                 portion = f"Full input ({network.input_dim} rows), columns {start_col}:{end_col}"
        #             elif network.split_strategy == "row_split":
        #                 rows_per_pe = network.input_dim // len(network.active_pes)
        #                 start_row = pe_idx * rows_per_pe
        #                 end_row = start_row + rows_per_pe if pe_idx < len(network.active_pes) - 1 else network.input_dim
        #                 portion = f"Rows {start_row}:{end_row}, full output ({network.layer_dims[0]} columns)"
        #             else:
        #                 portion = "Unknown distribution"
                    
        #             f.write(f"{name:15} {str(pe):12} {portion:50}\n")
        
        # f.write("\nData Distribution for Arithmetic Network:\n")
        # f.write("-" * 80 + "\n")
        # f.write(f"{'Network':15} {'PE':12} {'Operation':15} {'Data Handled':50}\n")
        # f.write("-" * 80 + "\n")
        
        # # Log arithmetic network data distributions
        # for name, network in llm.networks.items():
        #     if hasattr(network, 'operation'):  # This is an arithmetic network
        #         operation = network.operation
        #         for pe in network.active_pes:
        #             # For attention, explain the Q, K, V handling
        #             if operation == "attention":
        #                 data_handling = f"Full Q, K, V tensors for seq_len={network.seq_len}, d_model={network.d_model}"
        #             elif operation == "matmul":
        #                 data_handling = f"Matrix multiplication with dimensions appropriate for seq_len={network.seq_len}"
        #             else:
        #                 data_handling = f"Element-wise operation on tensors of shape [seq_len={network.seq_len}, d_model={network.d_model}]"
                    
        #             f.write(f"{name:15} {str(pe):12} {operation:15} {data_handling:50}\n")
        
        # Write PE count by network to log
        f.write("\nPE Count by Network:\n")
        for name, count in network_pe_count.items():
            utilization = (count / (llm.noc_rows * llm.noc_cols)) * 100
            f.write(f"{name}: {count} PEs ({utilization:.2f}% of NoC)\n")
        
        # Write output shapes to log
        f.write("\nOutput shapes:\n")
        for name, tensor in outputs.items():
            f.write(f"{name}: {tensor.shape}\n")
        
        # Get and enhance traffic statistics
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
        
        # Write traffic statistics to log
        # f.write("\nTraffic Statistics:\n")
        # if not traffic_table.empty:
        #     # Set pandas display options for log file
        #     pd.set_option('display.max_rows', None)  # Show all rows
        #     pd.set_option('display.width', 200)
            
        #     # Write full combined traffic table with wait IDs
            f.write("\nCombined Traffic Table:\n")
            display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'cycles', 'wait_ids', 'description']
            # f.write(traffic_table[display_cols].to_string() + "\n")
            
        #     # Group by source-destination pairs with network information
        #     grouped = traffic_table.groupby(['src_pe_with_network', 'dest_pe_with_network']).agg({
        #         'bytes': 'sum',
        #         'task_id': 'count'
        #     }).reset_index()
            
        #     # Rename columns for clarity
        #     grouped.columns = ['Source PE (Network)', 'Destination PE (Network)', 'Total Bytes', 'Total Tasks']
            
        #     # Add average bytes per task
        #     grouped['Avg Bytes/Task'] = grouped['Total Bytes'] / grouped['Total Tasks']
            
        #     # Format bytes with commas
        #     grouped['Total Bytes'] = grouped['Total Bytes'].apply(lambda x: f"{x:,}")
        #     grouped['Avg Bytes/Task'] = grouped['Avg Bytes/Task'].apply(lambda x: f"{x:,.2f}")
            
            # f.write("\nTraffic by PE pairs (Full Table):\n")
            # f.write(grouped.to_string(index=False) + "\n")
            
            # Group by source and destination networks
            network_traffic = traffic_table.groupby(['src_network', 'dest_network']).agg({
                'bytes': 'sum',
                'task_id': 'count'
            }).reset_index()
            
            # f.write("\nTraffic Between Networks:\n")
            # f.write(network_traffic.to_string(index=False) + "\n")
            
            # Write overall statistics
            # total_bytes = traffic_table['bytes'].sum()
            # total_tasks = len(traffic_table)
            # avg_bytes_per_task = total_bytes / total_tasks if total_tasks > 0 else 0
            
            # f.write(f"\nOverall Statistics:\n")
            # f.write(f"Total bytes transferred: {total_bytes:,}\n")
            # f.write(f"Total communication tasks: {total_tasks}\n")
            # f.write(f"Average bytes per task: {avg_bytes_per_task:.2f}\n")
        else:
            f.write("No traffic recorded\n")
        
        # Write detailed traffic statistics to log file
        if not traffic_table.empty:
            f.write("\nDetailed Traffic Statistics:\n")
            f.write("=" * 80 + "\n\n")
            
            # Log traffic for each network
            for name, network in llm.networks.items():
                f.write(f"{name} Network Traffic Summary:\n")
                f.write("-" * 40 + "\n")
                
                # Get network-specific traffic by filtering on src_network or dest_network
                network_traffic_df = traffic_table[
                    (traffic_table['src_network'] == name) | 
                    (traffic_table['dest_network'] == name)
                ]
                
                if not network_traffic_df.empty:
                    # Calculate total bytes and tasks
                    total_bytes = network_traffic_df['bytes'].sum()
                    total_tasks = len(network_traffic_df)
                    avg_bytes_per_task = total_bytes / total_tasks if total_tasks > 0 else 0
                    
                    f.write(f"Total bytes transferred: {total_bytes:,} bytes\n")
                    f.write(f"Total communication tasks: {total_tasks}\n")
                    f.write(f"Average bytes per task: {avg_bytes_per_task:.2f} bytes\n\n")
                    
                    # Print summarized traffic table grouped by source/destination
                    f.write(f"{name} Traffic Summary:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Group by source and destination networks for this specific network
                    network_summary = network_traffic_df.groupby(['src_network', 'dest_network']).agg({
                        'bytes': 'sum',
                        'task_id': 'count'
                    }).reset_index()
                    
                    for _, row in network_summary.iterrows():
                        f.write(f"{row['src_network']} -> {row['dest_network']}: {row['bytes']:,} bytes ({row['task_id']} tasks)\n")
                    f.write("\n")
                else:
                    f.write("No traffic recorded for this network\n\n")
            
            # Write traffic between networks summary to log
            # f.write("\nTraffic Between Networks:\n")
            # for _, row in network_traffic.iterrows():
            #     f.write(f"{row['src_network']} -> {row['dest_network']}: {row['bytes']} bytes ({row['task_id']} tasks)\n")
                
            # Log overall network statistics
            f.write("\nOverall Network Statistics:\n")
            f.write("=" * 40 + "\n")
            total_bytes = traffic_table['bytes'].sum()
            total_tasks = len(traffic_table)
            avg_bytes_per_task = total_bytes / total_tasks if total_tasks > 0 else 0
            
            f.write(f"Total bytes across all networks: {total_bytes:,} bytes\n")
            f.write(f"Total communication tasks across all networks: {total_tasks}\n")
            f.write(f"Average bytes per task across all networks: {avg_bytes_per_task:.2f} bytes\n\n")
            
            # Add PE mapping details
            f.write("=" * 80 + "\n")
            f.write("                          PE MAPPING DETAILS                          \n")
            f.write("=" * 80 + "\n")
            pe_mapping_df = llm.get_pe_mapping_details()
            
            # Add descriptive column headers
            if not pe_mapping_df.empty:
                pe_mapping_df.columns = [
                    'Network #', 
                    'Network Name',
                    'Network Type',
                    'PE Coords', 
                    'Layer ID', 
                    'PE Idx', 
                    'Split Type', 
                    'Weight Range', 
                    'Shape'
                ]
                
                # Add descriptive split dimension values
                split_dim_names = {0: 'Row', 1: 'Col', 2: 'Hybrid'}
                if 'Split Type' in pe_mapping_df.columns:
                    pe_mapping_df['Split Type'] = pe_mapping_df['Split Type'].map(
                        lambda x: split_dim_names.get(x, str(x))
                    )
            
            f.write(pe_mapping_df.to_string(index=False) + "\n\n")
            
            # Add combined traffic table to log
            f.write("\nCombined Traffic Table (All Rows):\n")
            f.write("-" * 80 + "\n")
            f.write(traffic_table[display_cols].to_string() + "\n")
            
        else:
            f.write("\nNo traffic recorded during test.\n")
        
        # # Add comparison with two-step approach
        # f.write("\n=== Comparison with Two-Step Approach ===\n")
        # f.write("The attention operation combines Q·K^T and (Q·K^T)·V into a single network operation.\n")
        # f.write("This reduces communication overhead compared to using separate matrix multiply operations.\n")
            
        # Add NoC Grid Layout at the end of the log file
        f.write("\n\nNoC Grid Layout (showing Network Names at each PE):\n")
        f.write("-" * 80 + "\n")
        
        # Create a grid representation of the NoC - now using full dimensions
        max_rows = llm.noc_rows
        max_cols = llm.noc_cols
        noc_grid = [[' ' for _ in range(max_cols)] for _ in range(max_rows)]
        
        # Fill in the grid with network names
        for name, network in llm.networks.items():
            for pe_coord in network.active_pes:
                x, y = pe_coord
                if 0 <= x < max_cols and 0 <= y < max_rows:
                    noc_grid[y][x] = name[:8]  # Use abbreviated name if too long
        
        # Print column headers
        header = '    |'
        for x in range(max_cols):
            header += f' {x:^8} |'
        f.write(header + "\n")
        f.write('-' * len(header) + "\n")
        
        # Write the grid to the log file
        for y in range(max_rows):
            row_str = f'{y:3d} |'
            for x in range(max_cols):
                cell_value = noc_grid[y][x] or "."
                row_str += f' {cell_value:^8} |'
            f.write(row_str + "\n")
    
    f = open(log_file, "a")
    f.write(f"\nTest completed successfully. Results saved to {log_file}")
    f.close()
    
    # Export the traffic table to a file
    traffic_filename = f"attention_operation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    export_path = export_traffic_table_to_file(llm, traffic_filename)
    print(f"Traffic table exported to: {export_path}")
    
    return outputs

# Run the test if this file is executed directly
if __name__ == "__main__":
    test_attention_operation() 