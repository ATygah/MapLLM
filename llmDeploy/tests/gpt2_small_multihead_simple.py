import torch
import sys
import os
import pandas as pd
from datetime import datetime
import traceback

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.llm import LLM
from llmDeploy.run_utils import export_traffic_table_to_file

def test_multihead_attention_simple():
    """
    A simplified test for multihead attention with minimal dimensions.
    """
    print("Starting simplified multihead attention test...")
    
    # Initialize the log file before running the test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "../final_traces/gpt2_small/multihead_simple")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"gpt2_small_multihead_simple_{timestamp}.log")
    
    with open(log_file, "w") as f:
        f.write("===== Testing Simplified Multihead Attention =====\n\n")
        
        # Very small dimensions for testing
        seq_len = 32
        d_model = 768
        n_heads = 12
        head_dim = d_model // n_heads  # 6 per head
        d_hidden = 3072  # Small hidden dimension
        noc_rows = 10
        noc_cols = 40
        f.write(f"Sequence length: {seq_len}\n")
        f.write(f"Model dimension (d_model): {d_model}\n")
        f.write(f"Number of attention heads: {n_heads}\n")
        f.write(f"Dimension per head: {head_dim}\n")
        f.write(f"Hidden dimension: {d_hidden}\n\n")
        f.write(f"NoC rows: {noc_rows}\n")
        f.write(f"NoC columns: {noc_cols}\n\n")
        
        print(f"Using dimensions: seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}, head_dim={head_dim}")
        
        # Create a minimal LLM instance
        f.write("Creating LLM instance...\n")
        llm = LLM(
            seq_len=seq_len,
            pe_memory_size=64 *1024,  # Increased memory size
            noc_rows=noc_rows,              # Increased grid size
            noc_cols=noc_cols,
            mapping_strategy="grid_wise",
            split_strategy="hybrid_split",
            data_type="float16",
            allow_wrapping=True
        )
        f.write("LLM instance created successfully.\n\n")
        
        # Create the Q, K, V projection networks
        f.write("Creating projection networks...\n")
        q_proj = llm.create_fc_network(
            name="q_proj",
            input_dim=d_model,
            output_dim=d_model,
            seq_len=seq_len
        )
        
        k_proj = llm.create_fc_network(
            name="k_proj",
            input_dim=d_model,
            output_dim=d_model,
            seq_len=seq_len
        )
        
        v_proj = llm.create_fc_network(
            name="v_proj",
            input_dim=d_model,
            output_dim=d_model,
            seq_len=seq_len
        )
        
        # Create attention heads
        f.write("Creating attention heads...\n")
        attention_heads = []
        for i in range(n_heads):
            head = llm.create_arithmetic_network(
                name=f"attention_head_{i}",
                seq_len=seq_len,
                d_model=head_dim,
                operation="attention"
            )
            attention_heads.append(head)
        
        # Create output projection
        f.write("Creating output projection...\n")
        output_proj = llm.create_fc_network(
            name="output_proj",
            input_dim=d_model,
            output_dim=d_model,
            seq_len=seq_len
        )
        
        # Write network configurations to log
        f.write("\nNetwork Configurations:\n")
        f.write(f"q_proj: input_dim={q_proj.input_dim}, output_dim={q_proj.layer_dims[0]}, seq_len={q_proj.seq_len}\n")
        f.write(f"k_proj: input_dim={k_proj.input_dim}, output_dim={k_proj.layer_dims[0]}, seq_len={k_proj.seq_len}\n")
        f.write(f"v_proj: input_dim={v_proj.input_dim}, output_dim={v_proj.layer_dims[0]}, seq_len={v_proj.seq_len}\n")
        
        for i, head in enumerate(attention_heads):
            f.write(f"attention_head_{i}: seq_len={seq_len}, d_model={head_dim}, operation=attention\n")
        
        f.write(f"output_proj: input_dim={output_proj.input_dim}, output_dim={output_proj.layer_dims[0]}, seq_len={output_proj.seq_len}\n\n")
        
        # Set execution order
        f.write("\nSetting execution order...\n")
        execution_order = [
            ["q_proj", "k_proj", "v_proj"],  # Run projections in parallel
        ]
        
        # Add all attention heads to execute in parallel
        attention_head_names = [f"attention_head_{i}" for i in range(n_heads)]
        execution_order.append(attention_head_names)
        
        # Add output projection
        execution_order.append(["output_proj"])
        
        llm.set_execution_order(execution_order)
        f.write(f"Execution order: {execution_order}\n\n")
        print(f"Execution order: {execution_order}")
        
        # Connect networks
        f.write("\nConnecting networks...\n")
        print("Connecting networks...")
        
        # Connect Q, K, V projections to each attention head with the appropriate slice
        for i, head in enumerate(attention_heads):
            head_name = f"attention_head_{i}"
            # Define which slice of the output should go to this head
            slice_start = i * head_dim
            slice_end = (i + 1) * head_dim
            
            # Connect Q projection to this attention head
            llm.connect_networks(
                "q_proj", head_name, 
                connection_type="attention_q",
                source_range=(slice_start, slice_end)
            )
            
            # Connect K projection to this attention head
            llm.connect_networks(
                "k_proj", head_name, 
                connection_type="attention_k",
                source_range=(slice_start, slice_end)
            )
            
            # Connect V projection to this attention head
            llm.connect_networks(
                "v_proj", head_name, 
                connection_type="attention_v",
                source_range=(slice_start, slice_end)
            )
            
            # Connect this attention head's output to the output projection
            llm.connect_networks(
                head_name, "output_proj",
                dest_range=(slice_start, slice_end)
            )
            
            f.write(f"Connected head {i}: q_proj[{slice_start}:{slice_end}] -> {head_name} -> output_proj[{slice_start}:{slice_end}]\n")
            print(f"Connected head {i}: q_proj[{slice_start}:{slice_end}] -> {head_name} -> output_proj[{slice_start}:{slice_end}]")
        
        # Create a mapping of PE coordinates to network names
        pe_to_network = {}
        for name, network in llm.networks.items():
            for pe_coord in network.active_pes:
                pe_to_network[pe_coord] = name
        
        # Create test input
        f.write("\nCreating test input...\n")
        print("Creating test input...")
        input_tensor = torch.randn(seq_len, d_model)
        inputs = {
            "q_proj": input_tensor,
            "k_proj": input_tensor,
            "v_proj": input_tensor
        }
        
        # Run inference and capture outputs
        f.write("Running inference...\n")
        print("Running inference...")
        try:
            outputs = llm.run_inference(inputs)
            f.write("Inference successful!\n\n")
            print("Inference successful!")
            
            # Print output shapes
            f.write("Output tensor shapes:\n")
            for name, tensor in outputs.items():
                f.write(f"{name}: {tensor.shape}\n")
                print(f"Output tensor shape for {name}: {tensor.shape}")
            
            # Write active PEs for each network to log
            f.write("\nActive Processing Elements:\n")
            f.write(f"q_proj: {q_proj.active_pes}\n")
            f.write(f"k_proj: {k_proj.active_pes}\n")
            f.write(f"v_proj: {v_proj.active_pes}\n")
            
            for i, head in enumerate(attention_heads):
                f.write(f"attention_head_{i}: {head.active_pes}\n")
                
            f.write(f"output_proj: {output_proj.active_pes}\n\n")
            
            # Get PE allocation table by network
            f.write("\nPE Allocation by Network:\n")
            all_pes = set()
            network_pe_count = {}
            
            for name, network in llm.networks.items():
                network_pe_count[name] = len(network.active_pes)
                all_pes.update(network.active_pes)
            
            total_pes = len(all_pes)
            f.write(f"Total unique PEs used: {total_pes}\n")
            
            # Write PE count by network to log
            f.write("\nPE Count by Network:\n")
            for name, count in network_pe_count.items():
                utilization = (count / (llm.noc_rows * llm.noc_cols)) * 100
                f.write(f"{name}: {count} PEs ({utilization:.2f}% of NoC)\n")
            
            # Get and log traffic statistics
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
            f.write("\nTraffic Statistics:\n")
            if not traffic_table.empty:
                # Write combined traffic table with network information
                f.write("\nCombined Traffic Table:\n")
                display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'wait_ids', 'description']
                f.write(traffic_table[display_cols].to_string() + "\n")
                
                # Group by source and destination networks
                network_traffic = traffic_table.groupby(['src_network', 'dest_network']).agg({
                    'bytes': 'sum',
                    'task_id': 'count'
                }).reset_index()
                
                # Write overall statistics
                total_bytes = traffic_table['bytes'].sum()
                total_tasks = len(traffic_table)
                avg_bytes_per_task = total_bytes / total_tasks if total_tasks > 0 else 0
                
                f.write(f"\nOverall Statistics:\n")
                f.write(f"Total bytes transferred: {total_bytes:,}\n")
                f.write(f"Total communication tasks: {total_tasks}\n")
                f.write(f"Average bytes per task: {avg_bytes_per_task:.2f}\n")
                
                # Add PE mapping details
                f.write("\n" + "=" * 80 + "\n")
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
            else:
                f.write("No traffic recorded\n")
                
            # Add NoC Grid Layout
            f.write("\n\nNoC Grid Layout (showing Network Names at each PE):\n")
            f.write("-" * 80 + "\n")
            
            # Create a grid representation of the NoC - only showing the relevant region
            max_rows = 10  # Show only first few rows
            max_cols = 10  # Show only first few columns
            max_rows = min(max_rows, llm.noc_rows)
            max_cols = min(max_cols, llm.noc_cols)
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
                
            # Export traffic table to file
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                traffic_filename = f"gpt2_small_multihead_simple_{timestamp}.txt"
                traces_dir = os.path.join(os.path.dirname(__file__), "../final_traces/gpt2_small/multihead_simple")
                export_path = export_traffic_table_to_file(llm, traffic_filename, directory=traces_dir)
                f.write(f"\nTraffic table exported to: {export_path}\n")
                print(f"Traffic table exported to: {export_path}")
            except Exception as e:
                error_msg = f"Error exporting traffic table: {e}"
                f.write(f"\n{error_msg}\n")
                print(error_msg)
                traceback.print_exc()
            
            return outputs
        except Exception as e:
            error_msg = f"Error during inference: {e}"
            f.write(f"\n{error_msg}\n")
            traceback.print_exc(file=f)
            print(error_msg)
            traceback.print_exc()
            return None
            
    # Append completion message to log file
    with open(log_file, "a") as f:
        f.write(f"\nTest completed. Results saved to {log_file}")
    
    print(f"Test completed. Results saved to {log_file}")

if __name__ == "__main__":
    test_multihead_attention_simple() 