import torch
import sys
import os
import pandas as pd
from datetime import datetime
import logging

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.llm import LLM

def test_network_allocation():
    """Test network allocation and traffic patterns."""
    print("\n===== Testing Network Allocation =====\n")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a log directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup log file with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"network_allocation_test_00.log")
    
    # Create an LLM instance with a 10x10 NoC
    llm = LLM(
        seq_len=6,
        pe_memory_size=4096,  # 4KB per PE
        noc_rows=10,
        noc_cols=10,
        mapping_strategy="column_wise",
        split_strategy="column_split",
        data_type="float16"
    )
    
    # Create a fully connected network
    fc1 = llm.create_fc_network(
        name="fc1",
        input_dim=32,
        output_dim=32,
        seq_len=6,
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    # Create a matrix multiplication network
    matmul = llm.create_arithmetic_network(
        name="matmul",
        seq_len=6,
        d_model=32,
        operation="matmul",
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    # Set execution order
    llm.set_execution_order([
        ["fc1"],
        ["matmul"]
    ])
    
    # Connect fc1 output to matmul input A
    llm.connect_networks("fc1", "matmul", "matmul_a")
    
    # Create a mapping of PE coordinates to network names
    pe_to_network = {}
    for name, network in llm.networks.items():
        for pe_coord in network.active_pes:
            pe_to_network[pe_coord] = name
    
    # Open log file for writing
    with open(log_file, "w") as f:
        # Log basic parameters
        f.write(f"Network Allocation Test Parameters:\n")
        f.write(f"Input Dimension: 32\n")
        f.write(f"Output Dimension: 32\n")
        f.write(f"Sequence Length: {llm.seq_len}\n")
        f.write(f"PE Memory Size: {llm.pe_memory_size} bytes\n")
        f.write(f"Mapping Strategy: {llm.mapping_strategy}\n")
        f.write(f"Split Strategy: {llm.split_strategy}\n")
        f.write(f"NoC Size: {llm.noc.rows}x{llm.noc.cols} ({llm.noc.rows*llm.noc.cols} PEs)\n")
        f.write(f"Data Type: {llm.data_type}\n\n")
        
        # Log PE allocation details
        f.write("=" * 80 + "\n")
        f.write("                          PE ALLOCATIONS                          \n")
        f.write("=" * 80 + "\n")
        
        # Print PE allocations to log
        f.write("\nPE Allocations by Network:\n")
        for name, network in llm.networks.items():
            f.write(f"{name} PE Allocations:" + "=" * 60 + "\n")
            if hasattr(network, 'mapper') and hasattr(network.mapper, 'get_pe_details'):
                pe_details = network.mapper.get_pe_details()
                f.write(pe_details.to_string() + "\n")
            f.write(f"{name} network active PEs: {network.active_pes}\n")
        
        # Create a visual representation of the NoC grid
        f.write("\nNoC Grid Layout (showing network assignment at each PE):\n")
        f.write("-" * 50 + "\n")
        
        # Create a grid representation of the NoC
        noc_grid = [[' ' for _ in range(llm.noc.cols)] for _ in range(llm.noc.rows)]
        
        # Map networks to symbols for the grid
        network_symbols = {
            'fc1': 'F',
            'matmul': 'M'
        }
        
        # Fill in the grid with network symbols
        for name, network in llm.networks.items():
            symbol = network_symbols.get(name, '?')
            for coords in network.active_pes:
                x, y = coords
                if 0 <= x < llm.noc.cols and 0 <= y < llm.noc.rows:
                    noc_grid[y][x] = symbol
        
        # Write the grid to the log file
        for y in range(min(10, llm.noc.rows)):
            row_str = '|'
            for x in range(min(10, llm.noc.cols)):
                row_str += f' {noc_grid[y][x] or "."} |'
            f.write(row_str + "\n")
        f.write("-" * 50 + "\n\n")
        
        # Create test inputs
        inputs = {
            "fc1": torch.randn(6, 32),  # Input for fc1
            "matmul_b": torch.randn(6, 32)  # Second input for matmul
        }
        
        # Run fc1 inference first
        fc1_outputs = fc1.run_inference(inputs["fc1"])
    
        # Get fc1's output PE coordinates for printing
        fc1_output_pes = list(fc1_outputs.keys())
        
        # Run matmul with fc1's output as input_a
        matmul_outputs = matmul.matrix_multiply(
            input_a=fc1_outputs,  # Pass the entire dictionary of outputs from fc1
            input_b=inputs["matmul_b"],  # Use the direct input tensor
            transpose_b=True,  # Default: Q @ K^T
            source_pe_b=(-1, 1)  # Virtual PE for input_b
        )
        
        # Log output shapes
        f.write("Output shapes:\n")
        f.write(f"fc1: {fc1_outputs[fc1_output_pes[0]][0].shape}\n")
        for pe_coords, (tensor, _, _) in matmul_outputs.items():
            f.write(f"matmul: {tensor.shape}\n")
        
        # Get traffic table
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
        
        # Log detailed traffic statistics
        f.write("\nDetailed Traffic Statistics:\n")
        f.write("=" * 80 + "\n\n")
        
        # Add combined traffic table to log
        f.write("\nCombined Traffic Table (All Networks):\n")
        f.write("=" * 80 + "\n")
        display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'cycles', 'wait_ids', 'description']
        f.write(traffic_table[display_cols].to_string() + "\n\n")
        
        # Log traffic for each network
        for name, network in llm.networks.items():
            f.write(f"{name} Network Traffic Summary:\n")
            f.write("-" * 40 + "\n")
            
            # Get network-specific traffic by filtering on src_network or dest_network
            if not traffic_table.empty:
                network_traffic = traffic_table[
                    (traffic_table['src_network'] == name) | 
                    (traffic_table['dest_network'] == name)
                ]
                
                # Calculate total bytes and tasks
                total_bytes = network_traffic['bytes'].sum()
                total_tasks = len(network_traffic)
                avg_bytes_per_task = total_bytes / total_tasks if total_tasks > 0 else 0
                
                f.write(f"Total bytes transferred: {total_bytes:,} bytes\n")
                f.write(f"Total communication tasks: {total_tasks}\n")
                f.write(f"Average bytes per task: {avg_bytes_per_task:.2f} bytes\n\n")
                
                # Print detailed traffic table
                f.write(f"{name} Traffic Table:\n")
                f.write("-" * 40 + "\n")
                # Use the enhanced PE columns that include network names
                display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'cycles', 'wait_ids', 'description']
                f.write(network_traffic[display_cols].to_string() + "\n\n")
                
                # Group by source-destination pairs for a summary
                f.write(f"{name} Communication Patterns:\n")
                f.write("-" * 40 + "\n")
                grouped = network_traffic.groupby(['src_pe_with_network', 'dest_pe_with_network'])['bytes'].sum().reset_index()
                f.write("Traffic between PE pairs:\n")
                for _, row in grouped.iterrows():
                    f.write(f"{row['src_pe_with_network']} -> {row['dest_pe_with_network']}: {row['bytes']} bytes\n")
                f.write("\n")
            else:
                f.write("No traffic recorded for this network\n\n")
        
        # Log overall network statistics
        f.write("Overall Network Statistics:\n")
        f.write("=" * 80 + "\n")
        total_bytes = traffic_table['bytes'].sum()
        total_tasks = len(traffic_table)
        avg_bytes_per_task = total_bytes / total_tasks if total_tasks > 0 else 0
        
        f.write(f"Total bytes across all networks: {total_bytes:,} bytes\n")
        f.write(f"Total communication tasks across all networks: {total_tasks}\n")
        f.write(f"Average bytes per task across all networks: {avg_bytes_per_task:.2f} bytes\n")
    
    # Print PE allocations
    print("\nPE Allocations:")
    for name, network in llm.networks.items():
        print(f"{name} network active PEs:", network.active_pes)
    
    # Print output shapes
    print("\nOutput shapes:")
    print(f"fc1 output shape at PE{fc1_output_pes[0]}:", fc1_outputs[fc1_output_pes[0]][0].shape)
    for pe_coords, (tensor, _, _) in matmul_outputs.items():
        print(f"matmul output shape at PE{pe_coords}:", tensor.shape)
    
    # Print traffic statistics with network names
    print("\nTraffic Statistics:")
    if not traffic_table.empty:
        # Print full combined traffic table with wait IDs
        print("\nCombined Traffic Table (All Networks):")
        pd.set_option('display.max_rows', 20)
        pd.set_option('display.width', 200)
        display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'cycles', 'wait_ids', 'description']
        print(traffic_table[display_cols].to_string())
        
        # Group by source-destination pairs
        grouped = traffic_table.groupby(['src_pe_with_network', 'dest_pe_with_network']).agg({
            'bytes': 'sum',
            'task_id': 'count'
        }).reset_index()
        
        # Rename columns for clarity
        grouped.columns = ['Source PE (Network)', 'Destination PE (Network)', 'Total Bytes', 'Total Tasks']
        
        # Add average bytes per task
        grouped['Avg Bytes/Task'] = grouped['Total Bytes'] / grouped['Total Tasks']
        
        # Format bytes with commas
        grouped['Total Bytes'] = grouped['Total Bytes'].apply(lambda x: f"{x:,}")
        grouped['Avg Bytes/Task'] = grouped['Avg Bytes/Task'].apply(lambda x: f"{x:,.2f}")
        
        print("\nTraffic by PE pairs:")
        print(grouped.to_string(index=False))
        
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
    
    print(f"\nLog file saved to: {log_file}")

if __name__ == "__main__":
    test_network_allocation() 