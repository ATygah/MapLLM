import torch
import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.llm import LLM

def test_triple_fc():
    """Test three fully connected networks connected in sequence through LLM."""
    print("\n===== Testing Triple FC Networks =====\n")
    
    # Create an LLM instance
    llm = LLM(
        seq_len=8,
        pe_memory_size=4096,  # 4KB per PE
        noc_rows=10,
        noc_cols=10,
        mapping_strategy="column_wise",
        split_strategy="column_split",
        data_type="float16"
    )
    
    # Create the first fully connected network
    fc1 = llm.create_fc_network(
        name="fc1",
        input_dim=64,
        output_dim=32,
        seq_len=8,
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    # Create the second fully connected network
    fc2 = llm.create_fc_network(
        name="fc2",
        input_dim=32,  # Must match fc1's output dimension
        output_dim=16,
        seq_len=8,
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    # Create the third fully connected network
    fc3 = llm.create_fc_network(
        name="fc3",
        input_dim=16,  # Must match fc2's output dimension
        output_dim=8,
        seq_len=8,
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    # Print network configurations
    print("\nNetwork Configurations:")
    print(f"fc1: input_dim={fc1.input_dim}, output_dim={fc1.layer_dims[0]}, seq_len={fc1.seq_len}")
    print(f"fc2: input_dim={fc2.input_dim}, output_dim={fc2.layer_dims[0]}, seq_len={fc2.seq_len}")
    print(f"fc3: input_dim={fc3.input_dim}, output_dim={fc3.layer_dims[0]}, seq_len={fc3.seq_len}")
    
    # Set execution order
    print("\nSetting execution order...")
    llm.set_execution_order([
        ["fc1"],  # First run fc1
        ["fc2"],  # Then run fc2
        ["fc3"]   # Finally run fc3
    ])
    
    # Print connection status before connecting
    print("Connections before:", llm.connections)
    
    # Connect networks
    print("\nConnecting networks...")
    llm.connect_networks("fc1", "fc2")
    llm.connect_networks("fc2", "fc3")
    
    # Print connection status after connecting
    print("Connections after:", llm.connections)
    
    # Create a mapping of PE coordinates to network names
    pe_to_network = {}
    for name, network in llm.networks.items():
        for pe_coord in network.active_pes:
            pe_to_network[pe_coord] = name
    
    # Create test input for fc1
    print("\nCreating test input...")
    input_tensor = torch.randn(8, 64)  # Input for fc1 [seq_len, input_dim]
    inputs = {
        "fc1": input_tensor
    }
    
    # Run inference through the LLM
    print("Running inference through LLM...")
    outputs = llm.run_inference(inputs)
    
    # Print active PEs
    print("\nActive Processing Elements:")
    print(f"fc1: {fc1.active_pes}")
    print(f"fc2: {fc2.active_pes}")
    print(f"fc3: {fc3.active_pes}")
    
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
        print("\nCombined Traffic Table (All Networks):")
        pd.set_option('display.max_rows', 20)
        pd.set_option('display.width', 200)
        display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'cycles', 'wait_ids', 'description']
        print(traffic_table[display_cols].to_string())
        
        # Print the traffic table
        print("\nDetailed Traffic Table (Top 10 rows):")
        print(f"Total rows: {len(traffic_table)}")
        display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'cycles', 'wait_ids', 'description']
        print(traffic_table[display_cols].head(10))
        
        # Group traffic by network
        if 'network_id' in traffic_table.columns:
            network_stats = traffic_table.groupby('network_id').agg({
                'bytes': 'sum',
                'task_id': 'count'
            }).reset_index()
            
            print("\nTraffic by network:")
            print(network_stats)
        
        # Analyze traffic between networks
        print("\nTraffic between networks:")
        network_traffic = {
            "fc1_to_fc2": 0,
            "fc2_to_fc3": 0
        }
        
        for idx, row in traffic_table.iterrows():
            if row['src_network'] == 'fc1' and row['dest_network'] == 'fc2':
                network_traffic["fc1_to_fc2"] += row['bytes']
                print(f"Traffic from fc1 PE{row['src_pe']} to fc2 PE{row['dest_pe']}: {row['bytes']} bytes (Task ID: {row['task_id']})")
            
            if row['src_network'] == 'fc2' and row['dest_network'] == 'fc3':
                network_traffic["fc2_to_fc3"] += row['bytes']
                print(f"Traffic from fc2 PE{row['src_pe']} to fc3 PE{row['dest_pe']}: {row['bytes']} bytes (Task ID: {row['task_id']})")
        
        print("\nTotal traffic between networks:")
        print(f"fc1 to fc2: {network_traffic['fc1_to_fc2']:,} bytes")
        print(f"fc2 to fc3: {network_traffic['fc2_to_fc3']:,} bytes")
        
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
    
    # Save results to log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"triple_fc_test_{timestamp}.log")
    
    with open(log_file, "w") as f:
        f.write("===== Triple FC Networks Test =====\n\n")
        f.write(f"First network (fc1): Input dim={64}, Output dim={32}\n")
        f.write(f"Second network (fc2): Input dim={32}, Output dim={16}\n")
        f.write(f"Third network (fc3): Input dim={16}, Output dim={8}\n\n")
        
        f.write("Active PEs:\n")
        f.write(f"fc1: {fc1.active_pes}\n")
        f.write(f"fc2: {fc2.active_pes}\n")
        f.write(f"fc3: {fc3.active_pes}\n\n")
        
        f.write("Output shapes:\n")
        for name, tensor in outputs.items():
            f.write(f"{name}: {tensor.shape}\n")
        
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
                    
                    # Print detailed traffic table
                    f.write(f"{name} Traffic Table:\n")
                    f.write("-" * 40 + "\n")
                    # Use the enhanced PE columns that include network names
                    display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'cycles', 'wait_ids', 'description']
                    f.write(network_traffic_df[display_cols].to_string() + "\n\n")
                    
                    # Group by source-destination pairs for a summary
                    f.write(f"{name} Communication Patterns:\n")
                    f.write("-" * 40 + "\n")
                    grouped = network_traffic_df.groupby(['src_pe_with_network', 'dest_pe_with_network'])['bytes'].sum().reset_index()
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
            
            # Add combined traffic table to log
            f.write("\nCombined Traffic Table (All Networks):\n")
            f.write("=" * 80 + "\n")
            display_cols = ['task_id', 'src_pe_with_network', 'dest_pe_with_network', 'bytes', 'cycles', 'wait_ids', 'description']
            f.write(traffic_table[display_cols].to_string() + "\n\n")
            
            # Log traffic between networks
            f.write("\nTraffic between networks:\n")
            network_traffic = {
                "fc1_to_fc2": 0,
                "fc2_to_fc3": 0
            }
            
            for idx, row in traffic_table.iterrows():
                if row['src_network'] == 'fc1' and row['dest_network'] == 'fc2':
                    network_traffic["fc1_to_fc2"] += row['bytes']
                if row['src_network'] == 'fc2' and row['dest_network'] == 'fc3':
                    network_traffic["fc2_to_fc3"] += row['bytes']
            
            f.write(f"fc1 to fc2: {network_traffic['fc1_to_fc2']:,} bytes\n")
            f.write(f"fc2 to fc3: {network_traffic['fc2_to_fc3']:,} bytes\n")
    
    print(f"\nLog saved to: {log_file}")
    print("\nTriple FC networks test completed successfully!")
    return outputs

if __name__ == "__main__":
    test_triple_fc() 