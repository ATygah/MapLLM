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
from llmDeploy.layer_mapper import FCLayerMapper
from llmDeploy.network_analysis import (
    analyze_network_traffic, create_pe_to_network_mapping, 
    calculate_network_bounds, calculate_pe_density,
    generate_network_layout_visualization,
    write_network_metrics_to_log,
    compare_mapping_strategies_metrics,
    rank_mapping_strategies,
    get_strategy_qualitative_assessment
)

def test_mlp_simple():
    """
    A simplified test for MLP with minimal dimensions.
    """
    print("Starting simplified MLP test...")
    
    # Initialize the log file before running the test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "../logs/map")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"mlp_simple_MAP_00.log")
    
    with open(log_file, "w") as f:
        f.write("===== Testing Simplified MLP =====\n\n")
        
        # Very small dimensions for testing
        seq_len = 1
        d_model = 768
        d_hidden = 3072  # Small hidden dimension
        
        f.write(f"Sequence length: {seq_len}\n")
        f.write(f"Model dimension (d_model): {d_model}\n")
        f.write(f"Hidden dimension: {d_hidden}\n\n")
        
        print(f"Using dimensions: seq_len={seq_len}, d_model={d_model}, d_hidden={d_hidden}")
        
        # Create a minimal LLM instance
        f.write("Creating LLM instance...\n")
        llm = LLM(
            seq_len=seq_len,
            pe_memory_size=64*1024,  # Increased memory size
            noc_rows=20,              # Increased grid size
            noc_cols=30,
            mapping_strategy="grid_wise",
            split_strategy="hybrid_split",
            data_type="float16",
            allow_wrapping=True
        )
        f.write("LLM instance created successfully.\n\n")
        
        # Create the MLP networks
        f.write("Creating MLP networks...\n")
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
        f.write(f"mlp1: input_dim={mlp1.input_dim}, output_dim={mlp1.layer_dims[0]}, seq_len={mlp1.seq_len}\n")
        f.write(f"mlp2: input_dim={mlp2.input_dim}, output_dim={mlp2.layer_dims[0]}, seq_len={mlp2.seq_len}\n\n")
        
        # Calculate and display effective NoC dimensions using the existing method
        f.write("\nEffective NoC Dimensions and Layer Positions:\n")
        f.write("-" * 50 + "\n")
        
        # Get all active PEs for calculation
        all_pes = set()
        for name, network in llm.networks.items():
            all_pes.update(network.active_pes)
        
        if all_pes:
            # Calculate positions for each network using the FCLayerMapper's method
            mlp1_bounds = mlp1.mapper.get_network_bounds()
            mlp2_bounds = mlp2.mapper.get_network_bounds()
            
            # Calculate overall dimensions
            x_coords = [x for x, y in all_pes]
            y_coords = [y for x, y in all_pes]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            effective_width = x_max - x_min + 1
            effective_height = y_max - y_min + 1
            effective_area = effective_width * effective_height
            utilization = (len(all_pes) / effective_area) * 100
            
            # Write overall effective NoC dimensions
            f.write("Overall Effective NoC Dimensions:\n")
            f.write(f"Width: {effective_width}, Height: {effective_height}, Area: {effective_area}\n")
            f.write(f"X-range: {x_min}-{x_max}, Y-range: {y_min}-{y_max}\n")
            f.write(f"Total PEs used: {len(all_pes)}\n")
            f.write(f"Utilization within effective area: {utilization:.2f}%\n\n")
            
            # Calculate using the FCLayerMapper's method for comparison
            # For the combined dimensions, we need a special approach
            class DummyNN:
                def __init__(self, active_pes):
                    self.active_pes = active_pes
            
            # Get effective dimensions directly from each network's mapper
            mlp1_dims = mlp1.mapper.get_effective_noc_dimensions()
            mlp2_dims = mlp2.mapper.get_effective_noc_dimensions()
            
            # For combined dimensions, we need to temporarily attach all PEs to a mapper
            # Save original neural_network reference
            original_nn = mlp1.mapper.neural_network
            
            # Create a temporary neural network with all PEs
            combined_nn = DummyNN(all_pes)
            mlp1.mapper.neural_network = combined_nn
            
            # Get combined dimensions
            combined_dims = mlp1.mapper.get_effective_noc_dimensions()
            
            # Restore original neural_network reference
            mlp1.mapper.neural_network = original_nn
            
            # Write network-specific effective dimensions from the method
            f.write("Effective Dimensions (from get_effective_noc_dimensions):\n")
            f.write(f"mlp1: Rows: {mlp1_dims[0]}, Columns: {mlp1_dims[1]}, Grid size: {mlp1_dims[2]}\n")
            f.write(f"mlp2: Rows: {mlp2_dims[0]}, Columns: {mlp2_dims[1]}, Grid size: {mlp2_dims[2]}\n")
            f.write(f"Combined: Rows: {combined_dims[0]}, Columns: {combined_dims[1]}, Grid size: {combined_dims[2]}\n\n")
            
            # Write layer-specific positions
            f.write("Layer Positions:\n")
            if mlp1_bounds:
                f.write(f"mlp1: X-range: {mlp1_bounds['x_range'][0]}-{mlp1_bounds['x_range'][1]} (width: {mlp1_bounds['width']}), "
                      f"Y-range: {mlp1_bounds['y_range'][0]}-{mlp1_bounds['y_range'][1]} (height: {mlp1_bounds['height']})\n")
            if mlp2_bounds:
                f.write(f"mlp2: X-range: {mlp2_bounds['x_range'][0]}-{mlp2_bounds['x_range'][1]} (width: {mlp2_bounds['width']}), "
                      f"Y-range: {mlp2_bounds['y_range'][0]}-{mlp2_bounds['y_range'][1]} (height: {mlp2_bounds['height']})\n")
            
            # Calculate and write PE density for each layer
            mlp1_density = (len(mlp1.active_pes) / mlp1_bounds['area']) * 100 if mlp1_bounds else 0
            mlp2_density = (len(mlp2.active_pes) / mlp2_bounds['area']) * 100 if mlp2_bounds else 0
            
            f.write("\nPE Density (percentage of PEs used within layer's bounding box):\n")
            f.write(f"mlp1: {len(mlp1.active_pes)} PEs in {mlp1_bounds['area']} possible positions = {mlp1_density:.2f}% density\n")
            f.write(f"mlp2: {len(mlp2.active_pes)} PEs in {mlp2_bounds['area']} possible positions = {mlp2_density:.2f}% density\n")
            f.write(f"Overall: {len(all_pes)} PEs in {effective_area} possible positions = {utilization:.2f}% density\n")
            f.write("-" * 50 + "\n\n")
        
        # Set execution order
        f.write("\nSetting execution order...\n")
        execution_order = [
            ["mlp1"],  # First layer
            ["mlp2"]   # Second layer
        ]
        
        llm.set_execution_order(execution_order)
        f.write(f"Execution order: {execution_order}\n\n")
        print(f"Execution order: {execution_order}")
        
        # Connect networks
        f.write("\nConnecting networks...\n")
        print("Connecting networks...")
        
        # Connect MLP layers
        llm.connect_networks("mlp1", "mlp2")
        f.write("Connected: mlp1 -> mlp2\n")
        print("Connected: mlp1 -> mlp2")
        
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
            "mlp1": input_tensor
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
            # try:
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     traffic_filename = f"mlp_simple_{timestamp}"
            #     traces_dir = os.path.join(os.path.dirname(__file__), "../traces")
            #     export_paths = export_traffic_table_to_file(llm, traffic_filename, directory=traces_dir)
            #     f.write(f"\nTraffic tables exported to:\n")
            #     f.write(f"- TXT format: {export_paths['txt']}\n")
            #     f.write(f"- TSV format: {export_paths['tsv']}\n")
            #     print(f"Traffic tables exported to: {export_paths['txt']} and {export_paths['tsv']}")
            # except Exception as e:
            #     error_msg = f"Error exporting traffic table: {e}"
            #     f.write(f"\n{error_msg}\n")
            #     print(error_msg)
            #     traceback.print_exc()
            
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

def test_mlp_compact():
    """
    A simplified test for MLP with compact mapping strategy.
    """
    print("Starting MLP test with compact mapping strategy...")
    
    # Initialize the log file before running the test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "../logs/map")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"mlp_compact_MAP_{timestamp}.log")
    
    with open(log_file, "w") as f:
        f.write("===== Testing MLP with Compact Mapping =====\n\n")
        
        # Very small dimensions for testing
        seq_len = 1
        d_model = 768
        d_hidden = 3072  # Small hidden dimension
        
        f.write(f"Sequence length: {seq_len}\n")
        f.write(f"Model dimension (d_model): {d_model}\n")
        f.write(f"Hidden dimension: {d_hidden}\n\n")
        
        print(f"Using dimensions: seq_len={seq_len}, d_model={d_model}, d_hidden={d_hidden}")
        
        # Create a minimal LLM instance with compact mapping
        f.write("Creating LLM instance with compact mapping...\n")
        llm = LLM(
            seq_len=seq_len,
            pe_memory_size=64*1024,  # Increased memory size
            noc_rows=30,              # Increased grid size
            noc_cols=30,
            mapping_strategy="compact",  # Using our new compact mapping strategy
            split_strategy="hybrid_split",
            data_type="float16",
            allow_wrapping=True
        )
        f.write("LLM instance created successfully.\n\n")
        
        # Create the MLP networks with compact mapping
        f.write("Creating MLP networks with compact mapping...\n")
        mlp1 = llm.create_fc_network(
            name="mlp1",
            input_dim=d_model,
            output_dim=d_hidden,
            seq_len=seq_len,
            mapping_strategy="compact",  # Using our new compact mapping strategy
            split_strategy="hybrid_split"
        )

        mlp2 = llm.create_fc_network(
            name="mlp2",
            input_dim=d_hidden,
            output_dim=d_model,
            seq_len=seq_len,
            mapping_strategy="compact",  # Using our new compact mapping strategy
            split_strategy="hybrid_split"
        )
        
        # Write network configurations to log
        f.write("\nNetwork Configurations:\n")
        f.write(f"mlp1: input_dim={mlp1.input_dim}, output_dim={mlp1.layer_dims[0]}, seq_len={mlp1.seq_len}\n")
        f.write(f"mlp2: input_dim={mlp2.input_dim}, output_dim={mlp2.layer_dims[0]}, seq_len={mlp2.seq_len}\n\n")
        
        # Calculate and display effective NoC dimensions using the existing method
        f.write("\nEffective NoC Dimensions and Layer Positions:\n")
        f.write("-" * 50 + "\n")
        
        # Get all active PEs for calculation
        all_pes = set()
        for name, network in llm.networks.items():
            all_pes.update(network.active_pes)
        
        if all_pes:
            # Calculate positions for each network using the FCLayerMapper's method
            mlp1_bounds = mlp1.mapper.get_network_bounds()
            mlp2_bounds = mlp2.mapper.get_network_bounds()
            
            # Calculate overall dimensions
            x_coords = [x for x, y in all_pes]
            y_coords = [y for x, y in all_pes]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            effective_width = x_max - x_min + 1
            effective_height = y_max - y_min + 1
            effective_area = effective_width * effective_height
            utilization = (len(all_pes) / effective_area) * 100
            
            # Write overall effective NoC dimensions
            f.write("Overall Effective NoC Dimensions:\n")
            f.write(f"Width: {effective_width}, Height: {effective_height}, Area: {effective_area}\n")
            f.write(f"X-range: {x_min}-{x_max}, Y-range: {y_min}-{y_max}\n")
            f.write(f"Total PEs used: {len(all_pes)}\n")
            f.write(f"Utilization within effective area: {utilization:.2f}%\n\n")
            
            # Calculate using the FCLayerMapper's method for comparison
            # For the combined dimensions, we need a special approach
            class DummyNN:
                def __init__(self, active_pes):
                    self.active_pes = active_pes
            
            # Get effective dimensions directly from each network's mapper
            mlp1_dims = mlp1.mapper.get_effective_noc_dimensions()
            mlp2_dims = mlp2.mapper.get_effective_noc_dimensions()
            
            # For combined dimensions, we need to temporarily attach all PEs to a mapper
            # Save original neural_network reference
            original_nn = mlp1.mapper.neural_network
            
            # Create a temporary neural network with all PEs
            combined_nn = DummyNN(all_pes)
            mlp1.mapper.neural_network = combined_nn
            
            # Get combined dimensions
            combined_dims = mlp1.mapper.get_effective_noc_dimensions()
            
            # Restore original neural_network reference
            mlp1.mapper.neural_network = original_nn
            
            # Write network-specific effective dimensions from the method
            f.write("Effective Dimensions (from get_effective_noc_dimensions):\n")
            f.write(f"mlp1: Rows: {mlp1_dims[0]}, Columns: {mlp1_dims[1]}, Grid size: {mlp1_dims[2]}\n")
            f.write(f"mlp2: Rows: {mlp2_dims[0]}, Columns: {mlp2_dims[1]}, Grid size: {mlp2_dims[2]}\n")
            f.write(f"Combined: Rows: {combined_dims[0]}, Columns: {combined_dims[1]}, Grid size: {combined_dims[2]}\n\n")
            
            # Write layer-specific positions
            f.write("Layer Positions:\n")
            if mlp1_bounds:
                f.write(f"mlp1: X-range: {mlp1_bounds['x_range'][0]}-{mlp1_bounds['x_range'][1]} (width: {mlp1_bounds['width']}), "
                      f"Y-range: {mlp1_bounds['y_range'][0]}-{mlp1_bounds['y_range'][1]} (height: {mlp1_bounds['height']})\n")
            if mlp2_bounds:
                f.write(f"mlp2: X-range: {mlp2_bounds['x_range'][0]}-{mlp2_bounds['x_range'][1]} (width: {mlp2_bounds['width']}), "
                      f"Y-range: {mlp2_bounds['y_range'][0]}-{mlp2_bounds['y_range'][1]} (height: {mlp2_bounds['height']})\n")
            
            # Calculate and write PE density for each layer
            mlp1_density = (len(mlp1.active_pes) / mlp1_bounds['area']) * 100 if mlp1_bounds else 0
            mlp2_density = (len(mlp2.active_pes) / mlp2_bounds['area']) * 100 if mlp2_bounds else 0
            
            f.write("\nPE Density (percentage of PEs used within layer's bounding box):\n")
            f.write(f"mlp1: {len(mlp1.active_pes)} PEs in {mlp1_bounds['area']} possible positions = {mlp1_density:.2f}% density\n")
            f.write(f"mlp2: {len(mlp2.active_pes)} PEs in {mlp2_bounds['area']} possible positions = {mlp2_density:.2f}% density\n")
            f.write(f"Overall: {len(all_pes)} PEs in {effective_area} possible positions = {utilization:.2f}% density\n")
            f.write("-" * 50 + "\n\n")
        
        # Set execution order
        f.write("\nSetting execution order...\n")
        execution_order = [
            ["mlp1"],  # First layer
            ["mlp2"]   # Second layer
        ]
        
        llm.set_execution_order(execution_order)
        f.write(f"Execution order: {execution_order}\n\n")
        print(f"Execution order: {execution_order}")
        
        # Connect networks
        f.write("\nConnecting networks...\n")
        print("Connecting networks...")
        
        # Connect MLP layers
        llm.connect_networks("mlp1", "mlp2")
        f.write("Connected: mlp1 -> mlp2\n")
        print("Connected: mlp1 -> mlp2")
        
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
            "mlp1": input_tensor
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
            # For compact mapping, we want to see the actual layout clearly
            max_rows = 20  # Show more rows to visualize compact layout
            max_cols = 20  # Show more columns to visualize compact layout
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
                traffic_filename = f"mlp_compact_{timestamp}"
                traces_dir = os.path.join(os.path.dirname(__file__), "../traces")
                os.makedirs(traces_dir, exist_ok=True)
                export_paths = export_traffic_table_to_file(llm, traffic_filename, directory=traces_dir)
                f.write(f"\nTraffic tables exported to:\n")
                f.write(f"- TXT format: {export_paths['txt']}\n")
                f.write(f"- TSV format: {export_paths['tsv']}\n")
                print(f"Traffic tables exported to: {export_paths['txt']} and {export_paths['tsv']}")
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

def test_mapping_strategies():
    """Test and compare different mapping strategies for MLP networks."""
    
    # Create log file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(os.path.dirname(__file__), f"../logs/map/mlp_strategy_comparison_{timestamp}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Test dimensions
    seq_len = 1
    d_model = 768
    d_hidden = 3072
    
    # Define mapping strategies to test
    mapping_strategies = ["column_wise", "row_wise", "grid_wise", "compact", "proximity"]
    
    # Store results for comparison
    results = {}
    
    with open(log_file, "w") as f:
        f.write("===== Mapping Strategy Comparison Test =====\n\n")
        f.write(f"Test configuration:\n")
        f.write(f"- Sequence length: {seq_len}\n")
        f.write(f"- Model dimension (d_model): {d_model}\n")
        f.write(f"- Hidden dimension (d_hidden): {d_hidden}\n")
        f.write(f"- Strategies tested: {', '.join(mapping_strategies)}\n\n")
        
        # Test each mapping strategy
        for strategy in mapping_strategies:
            f.write(f"\n{'=' * 50}\n")
            f.write(f"TESTING STRATEGY: {strategy.upper()}\n")
            f.write(f"{'=' * 50}\n\n")
            
            print(f"Testing {strategy} strategy...")
            
            # Create a fresh LLM instance for this strategy
            # Using much larger NOC dimensions to accommodate all strategies
            llm = LLM(
                seq_len=seq_len,
                pe_memory_size=64*1024,
                noc_rows=100,
                noc_cols=100,
                mapping_strategy=strategy,
                split_strategy="hybrid_split",
                data_type="float16",
                allow_wrapping=True
            )
            
            f.write(f"Creating networks with {strategy} mapping strategy...\n")
            
            # Create two identical MLP networks with this strategy
            mlp1 = llm.create_fc_network(
                name="mlp1",
                input_dim=d_model,
                output_dim=d_hidden,
                seq_len=seq_len,
                mapping_strategy=strategy,
                split_strategy="hybrid_split"
            )
            
            mlp2 = llm.create_fc_network(
                name="mlp2",
                input_dim=d_hidden,
                output_dim=d_model,
                seq_len=seq_len,
                mapping_strategy=strategy,
                split_strategy="hybrid_split"
            )
            
            # Set execution order and connect networks
            execution_order = [["mlp1"], ["mlp2"]]
            llm.set_execution_order(execution_order)
            llm.connect_networks("mlp1", "mlp2")
            
            # Collect all active PEs
            all_pes = set()
            for name, network in llm.networks.items():
                all_pes.update(network.active_pes)
            
            # Get the bounds for each network using the reusable function
            mlp1_bounds = calculate_network_bounds(mlp1.active_pes)
            mlp2_bounds = calculate_network_bounds(mlp2.active_pes)
            
            # Calculate overall area metrics
            x_coords = [x for x, y in all_pes]
            y_coords = [y for x, y in all_pes]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            effective_width = x_max - x_min + 1
            effective_height = y_max - y_min + 1
            effective_area = effective_width * effective_height
            pe_count = len(all_pes)
            utilization = (pe_count / effective_area) * 100
            
            # Calculate per-network metrics using the reusable function
            mlp1_pe_count = len(mlp1.active_pes)
            mlp1_area = mlp1_bounds['area'] if mlp1_bounds else 0
            mlp1_density = calculate_pe_density(mlp1.active_pes, mlp1_bounds)
            
            mlp2_pe_count = len(mlp2.active_pes)
            mlp2_area = mlp2_bounds['area'] if mlp2_bounds else 0
            mlp2_density = calculate_pe_density(mlp2.active_pes, mlp2_bounds)
            
            # Log the area metrics
            f.write("\n--- Area Metrics ---\n")
            f.write(f"Overall effective NoC dimensions: {effective_width}×{effective_height}\n")
            f.write(f"Total effective area: {effective_area} positions\n")
            f.write(f"Total PEs used: {pe_count}\n")
            f.write(f"Overall utilization: {utilization:.2f}%\n\n")
            
            f.write(f"mlp1 dimensions: {mlp1_bounds['width']}×{mlp1_bounds['height']} (area: {mlp1_area})\n")
            f.write(f"mlp1 utilization: {mlp1_density:.2f}%\n")
            f.write(f"mlp2 dimensions: {mlp2_bounds['width']}×{mlp2_bounds['height']} (area: {mlp2_area})\n")
            f.write(f"mlp2 utilization: {mlp2_density:.2f}%\n")
            
            # Create test input
            input_tensor = torch.randn(seq_len, d_model)
            inputs = {"mlp1": input_tensor}
            
            try:
                # Run inference to get traffic data
                f.write("\nRunning inference to gather traffic statistics...\n")
                outputs = llm.run_inference(inputs)
                
                # Get traffic statistics
                traffic_table = llm.noc.scheduler.get_traffic_table()
                
                if not traffic_table.empty:
                    # Create PE to network mapping
                    pe_to_network = create_pe_to_network_mapping(llm)
                    
                    # Use the network analysis module to analyze traffic
                    metrics = analyze_network_traffic(traffic_table, pe_to_network)
                    
                    # Write metrics to log
                    write_network_metrics_to_log(metrics, f)
                    
                    # Generate network layout visualization
                    generate_network_layout_visualization(llm, f)
                
                # Store the results for this strategy
                results[strategy] = {
                    "effective_area": effective_area,
                    "pe_count": pe_count,
                    "utilization": utilization,
                    "mlp1_density": mlp1_density,
                    "mlp2_density": mlp2_density
                }
                
                # Add traffic metrics to results if available
                if not traffic_table.empty and metrics.get("metrics_available", False):
                    results[strategy].update({
                        "avg_manhattan_distance": metrics.get("avg_manhattan_distance", 0),
                        "total_bytes": metrics.get("total_bytes", 0),
                        "weighted_distance": metrics.get("weighted_distance", 0),
                        "cross_network_avg_manhattan": metrics.get("cross_network_avg_manhattan", 0),
                        "avg_hops": metrics.get("avg_hops", 0),
                        "max_hops": metrics.get("max_hops", 0),
                        "cross_network_avg_hops": metrics.get("cross_network_avg_hops", 0),
                        "cross_network_max_hops": metrics.get("cross_network_max_hops", 0),
                        "total_hops": metrics.get("total_hops", 0),
                        "cross_network_total_hops": metrics.get("cross_network_total_hops", 0),
                        "intra_network_avg_manhattan": metrics.get("intra_network_avg_manhattan", 0),
                        "intra_network_avg_hops": metrics.get("intra_network_avg_hops", 0),
                        "intra_network_max_hops": metrics.get("intra_network_max_hops", 0),
                        "intra_network_total_hops": metrics.get("intra_network_total_hops", 0),
                        "intra_network_bytes": metrics.get("intra_network_bytes", 0)
                    })
                
            except Exception as e:
                error_msg = f"Error during inference for {strategy} strategy: {e}"
                f.write(f"\n{error_msg}\n")
                traceback.print_exc(file=f)
                print(error_msg)
        
        # Comparative analysis of all strategies
        compare_mapping_strategies_metrics(results, f)
        
        # Strategy rankings
        rank_mapping_strategies(results, f)
        
        # Qualitative assessment of strategies
        f.write("\nQualitative Assessment:\n\n")
        assessments = get_strategy_qualitative_assessment()
        
        for strategy in mapping_strategies:
            if strategy in assessments:
                f.write(f"{strategy.upper()}:\n")
                f.write("Pros:\n")
                for pro in assessments[strategy]["pros"]:
                    f.write(f"- {pro}\n")
                
                f.write("Cons:\n")
                for con in assessments[strategy]["cons"]:
                    f.write(f"- {con}\n")
                
                f.write("\n")
        
        # Overall recommendation
        f.write("\nOverall Recommendation:\n")
        f.write("Based on the comparative analysis, the following recommendations can be made:\n\n")
        
        # This is a placeholder - in a real test, this would be determined by the actual results
        f.write("1. For maximum area efficiency: compact mapping is typically best\n")
        f.write("2. For predictable layouts with simple implementation: column_wise or row_wise\n")
        f.write("3. For the best balance of communication and area efficiency: grid_wise is often a good compromise\n")
        f.write("4. For networks that need to communicate heavily between layers: proximity mapping is ideal\n")
        
        # Final notes
        f.write("\nNotes:\n")
        f.write("- The best mapping strategy may depend on the specific network topology and communication patterns\n")
        f.write("- Larger networks may benefit more from sophisticated mapping strategies\n")
        f.write("- The tradeoff between implementation complexity and performance should be considered\n")
        f.write("- Proximity mapping is particularly useful when minimizing communication distance is critical\n")
        
        f.write("\nTest completed. Results saved to this log file.")
    
    print(f"Mapping strategy comparison test completed. Results saved to {log_file}")
    return results

def test_mapping_strategies_basic():
    """
    Test and compare different mapping strategies for MLP networks.
    This is a simpler version without using the helper functions.
    """
    # Create log file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(os.path.dirname(__file__), f"../logs/map/mlp_strategy_basic_{timestamp}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Test dimensions
    seq_len = 1
    d_model = 768
    d_hidden = 3072
    
    # Define mapping strategies to test
    mapping_strategies = ["column_wise", "row_wise", "grid_wise", "compact"]
    
    # Store results for comparison
    results = {}
    
    with open(log_file, "w") as f:
        f.write("===== Basic Mapping Strategy Comparison Test =====\n\n")
        f.write(f"Test configuration:\n")
        f.write(f"- Sequence length: {seq_len}\n")
        f.write(f"- Model dimension (d_model): {d_model}\n")
        f.write(f"- Hidden dimension (d_hidden): {d_hidden}\n")
        f.write(f"- Strategies tested: {', '.join(mapping_strategies)}\n\n")
        
        # Test each mapping strategy
        for strategy in mapping_strategies:
            f.write(f"\n{'=' * 50}\n")
            f.write(f"TESTING STRATEGY: {strategy.upper()}\n")
            f.write(f"{'=' * 50}\n\n")
            
            print(f"Testing {strategy} strategy...")
            
            # Create a fresh LLM instance for this strategy
            llm = LLM(
                seq_len=seq_len,
                pe_memory_size=64*1024,
                noc_rows=30,
                noc_cols=30,
                mapping_strategy=strategy,
                split_strategy="hybrid_split",
                data_type="float16",
                allow_wrapping=True
            )
            
            f.write(f"Creating networks with {strategy} mapping strategy...\n")
            
            # Create two identical MLP networks with this strategy
            mlp1 = llm.create_fc_network(
                name="mlp1",
                input_dim=d_model,
                output_dim=d_hidden,
                seq_len=seq_len,
                mapping_strategy=strategy,
                split_strategy="hybrid_split"
            )
            
            mlp2 = llm.create_fc_network(
                name="mlp2",
                input_dim=d_hidden,
                output_dim=d_model,
                seq_len=seq_len,
                mapping_strategy=strategy,
                split_strategy="hybrid_split"
            )
            
            # Set execution order and connect networks
            execution_order = [["mlp1"], ["mlp2"]]
            llm.set_execution_order(execution_order)
            llm.connect_networks("mlp1", "mlp2")
            
            # Collect all active PEs
            all_pes = set()
            for name, network in llm.networks.items():
                all_pes.update(network.active_pes)
                f.write(f"Network {name} uses {len(network.active_pes)} PEs\n")
            
            # Calculate overall area metrics
            x_coords = [x for x, y in all_pes]
            y_coords = [y for x, y in all_pes]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            effective_width = x_max - x_min + 1
            effective_height = y_max - y_min + 1
            effective_area = effective_width * effective_height
            pe_count = len(all_pes)
            utilization = (pe_count / effective_area) * 100
            
            # For individual networks, calculate bounds
            # mlp1 bounds
            mlp1_x = [x for x, y in mlp1.active_pes]
            mlp1_y = [y for x, y in mlp1.active_pes]
            mlp1_x_min, mlp1_x_max = min(mlp1_x), max(mlp1_x)
            mlp1_y_min, mlp1_y_max = min(mlp1_y), max(mlp1_y)
            mlp1_width = mlp1_x_max - mlp1_x_min + 1
            mlp1_height = mlp1_y_max - mlp1_y_min + 1
            mlp1_area = mlp1_width * mlp1_height
            mlp1_density = (len(mlp1.active_pes) / mlp1_area) * 100
            
            # mlp2 bounds
            mlp2_x = [x for x, y in mlp2.active_pes]
            mlp2_y = [y for x, y in mlp2.active_pes]
            mlp2_x_min, mlp2_x_max = min(mlp2_x), max(mlp2_x)
            mlp2_y_min, mlp2_y_max = min(mlp2_y), max(mlp2_y)
            mlp2_width = mlp2_x_max - mlp2_x_min + 1
            mlp2_height = mlp2_y_max - mlp2_y_min + 1
            mlp2_area = mlp2_width * mlp2_height
            mlp2_density = (len(mlp2.active_pes) / mlp2_area) * 100
            
            # Log the area metrics
            f.write("\n--- Area Metrics ---\n")
            f.write(f"Overall effective NoC dimensions: {effective_width}×{effective_height}\n")
            f.write(f"Total effective area: {effective_area} positions\n")
            f.write(f"Total PEs used: {pe_count}\n")
            f.write(f"Overall utilization: {utilization:.2f}%\n\n")
            
            f.write(f"mlp1 dimensions: {mlp1_width}×{mlp1_height} (area: {mlp1_area})\n")
            f.write(f"mlp1 utilization: {mlp1_density:.2f}%\n")
            f.write(f"mlp2 dimensions: {mlp2_width}×{mlp2_height} (area: {mlp2_area})\n")
            f.write(f"mlp2 utilization: {mlp2_density:.2f}%\n")
            
            # Create test input
            input_tensor = torch.randn(seq_len, d_model)
            inputs = {"mlp1": input_tensor}
            
            try:
                # Run inference to get traffic data
                f.write("\nRunning inference to gather traffic statistics...\n")
                outputs = llm.run_inference(inputs)
                
                # Get traffic statistics
                traffic_table = llm.noc.scheduler.get_traffic_table()
                
                if not traffic_table.empty:
                    # Create PE to network mapping
                    pe_to_network = {}
                    for name, network in llm.networks.items():
                        for pe_coord in network.active_pes:
                            pe_to_network[pe_coord] = name
                    
                    # Enhance the traffic table with network information
                    src_networks = []
                    dest_networks = []
                    manhattan_distances = []
                    
                    for _, row in traffic_table.iterrows():
                        # Extract PE coordinates from string like "(0, 0)"
                        src_pe_str = row['src_pe'].strip('()')
                        dest_pe_str = row['dest_pe'].strip('()') if row['dest_pe'] != "None" else ""
                        
                        # Get source network
                        if src_pe_str and ',' in src_pe_str:
                            src_x, src_y = map(int, src_pe_str.split(','))
                            src_pe = (src_x, src_y)
                            src_network = pe_to_network.get(src_pe, "external")
                        else:
                            src_network = "external"
                        src_networks.append(src_network)
                        
                        # Get destination network
                        if dest_pe_str and ',' in dest_pe_str:
                            dest_x, dest_y = map(int, dest_pe_str.split(','))
                            dest_pe = (dest_x, dest_y)
                            dest_network = pe_to_network.get(dest_pe, "external")
                            
                            # Calculate Manhattan distance if both PEs are defined
                            if src_pe_str and dest_pe_str and ',' in src_pe_str and ',' in dest_pe_str:
                                manhattan_dist = abs(dest_x - src_x) + abs(dest_y - src_y)
                                manhattan_distances.append(manhattan_dist)
                            else:
                                manhattan_distances.append(0)
                        else:
                            dest_network = "external"
                            manhattan_distances.append(0)
                        dest_networks.append(dest_network)
                    
                    # Add the network and distance information to the traffic table
                    traffic_table['src_network'] = src_networks
                    traffic_table['dest_network'] = dest_networks
                    traffic_table['manhattan_distance'] = manhattan_distances
                    
                    # Calculate traffic statistics
                    total_bytes = traffic_table['bytes'].sum()
                    total_transfers = len(traffic_table)
                    
                    # Calculate weighted Manhattan distance
                    traffic_table['weighted_distance'] = traffic_table['bytes'] * traffic_table['manhattan_distance']
                    total_weighted_distance = traffic_table['weighted_distance'].sum()
                    avg_manhattan_distance = total_weighted_distance / total_bytes if total_bytes > 0 else 0
                    
                    # Calculate cross-network vs. intra-network statistics
                    cross_network_traffic = traffic_table[traffic_table['src_network'] != traffic_table['dest_network']]
                    intra_network_traffic = traffic_table[traffic_table['src_network'] == traffic_table['dest_network']]
                    
                    cross_network_bytes = cross_network_traffic['bytes'].sum() if not cross_network_traffic.empty else 0
                    intra_network_bytes = intra_network_traffic['bytes'].sum() if not intra_network_traffic.empty else 0
                    
                    cross_network_weighted_distance = cross_network_traffic['weighted_distance'].sum() if not cross_network_traffic.empty else 0
                    intra_network_weighted_distance = intra_network_traffic['weighted_distance'].sum() if not intra_network_traffic.empty else 0
                    
                    cross_network_avg_distance = cross_network_weighted_distance / cross_network_bytes if cross_network_bytes > 0 else 0
                    intra_network_avg_distance = intra_network_weighted_distance / intra_network_bytes if intra_network_bytes > 0 else 0
                    
                    # Write traffic statistics to log
                    f.write("\n--- Traffic Statistics ---\n")
                    f.write(f"Total bytes transferred: {total_bytes:,}\n")
                    f.write(f"Total transfers: {total_transfers}\n")
                    f.write(f"Average Manhattan distance (weighted by bytes): {avg_manhattan_distance:.2f}\n")
                    f.write(f"Total weighted distance (bytes × distance): {total_weighted_distance:,}\n\n")
                    
                    f.write("Network Traffic Breakdown:\n")
                    f.write(f"Cross-network bytes: {cross_network_bytes:,} ({cross_network_bytes/total_bytes*100:.2f}% of total)\n")
                    f.write(f"Intra-network bytes: {intra_network_bytes:,} ({intra_network_bytes/total_bytes*100:.2f}% of total)\n")
                    f.write(f"Cross-network avg distance: {cross_network_avg_distance:.2f}\n")
                    f.write(f"Intra-network avg distance: {intra_network_avg_distance:.2f}\n\n")
                    
                    # Add a simple NoC grid visualization
                    f.write("\n--- NoC Grid Layout ---\n")
                    max_display_size = 20  # Limit display size
                    display_rows = min(effective_height, max_display_size)
                    display_cols = min(effective_width, max_display_size)
                    
                    # Only show the effective region
                    start_x = x_min
                    start_y = y_min
                    
                    # Create a grid with network names
                    noc_grid = [[' ' for _ in range(display_cols)] for _ in range(display_rows)]
                    
                    for name, network in llm.networks.items():
                        for pe_x, pe_y in network.active_pes:
                            grid_x = pe_x - start_x
                            grid_y = pe_y - start_y
                            if 0 <= grid_x < display_cols and 0 <= grid_y < display_rows:
                                noc_grid[grid_y][grid_x] = name[:3]  # First 3 chars of network name
                    
                    # Print column headers
                    f.write("    ")
                    for i in range(display_cols):
                        col_idx = start_x + i
                        f.write(f"{col_idx:^3}")
                    f.write("\n")
                    
                    # Add horizontal separator line with bisection marker
                    f.write("   +")
                    h_bisect_pos = int((avg_manhattan_distance - x_min) * 3)
                    for i in range(display_cols * 3):
                        if i == h_bisect_pos:
                            f.write("H")
                        else:
                            f.write("-")
                    f.write("+\n")
                    
                    # Draw the grid with vertical bisection marker
                    v_bisect_pos = int(avg_manhattan_distance - y_min)
                    for y in range(display_rows):
                        if y == v_bisect_pos:
                            marker = "V"
                        else:
                            marker = "|"
                        f.write(f"{y+y_min:2d} {marker}")
                        for x in range(display_cols):
                            f.write(f" {noc_grid[y][x]} ")
                        f.write("|\n")
                    
                    # Add bottom separator line
                    f.write("   +")
                    for i in range(display_cols * 3):
                        f.write("-")
                    f.write("+\n\n")
                    
                    # Store results for this strategy
                    results[strategy] = {
                        "effective_area": effective_area,
                        "pe_count": pe_count,
                        "utilization": utilization,
                        "avg_manhattan_distance": avg_manhattan_distance,
                        "total_bytes": total_bytes,
                        "weighted_distance": total_weighted_distance,
                        "cross_network_avg_distance": cross_network_avg_distance,
                        "intra_network_avg_distance": intra_network_avg_distance,
                        "cross_network_bytes": cross_network_bytes,
                        "intra_network_bytes": intra_network_bytes
                    }
                
            except Exception as e:
                error_msg = f"Error during inference for {strategy} strategy: {e}"
                f.write(f"\n{error_msg}\n")
                traceback.print_exc(file=f)
                print(error_msg)
        
        # Compare all strategies side by side
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("                      STRATEGY COMPARISON SUMMARY                      \n")
        f.write("=" * 80 + "\n\n")
        
        if results:
            # Create a table of metrics
            f.write("Area Efficiency:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Strategy':<15} {'Area':<10} {'PEs':<8} {'Utilization':<15}\n")
            f.write("-" * 50 + "\n")
            for strategy, metrics in results.items():
                f.write(f"{strategy:<15} {metrics['effective_area']:<10} {metrics['pe_count']:<8} {metrics['utilization']:.2f}%\n")
            
            f.write("\nCommunication Efficiency:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Strategy':<15} {'Avg Distance':<15} {'Total Bytes':<15} {'Weighted Dist':<15}\n")
            f.write("-" * 70 + "\n")
            for strategy, metrics in results.items():
                if 'avg_manhattan_distance' in metrics:
                    f.write(f"{strategy:<15} {metrics['avg_manhattan_distance']:.2f} {metrics['total_bytes']:15,} {metrics['weighted_distance']:15,}\n")
            
            f.write("\nNetwork Communication Breakdown:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Strategy':<15} {'Cross-Net %':<15} {'Cross Dist':<15} {'Intra Dist':<15}\n")
            f.write("-" * 80 + "\n")
            for strategy, metrics in results.items():
                if 'cross_network_bytes' in metrics and 'total_bytes' in metrics:
                    cross_pct = (metrics['cross_network_bytes'] / metrics['total_bytes']) * 100
                    f.write(f"{strategy:<15} {cross_pct:6.2f}% {metrics['cross_network_avg_distance']:13.2f} {metrics['intra_network_avg_distance']:13.2f}\n")
            
            # Rank the strategies
            f.write("\n\nStrategy Rankings:\n")
            
            # Rank by area efficiency (higher utilization is better)
            area_ranking = sorted(results.keys(), key=lambda s: results[s]['utilization'], reverse=True)
            f.write("\nArea Efficiency Ranking (higher is better):\n")
            for i, strategy in enumerate(area_ranking):
                f.write(f"{i+1}. {strategy} (Utilization: {results[strategy]['utilization']:.2f}%)\n")
            
            # Rank by communication efficiency (lower weighted distance is better)
            if all('weighted_distance' in results[s] for s in results):
                comm_ranking = sorted(results.keys(), key=lambda s: results[s]['weighted_distance'])
                f.write("\nCommunication Efficiency Ranking (lower is better):\n")
                for i, strategy in enumerate(comm_ranking):
                    f.write(f"{i+1}. {strategy} (Weighted Distance: {results[strategy]['weighted_distance']:,})\n")
            
            # Rank by cross-network efficiency (lower cross-network distance is better)
            if all('cross_network_avg_distance' in results[s] for s in results):
                cross_ranking = sorted(results.keys(), key=lambda s: results[s]['cross_network_avg_distance'])
                f.write("\nCross-Network Efficiency Ranking (lower is better):\n")
                for i, strategy in enumerate(cross_ranking):
                    f.write(f"{i+1}. {strategy} (Cross-Network Avg Distance: {results[strategy]['cross_network_avg_distance']:.2f})\n")
        else:
            f.write("No results available for comparison.\n")
        
        # Qualitative assessment of strategies
        f.write("\nQualitative Assessment:\n\n")
        
        for strategy in mapping_strategies:
            f.write(f"{strategy.upper()}:\n")
            
            if strategy == "column_wise":
                f.write("Pros:\n")
                f.write("- Simple to implement\n")
                f.write("- Predictable layout\n")
                f.write("- Good for networks with few PEs per layer\n")
                f.write("Cons:\n")
                f.write("- Poor utilization for large networks\n")
                f.write("- May lead to long inter-network distances\n")
                f.write("- Doesn't optimize for communication patterns\n\n")
            
            elif strategy == "row_wise":
                f.write("Pros:\n")
                f.write("- Simple to implement\n")
                f.write("- Predictable layout\n")
                f.write("- Good for networks with few PEs per layer\n")
                f.write("Cons:\n")
                f.write("- Poor utilization for large networks\n")
                f.write("- May lead to long inter-network distances\n")
                f.write("- Doesn't optimize for communication patterns\n\n")
            
            elif strategy == "grid_wise":
                f.write("Pros:\n")
                f.write("- Better area utilization\n")
                f.write("- More flexible arrangement\n")
                f.write("- Works well with hybrid split strategy\n")
                f.write("Cons:\n")
                f.write("- More complex implementation\n")
                f.write("- May still have suboptimal communication patterns\n")
                f.write("- Can be unpredictable for different network sizes\n\n")
            
            elif strategy == "compact":
                f.write("Pros:\n")
                f.write("- Optimized for area efficiency\n")
                f.write("- Keeps related PEs close together\n")
                f.write("- Predictable layout shape\n")
                f.write("Cons:\n")
                f.write("- Higher intra-network distances\n")
                f.write("- May create irregular shapes\n")
                f.write("- Trade-off between internal and cross-network efficiency\n\n")
        
        # Overall recommendation
        f.write("\nOverall Recommendation:\n")
        f.write("Based on the comparative analysis, the following recommendations can be made:\n\n")
        
        # This is a placeholder - in a real test, this would be determined by the actual results
        f.write("1. For maximum area efficiency: compact mapping is typically best\n")
        f.write("2. For predictable layouts with simple implementation: column_wise or row_wise\n")
        f.write("3. For the best balance of communication and area efficiency: grid_wise is often a good compromise\n")
        f.write("4. For networks that need to communicate heavily between layers: proximity mapping (not included in this test) would be ideal\n")
        
        # Final notes
        f.write("\nNotes:\n")
        f.write("- The best mapping strategy may depend on the specific network topology and communication patterns\n")
        f.write("- Larger networks may benefit more from sophisticated mapping strategies\n")
        f.write("- The tradeoff between implementation complexity and performance should be considered\n")
        
        f.write("\nTest completed. Results saved to this log file.")
    
    print(f"Basic mapping strategy comparison test completed. Results saved to {log_file}")
    return results

def test_bisection_bandwidth():
    """
    Test the bisection bandwidth metrics for different mapping strategies.
    """
    print("Testing bisection bandwidth of different mapping strategies...")
    
    # Create a log file for this test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(os.path.dirname(__file__), f"../logs/map/mlp_bisection_bandwidth_{timestamp}.log")
    os.makedirs(os.path.join(os.path.dirname(__file__), "../logs/map"), exist_ok=True)
    print(f"Writing results to {log_file}")
    
    # Test dimensions
    seq_len = 1
    d_model = 768
    d_hidden = 3072
    
    # Define mapping strategies to test
    mapping_strategies = ["column_wise", "row_wise", "grid_wise", "compact", "proximity"]
    
    # Store results for comparison
    results = {}
    
    with open(log_file, "w") as f:
        f.write("===== Bisection Bandwidth Test =====\n\n")
        f.write(f"Test configuration:\n")
        f.write(f"- Sequence length: {seq_len}\n")
        f.write(f"- Model dimension (d_model): {d_model}\n")
        f.write(f"- Hidden dimension (d_hidden): {d_hidden}\n")
        f.write(f"- Strategies tested: {', '.join(mapping_strategies)}\n\n")
        
        # Test each mapping strategy
        for strategy in mapping_strategies:
            f.write(f"\n{'=' * 50}\n")
            f.write(f"TESTING STRATEGY: {strategy.upper()}\n")
            f.write(f"{'=' * 50}\n\n")
            
            print(f"Testing bisection bandwidth for {strategy} strategy...")
            
            # Create a fresh LLM instance for this strategy
            llm = LLM(
                seq_len=seq_len,
                pe_memory_size=64*1024,
                noc_rows=100,
                noc_cols=100,
                mapping_strategy=strategy,
                split_strategy="hybrid_split",
                data_type="float16",
                allow_wrapping=True
            )
            
            f.write(f"Creating networks with {strategy} mapping strategy...\n")
            
            # Create two identical MLP networks with this strategy
            mlp1 = llm.create_fc_network(
                name="mlp1",
                input_dim=d_model,
                output_dim=d_hidden,
                seq_len=seq_len,
                mapping_strategy=strategy,
                split_strategy="hybrid_split"
            )
            
            mlp2 = llm.create_fc_network(
                name="mlp2",
                input_dim=d_hidden,
                output_dim=d_model,
                seq_len=seq_len,
                mapping_strategy=strategy,
                split_strategy="hybrid_split"
            )
            
            # Set execution order and connect networks
            execution_order = [["mlp1"], ["mlp2"]]
            llm.set_execution_order(execution_order)
            llm.connect_networks("mlp1", "mlp2")
            
            # Collect all active PEs
            all_pes = set()
            for name, network in llm.networks.items():
                all_pes.update(network.active_pes)
                f.write(f"Network {name} uses {len(network.active_pes)} PEs\n")
            
            # Calculate overall area metrics
            x_coords = [x for x, y in all_pes]
            y_coords = [y for x, y in all_pes]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            effective_width = x_max - x_min + 1
            effective_height = y_max - y_min + 1
            
            # Create test input
            input_tensor = torch.randn(seq_len, d_model)
            inputs = {"mlp1": input_tensor}
            
            try:
                # Run inference to get traffic data
                f.write("\nRunning inference to gather traffic statistics...\n")
                outputs = llm.run_inference(inputs)
                
                # Get traffic statistics
                traffic_table = llm.noc.scheduler.get_traffic_table()
                
                if not traffic_table.empty:
                    # Create PE to network mapping
                    pe_to_network = {}
                    for name, network in llm.networks.items():
                        for pe_coord in network.active_pes:
                            pe_to_network[pe_coord] = name
                    
                    # Enhance the traffic table with network information
                    src_networks = []
                    dest_networks = []
                    manhattan_distances = []
                    src_coords = []
                    dest_coords = []
                    
                    for _, row in traffic_table.iterrows():
                        # Extract PE coordinates from string like "(0, 0)"
                        src_pe_str = row['src_pe'].strip('()')
                        dest_pe_str = row['dest_pe'].strip('()') if row['dest_pe'] != "None" else ""
                        
                        # Get source network
                        if src_pe_str and ',' in src_pe_str:
                            src_x, src_y = map(int, src_pe_str.split(','))
                            src_pe = (src_x, src_y)
                            src_network = pe_to_network.get(src_pe, "external")
                        else:
                            src_network = "external"
                        src_networks.append(src_network)
                        src_coords.append(src_pe)
                        
                        # Get destination network
                        if dest_pe_str and ',' in dest_pe_str:
                            dest_x, dest_y = map(int, dest_pe_str.split(','))
                            dest_pe = (dest_x, dest_y)
                            dest_network = pe_to_network.get(dest_pe, "external")
                            
                            # Calculate Manhattan distance if both PEs are defined
                            if src_pe_str and dest_pe_str and ',' in src_pe_str and ',' in dest_pe_str:
                                manhattan_dist = abs(dest_x - src_x) + abs(dest_y - src_y)
                                manhattan_distances.append(manhattan_dist)
                            else:
                                manhattan_distances.append(0)
                        else:
                            dest_network = "external"
                            manhattan_distances.append(0)
                        dest_networks.append(dest_network)
                        dest_coords.append(dest_pe)
                    
                    # Add the network and distance information to the traffic table
                    traffic_table['src_network'] = src_networks
                    traffic_table['dest_network'] = dest_networks
                    traffic_table['manhattan_distance'] = manhattan_distances
                    traffic_table['src_coords'] = src_coords
                    traffic_table['dest_coords'] = dest_coords
                    
                    # Calculate bisection bandwidth for different partition strategies
                    f.write("\n--- Bisection Bandwidth Analysis ---\n")
                    
                    # Method 1: Horizontal bisection (split by x-coordinate)
                    median_x = (x_min + x_max) / 2
                    left_half = {pe for pe in all_pes if pe[0] < median_x}
                    right_half = {pe for pe in all_pes if pe[0] >= median_x}
                    
                    # Filter traffic table for cross-partition communication
                    horizontal_cross_traffic = traffic_table[
                        ((traffic_table['src_coords'].apply(lambda x: x in left_half if x else False)) & 
                         (traffic_table['dest_coords'].apply(lambda x: x in right_half if x else False))) |
                        ((traffic_table['src_coords'].apply(lambda x: x in right_half if x else False)) & 
                         (traffic_table['dest_coords'].apply(lambda x: x in left_half if x else False)))
                    ]
                    
                    horizontal_bisection_bytes = horizontal_cross_traffic['bytes'].sum() if not horizontal_cross_traffic.empty else 0
                    
                    # Method 2: Vertical bisection (split by y-coordinate)
                    median_y = (y_min + y_max) / 2
                    top_half = {pe for pe in all_pes if pe[1] < median_y}
                    bottom_half = {pe for pe in all_pes if pe[1] >= median_y}
                    
                    vertical_cross_traffic = traffic_table[
                        ((traffic_table['src_coords'].apply(lambda x: x in top_half if x else False)) & 
                         (traffic_table['dest_coords'].apply(lambda x: x in bottom_half if x else False))) |
                        ((traffic_table['src_coords'].apply(lambda x: x in bottom_half if x else False)) & 
                         (traffic_table['dest_coords'].apply(lambda x: x in top_half if x else False)))
                    ]
                    
                    vertical_bisection_bytes = vertical_cross_traffic['bytes'].sum() if not vertical_cross_traffic.empty else 0
                    
                    # Method 3: Network bisection (separate the two networks)
                    mlp1_pes = set(mlp1.active_pes)
                    mlp2_pes = set(mlp2.active_pes)
                    
                    network_cross_traffic = traffic_table[
                        ((traffic_table['src_coords'].apply(lambda x: x in mlp1_pes if x else False)) & 
                         (traffic_table['dest_coords'].apply(lambda x: x in mlp2_pes if x else False))) |
                        ((traffic_table['src_coords'].apply(lambda x: x in mlp2_pes if x else False)) & 
                         (traffic_table['dest_coords'].apply(lambda x: x in mlp1_pes if x else False)))
                    ]
                    
                    network_bisection_bytes = network_cross_traffic['bytes'].sum() if not network_cross_traffic.empty else 0
                    
                    # Calculate total traffic for comparison
                    total_bytes = traffic_table['bytes'].sum()
                    
                    # Write bisection results
                    f.write(f"Total bytes transferred: {total_bytes:,}\n\n")
                    
                    f.write("Horizontal Bisection (split at x = {:.1f}):\n".format(median_x))
                    f.write(f"Left half: {len(left_half)} PEs, Right half: {len(right_half)} PEs\n")
                    f.write(f"Cross-bisection bytes: {horizontal_bisection_bytes:,} ({horizontal_bisection_bytes/total_bytes*100:.2f}% of total)\n")
                    f.write(f"Number of cross-bisection transfers: {len(horizontal_cross_traffic)}\n\n")
                    
                    f.write("Vertical Bisection (split at y = {:.1f}):\n".format(median_y))
                    f.write(f"Top half: {len(top_half)} PEs, Bottom half: {len(bottom_half)} PEs\n")
                    f.write(f"Cross-bisection bytes: {vertical_bisection_bytes:,} ({vertical_bisection_bytes/total_bytes*100:.2f}% of total)\n")
                    f.write(f"Number of cross-bisection transfers: {len(vertical_cross_traffic)}\n\n")
                    
                    f.write("Network Bisection (mlp1 vs mlp2):\n")
                    f.write(f"mlp1: {len(mlp1_pes)} PEs, mlp2: {len(mlp2_pes)} PEs\n")
                    f.write(f"Cross-bisection bytes: {network_bisection_bytes:,} ({network_bisection_bytes/total_bytes*100:.2f}% of total)\n")
                    f.write(f"Number of cross-bisection transfers: {len(network_cross_traffic)}\n\n")
                    
                    # Visualization of bisections
                    f.write("\n--- Bisection Visualization ---\n")
                    f.write("NoC grid showing network partitions and bisection lines:\n")
                    
                    # Create a grid visualization
                    grid_width = min(effective_width, 30)
                    grid_height = min(effective_height, 30)
                    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]
                    
                    # Mark network partitions
                    for x, y in all_pes:
                        grid_x = x - x_min
                        grid_y = y - y_min
                        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                            if (x, y) in mlp1_pes:
                                grid[grid_y][grid_x] = '1'
                            elif (x, y) in mlp2_pes:
                                grid[grid_y][grid_x] = '2'
                    
                    # Add column headers
                    f.write("    ")
                    for i in range(grid_width):
                        f.write(f"{i+x_min:^3}")
                    f.write("\n")
                    
                    # Add horizontal separator line with bisection marker
                    f.write("   +")
                    h_bisect_pos = int((median_x - x_min) * 3)
                    for i in range(grid_width * 3):
                        if i == h_bisect_pos:
                            f.write("H")
                        else:
                            f.write("-")
                    f.write("+\n")
                    
                    # Draw the grid with vertical bisection marker
                    v_bisect_pos = int(median_y - y_min)
                    for y in range(grid_height):
                        if y == v_bisect_pos:
                            marker = "V"
                        else:
                            marker = "|"
                        f.write(f"{y+y_min:2d} {marker}")
                        for x in range(grid_width):
                            f.write(f" {grid[y][x]} ")
                        f.write("|\n")
                    
                    # Add bottom separator line
                    f.write("   +")
                    for i in range(grid_width * 3):
                        f.write("-")
                    f.write("+\n\n")
                    
                    # Store the results
                    results[strategy] = {
                        "total_bytes": total_bytes,
                        "horizontal_bisection_bytes": horizontal_bisection_bytes,
                        "horizontal_bisection_percentage": (horizontal_bisection_bytes/total_bytes*100) if total_bytes > 0 else 0,
                        "vertical_bisection_bytes": vertical_bisection_bytes,
                        "vertical_bisection_percentage": (vertical_bisection_bytes/total_bytes*100) if total_bytes > 0 else 0,
                        "network_bisection_bytes": network_bisection_bytes,
                        "network_bisection_percentage": (network_bisection_bytes/total_bytes*100) if total_bytes > 0 else 0,
                    }
                
            except Exception as e:
                error_msg = f"Error during inference for {strategy} strategy: {e}"
                f.write(f"\n{error_msg}\n")
                traceback.print_exc(file=f)
                print(error_msg)
        
        # Comparative analysis of bisection bandwidth for all strategies
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("                    BISECTION BANDWIDTH COMPARISON                     \n")
        f.write("=" * 80 + "\n\n")
        
        if results:
            # Create comparative tables
            f.write("Horizontal Bisection Bandwidth (left/right split):\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Strategy':<15} {'Bytes':<15} {'% of Total':<15} {'Rank':<10}\n")
            f.write("-" * 70 + "\n")
            
            # Sort strategies by horizontal bisection percentage (lower is better for bandwidth efficiency)
            h_ranking = sorted(results.keys(), key=lambda s: results[s]['horizontal_bisection_percentage'])
            for i, strategy in enumerate(h_ranking):
                f.write(f"{strategy:<15} {results[strategy]['horizontal_bisection_bytes']:15,} "
                      f"{results[strategy]['horizontal_bisection_percentage']:15.2f}% {i+1:10}\n")
            
            f.write("\nVertical Bisection Bandwidth (top/bottom split):\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Strategy':<15} {'Bytes':<15} {'% of Total':<15} {'Rank':<10}\n")
            f.write("-" * 70 + "\n")
            
            # Sort strategies by vertical bisection percentage
            v_ranking = sorted(results.keys(), key=lambda s: results[s]['vertical_bisection_percentage'])
            for i, strategy in enumerate(v_ranking):
                f.write(f"{strategy:<15} {results[strategy]['vertical_bisection_bytes']:15,} "
                      f"{results[strategy]['vertical_bisection_percentage']:15.2f}% {i+1:10}\n")
            
            f.write("\nNetwork Bisection Bandwidth (mlp1/mlp2 split):\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Strategy':<15} {'Bytes':<15} {'% of Total':<15} {'Rank':<10}\n")
            f.write("-" * 70 + "\n")
            
            # Sort strategies by network bisection percentage
            n_ranking = sorted(results.keys(), key=lambda s: results[s]['network_bisection_percentage'])
            for i, strategy in enumerate(n_ranking):
                f.write(f"{strategy:<15} {results[strategy]['network_bisection_bytes']:15,} "
                      f"{results[strategy]['network_bisection_percentage']:15.2f}% {i+1:10}\n")
            
            # Calculate average rank across all bisection methods
            avg_ranks = {}
            for strategy in results.keys():
                h_rank = h_ranking.index(strategy) + 1
                v_rank = v_ranking.index(strategy) + 1
                n_rank = n_ranking.index(strategy) + 1
                avg_ranks[strategy] = (h_rank + v_rank + n_rank) / 3
            
            # Overall ranking
            f.write("\nOverall Bisection Bandwidth Ranking (lower is better):\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Strategy':<15} {'Avg Rank':<15} {'Best Bisection':<20} {'Worst Bisection':<20}\n")
            f.write("-" * 70 + "\n")
            
            for strategy, avg_rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
                # Find best and worst bisection methods for this strategy
                bisections = {
                    'Horizontal': results[strategy]['horizontal_bisection_percentage'],
                    'Vertical': results[strategy]['vertical_bisection_percentage'],
                    'Network': results[strategy]['network_bisection_percentage']
                }
                best_bisection = min(bisections.items(), key=lambda x: x[1])
                worst_bisection = max(bisections.items(), key=lambda x: x[1])
                
                f.write(f"{strategy:<15} {avg_rank:<15.2f} "
                      f"{best_bisection[0]} ({best_bisection[1]:.2f}%) {worst_bisection[0]:5} ({worst_bisection[1]:.2f}%)\n")
        
        # Summary and conclusions
        f.write("\n--- Summary and Conclusions ---\n\n")
        
        if results:
            best_strategy = min(avg_ranks.items(), key=lambda x: x[1])[0]
            
            f.write(f"Based on the bisection bandwidth analysis, {best_strategy} appears to be the most efficient mapping strategy ")
            f.write("for minimizing cross-partition communication.\n\n")
            
            f.write("Key observations:\n")
            
            # Pattern observations
            if all(results[s]['horizontal_bisection_percentage'] < results[s]['vertical_bisection_percentage'] 
                   for s in ["column_wise", "row_wise"]):
                f.write("- Column-wise and row-wise mappings tend to have better horizontal than vertical bisection bandwidth.\n")
            
            if "compact" in results and "grid_wise" in results:
                if results["compact"]["network_bisection_percentage"] < results["grid_wise"]["network_bisection_percentage"]:
                    f.write("- Compact mapping provides better network-to-network bisection bandwidth than grid-wise mapping.\n")
                else:
                    f.write("- Grid-wise mapping provides better network-to-network bisection bandwidth than compact mapping.\n")
            
            # General conclusions
            f.write("\nGeneral conclusions:\n")
            f.write("1. The choice of mapping strategy significantly impacts bisection bandwidth requirements.\n")
            f.write("2. Network-to-network communication patterns are the most critical for overall performance.\n")
            f.write("3. Mapping strategies that place communicating elements closer together reduce bandwidth requirements.\n")
            
            # Recommendations
            f.write("\nRecommendations:\n")
            f.write(f"- For this specific workload, {best_strategy} mapping is recommended for optimal bisection bandwidth.\n")
            f.write("- When designing NoC topologies, the bisection bandwidth should be provisioned according to the chosen mapping strategy.\n")
            f.write("- Consider the dominant communication patterns in the workload when selecting a mapping strategy.\n")
        else:
            f.write("Insufficient data to draw conclusions. Ensure that the test runs successfully for all mapping strategies.\n")
        
        f.write("\nTest completed. Results saved to this log file.")
    
    print(f"Bisection bandwidth test completed. Results saved to {log_file}")
    return results

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "test_mlp_simple":
            test_mlp_simple()
        elif test_name == "test_mlp_compact":
            test_mlp_compact()
        elif test_name == "test_mapping_strategies":
            test_mapping_strategies()
        elif test_name == "test_mapping_strategies_basic":
            test_mapping_strategies_basic()
        elif test_name == "test_bisection_bandwidth":
            test_bisection_bandwidth()
        else:
            print(f"Unknown test name: {test_name}")
            print("Available tests: test_mlp_simple, test_mlp_compact, test_mapping_strategies, test_mapping_strategies_basic, test_bisection_bandwidth")
    else:
        # Default behavior when no arguments are provided
        test_mapping_strategies()  # Run the standard test by default
        # Uncomment to run other tests:
        #test_mlp_simple()  # Original grid_wise mapping test
        # test_mlp_compact()   # New compact mapping test
        # test_mapping_strategies_basic()  # Basic comparison without helper functions
        # test_bisection_bandwidth()  # Test bisection bandwidth