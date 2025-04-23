#!/usr/bin/env python
"""
Example script demonstrating the sequential LLM architecture.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

from llmDeploy import LLM

def simple_sequential_example():
    """
    Create a sequential LLM with multiple FC layers.
    """
    print("Running Sequential LLM example...")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define dimensions for the network
    input_dim = 3
    hidden_dim1 = 6
    hidden_dim2 = 6
    output_dim = 3
    
    # Create the layer dimensions
    #layer_dims = [input_dim, hidden_dim1, hidden_dim2, output_dim]
    layer_dims = [input_dim, hidden_dim1, hidden_dim2, output_dim]  # Use 3 networks instead of 2
    
    # Setup log file
    log_file = "noc_simulation.log"
    
    # Open log file for writing
    with open(log_file, "w") as f:
        # Log basic parameters
        f.write(f"Sequential LLM Simulation Parameters:\n")
        f.write(f"Input Dimension: {input_dim}\n")
        f.write(f"Hidden Dimension 1: {hidden_dim1}\n")
        f.write(f"Hidden Dimension 2: {hidden_dim2}\n")
        f.write(f"Output Dimension: {output_dim}\n")
        f.write(f"Total Layers: {len(layer_dims) - 1}\n")
        
        # Create the LLM
        llm = LLM(
            layer_dims=layer_dims,
            seq_len=4,
            pe_memory_size=12,  # 64 KB
            mapping_strategy="column_wise",
            split_strategy="column_split",
            row_aggregation_enabled = True,
            column_aggregation_enabled = False,
            reuse_pe_for_aggregation=True,
            data_type="float16",
            channel_bandwidth=32.0,
            noc_rows=1000,
            noc_cols=1000
        )
        
        # Log NoC configuration
        f.write(f"Sequence Length: {llm.seq_len}\n")
        f.write(f"PE Memory Size: {llm.pe_memory_size} bytes\n")
        f.write(f"Mapping Strategy: {llm.mapping_strategy}\n")
        f.write(f"Split Strategy: {llm.split_strategy}\n")
        f.write(f"Reuse PE for Aggregation: {llm.reuse_pe_for_aggregation}\n")
        f.write(f"NoC Size: {llm.noc.rows}x{llm.noc.cols} ({llm.noc.rows*llm.noc.cols} PEs)\n")
        f.write(f"Data Type: {llm.data_type}\n")
        f.write(f"Channel Bandwidth: {llm.channel_bandwidth} B/cycle\n\n")
        
        # Log PE mapping details
        f.write("=" * 80 + "\n")
        f.write("                          PE MAPPING DETAILS                          \n")
        f.write("=" * 80 + "\n")
        pe_mapping = llm.get_pe_mapping_details()
        
        # Add descriptive column headers
        if not pe_mapping.empty:
            pe_mapping.columns = [
                'Network #', 
                'PE Coords', 
                'Layer ID', 
                'PE Idx', 
                'Split Type', 
                'Weight Range', 
                'Shape'
            ]
            
            # Add descriptive split dimension values
            split_dim_names = {0: 'Row', 1: 'Col', 2: 'Hybrid'}
            pe_mapping['Split Type'] = pe_mapping['Split Type'].map(
                lambda x: split_dim_names.get(x, str(x))
            )
        
        f.write(pe_mapping.to_string(index=False) + "\n\n")
        
        # Create a visual representation of the NoC grid
        f.write("NoC Grid Layout (showing Network # at each PE):\n")
        f.write("-" * 50 + "\n")
        
        # Create a grid representation of the NoC
        noc_grid = [[' ' for _ in range(llm.noc.cols)] for _ in range(llm.noc.rows)]
        
        # Fill in the grid with network indices
        for _, row in pe_mapping.iterrows():
            # Extract coordinates from PE Coords which may be a tuple or string
            coords_str = row['PE Coords']
            
            # Handle different types of coordinates
            if isinstance(coords_str, str):
                # If it's a string like "(0, 1)", parse it
                if coords_str.startswith('(') and coords_str.endswith(')'):
                    parts = coords_str.strip('()').split(',')
                    if len(parts) == 2:
                        try:
                            x = int(parts[0].strip())
                            y = int(parts[1].strip())
                            if 0 <= x < llm.noc.cols and 0 <= y < llm.noc.rows:
                                noc_grid[y][x] = str(row['Network #'])
                        except ValueError:
                            # Skip if conversion fails
                            pass
            elif isinstance(coords_str, tuple) and len(coords_str) == 2:
                # If it's already a tuple like (0, 1)
                x, y = coords_str
                if 0 <= x < llm.noc.cols and 0 <= y < llm.noc.rows:
                    noc_grid[y][x] = str(row['Network #'])
        
        # Write the grid to the log file
        for y in range(min(16, llm.noc.rows)):  # Limit to 16 rows for readability
            row_str = '|'
            for x in range(min(16, llm.noc.cols)):  # Limit to 16 columns for readability
                row_str += f' {noc_grid[y][x] or "."} |'
            f.write(row_str + "\n")
        
        # If grid is larger than 16x16, indicate truncation
        if llm.noc.rows > 16 or llm.noc.cols > 16:
            f.write("(Grid truncated for display - showing only 16x16 portion)\n")
        
        f.write("-" * 50 + "\n\n")
        
        # Print the network structure
        llm.print_network_structure()
        
        # Get network details
        network_details = llm.get_network_details()
        print(f"\nNetwork has {network_details['total_layers']} layers with dimensions:")
        for i, dim in enumerate(network_details['layer_dims']):
            print(f"  Layer {i+1}: {dim[0]} -> {dim[1]}")
        
        # Log network details
        f.write(f"Network Structure:\n")
        for i, dim in enumerate(network_details['layer_dims']):
            f.write(f"  Layer {i+1}: {dim[0]} -> {dim[1]}\n")
        
        # Create input tensor for inference
        batch_size = 1
        seq_len = 4
        input_tensor = torch.randn(seq_len, input_dim)
        
        # Run inference
        print("\nRunning inference through the network...")
        f.write("\nRunning inference through the network...\n")
        output = llm.run_inference(input_tensor)
        
        # Print output shape
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        
        # Log tensor shapes
        f.write(f"Input shape: {input_tensor.shape}\n")
        f.write(f"Output shape: {output.shape}\n")
        
        # Print PE utilization
        pe_utilization = llm.get_pe_utilization()
        print(f"PE Utilization: {pe_utilization:.2f}%")
        print(f"Total used PEs: {len(llm._get_all_pes())}")
        
        # Log PE utilization
        f.write("\nPE Utilization:\n")
        f.write(f"Total PEs in NoC: {llm.noc.rows * llm.noc.cols}\n")
        f.write(f"Total used PEs: {len(llm._get_all_pes())}\n")
        f.write(f"PE Utilization: {pe_utilization:.2f}%\n\n")
        
        # Log each network's traffic table if available
        f.write("\nTraffic Details:\n")
        
        # First show the complete unfiltered traffic table
        f.write("\nComplete Traffic Table (All Networks):\n")
        complete_traffic_table = llm.noc.scheduler.get_traffic_table()
        f.write(complete_traffic_table.to_string() + "\n\n")
        
        # Then show individual network traffic tables
        for i, network in enumerate(llm.networks):
            f.write(f"\nNetwork {i+1} Traffic:\n")
            if hasattr(network, 'get_traffic_table'):
                try:
                    traffic_table = network.get_traffic_table()
                    
                    # Add network_id column if not present
                    if 'network_id' not in traffic_table.columns:
                        # For backward compatibility, add network index
                        traffic_table['network_id'] = i
                    
                    # Filter to only show tasks for this network
                    network_traffic = traffic_table[
                        (traffic_table['network_id'] == i) | 
                        (traffic_table['network_id'] == str(i))
                    ]
                    
                    # If no filtered results, show all (might be using older version)
                    if len(network_traffic) == 0:
                        network_traffic = traffic_table
                        f.write("Note: Traffic may include data from multiple networks\n")
                    
                    f.write(network_traffic.to_string() + "\n")
                    
                    # Count tasks by type if description field exists
                    if 'description' in network_traffic.columns:
                        task_types = network_traffic['description'].str.extract(r'(->|computation|collection)').value_counts()
                        f.write("\nTask Distribution:\n")
                        f.write(task_types.to_string() + "\n")
                except Exception as e:
                    f.write(f"Could not get traffic table: {str(e)}\n")
            else:
                f.write("Traffic table not available for this network\n")
        
        print(f"Simulation results saved to {log_file}")
        
    print("\nSequential LLM example completed successfully!")
    return llm, input_tensor, output

def analyze_layer_depth():
    """
    Analyze how adding more layers affects PE utilization.
    """
    print("\nAnalyzing effect of layer depth on PE utilization...")
    
    # Setup log file
    log_file = "layer_depth_analysis.log"
    
    with open(log_file, "w") as f:
        f.write("Layer Depth Analysis Results:\n\n")
        
        # Parameters
        base_dim = 64
        max_layers = 6
        layer_counts = list(range(1, max_layers + 1))
        utilizations = []
        
        f.write(f"Base dimension: {base_dim}\n")
        f.write(f"Testing layer counts from 1 to {max_layers}\n\n")
        f.write("Layer Count | Layer Dimensions | PE Utilization | Total PEs Used\n")
        f.write("-" * 70 + "\n")
        
        for num_layers in layer_counts:
            # Create layers with alternating growing and shrinking dimensions
            layer_dims = [base_dim]  # Start with input dimension
            current_dim = base_dim
            
            for i in range(num_layers):
                if i % 2 == 0:
                    # Expand dimension
                    next_dim = current_dim * 2
                else:
                    # Contract dimension
                    next_dim = current_dim // 2
                    
                layer_dims.append(next_dim)
                current_dim = next_dim
            
            # Create LLM with these layer dimensions
            llm = LLM(
                layer_dims=layer_dims,
                seq_len=4,
                pe_memory_size=64 * 1024,  # 64 KB
                mapping_strategy="column_wise",
                split_strategy="column_split",
                reuse_pe_for_aggregation=True,
                data_type="float16"
            )
            
            # Get PE utilization
            utilization = llm.get_pe_utilization()
            utilizations.append(utilization)
            
            # Print and log results
            print(f"Layers: {num_layers}, Dimensions: {layer_dims}, Utilization: {utilization:.2f}%")
            f.write(f"{num_layers} | {layer_dims} | {utilization:.2f}% | {len(llm._get_all_pes())}\n")
        
        f.write("\nAnalysis completed successfully.\n")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(layer_counts, utilizations, marker='o', linestyle='-', linewidth=2)
    plt.title('PE Utilization vs Layer Depth')
    plt.xlabel('Number of Layers')
    plt.ylabel('PE Utilization (%)')
    plt.grid(True)
    plt.savefig('layer_depth_analysis.png')
    print(f"Layer depth analysis plot saved to layer_depth_analysis.png")
    print(f"Layer depth analysis log saved to {log_file}")
    
    return layer_counts, utilizations

if __name__ == "__main__":
    # Run the simple sequential example
    llm, input_tensor, output = simple_sequential_example()
    
    # Run the layer depth analysis
    #layer_counts, utilizations = analyze_layer_depth()
    
    print("All examples completed successfully!") 