#!/usr/bin/env python3
"""
Example script demonstrating how to use the llmDeploy package.
"""

import torch
import matplotlib.pyplot as plt
from llmDeploy import (
    FCNeuralNetwork,
    run_example,
    analyze_pe_memory_impact
)

def simple_example():
    """Run a simple example of a neural network on NoC."""
    # Example parameters
    input_dim = 6
    hidden_dim = 12
    output_dim = 6
    seq_len = 1
    mem=6

    # Create input tensor
    input_tensor = torch.randn(seq_len, input_dim)
    
    # Create neural network with column-wise mapping and hybrid split
    print("\nCreating neural network with column-wise mapping and hybrid split...")
    nn = FCNeuralNetwork(
        input_dim=input_dim,
        layer_dims=[hidden_dim, output_dim],
        seq_len=seq_len,
        pe_memory_size=mem,
        mapping_strategy="column_wise",
        split_strategy="hybrid_split",
        reuse_pe_for_aggregation=True,
        data_type="float16",
        channel_bandwidth=32.0
    )
    
    # Print some information about the network
    print(f"NoC Dimensions: {nn.noc.rows}x{nn.noc.cols}")
    print(f"Active PEs: {len(nn.active_pes)}")
    
    # Run inference
    print("\nRunning inference...")
    pe_outputs = nn.run_inference(input_tensor)
    
    # Get traffic table
    traffic_table = nn.get_traffic_table()
    print(f"\nTotal traffic tasks: {len(traffic_table)}")
    print(f"Total cycles: {traffic_table['cycles'].sum()}")
    
    # Print PE utilization
    utilization = nn.get_pe_utilization(use_effective_dimensions=True)
    print(f"\nPE Utilization: {utilization['total_utilization']:.2f}%")
    
    # Print a few traffic tasks for inspection
    print("\nSample traffic tasks:")
    print(traffic_table.head(5))
    
    return nn, pe_outputs, traffic_table

def run_analyses():
    """Run various analyses on neural network configurations."""
    # Run basic example with logging
    print("\nRunning basic example with different configurations...")
    run_example(log_file="noc_simulation.log")
    
    # Analyze PE memory impact
    print("\nAnalyzing PE memory impact...")
    memory_analysis = analyze_pe_memory_impact(
        input_dim=512,
        hidden_dim=2048,
        output_dim=512,
        memory_sizes=[16*1024, 32*1024, 64*1024, 128*1024],
        mapping_strategy="column_wise",
        split_strategy="column_split",  # Using column_split which is more stable
        reuse_pe_for_aggregation=True,  # Reuse PEs to avoid aggregation PE assignment issues
        save_plot=True,
        plot_filename="pe_memory_analysis.png"
    )
    
    return memory_analysis

if __name__ == "__main__":
    print("Running simple neural network on NoC example...")
    nn, pe_outputs, traffic_table = simple_example()
    
    print("\nRunning analyses...")
    memory_analysis = run_analyses()
    
    print("\nExample completed successfully!") 