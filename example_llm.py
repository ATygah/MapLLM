#!/usr/bin/env python
"""
Example script demonstrating the LLM class that manages NoC resources.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

from llmDeploy import LLM

def simple_llm_example():
    """
    Create an LLM with multiple neural networks and attention modules.
    This demonstrates how the LLM tracks PE usage across components.
    """
    print("Running LLM with PE tracking example...")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parameters
    noc_rows = 16
    noc_cols = 16
    pe_memory_size = 64 * 1024  # 64 KB
    
    print(f"Creating LLM with NoC dimensions: {noc_rows}x{noc_cols}")
    
    # Create LLM
    llm = LLM(
        noc_rows=noc_rows,
        noc_cols=noc_cols,
        pe_memory_size=pe_memory_size
    )
    
    # Create a feed-forward neural network
    print("Creating feed-forward neural network...")
    input_dim = 64
    hidden_dims = [128]
    ff_network = llm.create_neural_network(
        input_dim=input_dim,
        layer_dims=hidden_dims,
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    # Check PE utilization after creating the feed-forward network
    ff_utilization = llm.get_pe_utilization()
    print(f"PE Utilization after creating feed-forward network: {ff_utilization:.2f}%")
    
    # Create an attention module
    print("Creating self-attention module...")
    head_size = 32
    num_heads = 2
    attention = llm.create_attention(
        input_dim=input_dim,
        head_size=head_size,
        num_heads=num_heads,
        seq_len=4,
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    # Check PE utilization after creating the attention module
    attn_utilization = llm.get_pe_utilization()
    print(f"PE Utilization after creating attention module: {attn_utilization:.2f}%")
    
    # Create another feed-forward network
    print("Creating second feed-forward neural network...")
    output_dim = 64
    ff_network2 = llm.create_neural_network(
        input_dim=input_dim,
        layer_dims=[output_dim],
        mapping_strategy="column_wise",
        split_strategy="column_split"
    )
    
    # Check PE utilization after creating the second feed-forward network
    final_utilization = llm.get_pe_utilization()
    print(f"Final PE Utilization: {final_utilization:.2f}%")
    
    # Create input tensor
    batch_size = 1
    seq_len = 4
    input_tensor = torch.randn(batch_size, seq_len, input_dim)
    
    # Run inference through the attention module
    print("Running inference through attention module...")
    attn_output = attention.run_inference(input_tensor)
    print(f"Attention output shape: {attn_output.shape}")
    
    # Run inference through the first feed-forward network
    # We'll use just the first sequence element for this example
    print("Running inference through feed-forward network...")
    ff_output = ff_network.run_inference(input_tensor[:, 0, :])
    print("Feed-forward network inference completed")
    
    print("LLM example completed successfully!")
    return llm, attention, ff_network, ff_network2

def analyze_pe_sharing():
    """
    Analyze how increasing the number of components affects PE utilization.
    """
    print("Analyzing PE sharing across networks...")
    
    # Parameters
    noc_rows = 16
    noc_cols = 16
    pe_memory_size = 64 * 1024  # 64 KB
    input_dim = 64
    hidden_dim = 128
    output_dim = 64
    
    # Create LLM
    llm = LLM(
        noc_rows=noc_rows,
        noc_cols=noc_cols,
        pe_memory_size=pe_memory_size
    )
    
    # Keep track of utilization as we add components
    component_counts = []
    utilizations = []
    
    # Add components one by one and track utilization
    max_components = 10
    for i in range(1, max_components + 1):
        component_type = "ff" if i % 2 == 0 else "attn"
        
        if component_type == "ff":
            # Add a feed-forward network
            ff_network = llm.create_neural_network(
                input_dim=input_dim,
                layer_dims=[hidden_dim if i % 4 != 0 else output_dim],
                mapping_strategy="column_wise",
                split_strategy="column_split"
            )
        else:
            # Add an attention module
            attention = llm.create_attention(
                input_dim=input_dim,
                head_size=32,
                num_heads=i % 4 + 1,  # Vary number of heads
                seq_len=4,
                mapping_strategy="column_wise",
                split_strategy="column_split"
            )
        
        # Record utilization
        component_counts.append(i)
        utilizations.append(llm.get_pe_utilization())
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(component_counts, utilizations, 'o-', linewidth=2)
    plt.xlabel('Number of Components')
    plt.ylabel('PE Utilization (%)')
    plt.title('PE Utilization vs. Number of Components')
    plt.grid(True)
    plt.savefig('pe_sharing_analysis.png')
    
    print("Analysis complete. Plot saved as 'pe_sharing_analysis.png'")
    return component_counts, utilizations

if __name__ == "__main__":
    # Run simple example
    llm, attention, ff_network, ff_network2 = simple_llm_example()
    
    # Run analysis
    component_counts, utilizations = analyze_pe_sharing()
    
    print("All examples completed successfully!") 