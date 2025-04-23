import torch
import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.llm import LLM
from llmDeploy.neural_network import FCNeuralNetwork, ArithmeticNetwork

def test_matrix_multiply_connection():
    """Test that matrix multiplication correctly uses source PEs from connected networks."""
    print("\n===== Testing Matrix Multiply Connection =====\n")
    
    # Create an LLM instance with a 10x10 NoC
    llm = LLM(
        layer_dims=[32, 32],  # Input and output dimensions
        seq_len=6,
        pe_memory_size=4096,  # 4KB per PE
        noc_rows=10,
        noc_cols=10,
        mapping_strategy="column_wise",
        split_strategy="column_split",
        data_type="float16"
    )
    
    # Create a fully connected network with explicit PE coordinates
    fc1 = llm.create_fc_network(
        name="fc1",
        input_dim=32,
        output_dim=32,
        seq_len=6
    )
    
    # Create a matrix multiplication network with explicit PE coordinates
    matmul = llm.create_arithmetic_network(
        name="matmul",
        seq_len=6,
        d_model=32,
        operation="matmul"
    )
    
    # Print active PEs for each network
    print("\nPE Allocations:")
    for name, network in llm.networks.items():
        print(f"{name} network active PEs:", network.active_pes)
    
    # Create test inputs
    fc1_input = torch.randn(6, 32)
    matmul_input_b = torch.randn(6, 32)
    
    # Run FC inference to get outputs
    print("\nRunning fc1 inference...")
    fc1_outputs = fc1.run_inference(fc1_input)
    
    # Extract PE information from fc1's output
    fc1_output_pes = list(fc1_outputs.keys())
    print(f"fc1 output PEs: {fc1_output_pes}")
    
    # Print details about the fc1 output
    for pe, (tensor, output_range, task_id) in fc1_outputs.items():
        print(f"fc1 output at PE{pe}:")
        print(f"  - Tensor shape: {tensor.shape}")
        print(f"  - Output range: {output_range}")
        print(f"  - Task ID: {task_id}")
    
    # Run matrix multiply with fc1's output as input_a
    print("\nRunning matrix multiply with fc1's output...")
    matmul_outputs = matmul.matrix_multiply(
        input_a=fc1_outputs,  # Pass the entire dictionary
        input_b=matmul_input_b,
        transpose_b=True,  # Default: Q @ K^T
        source_pe_b=(-1, 1)  # Virtual PE for input_b
    )
    
    # Print details of the traffic table
    traffic_table = llm.noc.scheduler.get_traffic_table()
    print("\nDetailed Traffic Table:")
    if not traffic_table.empty:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print(traffic_table)
    
    # Analyze traffic from fc1 to matmul
    print("\nTraffic Analysis:")
    if not traffic_table.empty:
        # Filter traffic from fc1's PE to matmul's PE
        fc1_pe = fc1_output_pes[0]
        matmul_pe = list(matmul.active_pes)[0]
        
        fc1_to_matmul_traffic = traffic_table[
            (traffic_table['src_pe'] == str(fc1_pe)) & 
            (traffic_table['dest_pe'] == str(matmul_pe))
        ]
        
        print(f"Traffic from fc1 PE{fc1_pe} to matmul PE{matmul_pe}:")
        print(fc1_to_matmul_traffic)
        
        # Also check for traffic from any source to matmul's PE
        to_matmul_traffic = traffic_table[traffic_table['dest_pe'] == str(matmul_pe)]
        print(f"\nAll traffic to matmul PE{matmul_pe}:")
        print(to_matmul_traffic)

if __name__ == "__main__":
    test_matrix_multiply_connection() 