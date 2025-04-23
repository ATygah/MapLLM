import torch
import sys
import os
import pandas as pd
import time
from typing import Dict, Tuple, Optional, List

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.pe_noc import NoCTopology
from llmDeploy.neural_network import ArithmeticNetwork

def test_arithmetic_network_matmul():
    """
    Test the ArithmeticNetwork's matrix multiplication capability with different split strategies.
    """
    print("\n===== Testing ArithmeticNetwork Matrix Multiplication =====\n")
    
    try:
        # Parameters
        rows, cols = 4, 4  # NoC grid size
        seq_len = 8        # Sequence length
        d_model = 16       # Model dimension
        memory_size = 1024 * 1024  # 1MB per PE
        
        # Initialize a NoC topology
        noc = NoCTopology(rows, cols, memory_size)
        
        # Create dictionary to store results for different strategies
        strategy_results = {}
        
        # Test different split strategies
        for split_strategy in ["column_split", "row_split", "hybrid_split"]:
            print(f"\n----- Testing {split_strategy} strategy -----")
            
            # Create the arithmetic network
            network = ArithmeticNetwork(
                noc=noc,
                seq_len=seq_len,
                d_model=d_model,
                mapping_strategy="grid_wise",
                split_strategy=split_strategy,
                data_type="float16",
                reuse_pe_for_aggregation=True,
                row_aggregation_enabled=True,
                column_aggregation_enabled=True
            )
            
            # Create sample input tensors
            input_a = torch.randn(seq_len, d_model)  # Q matrix
            input_b = torch.randn(seq_len, d_model)  # K matrix
            
            # Run matrix multiplication with transpose (Q @ K^T)
            print(f"\nRunning matrix multiplication with {split_strategy} strategy...")
            
            # Check if matrix_multiply method exists and print its attributes
            print(f"Has matrix_multiply attribute: {hasattr(network, 'matrix_multiply')}")
            if hasattr(network, 'matrix_multiply'):
                print(f"matrix_multiply is callable: {callable(getattr(network, 'matrix_multiply'))}")
                
            start_time = time.time()
            pe_outputs = network.matrix_multiply(input_a, input_b, transpose_b=True)
            end_time = time.time()
            
            # Print PE outputs
            print("\nPE Outputs:")
            network.print_pe_outputs(pe_outputs)
            
            # Print traffic statistics
            traffic_table = network.get_traffic_table()
            print("\nTraffic Statistics:")
            total_bytes = traffic_table['bytes'].sum()
            total_tasks = len(traffic_table)
            print(f"Total bytes transferred: {total_bytes} bytes")
            print(f"Total communication tasks: {total_tasks}")
            
            # Print utilization statistics
            utilization = network.get_pe_utilization(use_effective_dimensions=True)
            print("\nPE Utilization:")
            print(f"Total PEs: {utilization['total_pes']}")
            print(f"Used computation PEs: {utilization['used_computation_pes']}")
            print(f"Computation utilization: {utilization['computation_utilization']:.2f}%")
            print(f"Total utilization: {utilization['total_utilization']:.2f}%")
            
            # Print time statistics
            print(f"\nExecution time: {(end_time - start_time) * 1000:.2f} ms")
            
            # Store results for comparison
            strategy_results[split_strategy] = {
                'total_bytes': total_bytes,
                'total_tasks': total_tasks,
                'computation_utilization': utilization['computation_utilization'],
                'execution_time': end_time - start_time
            }
            
            # Reference calculation to verify correctness
            reference_output = input_a @ input_b.transpose(0, 1)
            
            # Clear the scheduler for the next test
            noc.scheduler.clear()
        
        # Compare strategies
        print("\n===== Strategy Comparison =====")
        comparison_df = pd.DataFrame.from_dict(strategy_results, orient='index')
        comparison_df.columns = ['Total Bytes', 'Total Tasks', 'Computation Utilization (%)', 'Execution Time (s)']
        print(comparison_df)
        
        print("\nDone!")
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        print(traceback.format_exc())

def test_elementwise_operations():
    """
    Test the ArithmeticNetwork's element-wise operations.
    """
    print("\n===== Testing ArithmeticNetwork Element-wise Operations =====\n")
    
    # Parameters
    rows, cols = 4, 4  # NoC grid size
    seq_len = 8        # Sequence length
    d_model = 16       # Model dimension
    memory_size = 1024 * 1024  # 1MB per PE
    
    # Initialize a NoC topology
    noc = NoCTopology(rows, cols, memory_size)
    
    # Create the arithmetic network
    network = ArithmeticNetwork(
        noc=noc,
        seq_len=seq_len,
        d_model=d_model,
        mapping_strategy="grid_wise",
        split_strategy="column_split",
        data_type="float16"
    )
    
    # Create sample input tensors
    input_a = torch.randn(seq_len, d_model)
    input_b = torch.randn(seq_len, d_model)
    
    # Test different element-wise operations
    for operation in ["add", "multiply"]:
        print(f"\n----- Testing element-wise {operation} operation -----")
        
        # Run element-wise operation
        start_time = time.time()
        pe_outputs = network.element_wise(input_a, input_b, operation=operation)
        end_time = time.time()
        
        # Print PE outputs
        print("\nPE Outputs:")
        network.print_pe_outputs(pe_outputs)
        
        # Print traffic statistics
        traffic_table = network.get_traffic_table()
        print("\nTraffic Statistics:")
        total_bytes = traffic_table['bytes'].sum()
        total_tasks = len(traffic_table)
        print(f"Total bytes transferred: {total_bytes} bytes")
        print(f"Total communication tasks: {total_tasks}")
        
        # Print time statistics
        print(f"\nExecution time: {(end_time - start_time) * 1000:.2f} ms")
        
        # Clear the scheduler for the next test
        noc.scheduler.clear()

if __name__ == "__main__":
    # Run the matrix multiplication test
    test_arithmetic_network_matmul()
    
    # Run the element-wise operations test
    #test_elementwise_operations() 