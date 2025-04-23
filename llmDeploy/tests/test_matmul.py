import torch
import sys
import os
import time
import datetime
from typing import Dict, Tuple, Optional, List
import pandas as pd

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.pe_noc import NoCTopology
from llmDeploy.neural_network import ArithmeticNetwork

def log_detailed_pe_table(network, log_file=None):
    """
    Logs a detailed PE table showing only active PEs and their attributes.
    Similar to the output in example_sequential_llm.py but without memory usage stats.
    """
    noc = network.noc
    active_pes = network.active_pes
    
    # Skip if no active PEs
    if not active_pes:
        log_message = "\n\n===== Detailed PE Table =====\nNo active PEs found.\n"
        if log_file:
            log_file.write(log_message)
        return
    
    # Collect PE table headers based on first PE's attributes
    first_pe = noc.get_pe(*next(iter(active_pes)))
    attribute_names = []
    for attr_name in dir(first_pe):
        # Skip hidden attributes, functions, and methods
        if attr_name.startswith('_') or callable(getattr(first_pe, attr_name)):
            continue
        # Skip common basic attributes that are less relevant
        if attr_name in ['input_buffers', 'output_buffers', 'pe_id', 'memory_banks', 'memory_size']:
            continue
        attribute_names.append(attr_name)
    
    # Add headers for basic information
    headers = ['PE']
    headers.extend(attribute_names)
    
    # Create header
    header_line = " | ".join(f"{h:<20}" for h in headers)
    separator_line = "-" * len(header_line)
    
    log_message = "\n\n===== Detailed PE Table (Active PEs Only) =====\n"
    log_message += separator_line + "\n"
    log_message += header_line + "\n"
    log_message += separator_line + "\n"
    
    # Add only active PE rows
    for pe_coords in active_pes:
        pe = noc.get_pe(*pe_coords)
        
        # Create row with basic info
        row_data = [
            str(pe_coords)
        ]
        
        # Add attribute values
        for attr_name in attribute_names:
            if hasattr(pe, attr_name):
                attr_value = getattr(pe, attr_name)
                if attr_value is None:
                    row_data.append("None")
                elif isinstance(attr_value, (list, tuple, dict)):
                    # Truncate collections
                    val_str = str(attr_value)
                    if len(val_str) > 17:
                        val_str = val_str[:14] + "..."
                    row_data.append(val_str)
                else:
                    # Truncate long strings
                    val_str = str(attr_value)
                    if len(val_str) > 17:
                        val_str = val_str[:14] + "..."
                    row_data.append(val_str)
            else:
                row_data.append("")
        
        # Format and add row
        formatted_row = " | ".join(f"{val:<20}" for val in row_data)
        log_message += formatted_row + "\n"
    
    log_message += separator_line + "\n"
    
    # Write to log file
    if log_file:
        log_file.write(log_message)

def log_detailed_traffic_statistics(network, split_strategy, log_file=None):
    """
    Logs traffic statistics in a simplified format.
    Just shows the PE details and raw traffic table.
    """
    # Get PE details
    if hasattr(network.mapper, 'get_pe_details'):
        log_message = f"\n===== PE Details ({split_strategy}) =====\n"
        pe_details = network.mapper.get_pe_details()
        if not pe_details.empty:
            log_message += pe_details.to_string(index=False) + "\n\n"
        else:
            log_message += "No PE details available.\n\n"
    else:
        log_message = f"\n===== PE Details ({split_strategy}) =====\n"
        log_message += "get_pe_details method not available on mapper.\n\n"
    
    # Get traffic table
    traffic_table = network.get_traffic_table()
    total_bytes = traffic_table['bytes'].sum() if not traffic_table.empty else 0
    total_tasks = len(traffic_table)
    
    # Add traffic table summary
    log_message += f"===== Traffic Table ({split_strategy}) =====\n"
    log_message += f"Total bytes transferred: {total_bytes} bytes\n"
    log_message += f"Total communication tasks: {total_tasks}\n\n"
    
    if traffic_table.empty:
        log_message += "No traffic recorded.\n"
    else:
        # Just show the raw traffic table
        log_message += traffic_table.to_string() + "\n"
    
    # Write to log file
    if log_file:
        log_file.write(log_message)

def run_and_evaluate_strategy(noc, strategy, seq_len, d_model, log_file=None):
    """Run a single matrix multiplication test with the given strategy."""
    log_message = f"\n----- Testing {strategy} strategy -----"
    if log_file:
        log_file.write(log_message + "\n")
    
    # Create the arithmetic network with appropriate dimensions
    network = ArithmeticNetwork(
        noc=noc,
        seq_len=seq_len,
        d_model=d_model,
        mapping_strategy="grid_wise",  # Should distribute PEs across the grid
        split_strategy=strategy,
        data_type="float16",
        reuse_pe_for_aggregation=True,
        row_aggregation_enabled=True,
        column_aggregation_enabled=True
    )
    
    # Create sample input tensors
    input_a = torch.randn(seq_len, d_model)  # Q matrix
    input_b = torch.randn(seq_len, d_model)  # K matrix
    
    log_message = f"\nRunning matrix multiplication with {strategy} strategy..."
    if log_file:
        log_file.write(log_message + "\n")
    
    has_attr = hasattr(network, 'matrix_multiply')
    is_callable = callable(getattr(network, 'matrix_multiply')) if has_attr else False
    
    log_message = f"Has matrix_multiply attribute: {has_attr}\nmatrix_multiply is callable: {is_callable}"
    if log_file:
        log_file.write(log_message + "\n")
    
    try:
        # Execute matrix multiplication
        start_time = time.time()
        pe_outputs = network.matrix_multiply(input_a, input_b, transpose_b=True)
        end_time = time.time()
        exec_time = (end_time - start_time) * 1000  # ms
        
        # Capture PE outputs to log
        log_message = "\nPE Outputs:"
        if log_file:
            log_file.write(log_message + "\n")
            
        # Capture PE outputs in a string for logging
        pe_output_lines = []
        for pe_coords, (pe_output, output_range, computation_task_id) in pe_outputs.items():
            if isinstance(output_range, tuple) and len(output_range) == 2 and isinstance(output_range[0], int):
                start = output_range[0] if output_range[0] is not None else "None"
                end = output_range[1] if output_range[1] is not None else "None"
                range_str = f"(:, {start}:{end})"
            elif isinstance(output_range, tuple) and len(output_range) == 2 and isinstance(output_range[0], tuple):
                row_range, col_range = output_range
                row_part = f"{row_range[0]}:{row_range[1]}" if None not in row_range else "None:None"
                col_part = f"{col_range[0]}:{col_range[1]}" if None not in col_range else "None:None"
                range_str = f"({row_part}, {col_part})"
            else:
                range_str = str(output_range)
            
            output_line = f"PE{pe_coords} output: {pe_output.shape}, tensor slice: {range_str}, task_id: {computation_task_id}"
            pe_output_lines.append(output_line)
            if log_file:
                log_file.write(output_line + "\n")
        
        # Print utilization statistics
        utilization = network.get_pe_utilization(use_effective_dimensions=True)
        compute_util = utilization['computation_utilization']
        total_util = utilization['total_utilization']
        
        log_message = (f"\nPE Utilization:\n"
                      f"Total PEs: {utilization['total_pes']}\n"
                      f"Used computation PEs: {utilization['used_computation_pes']}\n"
                      f"Computation utilization: {compute_util:.2f}%\n"
                      f"Total utilization: {total_util:.2f}%")
        if log_file:
            log_file.write(log_message + "\n")
        
        log_message = f"\nExecution time: {exec_time:.2f} ms"
        if log_file:
            log_file.write(log_message + "\n")
        
        # Log detailed PE table
        log_detailed_pe_table(network, log_file)
        
        # Log detailed traffic statistics
        log_detailed_traffic_statistics(network, strategy, log_file)
        
        # Get traffic table for basic statistics
        traffic_table = network.get_traffic_table()
        total_bytes = traffic_table['bytes'].sum() if not traffic_table.empty else 0
        total_tasks = len(traffic_table)
        
        # Return results for summary table
        return {
            'bytes': total_bytes,
            'tasks': total_tasks,
            'util': compute_util,
            'time': exec_time
        }
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        log_message = f"Error in {strategy} test: {e}\n{error_trace}"
        if log_file:
            log_file.write(log_message + "\n")
        return {
            'bytes': 0,
            'tasks': 0,
            'util': 0,
            'time': 0
        }

def test_arithmetic_network_matmul():
    """
    Test the ArithmeticNetwork's matrix multiplication capability with different split strategies.
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = "00"
    log_filename = os.path.join(logs_dir, f"matmul_test_{timestamp}.log")
    
    with open(log_filename, 'w') as log_file:
        header = f"\n===== Testing ArithmeticNetwork Matrix Multiplication =====\n"
        header += f"Test started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        log_file.write(header)
        
        # Parameters
        rows, cols = 10, 10     # NoC grid size
        seq_len = 6          # Sequence length
        d_model = 128         # Model dimension
        memory_size = 12  # 36 bytes per PE
        
        # Log test parameters
        params = (f"Test Parameters:\n"
                 f"- NoC grid size: {rows}x{cols}\n"
                 f"- Sequence length: {seq_len}\n"
                 f"- Model dimension: {d_model}\n"
                 f"- Memory size per PE: {memory_size} bytes\n")
        log_file.write(params + "\n")
        
        # Initialize a NoC topology
        noc = NoCTopology(rows, cols, memory_size)
        
        # Dictionary to store results for comparison
        strategy_data = {}
        
        # Test column_split strategy and save results
        strategy_data["column_split"] = run_and_evaluate_strategy(noc, "column_split", seq_len, d_model, log_file)
        result_msg = f"\nColumn Split Results - Bytes: {strategy_data['column_split']['bytes']}, Tasks: {strategy_data['column_split']['tasks']}, Utilization: {strategy_data['column_split']['util']:.2f}%, Time: {strategy_data['column_split']['time']:.2f} ms"
        log_file.write(result_msg + "\n")
        
        # Test row_split strategy and save results
        strategy_data["row_split"] = run_and_evaluate_strategy(noc, "row_split", seq_len, d_model, log_file)
        result_msg = f"\nRow Split Results - Bytes: {strategy_data['row_split']['bytes']}, Tasks: {strategy_data['row_split']['tasks']}, Utilization: {strategy_data['row_split']['util']:.2f}%, Time: {strategy_data['row_split']['time']:.2f} ms"
        log_file.write(result_msg + "\n")
        
        # Test hybrid_split strategy and save results
        strategy_data["hybrid_split"] = run_and_evaluate_strategy(noc, "hybrid_split", seq_len, d_model, log_file)
        result_msg = f"\nHybrid Split Results - Bytes: {strategy_data['hybrid_split']['bytes']}, Tasks: {strategy_data['hybrid_split']['tasks']}, Utilization: {strategy_data['hybrid_split']['util']:.2f}%, Time: {strategy_data['hybrid_split']['time']:.2f} ms"
        log_file.write(result_msg + "\n")
        
        # Ensure flush to file
        log_file.flush()
        
        # Print summary table
        summary_header = "\n===== Matrix Multiplication Strategy Summary =====\n"
        summary_header += "----------------------------------------------------------\n"
        summary_header += f"{'Strategy':<15} {'Bytes':<10} {'Tasks':<8} {'Compute %':<10} {'Time (ms)':<10}\n"
        summary_header += "----------------------------------------------------------"
        log_file.write(summary_header + "\n")
        
        summary_rows = ""
        for strategy, data in strategy_data.items():
            row = f"{strategy:<15} {data['bytes']:<10} {data['tasks']:<8} {data['util']:<10.2f} {data['time']:<10.2f}"
            summary_rows += row + "\n"
        
        summary_footer = "----------------------------------------------------------"
        log_file.write(summary_rows)
        log_file.write(summary_footer + "\n")
        
        # Final message
        log_file.write("\nTest completed at: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
    return log_filename

if __name__ == "__main__":
    try:
        log_file = test_arithmetic_network_matmul()
        print(f"Test results saved to: {log_file}")  # Keeping only this print statement for user feedback
    except Exception as e:
        import traceback
        print(f"Critical Error: {e}")  # Keeping error reporting
        print(traceback.format_exc()) 