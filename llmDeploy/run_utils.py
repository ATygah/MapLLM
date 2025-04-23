import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import logging
from .neural_network import FCNeuralNetwork
import math

def run_example(log_file="noc_simulation.log", data_type="float16", channel_bandwidth=32.0, row_aggregation_enabled=True):
    """Run a simple example with a FC neural network."""
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example parameters
    input_dim = 768
    hidden_dim = 3072
    output_dim = 768
    seq_len = 1
    
    # Create input tensor
    input_tensor = torch.randn(seq_len, input_dim)
    
    # Create neural network with different configurations for comparison
    configurations = [
        {
            "name": "Column-wise, Column-split",
            "mapping_strategy": "column_wise",
            "split_strategy": "column_split",
            "reuse_pe_for_aggregation": True
        },
        {
            "name": "Column-wise, Row-split",
            "mapping_strategy": "column_wise",
            "split_strategy": "row_split",
            "reuse_pe_for_aggregation": True
        },
        {
            "name": "Column-wise, Hybrid-split",
            "mapping_strategy": "column_wise",
            "split_strategy": "hybrid_split",
            "reuse_pe_for_aggregation": True
        },
        {
            "name": "Grid-wise, Hybrid-split",
            "mapping_strategy": "grid_wise",
            "split_strategy": "hybrid_split",
            "reuse_pe_for_aggregation": True
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nRunning configuration: {config['name']}")
        nn = FCNeuralNetwork(
            input_dim=input_dim,
            layer_dims=[hidden_dim, output_dim],
            seq_len=seq_len,
            mapping_strategy=config["mapping_strategy"],
            split_strategy=config["split_strategy"],
            reuse_pe_for_aggregation=config["reuse_pe_for_aggregation"],
            row_aggregation_enabled=row_aggregation_enabled,
            data_type=data_type,
            channel_bandwidth=channel_bandwidth
        )
        
        # Run inference
        pe_outputs = nn.run_inference(input_tensor)
        
        # Get traffic table
        traffic_table = nn.get_traffic_table()
        
        # Get PE utilization
        utilization = nn.get_pe_utilization(use_effective_dimensions=True)
        
        # Log results
        logging.info(f"Configuration: {config['name']}")
        logging.info(f"NoC Dimensions: {nn.noc.rows}x{nn.noc.cols}")
        logging.info(f"Active PEs: {len(nn.active_pes)}")
        logging.info(f"Utilization: {utilization['total_utilization']:.2f}%")
        logging.info(f"Total Traffic Tasks: {len(traffic_table)}")
        logging.info(f"Total Cycles: {traffic_table['cycles'].sum()}")
        
        # Print results
        print(f"NoC Dimensions: {nn.noc.rows}x{nn.noc.cols}")
        print(f"Active PEs: {len(nn.active_pes)}")
        print(f"Utilization: {utilization['total_utilization']:.2f}%")
        print(f"Total Traffic Tasks: {len(traffic_table)}")
        print(f"Total Cycles: {traffic_table['cycles'].sum()}")
        
        # Store results for comparison
        results.append({
            "config_name": config["name"],
            "noc_dims": f"{nn.noc.rows}x{nn.noc.cols}",
            "active_pes": len(nn.active_pes),
            "utilization": utilization['total_utilization'],
            "traffic_tasks": len(traffic_table),
            "total_cycles": traffic_table['cycles'].sum()
        })
    
    # Create comparison table
    results_df = pd.DataFrame(results)
    print("\nComparison of Configurations:")
    print(results_df)
    
    return results_df

def analyze_pe_memory_impact(
    input_dim=768, 
    hidden_dim=3072, 
    output_dim=768, 
    seq_len=1,
    data_type="float16", 
    channel_bandwidth=32.0,
    memory_sizes=None,
    mapping_strategy="column_wise",
    split_strategy="hybrid_split",
    reuse_pe_for_aggregation=False,
    row_aggregation_enabled=True,
    save_plot=True,
    plot_filename="pe_memory_analysis.png",
    use_effective_dimensions=True
):
    """
    Analyze the impact of PE memory size on NoC resource usage and performance.
    """
    if memory_sizes is None:
        memory_sizes = [
            16 * 1024,    # 16 KB
            32 * 1024,    # 32 KB
            64 * 1024,    # 64 KB
            128 * 1024,   # 128 KB
            256 * 1024,   # 256 KB
            512 * 1024,   # 512 KB
            1024 * 1024   # 1 MB
        ]
    
    results = []
    
    # Prepare input tensor
    input_tensor = torch.randn(seq_len, input_dim)
    
    # Store original dimensions for scaling
    orig_input_dim = input_dim
    orig_hidden_dim = hidden_dim
    orig_output_dim = output_dim
    
    # Run analysis for each memory size
    for memory_size in memory_sizes:
        print(f"\nAnalyzing PE memory size: {memory_size/1024:.1f} KB")
        
        # Scale dimensions if using effective dimensions
        if use_effective_dimensions:
            # Calculate scaling factor based on memory size
            base_memory = 64 * 1024  # 64 KB as reference
            scale = math.sqrt(memory_size / base_memory)
            
            # Scale dimensions
            input_dim = int(orig_input_dim * scale)
            hidden_dim = int(orig_hidden_dim * scale)
            output_dim = int(orig_output_dim * scale)
            
            print(f"Using effective dimensions: input={input_dim}, hidden={hidden_dim}, output={output_dim}")
            
            # Re-initialize input tensor with new dimensions
            input_tensor = torch.randn(seq_len, input_dim)
        
        try:
            nn = FCNeuralNetwork(
                input_dim=input_dim,
                layer_dims=[hidden_dim, output_dim],
                seq_len=seq_len,
                mapping_strategy=mapping_strategy,
                split_strategy=split_strategy,
                reuse_pe_for_aggregation=reuse_pe_for_aggregation,
                row_aggregation_enabled=row_aggregation_enabled,
                data_type=data_type,
                channel_bandwidth=channel_bandwidth,
                pe_memory_size=memory_size
            )
            
            # Get dimensions and utilization
            if use_effective_dimensions:
                utilization = nn.get_pe_utilization(use_effective_dimensions=True)
                effective_rows = utilization.get('effective_rows', 0)
                effective_cols = utilization.get('effective_cols', 0)
                effective_total_pes = utilization.get('total_pes', 0)
                noc_dims = f"{effective_rows}x{effective_cols} (effective)"
                noc_area = effective_total_pes
            else:
                noc_dims = f"{nn.noc.rows}x{nn.noc.cols}"
                noc_area = nn.noc.rows * nn.noc.cols
            
            # Run a simple inference to get traffic stats
            pe_outputs = nn.run_inference(input_tensor)
            traffic_table = nn.get_traffic_table()
            
            # Calculate metrics
            total_cycles = traffic_table['cycles'].sum()
            active_pes = len(nn.active_pes)
            utilization_pct = active_pes / noc_area * 100 if noc_area > 0 else 0
            
            # Store results
            results.append({
                "memory_size_kb": memory_size / 1024,
                "noc_dims": noc_dims,
                "noc_area": noc_area,
                "active_pes": active_pes,
                "utilization_pct": utilization_pct,
                "total_cycles": total_cycles,
                "traffic_tasks": len(traffic_table)
            })
            
            # Print results
            print(f"NoC Dimensions: {noc_dims}")
            print(f"NoC Area (total PEs): {noc_area}")
            print(f"Active PEs: {active_pes}")
            print(f"Utilization: {utilization_pct:.2f}%")
            print(f"Total Traffic Tasks: {len(traffic_table)}")
            print(f"Total Cycles: {total_cycles}")
        except ValueError as e:
            print(f"Error during inference: {e}")
            total_cycles = float('nan')
            traffic_tasks = 0
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Generate plot
    if save_plot:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot NoC area and utilization
        color = 'tab:blue'
        ax1.set_xlabel('PE Memory Size (KB)')
        ax1.set_ylabel('NoC Area (# of PEs)', color=color)
        ax1.plot(results_df['memory_size_kb'], results_df['noc_area'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for utilization
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Utilization (%)', color=color)
        ax2.plot(results_df['memory_size_kb'], results_df['utilization_pct'], color=color, marker='s')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Create third y-axis for cycles
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))
        color = 'tab:green'
        ax3.set_ylabel('Total Cycles', color=color)
        ax3.plot(results_df['memory_size_kb'], results_df['total_cycles'], color=color, marker='^')
        ax3.tick_params(axis='y', labelcolor=color)
        
        # Add title and save
        mapping_name = mapping_strategy.replace('_', ' ').title()
        split_name = split_strategy.replace('_', ' ').title()
        plt.title(f'Effect of PE Memory Size on NoC Metrics\n({mapping_name}, {split_name})')
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
    
    print("\nPE Memory Size Analysis Results:")
    print(results_df)
    
    return results_df

def analyze_split_strategies(
    input_dim=768, 
    hidden_dim=3072, 
    output_dim=768, 
    seq_len=1,
    pe_memory_size=64 * 1024,  # 64 KB
    data_type="float16", 
    channel_bandwidth=32.0,
    mapping_strategy="column_wise",
    reuse_pe_for_aggregation=False,
    row_aggregation_enabled=True,
    save_plot=True,
    plot_filename="split_strategy_analysis.png",
    use_effective_dimensions=True
):
    """
    Analyze the performance of different split strategies.
    """
    # Define split strategies
    split_strategies = ["column_split", "row_split", "hybrid_split"]
    
    results = []
    
    # Prepare input tensor
    input_tensor = torch.randn(seq_len, input_dim)
    
    for split_strategy in split_strategies:
        print(f"\nAnalyzing split strategy: {split_strategy}")
        
        try:
            nn = FCNeuralNetwork(
                input_dim=input_dim,
                layer_dims=[hidden_dim, output_dim],
                seq_len=seq_len,
                pe_memory_size=pe_memory_size,
                mapping_strategy=mapping_strategy,
                split_strategy=split_strategy,
                reuse_pe_for_aggregation=reuse_pe_for_aggregation,
                row_aggregation_enabled=row_aggregation_enabled,
                data_type=data_type,
                channel_bandwidth=channel_bandwidth
            )
            
            # Run inference and gather statistics
            pe_outputs = nn.run_inference(input_tensor)
            traffic_table = nn.get_traffic_table()
            
            # Calculate metrics
            total_cycles = traffic_table['cycles'].sum()
            active_pes = len(nn.active_pes)
            utilization_pct = active_pes / (nn.noc.rows * nn.noc.cols) * 100 if (nn.noc.rows * nn.noc.cols) > 0 else 0
            
            # Store results
            results.append({
                "split_strategy": split_strategy,
                "noc_dims": f"{nn.noc.rows}x{nn.noc.cols}",
                "noc_area": nn.noc.rows * nn.noc.cols,
                "active_pes": active_pes,
                "utilization_pct": utilization_pct,
                "total_cycles": total_cycles,
                "traffic_tasks": len(traffic_table)
            })
            
            # Print results
            print(f"NoC Dimensions: {nn.noc.rows}x{nn.noc.cols}")
            print(f"NoC Area (total PEs): {nn.noc.rows * nn.noc.cols}")
            print(f"Active PEs: {active_pes}")
            print(f"Utilization: {utilization_pct:.2f}%")
            print(f"Total Traffic Tasks: {len(traffic_table)}")
            print(f"Total Cycles: {total_cycles}")
        except ValueError as e:
            print(f"Error during inference: {e}")
            total_cycles = float('nan')
            traffic_tasks = 0
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Generate plot
    if save_plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        x = range(len(split_strategies))
        width = 0.2
        
        # Plot bars for different metrics
        ax.bar([i-width for i in x], results_df['noc_area'], width, label='NoC Area (PEs)', color='blue')
        ax.bar([i for i in x], results_df['active_pes'], width, label='Active PEs', color='green')
        ax.bar([i+width for i in x], results_df['utilization_pct'], width, label='Utilization (%)', color='red')
        
        # Create second y-axis for cycles
        ax2 = ax.twinx()
        ax2.plot(x, results_df['total_cycles'], 'o-', color='purple', label='Total Cycles')
        ax2.set_ylabel('Total Cycles', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        
        # Set labels and title
        ax.set_xlabel('Split Strategy')
        ax.set_ylabel('Count / Percentage')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in split_strategies])
        ax.legend()
        
        # Add title and save
        mapping_name = mapping_strategy.replace('_', ' ').title()
        plt.title(f'Comparison of Split Strategies ({mapping_name} Mapping)')
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
    
    print("\nSplit Strategy Analysis Results:")
    print(results_df)
    
    return results_df

def analyze_network_dimensions(
    base_dim=64,
    scaling_factors=None,
    seq_len=1,
    pe_memory_size=64 * 1024,  # 64 KB
    data_type="float16", 
    channel_bandwidth=32.0,
    mapping_strategy="column_wise",
    split_strategy="hybrid_split",
    reuse_pe_for_aggregation=False,
    row_aggregation_enabled=True,
    save_plot=True,
    plot_filename="network_dimension_analysis.png",
    use_effective_dimensions=True
):
    """
    Analyze the impact of network dimensions on NoC resource usage and performance.
    """
    if scaling_factors is None:
        scaling_factors = [1, 2, 4, 8, 16, 32, 64]
    
    results = []
    
    for scale in scaling_factors:
        print(f"\nAnalyzing network scale factor: {scale}")
        
        # Scale dimensions
        input_dim = base_dim * scale
        hidden_dim = base_dim * scale * 4  # Typical MLP expansion
        output_dim = base_dim * scale
        
        print(f"Dimensions: input={input_dim}, hidden={hidden_dim}, output={output_dim}")
        
        # Create input tensor
        input_tensor = torch.randn(seq_len, input_dim)
        
        try:
            nn = FCNeuralNetwork(
                input_dim=input_dim,
                layer_dims=[hidden_dim, output_dim],
                seq_len=seq_len,
                pe_memory_size=pe_memory_size,
                mapping_strategy=mapping_strategy,
                split_strategy=split_strategy,
                reuse_pe_for_aggregation=reuse_pe_for_aggregation,
                row_aggregation_enabled=row_aggregation_enabled,
                data_type=data_type,
                channel_bandwidth=channel_bandwidth
            )
            
            # Run inference and gather statistics
            pe_outputs = nn.run_inference(input_tensor)
            traffic_table = nn.get_traffic_table()
            total_cycles = traffic_table['cycles'].sum()
            traffic_tasks = len(traffic_table)
            
            # Calculate metrics
            active_pes = len(nn.active_pes)
            utilization_pct = active_pes / (nn.noc.rows * nn.noc.cols) * 100 if (nn.noc.rows * nn.noc.cols) > 0 else 0
            
            # Store results
            results.append({
                "scale_factor": scale,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "noc_dims": f"{nn.noc.rows}x{nn.noc.cols}",
                "noc_area": nn.noc.rows * nn.noc.cols,
                "active_pes": active_pes,
                "utilization_pct": utilization_pct,
                "total_cycles": total_cycles,
                "traffic_tasks": traffic_tasks
            })
            
            # Print results
            print(f"NoC Dimensions: {nn.noc.rows}x{nn.noc.cols}")
            print(f"NoC Area (total PEs): {nn.noc.rows * nn.noc.cols}")
            print(f"Active PEs: {active_pes}")
            print(f"Utilization: {utilization_pct:.2f}%")
            print(f"Total Traffic Tasks: {traffic_tasks}")
            print(f"Total Cycles: {total_cycles}")
        except ValueError as e:
            print(f"Error during inference: {e}")
            total_cycles = float('nan')
            traffic_tasks = 0
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Generate plot
    if save_plot:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot NoC area
        color = 'tab:blue'
        ax1.set_xlabel('Network Scale Factor')
        ax1.set_ylabel('NoC Area (# of PEs)', color=color)
        ax1.plot(results_df['scale_factor'], results_df['noc_area'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for utilization
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Utilization (%)', color=color)
        ax2.plot(results_df['scale_factor'], results_df['utilization_pct'], color=color, marker='s')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Create third y-axis for cycles
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))
        color = 'tab:green'
        ax3.set_ylabel('Total Cycles (log scale)', color=color)
        ax3.semilogy(results_df['scale_factor'], results_df['total_cycles'], color=color, marker='^')
        ax3.tick_params(axis='y', labelcolor=color)
        
        # Add title and save
        mapping_name = mapping_strategy.replace('_', ' ').title()
        split_name = split_strategy.replace('_', ' ').title()
        plt.title(f'Effect of Network Dimensions on NoC Metrics\n({mapping_name}, {split_name})')
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
    
    print("\nNetwork Dimension Analysis Results:")
    print(results_df)
    
    return results_df

def analyze_mapping_strategies(
    input_dim=768, 
    hidden_dim=3072, 
    output_dim=768, 
    seq_len=1,
    pe_memory_size=64 * 1024,  # 64 KB
    data_type="float16", 
    channel_bandwidth=32.0,
    split_strategy="hybrid_split",
    reuse_pe_for_aggregation=False,
    row_aggregation_enabled=True,
    save_plot=True,
    plot_filename="mapping_strategy_analysis.png",
    use_effective_dimensions=True
):
    """
    Analyze the performance of different mapping strategies.
    """
    # Define mapping strategies
    mapping_strategies = ["column_wise", "row_wise", "grid_wise"]
    
    results = []
    
    # Prepare input tensor
    input_tensor = torch.randn(seq_len, input_dim)
    
    for mapping_strategy in mapping_strategies:
        print(f"\nAnalyzing mapping strategy: {mapping_strategy}")
        
        try:
            nn = FCNeuralNetwork(
                input_dim=input_dim,
                layer_dims=[hidden_dim, output_dim],
                seq_len=seq_len,
                pe_memory_size=pe_memory_size,
                mapping_strategy=mapping_strategy,
                split_strategy=split_strategy,
                reuse_pe_for_aggregation=reuse_pe_for_aggregation,
                row_aggregation_enabled=row_aggregation_enabled,
                data_type=data_type,
                channel_bandwidth=channel_bandwidth
            )
            
            # Run inference and gather statistics
            pe_outputs = nn.run_inference(input_tensor)
            traffic_table = nn.get_traffic_table()
            
            # Calculate metrics
            total_cycles = traffic_table['cycles'].sum()
            active_pes = len(nn.active_pes)
            utilization_pct = active_pes / (nn.noc.rows * nn.noc.cols) * 100 if (nn.noc.rows * nn.noc.cols) > 0 else 0
            
            # Store results
            results.append({
                "mapping_strategy": mapping_strategy,
                "noc_dims": f"{nn.noc.rows}x{nn.noc.cols}",
                "noc_area": nn.noc.rows * nn.noc.cols,
                "active_pes": active_pes,
                "utilization_pct": utilization_pct,
                "total_cycles": total_cycles,
                "traffic_tasks": len(traffic_table)
            })
            
            # Print results
            print(f"NoC Dimensions: {nn.noc.rows}x{nn.noc.cols}")
            print(f"NoC Area (total PEs): {nn.noc.rows * nn.noc.cols}")
            print(f"Active PEs: {active_pes}")
            print(f"Utilization: {utilization_pct:.2f}%")
            print(f"Total Traffic Tasks: {len(traffic_table)}")
            print(f"Total Cycles: {total_cycles}")
        except ValueError as e:
            print(f"Error during inference: {e}")
            total_cycles = float('nan')
            traffic_tasks = 0
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Generate plot
    if save_plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        x = range(len(mapping_strategies))
        width = 0.2
        
        # Plot bars for different metrics
        ax.bar([i-width for i in x], results_df['noc_area'], width, label='NoC Area (PEs)', color='blue')
        ax.bar([i for i in x], results_df['active_pes'], width, label='Active PEs', color='green')
        ax.bar([i+width for i in x], results_df['utilization_pct'], width, label='Utilization (%)', color='red')
        
        # Create second y-axis for cycles
        ax2 = ax.twinx()
        ax2.plot(x, results_df['total_cycles'], 'o-', color='purple', label='Total Cycles')
        ax2.set_ylabel('Total Cycles', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        
        # Set labels and title
        ax.set_xlabel('Mapping Strategy')
        ax.set_ylabel('Count / Percentage')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in mapping_strategies])
        ax.legend()
        
        # Add title and save
        split_name = split_strategy.replace('_', ' ').title()
        plt.title(f'Comparison of Mapping Strategies ({split_name} Split)')
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
    
    print("\nMapping Strategy Analysis Results:")
    print(results_df)
    
    return results_df

def run_all_analyses(output_dir="analysis_results", use_effective_dimensions=True, row_aggregation_enabled=True):
    """
    Run all analyses and save results to output directory.
    
    Args:
        output_dir: Directory to save plot files in
        use_effective_dimensions: Whether to use effective dimensions for utilization calculations
        row_aggregation_enabled: Whether to aggregate partial results (True) or pass unaggregated results (False)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Run memory analysis
    memory_plot = os.path.join(output_dir, "memory_analysis.png")
    analyze_pe_memory_impact(
        save_plot=True, 
        plot_filename=memory_plot,
        use_effective_dimensions=use_effective_dimensions,
        row_aggregation_enabled=row_aggregation_enabled
    )
    
    # Run split strategy analysis
    split_plot = os.path.join(output_dir, "split_analysis.png")
    analyze_split_strategies(
        save_plot=True, 
        plot_filename=split_plot,
        use_effective_dimensions=use_effective_dimensions,
        row_aggregation_enabled=row_aggregation_enabled
    )
    
    # Run network dimension analysis
    dim_plot = os.path.join(output_dir, "dimension_analysis.png")
    analyze_network_dimensions(
        save_plot=True, 
        plot_filename=dim_plot,
        use_effective_dimensions=use_effective_dimensions,
        row_aggregation_enabled=row_aggregation_enabled
    )
    
    # Run mapping strategy analysis
    map_plot = os.path.join(output_dir, "mapping_analysis.png")
    analyze_mapping_strategies(
        save_plot=True, 
        plot_filename=map_plot,
        use_effective_dimensions=use_effective_dimensions,
        row_aggregation_enabled=row_aggregation_enabled
    )
    
    print(f"\nAll analyses complete. Results saved to {output_dir}/")
    return {
        "memory_plot": memory_plot,
        "split_plot": split_plot,
        "dimension_plot": dim_plot,
        "mapping_plot": map_plot
    }

def export_traffic_table_to_file(network, filename=None, directory=None):
    """
    Export traffic table to both txt and tsv files.
    
    Args:
        network: Neural network object or LLM object with traffic data
        filename: Base name of output file without extension (default: traffic_table_{timestamp})
        directory: Directory to save the file (default: 'llmDeploy/traces')
    
    Returns:
        Dictionary: Paths to the exported files (txt and tsv)
    """
    import os
    import re
    from datetime import datetime
    
    # Get traffic table - handle either LLM objects or network objects
    if hasattr(network, 'get_traffic_table'):
        traffic_table = network.get_traffic_table()
    elif hasattr(network, 'noc') and hasattr(network.noc, 'scheduler'):
        # If passed an LLM object, get the traffic table from its NoC scheduler
        traffic_table = network.noc.scheduler.get_traffic_table()
    else:
        raise ValueError("Network object must have a get_traffic_table method or a noc attribute with a scheduler")
    
    # Get NoC dimensions
    if hasattr(network, 'noc'):
        noc_rows = network.noc.rows
        noc_cols = network.noc.cols
    else:
        noc_rows = network.rows if hasattr(network, 'rows') else 1
        noc_cols = network.cols if hasattr(network, 'cols') else 1
    
    # Set default directory inside llmDeploy
    if directory is None:
        directory = os.path.join('llmDeploy', 'traces')
    
    # Generate base filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f'traffic_table_{timestamp}'
    else:
        # Remove any extension if present
        base_filename = os.path.splitext(filename)[0]
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Define output filenames with full paths
    txt_filename = f"{base_filename}.txt"
    tsv_filename = f"{base_filename}.tsv"
    txt_output_path = os.path.join(directory, txt_filename)
    tsv_output_path = os.path.join(directory, tsv_filename)
    
    # Process the traffic data
    rows = []
    
    # Extract columns we care about
    src_col = 'src_pe' if 'src_pe' in traffic_table.columns else 'source'
    dest_col = 'dest_pe' if 'dest_pe' in traffic_table.columns else 'destination'
    data_size_col = 'bytes' if 'bytes' in traffic_table.columns else 'data_size'
    task_id_col = 'task_id'
    wait_ids_col = 'wait_ids' if 'wait_ids' in traffic_table.columns else None
    description_col = 'description' if 'description' in traffic_table.columns else None
    
    # Process each row to convert PE coordinates to numbers for TXT file
    for _, row in traffic_table.iterrows():
        try:
            # Get task ID
            task_id = row[task_id_col] if task_id_col in row else "unknown"
            
            # Source PE processing
            src = row[src_col]
            if isinstance(src, tuple) and len(src) == 2:
                # For (x,y) format where x is column, y is row
                src_pe = src[1] * noc_cols + src[0]
            elif isinstance(src, str) and '(' in src:
                # Extract coordinates from string format like "(0, 0)"
                match = re.search(r'\((\d+),\s*(\d+)\)', src)
                if match:
                    # x is match.group(1), y is match.group(2)
                    src_pe = int(match.group(2)) * noc_cols + int(match.group(1))
                else:
                    src_pe = -1
            else:
                src_pe = -1
                
            # Destination PE processing
            dest = row[dest_col]
            if isinstance(dest, tuple) and len(dest) == 2:
                # For (x,y) format where x is column, y is row
                dest_pe = dest[1] * noc_cols + dest[0]
            elif isinstance(dest, str) and '(' in dest:
                # Extract coordinates from string format like "(0, 0)"
                match = re.search(r'\((\d+),\s*(\d+)\)', dest)
                if match:
                    # x is match.group(1), y is match.group(2)
                    dest_pe = int(match.group(2)) * noc_cols + int(match.group(1))
                else:
                    dest_pe = -1
            else:
                dest_pe = -1
            
            # Data size processing
            data_size = int(row[data_size_col]) if data_size_col in row else 0
            
            # Wait IDs processing
            wait_ids = str(row[wait_ids_col]) if wait_ids_col and wait_ids_col in row and row[wait_ids_col] is not None else "None"
            
            # Store as tuple - reordered as specified: task_id, source_pe, dest_pe, data_size, wait_ids
            rows.append((task_id, src_pe, dest_pe, data_size, wait_ids))
        except Exception as e:
            print(f"Error processing row: {e}")
    
    # Write to TXT file with UTF-8 encoding and LF line endings
    # This is the C++ parsing-friendly format with flattened PE coordinates
    with open(txt_output_path, 'w', encoding='utf-8', newline='\n') as f:
        # Write header as comments
        f.write(f"# Traffic table exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# NoC dimensions: {noc_rows}x{noc_cols}\n")
        f.write(f"# Columns: task_id\tsource_pe\tdest_pe\tdata_size\twait_ids\n\n")
        
        # Write data rows with tab separation for C++ parsing in the specified order
        for task_id, src_pe, dest_pe, data_size, wait_ids in rows:
            line = f"{task_id}\t{src_pe}\t{dest_pe}\t{data_size}\t{wait_ids}\n"
            f.write(line)
    
    # Write to TSV file with UTF-8 encoding and LF line endings
    # This is the human-readable format with all original columns including PE coordinates and descriptions
    with open(tsv_output_path, 'w', encoding='utf-8', newline='\n') as f:
        # Write header as comments
        f.write(f"# Traffic table exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# NoC dimensions: {noc_rows}x{noc_cols}\n")
        
        # Get all columns from the original traffic table
        columns = list(traffic_table.columns)
        
        # Write header row with column names
        f.write('\t'.join(columns) + '\n')
        
        # Write all rows with all columns from the original traffic table
        for _, row in traffic_table.iterrows():
            values = []
            for col in columns:
                # Ensure all values are strings
                val = str(row[col]) if row[col] is not None else ""
                values.append(val)
            f.write('\t'.join(values) + '\n')
    
    # Verify the files were written correctly
    txt_tab_count = 0
    tsv_tab_count = 0
    with open(txt_output_path, 'rb') as f:
        content = f.read()
        txt_tab_count = content.count(b'\t')
    
    with open(tsv_output_path, 'rb') as f:
        content = f.read()
        tsv_tab_count = content.count(b'\t')
    
    print(f"Traffic table exported to {txt_output_path} with {txt_tab_count} tab characters")
    print(f"Enhanced traffic table exported to {tsv_output_path} with {tsv_tab_count} tab characters")
    
    return {
        "txt": txt_output_path,
        "tsv": tsv_output_path
    }

def create_enhanced_traffic_table(network):
    """
    Create an enhanced traffic table with network names added directly to PE coordinates.
    
    Args:
        network: Neural network object or LLM object with traffic data and PE mapping
    
    Returns:
        DataFrame: The enhanced traffic table with network names included in src_pe and dest_pe fields
    """
    import pandas as pd
    import re
    
    # Get traffic table - handle either LLM objects or network objects
    if hasattr(network, 'get_traffic_table'):
        traffic_table = network.get_traffic_table()
    elif hasattr(network, 'noc') and hasattr(network.noc, 'scheduler'):
        # If passed an LLM object, get the traffic table from its NoC scheduler
        traffic_table = network.noc.scheduler.get_traffic_table()
    else:
        raise ValueError("Network object must have a get_traffic_table method or a noc attribute with a scheduler")
    
    # Check if traffic table is available
    if traffic_table is None or traffic_table.empty:
        return pd.DataFrame()  # Return empty DataFrame if no traffic data
    
    # Get PE mapping details - handle either objects with get_pe_mapping_details or LLM objects
    if hasattr(network, 'get_pe_mapping_details'):
        pe_mapping_df = network.get_pe_mapping_details()
    elif hasattr(network, 'llm') and hasattr(network.llm, 'get_pe_mapping_details'):
        pe_mapping_df = network.llm.get_pe_mapping_details()
    else:
        # If mapping details are not available, return unmodified traffic table
        return traffic_table.copy()
    
    # Create a copy of the traffic table to modify
    enhanced_traffic_table = traffic_table.copy()
    
    # Create mapping from PEs to network names
    pe_to_network = {}
    if not pe_mapping_df.empty:
        # Process the dataframe to create PE to network mapping
        for _, row in pe_mapping_df.iterrows():
            network_name = row['network_name']
            pe_coord_str = str(row['pe_coords'])
            
            # Extract coordinates from string like "(x, y)"
            try:
                pe_coord_str = pe_coord_str.strip('()')
                if ',' in pe_coord_str:
                    x, y = map(int, pe_coord_str.split(','))
                    pe_to_network[(x, y)] = network_name
            except Exception as e:
                print(f"Warning: Could not parse PE coordinates: {pe_coord_str}, Error: {str(e)}")
    
    # Extract columns we care about
    src_col = 'src_pe' if 'src_pe' in enhanced_traffic_table.columns else 'source'
    dest_col = 'dest_pe' if 'dest_pe' in enhanced_traffic_table.columns else 'destination'
    
    # Add network information directly to PE coordinates
    for idx, row in enhanced_traffic_table.iterrows():
        # Extract source PE coordinates and add network info
        src_pe_str = str(row[src_col]).strip('()')
        if src_pe_str and ',' in src_pe_str:
            try:
                # Handle different formats of PE coordinates
                src_pe_match = re.search(r'(\d+),\s*(\d+)', src_pe_str)
                if src_pe_match:
                    src_x, src_y = map(int, src_pe_match.groups())
                    src_pe = (src_x, src_y)
                    src_network = pe_to_network.get(src_pe, "external")
                    enhanced_traffic_table.at[idx, src_col] = f"({src_x}, {src_y}) ({src_network})"
            except Exception as e:
                print(f"Warning: Could not enhance source PE: {src_pe_str}, Error: {str(e)}")
        
        # Extract destination PE coordinates and add network info
        dest_pe_str = str(row[dest_col]).strip('()') if row[dest_col] != "None" and row[dest_col] is not None else ""
        if dest_pe_str and ',' in dest_pe_str:
            try:
                # Handle different formats of PE coordinates
                dest_pe_match = re.search(r'(\d+),\s*(\d+)', dest_pe_str)
                if dest_pe_match:
                    dest_x, dest_y = map(int, dest_pe_match.groups())
                    dest_pe = (dest_x, dest_y)
                    dest_network = pe_to_network.get(dest_pe, "external")
                    enhanced_traffic_table.at[idx, dest_col] = f"({dest_x}, {dest_y}) ({dest_network})"
            except Exception as e:
                print(f"Warning: Could not enhance destination PE: {dest_pe_str}, Error: {str(e)}")
    
    return enhanced_traffic_table

def generate_dependency_graph(traffic_table, output_filename=None, title="Task Dependency Graph", figsize=(16, 10)):
    """
    Generate a task dependency graph visualization from a traffic table.
    
    Args:
        traffic_table (DataFrame): The traffic table with task dependencies.
            Must have columns: task_id, src_pe, dest_pe, data_size, wait_ids
        output_filename (str, optional): Path to save the output image. If None, will just display the plot.
        title (str, optional): Title for the graph.
        figsize (tuple, optional): Figure size as (width, height) in inches.
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import re
    import pandas as pd
    
    # Check if traffic table is empty or None
    if traffic_table is None or traffic_table.empty:
        print("Warning: Traffic table is empty or None. Cannot generate dependency graph.")
        return None
    
    # Standardize column names
    column_mapping = {
        'task_id': 'task_id',
        'id': 'task_id',
        'src_pe': 'src_pe',
        'source': 'src_pe',
        'source_pe': 'src_pe',
        'dest_pe': 'dest_pe',
        'destination': 'dest_pe',
        'destination_pe': 'dest_pe',
        'data_size': 'data_size',
        'bytes': 'data_size',
        'size': 'data_size',
        'wait_ids': 'wait_ids',
        'dependencies': 'wait_ids',
        'description': 'description'
    }
    
    # Rename columns if needed
    for orig_col, std_col in column_mapping.items():
        if orig_col in traffic_table.columns and std_col not in traffic_table.columns:
            traffic_table = traffic_table.rename(columns={orig_col: std_col})
    
    # Make sure required columns exist
    required_cols = ['task_id', 'src_pe', 'dest_pe', 'data_size', 'wait_ids']
    missing_cols = [col for col in required_cols if col not in traffic_table.columns]
    if missing_cols:
        raise ValueError(f"Traffic table is missing required columns: {', '.join(missing_cols)}")
    
    # Parse task data
    tasks = []
    for idx, row in traffic_table.iterrows():
        task_id = int(row['task_id'])
        src_pe = str(row['src_pe'])
        dest_pe = str(row['dest_pe'])
        
        # Parse dependencies
        dependencies = []
        wait_ids = row['wait_ids']
        
        if isinstance(wait_ids, str) and wait_ids != "None" and wait_ids is not None:
            # Handle multiple comma-separated dependencies
            for dep_part in wait_ids.replace(" ", "").split(','):
                if dep_part.isdigit():
                    dependencies.append(int(dep_part))
        
        # Determine if task is computation or communication
        is_computation = src_pe == dest_pe and "external" not in src_pe.lower()
        task_type = "Computation" if is_computation else "Communication"
        
        # Get description if available
        description = row.get('description', f"Task {task_id}: {src_pe} -> {dest_pe}")
        
        # Add to tasks list
        tasks.append({
            'task_id': task_id,
            'src_pe': src_pe,
            'dest_pe': dest_pe,
            'dependencies': dependencies,
            'is_computation': is_computation,
            'type': task_type,
            'description': description
        })
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for task in tasks:
        task_id = task['task_id']
        G.add_node(task_id, label=str(task_id), type=task['type'])
    
    # Add edges
    for task in tasks:
        task_id = task['task_id']
        for dep in task['dependencies']:
            G.add_edge(dep, task_id)
    
    # Compute node layers for horizontal positioning
    node_layers = {}
    root_nodes = [task['task_id'] for task in tasks if not task['dependencies']]
    
    def assign_layers(node, current_layer=0):
        if node in node_layers:
            node_layers[node] = max(node_layers[node], current_layer)
        else:
            node_layers[node] = current_layer
        
        # Process children
        for successor in G.successors(node):
            assign_layers(successor, current_layer + 1)
    
    # Assign initial layers
    for root in root_nodes:
        assign_layers(root)
    
    # Find the critical path (longest path through the graph)
    critical_path = []
    if tasks:
        # Create a topological sort
        topo_sort = list(nx.topological_sort(G))
        
        # Dictionary to store longest path lengths to each node
        longest_paths = {node: 0 for node in G.nodes()}
        predecessors = {node: None for node in G.nodes()}
        
        # Compute longest paths
        for node in topo_sort:
            for successor in G.successors(node):
                if longest_paths[node] + 1 > longest_paths[successor]:
                    longest_paths[successor] = longest_paths[node] + 1
                    predecessors[successor] = node
        
        # Find node with maximum path length
        end_node = max(longest_paths.items(), key=lambda x: x[1])[0]
        
        # Reconstruct the critical path
        node = end_node
        while node is not None:
            critical_path.append(node)
            node = predecessors[node]
        
        critical_path.reverse()
    
    # Create a hierarchical layout based on node_layers
    # Widen the horizontal spacing to make arrows more distinct
    pos = {}
    for node, layer in node_layers.items():
        pos[node] = (layer * 2.5, 0)  # Initial position with wider horizontal spacing
    
    # Get all nodes at each layer
    layer_nodes = {}
    for node, layer in node_layers.items():
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node)
    
    # Sort layers
    sorted_layers = sorted(layer_nodes.keys())
    
    # Position nodes in each layer
    y_spacing = 1.5  # Increased vertical spacing between nodes
    for layer in sorted_layers:
        nodes = layer_nodes[layer]
        
        # Sort nodes within layer to minimize edge crossings
        nodes.sort()
        
        # Assign y positions
        total_height = (len(nodes) - 1) * y_spacing
        for i, node in enumerate(nodes):
            y_pos = -total_height/2 + i * y_spacing
            pos[node] = (layer * 2.5, y_pos)  # Wider horizontal spacing
    
    # Set up the figure with a professional look
    fig = plt.figure(figsize=figsize)
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Define better colors for computation and communication tasks
    computation_color = "#009900"  # Green for computation tasks
    communication_color = "#0099CC"  # Blue for communication tasks
    critical_path_color = "#CC0000"  # Red for critical path
    
    # Assign colors to nodes
    node_colors = []
    for node in G.nodes():
        if node in critical_path:
            # Highlight critical path
            node_colors.append(critical_path_color)
        elif G.nodes[node]['type'] == 'Computation':
            node_colors.append(computation_color)
        else:
            node_colors.append(communication_color)
    
    # Draw curved edges with custom style to avoid overlap
    edge_list = list(G.edges())
    edge_colors = []
    edge_styles = []
    edge_widths = []
    
    for u, v in edge_list:
        if u in critical_path and v in critical_path:
            edge_colors.append(critical_path_color)
            edge_styles.append('solid')
            edge_widths.append(2.0)
        else:
            edge_colors.append('gray')
            edge_styles.append('solid')
            edge_widths.append(1.5)
    
    # FIRST draw all edges before drawing nodes
    for i, (u, v) in enumerate(edge_list):
        # Calculate curved path for the edge
        # The higher the rad, the more curved the edge
        rad = 0.3
        
        # Create more curvature for some edges to differentiate
        # Adjust curvature based on source and target nodes
        if u % 2 == 0:  # Even source nodes
            rad = 0.2
        else:  # Odd source nodes
            rad = 0.3
            
        # Draw the curved edge
        nx.draw_networkx_edges(G, 
                              pos,
                              edgelist=[(u, v)],
                              width=edge_widths[i],
                              edge_color=[edge_colors[i]],
                              style=edge_styles[i],
                              arrows=True,  # Ensure arrows are drawn
                              arrowstyle='-|>',  # Simple arrow style
                              arrowsize=30,  # Very large arrows
                              connectionstyle=f'arc3, rad={rad}',
                              node_size=800)  # Smaller nodes for arrow visibility
    
    # THEN draw nodes on top of edges, but with a slightly smaller size than used for edge calculation
    nx.draw_networkx_nodes(G, 
                          pos, 
                          node_color=node_colors,
                          node_size=1000,  # Smaller size so arrows remain visible
                          alpha=1.0,  # Fully opaque
                          edgecolors='black',  # White border for better contrast
                          linewidths=2)  # Border width
    
    # Use a slightly higher font size for better readability
    nx.draw_networkx_labels(G, 
                           pos, 
                           labels={node: G.nodes[node]['label'] for node in G.nodes()},
                           font_size=12,  # Slightly larger font
                           font_color='white',
                           font_weight='bold')
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=computation_color, markersize=10, label='Computation Task'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=communication_color, markersize=10, label='Communication Task'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=critical_path_color, markersize=10, label='Critical Path')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Set clean background and axis appearance
    plt.grid(False)
    plt.axis('off')
    
    # Add a caption with publication-style formatting
    caption = (
        f"Task dependency graph for NoC traffic. "
        "Tasks flow from left to right, with distinct curved edges showing individual dependencies. "
        "Green nodes represent computation tasks, blue nodes represent communication tasks, "
        "and the critical path is highlighted in red."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    
    # Add publication-friendly metadata
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for caption
    
    # Save the figure if an output filename is provided
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Task dependency graph saved as '{output_filename}'")
    
    # Print summary information
    print(f"Critical path identified: {' → '.join(map(str, critical_path))}")
    
    # Print all task dependencies to verify correct parsing
    print("\nTask Dependencies:")
    for task in tasks:
        if task['dependencies']:
            dep_str = ', '.join(map(str, task['dependencies']))
            print(f"Task {task['task_id']} depends on: {dep_str}")
    
    return fig 