import torch
import sys
import os
import pandas as pd
import traceback
from datetime import datetime

# Add debug print at the very start
print("Script starting...")

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print("Added parent directory to path")

try:
    from llmDeploy.models import GPT2Small
    from llmDeploy.run_utils import export_traffic_table_to_file, create_enhanced_traffic_table, generate_dependency_graph
    from llmDeploy.network_analysis import (
        analyze_network_traffic, create_pe_to_network_mapping, 
        calculate_network_bounds, calculate_pe_density,
        generate_network_layout_visualization,
        write_network_metrics_to_log,
        compare_mapping_strategies_metrics,
        rank_mapping_strategies,
        get_strategy_qualitative_assessment,
        plot_mapping_strategy_metrics,
        calculate_model_utilization
    )
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_gpt2_small(optimization_level=1, export_traffic=True, type="full", mapping_strategies=None):
    """
    Test the GPT2-small model implementation.
    This creates a GPT2-small model and runs a simple inference.
    
    Args:
        optimization_level: Level of optimization to use (0=none, 1=skip embeddings and normalizations)
        export_traffic: Whether to export traffic table to separate files
        type: Type of test being run
        mapping_strategies: List of mapping strategies to test. If None, only uses "grid_wise"
    """
    try:
        # Initialize the log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Timestamp: {timestamp}")
        
        log_dir = os.path.join(os.path.dirname(__file__), "../final_traces/gpt2_small/paper_disc")
        print(f"Log directory: {log_dir}")
        
        # Create log directory
        try:
            os.makedirs(log_dir, exist_ok=True)
            print(f"Log directory created/exists: {os.path.exists(log_dir)}")
        except Exception as e:
            print(f"Error creating log directory: {e}")
            traceback.print_exc()
        
        log_file = os.path.join(log_dir, f"gpt2_small_{type}_test_opt{optimization_level}_{timestamp}.log")
        print(f"Log file will be: {log_file}")
        
        noc_rows = 200
        noc_cols = 200
        pe_size = 64 * 1024
        
        # Set default mapping strategies if none provided
        if mapping_strategies is None:
            mapping_strategies = ["grid_wise"]
        
        # Create test config path
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/models/gpt2_small.yaml"))
        print(f"Config path: {config_path}")
        print(f"Config file exists: {os.path.exists(config_path)}")
        
        if not os.path.exists(config_path):
            print(f"ERROR: Config file does not exist: {config_path}")
            print("Checking directories...")
            config_dir = os.path.dirname(config_path)
            print(f"Config directory exists: {os.path.exists(config_dir)}")
            if os.path.exists(config_dir):
                print(f"Files in config dir: {os.listdir(config_dir)}")
                print(f"Files in models dir: {os.listdir(os.path.join(config_dir, 'models'))}")
            return None, {}
        
        # Store results for strategy comparison
        results = {}
        
        with open(log_file, "w") as f:
            print(f"Log file opened for writing: {log_file}")
            f.write(f"===== Testing GPT2-small Model (Optimization Level {optimization_level}) =====\n\n")
            f.write(f"NoC rows: {noc_rows}\n")
            f.write(f"NoC columns: {noc_cols}\n\n")
            f.write(f"Using configuration from: {config_path}\n\n")
            
            # Use a reduced sequence length for testing
            test_seq_len = 32
            
            # Create a custom config with the test sequence length
            import yaml
            try:
                with open(config_path, 'r') as config_file:
                    config = yaml.safe_load(config_file)
                print(f"Config loaded successfully")
                
                # Override sequence length in the config
                test_config_path = os.path.join(log_dir, f"gpt2_small_test_config_{timestamp}.yaml")
                config['seq_len'] = test_seq_len
                config['max_seq_len'] = test_seq_len
                config['num_layers'] = 1
                
                with open(test_config_path, 'w') as test_config_file:
                    yaml.dump(config, test_config_file)
                print(f"Test config created at: {test_config_path}")
                f.write(f"Created test config with seq_len={test_seq_len} at {test_config_path}\n\n")
            except Exception as e:
                error_msg = f"Error creating test config: {str(e)}"
                print(error_msg)
                f.write(error_msg + "\n")
                traceback.print_exc()
                f.write(traceback.format_exc() + "\n")
                return None, {}
            
            # Test each mapping strategy
            f.write(f"Testing {len(mapping_strategies)} mapping strategies: {', '.join(mapping_strategies)}\n\n")
            print(f"Testing {len(mapping_strategies)} mapping strategies: {', '.join(mapping_strategies)}")
            
            for strategy in mapping_strategies:
                f.write(f"\n===== Testing Mapping Strategy: {strategy} =====\n\n")
                print(f"Testing mapping strategy: {strategy}")
                
                try:
                    # Create a GPT2-small model with the current mapping strategy
                    f.write(f"Creating GPT2-small model with optimization level {optimization_level} and mapping strategy '{strategy}'...\n")
                    print(f"Creating GPT2-small model with optimization level {optimization_level} and mapping strategy '{strategy}'...")
                    
                    model = GPT2Small(
                        noc_rows=noc_rows,
                        noc_cols=noc_cols,
                        pe_memory_size=pe_size,
                        mapping_strategy=strategy,
                        split_strategy="hybrid_split",
                        data_type="float16",
                        config_path=test_config_path,
                        allow_wrapping=False,
                        optimization_level=optimization_level,
                        reuse_pe_for_aggregation=True,
                        row_aggregation_enabled=True,
                        column_aggregation_enabled=False
                    )
                    print(f"GPT2-small model created successfully with {strategy} strategy")
                    f.write(f"GPT2-small model created successfully with {strategy} strategy.\n\n")
                    
                    # Log model configuration
                    f.write("Model Configuration:\n")
                    f.write(f"Model name: {model.model_name}\n")
                    f.write(f"Vocab size: {model.vocab_size}\n")
                    f.write(f"Embedding dimension: {model.embed_dim}\n")
                    f.write(f"Sequence length: {model.seq_len}\n")
                    f.write(f"Number of heads: {model.num_heads}\n")
                    f.write(f"Number of layers: {model.num_layers}\n")
                    f.write(f"MLP expansion factor: {model.mlp_expansion_factor}\n")
                    f.write(f"Data type: {model.dtype}\n")
                    f.write(f"Optimization level: {model.optimization_level}\n")
                    if hasattr(model, 'positional_encoding'):
                        f.write(f"Positional encoding: {model.positional_encoding}\n\n")
                    else:
                        f.write("\n")
                    
                    # Calculate and log PE utilization
                    print("Calculating model utilization...")
                    utilization = calculate_model_utilization(model)
                    f.write(f"Model utilization: {utilization:.2f}%\n\n")
                    print(f"Model utilization: {utilization:.2f}%\n")
                    
                    # Calculate network bounds
                    all_pes = set()
                    for network_name, network in model.networks.items():
                        if hasattr(network, 'active_pes'):
                            all_pes.update(network.active_pes)
                    
                    if all_pes:
                        bounds = calculate_network_bounds(all_pes)
                        pe_count = len(all_pes)
                        effective_area = bounds["width"] * bounds["height"]
                        
                        f.write("Network bounds information:\n")
                        f.write(f"Width: {bounds['width']}, Height: {bounds['height']}\n")
                        f.write(f"X range: {bounds['x_range']}, Y range: {bounds['y_range']}\n")
                        f.write(f"PE count: {pe_count}, Effective area: {effective_area}\n\n")
                        
                    # Create a test input tensor
                    f.write("Creating test input...\n")
                    print("Creating test input...")
                    
                    if optimization_level == 0:
                        # For regular model, input is one-hot encoded token IDs
                        input_tensor = torch.randint(0, model.vocab_size, (test_seq_len,))
                        one_hot_input = torch.nn.functional.one_hot(input_tensor, num_classes=model.vocab_size).float()
                        f.write(f"Input tensor shape: {one_hot_input.shape} (one-hot encoded tokens)\n\n")
                        input_data = one_hot_input
                    else:
                        # For optimized model, input is already in embedding space
                        input_data = torch.randn(test_seq_len, model.embed_dim)
                        f.write(f"Input tensor shape: {input_data.shape} (pre-embedded vectors)\n\n")
                    
                    # Run inference
                    f.write("Running inference to gather traffic statistics...\n")
                    print(f"Running inference with {strategy} strategy...")
                    
                    outputs = model.run_inference(input_data)
                    f.write("Inference successful!\n\n")
                    print(f"Inference successful with {strategy} strategy!\n")
                    
                    # Get traffic statistics
                    traffic_table = None
                    metrics = {}
                    
                    if hasattr(model, 'llm') and hasattr(model.llm, 'noc') and hasattr(model.llm.noc, 'scheduler'):
                        traffic_table = model.llm.noc.scheduler.get_traffic_table()
                        
                        if not traffic_table.empty:
                            # Create PE to network mapping
                            pe_to_network = create_pe_to_network_mapping(model.llm)
                            
                            # Use the network analysis module to analyze traffic
                            metrics = analyze_network_traffic(traffic_table, pe_to_network)
                            
                            # Write metrics to log
                            f.write("Traffic statistics:\n")
                            write_network_metrics_to_log(metrics, f)
                            
                            # Export the traffic tables
                            if export_traffic:
                                enhanced_traffic_table = create_enhanced_traffic_table(model.llm)
                                
                                if not enhanced_traffic_table.empty:
                                    # Log the enhanced traffic table
                                    enhanced_traffic_file = os.path.join(
                                        log_dir, 
                                        f"gpt2_small_traffic_enhanced_{type}_{strategy}_opt{optimization_level}_{timestamp}.tsv"
                                    )
                                    enhanced_traffic_table.to_csv(enhanced_traffic_file, sep='\t', index=False)
                                    f.write(f"Enhanced traffic table exported to: {enhanced_traffic_file}\n")
                                    
                                    # Export basic traffic table
                                    traffic_file = os.path.join(
                                        log_dir, 
                                        f"gpt2_small_traffic_{type}_{strategy}_opt{optimization_level}_{timestamp}.tsv"
                                    )
                                    export_path = export_traffic_table_to_file(model.llm, traffic_file)
                                    f.write(f"Traffic table exported to: {export_path}\n\n")
                    
                    # Store the results for this strategy
                    results[strategy] = {
                        "utilization": utilization
                    }
                    
                    # Add PE counts and area information if available
                    if all_pes:
                        results[strategy].update({
                            "effective_area": effective_area,
                            "pe_count": pe_count
                        })
                    
                    # Add traffic metrics to results if available
                    if traffic_table is not None and not traffic_table.empty and metrics.get("metrics_available", False):
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
                    
                    # Get the final output 
                    if model.optimization_level == 0 and "final_ln" in outputs:
                        final_output = outputs["final_ln"]
                        f.write(f"Final output shape: {final_output.shape}\n")
                    else:
                        # For optimized model, get the last layer's output
                        last_layer = f"layer_{model.num_layers-1}_mlp2"
                        if last_layer in outputs:
                            final_output = outputs[last_layer]
                            f.write(f"Final output shape: {final_output.shape}\n")
                    
                    # Log PE utilization directly from llm
                    if hasattr(model, 'llm') and hasattr(model.llm, 'get_pe_utilization'):
                        pe_utilization = model.llm.get_pe_utilization()
                        f.write("\nPE Utilization:\n")
                        if isinstance(pe_utilization, dict):
                            # Handle the case where pe_utilization is a dictionary
                            f.write(f"Total PEs used: {pe_utilization['total_pes_used']}\n")
                            f.write(f"PE utilization: {pe_utilization['pe_utilization']:.2f}%\n\n")
                        else:
                            # Handle the case where pe_utilization is a float
                            f.write(f"PE utilization: {pe_utilization:.2f}%\n\n")
                    
                    # Log PE mapping details
                    if hasattr(model, 'llm') and hasattr(model.llm, 'get_pe_mapping_details'):
                        pe_mapping_df = model.llm.get_pe_mapping_details()
                        if not pe_mapping_df.empty:
                            f.write("\nPE Mapping Details:\n")
                            f.write(pe_mapping_df.to_string() + "\n\n")
                
                except Exception as e:
                    error_msg = f"Error during inference for {strategy} strategy: {e}"
                    print(error_msg)
                    f.write(f"\n{error_msg}\n")
                    traceback.print_exc()
                    f.write(traceback.format_exc() + "\n")
            
            # If multiple strategies were tested, perform comparative analysis
            if len(mapping_strategies) > 1:
                f.write("\n\n===== Comparative Analysis of Mapping Strategies =====\n\n")
                
                # Comparative analysis of all strategies
                compare_mapping_strategies_metrics(results, f)
                
                # Generate visualizations if needed
                plot_dir = os.path.join(log_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_prefix = f"gpt2_small_{type}_opt{optimization_level}"
                plot_files = plot_mapping_strategy_metrics(results, plot_dir, plot_prefix)
                
                if plot_files:
                    f.write("Generated the following visualization plots:\n")
                    for plot_type, plot_path in plot_files.items():
                        rel_path = os.path.relpath(plot_path, os.path.dirname(log_file))
                        f.write(f"- {plot_type}: {rel_path}\n")
                    f.write("\nYou can find these plots in the plots directory.\n")
                    print(f"Generated {len(plot_files)} visualization plots in {plot_dir}")
                else:
                    f.write("No visualization plots could be generated.\n")
                    print("No visualization plots could be generated.")
        
        print(f"Test completed. Log file: {log_file}")
        
        # Return the results and visualization paths
        return results, plot_files if 'plot_files' in locals() else {}
    except Exception as e:
        print(f"Unhandled exception in test_gpt2_small: {e}")
        traceback.print_exc()
        return None, {}

if __name__ == "__main__":
    try:
        # Default test with optimization level 1 and traffic export enabled
        print("Starting GPT2-small test...")
        
        # Test multiple mapping strategies
        mapping_strategies = ["row_wise", "grid_wise", "compact", "proximity"]
        
        results, plot_files = test_gpt2_small(optimization_level=1, export_traffic=True, type="full", mapping_strategies=mapping_strategies)
        
        if plot_files:
            print(f"\nGenerated {len(plot_files)} plots comparing mapping strategies:")
            for plot_type, plot_path in plot_files.items():
                print(f"- {plot_type}: {os.path.basename(plot_path)}")
    except Exception as e:
        print(f"Exception in main: {e}")
        traceback.print_exc()