import torch
import sys
import os
import pandas as pd
import traceback
from datetime import datetime

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llmDeploy.models import GPT2XL
from llmDeploy.run_utils import export_traffic_table_to_file, create_enhanced_traffic_table, generate_dependency_graph

def test_gpt2_xl(optimization_level=1, export_traffic=True, type="full"):
    """
    Test the GPT2-XL model implementation.
    This creates a GPT2-XL model and runs a simple inference.
    
    Args:
        optimization_level: Level of optimization to use (0=none, 1=skip embeddings and normalizations)
        export_traffic: Whether to export traffic table to separate files
    """
    # Initialize the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "../final_traces")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"gpt2_xl_{type}_test_opt{optimization_level}_00.log")
    
    with open(log_file, "w") as f:
        f.write(f"===== Testing GPT2-XL Model (Optimization Level {optimization_level}) =====\n\n")
        
        # Create test config path
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/models/gpt2_xl.yaml"))
        f.write(f"Using configuration from: {config_path}\n\n")
        
        # Create a GPT2-XL model
        f.write(f"Creating GPT2-XL model with optimization level {optimization_level}...\n")
        
        # Use a reduced sequence length for testing
        test_seq_len = 1
        
        # Create a custom config with the test sequence length
        import yaml
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        
        # Override sequence length in the config
        test_config_path = os.path.join(log_dir, f"gpt2_xl_test_config_{timestamp}.yaml")
        config['seq_len'] = test_seq_len
        config['max_seq_len'] = test_seq_len
        
        with open(test_config_path, 'w') as test_config_file:
            yaml.dump(config, test_config_file)
            
        f.write(f"Created test config with seq_len={test_seq_len} at {test_config_path}\n")
        
        model = GPT2XL(
            noc_rows=400,
            noc_cols=400,
            pe_memory_size=64 * 1024,
            mapping_strategy="grid_wise",
            split_strategy="hybrid_split",
            data_type="float16",
            config_path=test_config_path,
            allow_wrapping=False,
            optimization_level=optimization_level,
            reuse_pe_for_aggregation=True,
            row_aggregation_enabled=True,
            column_aggregation_enabled=True
        )
        
        f.write("GPT2-XL model created successfully.\n\n")
        
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
        f.write(f"Optimization level: {model.optimization_level}\n\n")
        
        # Create a test input tensor
        f.write("Creating test input...\n")
        
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
        f.write("Running inference...\n")
        try:
            outputs = model.run_inference(input_data)
            f.write("Inference successful!\n\n")
            
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
            else:
                f.write("PE utilization information not available.\n\n")
            
            # Log PE mapping details
            if hasattr(model, 'llm') and hasattr(model.llm, 'get_pe_mapping_details'):
                pe_mapping_df = model.llm.get_pe_mapping_details()
                if not pe_mapping_df.empty:
                    f.write("\nPE Mapping Details:\n")
                    f.write(pe_mapping_df.to_string() + "\n\n")
                else:
                    f.write("PE mapping details not available (empty dataframe).\n\n")
            else:
                f.write("PE mapping details not available.\n\n")
            
            # Log traffic table
            if hasattr(model, 'llm') and hasattr(model.llm, 'noc') and hasattr(model.llm.noc, 'scheduler'):
                # Use the create_enhanced_traffic_table function to get enhanced traffic table
                enhanced_traffic_table = create_enhanced_traffic_table(model.llm)
                
                if not enhanced_traffic_table.empty:
                    # Log the enhanced traffic table
                    enhanced_traffic_file = os.path.join(log_dir, f"gpt2_xl_traffic_enhanced_{type}_test_opt{optimization_level}_{timestamp}.tsv")
                    enhanced_traffic_table.to_csv(enhanced_traffic_file, sep='\t', index=False)
                    f.write(f"Enhanced traffic table exported to: {enhanced_traffic_file}\n")
                    
                    # Optionally export the traffic tables to files
                    if export_traffic:
                        try:
                            # Export basic traffic table using the export_traffic_table_to_file function
                            traffic_file = os.path.join(log_dir, f"gpt2_xl_traffic_{type}_test_opt{optimization_level}_{timestamp}.tsv")
                            export_path = export_traffic_table_to_file(model.llm, traffic_file)
                            f.write(f"Traffic table exported to: {export_path}\n")
                            
                            # Export enhanced traffic table
                            
                            
                            # Generate and save dependency graph
                            # graph_file = os.path.join(log_dir, f"gpt2_xl_dependency_graph_opt{optimization_level}_{timestamp}.png")
                            # title = f"GPT2-XL Task Dependency Graph (Opt Level {optimization_level})"
                            # _ = generate_dependency_graph(
                            #     enhanced_traffic_table, 
                            #     output_filename=graph_file,
                            #     title=title
                            # )
                            #f.write(f"Task dependency graph generated and saved to: {graph_file}\n")
                        except Exception as e:
                            f.write(f"Error exporting traffic data or generating graph: {str(e)}\n")
                            f.write(f"Traceback: {traceback.format_exc()}\n")
                else:
                    f.write("Traffic table not available (empty or None).\n\n")
            else:
                missing_components = []
                if not hasattr(model, 'llm'):
                    missing_components.append("llm")
                elif not hasattr(model.llm, 'noc'):
                    missing_components.append("noc")
                elif not hasattr(model.llm.noc, 'scheduler'):
                    missing_components.append("scheduler")
                f.write(f"Traffic table not available (missing components: {', '.join(missing_components)}).\n\n")
                    
        except Exception as e:
            f.write(f"Error during inference: {str(e)}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")
    
    print(f"Test completed. Log file: {log_file}")

if __name__ == "__main__":
    # Default test with optimization level 1 and traffic export enabled
    test_gpt2_xl(optimization_level=1, export_traffic=True, type="full") 