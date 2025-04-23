import yaml
import importlib
import os
import sys
from pathlib import Path
from layer_resources import ModelResources, LayerResources
from embed import calculate_embedding_memory, calculate_positional_embedding_costs
from attn import calculate_attention_costs
from norm_short import calculate_norm_params_costs, calculate_shortcut_costs
from fc import calculate_ff_costs, calculate_mlp_costs, calculate_activation_costs

#TODO: Add a function to re-compute the parameters, memory of weights, FLOPS and activation memory of a layer.
#      It should include an update mechanism so that we don't re-create layers each time but only update the existing ones.

class LLMResourceCalculator:
    """Main class to calculate and track resources for LLM models"""
    
    def __init__(self, base_config_path=None):
        # One singular base_config to store the base configuration.
        self.base_config = None
        # We had self.model_config to store multiple model configs.
        self.model_configs = {}
        if base_config_path:
            self.load_base_config(base_config_path)
        self.models = {}  # Store computed model resources
            
    def load_base_config(self, base_config_path):
        """Load base configuration from YAML file"""
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        return self.base_config
    
    def load_model_config(self, model_config_path):
        """Load model-specific configuration from YAML file"""
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Store model config by its name, None if nothing found.
        model_name = model_config.get('model_name')
        if not model_name:
            raise ValueError(f"Model configuration in {model_config_path} must include 'model_name'")
        
        self.model_configs[model_name] = model_config
        return model_config
    
    def load_model_configs_from_directory(self, directory_path):
        """Load all model configs from a directory"""
        directory = Path(directory_path)
        if not directory.is_dir():
            raise ValueError(f"{directory_path} is not a valid directory")
        
        # Load all YAML files in the directory
        for config_file in directory.glob("*.yaml"):
            self.load_model_config(config_file)
            
        return self.model_configs
            
    def calculate_model_resources(self, model_name, batch_size=None, seq_len=None):
        """Calculate resources for a specified model"""
        if not self.base_config:
            raise ValueError("Base configuration not loaded. Call load_base_config first.")
            
        # Ensure model config is loaded
        if model_name not in self.model_configs:
            # Try to find the model config by searching for files with this name pattern
            model_dir = Path("config/models")
            possible_files = list(model_dir.glob(f"*{model_name.lower().replace('-', '_')}*.yaml"))
            
            if not possible_files:
                raise ValueError(f"Model {model_name} configuration not found. Load it first with load_model_config().")
            
            # Load the first matching config
            self.load_model_config(possible_files[0])
            
        # Get model config
        model_config = self.model_configs[model_name]
        
        # Override batch_size and seq_len if provided
        if batch_size is not None:
            model_config['batch_size'] = batch_size
        if seq_len is not None:
            model_config['seq_len'] = seq_len
            
        # Process layers according to architecture
        # We get the architecture name for this model. Later use it to
        # extract the architecture's layers from the base_config.
        arch_name = model_config.get('architecture', 'default')
        if arch_name not in self.base_config['architectures']:
            raise ValueError(f"Architecture {arch_name} not found in base configuration")
        
        # Extract the architecture's layers from the base_config.
        architecture = self.base_config['architectures'][arch_name]
        
        # Create model resources object
        resources = ModelResources(model_name, model_config)
        
        # Process the architecture.
        # Populate..
        self._process_architecture(architecture, resources, model_config)
        
        # Store the results
        self.models[model_name] = resources
        return resources
        
    def _process_architecture(self, architecture, resources, model_config):
        """Process architecture definition and calculate layer resources"""
        position = 0

        for layer_def in architecture:
            layer_name = layer_def['name']
            layer_type = layer_def['layer_type']

            # Handle repeated blocks
            if layer_def.get('is_repeated', False):
                repeat_param = layer_def['repeat_param']
                if repeat_param not in model_config:
                    raise ValueError(f"Repeat parameter {repeat_param} not found in model config")
                    
                repeat_count = model_config[repeat_param]
                
                for i in range(repeat_count):
                    block_name = f"{layer_name}_{i}"
                    block = ModelResources(model_name=block_name, position=position)
                    
                    # Process sublayers
                    self._process_architecture(
                        layer_def['sublayers'], 
                        block, 
                        model_config
                    )
                    
                    # Add block to resources
                    resources.add_block(block_name, block)
                    position += 1
            else:
                # Calculate resources for this layer
                # Build parameters dict for function calls
                if(layer_type == 'token_embedding'):
                    flops, param_count, static_memory, activation_memory = calculate_embedding_memory(
                        vocab_size=model_config['vocab_size'],
                        batch_size=model_config['batch_size'],
                        seq_len=model_config['seq_len'],
                        embedding_dim=model_config['embed_dim'],
                        dtype=model_config['dtype']
                    )
                elif(layer_type == 'positional_embedding'):
                    # Activation memory is 0 because positional encodings are added to the input embeddings
                    flops, param_count, static_memory, activation_memory = calculate_positional_embedding_costs(
                        max_seq_len=model_config['max_seq_len'],
                        batch_size=model_config['batch_size'],
                        seq_len=model_config['seq_len'],
                        embedding_dim=model_config['embed_dim'],
                        dtype=model_config['dtype']
                    )
                elif(layer_type == 'attention'):
                    flops, param_count, static_memory, activation_memory = calculate_attention_costs(
                        attention_type=model_config['attention_type'],
                        batch_size=model_config['batch_size'],
                        q_seq_length=model_config['seq_len'],
                        kv_seq_length=model_config['kv_seq_len'],
                        d_model=model_config['embed_dim'],
                        num_heads=model_config['num_heads'],
                        dtype=model_config['dtype']
                    )
                elif(layer_type == 'mlp'):
                    flops, param_count, static_memory, activation_memory = calculate_mlp_costs(
                        embed_dim=model_config['embed_dim'],           
                        batch_size=model_config['batch_size'],
                        seq_len=model_config['seq_len'],
                        expansion_factor=model_config['mlp_expansion_factor'],
                        activation_type=model_config['mlp_activation'],
                        dtype=model_config['dtype']
                    )
                elif(layer_type == 'layernorm'):
                    # TODO: Fix this part by taking into account the conv part
                    flops, param_count, static_memory, activation_memory = calculate_norm_params_costs(
                        norm_type=model_config['norm_type'],
                        batch_size=model_config['batch_size'],
                        seq_len=model_config['seq_len'],
                        input_dim=model_config['embed_dim'],
                        dtype=model_config['dtype']
                    )
                elif(layer_type == 'output_head'):
                    # Check if token_embedding exists in the resources
                    has_token_embedding = any(
                        layer.layer_type == 'token_embedding' 
                        for layer in resources.layers.values()
                    )
                    
                    if not has_token_embedding:
                        # TODO: It will reuse the input but what will be the activation memory? and FLOPS?
                        flops, param_count, static_memory, activation_memory = calculate_ff_costs(
                            vocab_size=model_config['vocab_size'],
                            batch_size=model_config['batch_size'],
                            seq_len=model_config['seq_len'],
                            embed_dim=model_config['embed_dim'],
                            dtype=model_config['dtype']
                        )
                elif(layer_type == 'conv_projection'):
                    pass
                elif(layer_type == 'shortcut'):
                    flops, param_count, static_memory, activation_memory = calculate_shortcut_costs(
                        batch_size=model_config['batch_size'],
                        seq_len=model_config['seq_len'],
                        embed_dim=model_config['embed_dim'],
                        dtype=model_config['dtype']
                    )

                # Add layer to resources
                resources.add_layer(
                    layer_name, 
                    layer_type, 
                    param_count, 
                    static_memory, 
                    flops, 
                    activation_memory,
                    position
                )
                position += 1
    
    def plot_model_resources(self, model_name, batch_size=None, seq_len=None, unit='M'):
        """Plot resources for a specified model"""

            
        #resources = self.models[model_name]
        
    #TODO: This function is broken. Fix it.
    def compare_models(self, model_names, batch_size=None, seq_len=None, unit='M'):
        """Compare resources across multiple models"""
        results = []
        
        for model_name in model_names:
            if model_name not in self.models:
                self.calculate_model_resources(model_name, batch_size, seq_len)
            results.append(self.models[model_name])
            
        # Determine scaling based on unit
        if unit == 'K':
            param_div, mem_div, flop_div = 1e3, 1024, 1e3
            param_unit, mem_unit, flop_unit = 'K', 'KB', 'KFLOPs'
        elif unit == 'M':
            param_div, mem_div, flop_div = 1e6, 1024**2, 1e6
            param_unit, mem_unit, flop_unit = 'M', 'MB', 'MFLOPs'
        elif unit == 'G':
            param_div, mem_div, flop_div = 1e9, 1024**3, 1e9
            param_unit, mem_unit, flop_unit = 'B', 'GB', 'GFLOPs'
        elif unit == 'T':
            param_div, mem_div, flop_div = 1e12, 1024**4, 1e12
            param_unit, mem_unit, flop_unit = 'T', 'TB', 'TFLOPs'
        else:
            raise ValueError(f"Invalid unit: {unit}")
            
        # Print comparison
        print(f"===== Model Comparison ({len(model_names)} models) =====")
        print(f"{'Model':<15} {'Params':<10} {'Static Memory':<15} {'FLOPs':<15} {'Activation Memory':<15}")
        print("-" * 70)
        
        for model in results:
            total_params = model.get_total_params() / param_div
            total_static_memory = model.get_total_static_memory() / mem_div
            total_flops = model.get_total_flops() / flop_div
            total_activation_memory = model.get_total_activation_memory() / mem_div
            
            print(f"{model.model_name:<15} {total_params:<10.2f} {total_static_memory:<15.2f} {total_flops:<15.2f} {total_activation_memory:<15.2f}")
            
        return results
        
    #TODO: This function is never run because I don't have GPUs. Fix it.
    def verify_with_real_model(self, model_name, real_model, real_input, batch_size=None, seq_len=None):
        """Verify calculated resources against a real PyTorch model"""
        import torch
        
        # Calculate theoretical resources
        if model_name not in self.models:
            self.calculate_model_resources(model_name, batch_size, seq_len)
            
        resources = self.models[model_name]
        
        # Count parameters in real model
        real_params = sum(p.numel() for p in real_model.parameters())
        
        # Try to estimate memory usage
        with torch.no_grad():
            # Warmup pass
            _ = real_model(real_input)
            
            # Memory before forward pass
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            
            # Forward pass
            output = real_model(real_input)
            
            # Memory after forward pass
            peak_mem = torch.cuda.max_memory_allocated()
            
        print(f"===== Resource Verification for {model_name} =====")
        print(f"Parameters:")
        print(f"  - Calculated: {resources.get_total_params():,}")
        print(f"  - Actual:     {real_params:,}")
        print(f"  - Difference: {abs(resources.get_total_params() - real_params):,} ({abs(resources.get_total_params() - real_params) / real_params * 100:.2f}%)")
        
        print(f"\nMemory Usage:")
        print(f"  - Calculated Static: {resources.get_total_static_memory() / (1024**2):.2f} MB")
        print(f"  - Calculated Activation: {resources.get_total_activation_memory() / (1024**2):.2f} MB")
        print(f"  - Total Calculated: {(resources.get_total_static_memory() + resources.get_total_activation_memory()) / (1024**2):.2f} MB")
        print(f"  - Actual Peak: {peak_mem / (1024**2):.2f} MB")
        
        return {
            "model_name": model_name,
            "calculated_params": resources.get_total_params(),
            "actual_params": real_params,
            "calculated_static_memory": resources.get_total_static_memory(),
            "calculated_activation_memory": resources.get_total_activation_memory(), 
            "actual_peak_memory": peak_mem
        }
