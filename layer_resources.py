import matplotlib.pyplot as plt
import numpy as np

class LayerResources:
    """Class to track computational resources for a single layer"""
    
    def __init__(self, name, layer_type, position=None):
        self.name = name
        self.layer_type = layer_type
        self.params = 0
        self.static_memory = 0  # Memory for weights in bytes
        self.flops = 0          # FLOPs for forward pass
        self.activation_memory = 0  # Memory for activations in bytes
        self.forward_time = 0   # Optional: measured time
        self.position = position

    def update(self, params=None, static_memory=None, flops=None, activation_memory=None, forward_time=None):
        """Update resource metrics"""
        if params is not None:
            self.params = params
        if static_memory is not None:
            self.static_memory = static_memory
        if flops is not None:
            self.flops = flops
        if activation_memory is not None:
            self.activation_memory = activation_memory
        if forward_time is not None:
            self.forward_time = forward_time
            
    def __str__(self):
        """String representation with formatted metrics"""
        return (f"Layer: {self.name} ({self.layer_type})\n"
                f"  Parameters: {self.params:,}\n"
                f"  Static Memory: {self.static_memory/1024**2:.2f} MB\n"
                f"  FLOPs: {self.flops/1e6:.2f} MFLOPs\n"
                f"  Activation Memory: {self.activation_memory/1024**2:.2f} MB")

class ModelResources:
    """Class to track computational resources for an entire model"""
    
    def __init__(self, model_name, model_config=None, position=None):
        self.model_name = model_name
        self.model_config = model_config or {}
        self.position = position
        self.layers = {}  # Dictionary of LayerResources objects
        self.blocks = {}  # Dictionary of ModelResources objects for repeated blocks
        
    def add_layer(self, name, layer_type, params=0, static_memory=0, flops=0, activation_memory=0, position=None):
        """Add a layer and its resource metrics"""
        layer = LayerResources(name, layer_type, position)
        layer.update(params, static_memory, flops, activation_memory)
        self.layers[name] = layer
        return layer
        
    def add_block(self, block_name, block_resources):
        """Add a reusable block (like a transformer block)"""
        if isinstance(block_resources, ModelResources):
            self.blocks[block_name] = block_resources
        else:
            raise TypeError("block_resources must be a ModelResources instance")
            
    def get_total_params(self):
        """Get total parameter count across all layers and blocks"""
        layer_params = sum(layer.params for layer in self.layers.values())
        block_params = sum(block.get_total_params() for block in self.blocks.values())
        return layer_params + block_params
        
    def get_total_static_memory(self):
        """Get total memory usage for weights"""
        layer_memory = sum(layer.static_memory for layer in self.layers.values())
        block_memory = sum(block.get_total_static_memory() for block in self.blocks.values())
        return layer_memory + block_memory
        
    def get_total_flops(self):
        """Get total FLOPs for a forward pass"""
        layer_flops = sum(layer.flops for layer in self.layers.values())
        block_flops = sum(block.get_total_flops() for block in self.blocks.values())
        return layer_flops + block_flops
        
    def get_total_activation_memory(self):
        """Get total memory usage for activations"""
        layer_memory = sum(layer.activation_memory for layer in self.layers.values())
        block_memory = sum(block.get_total_activation_memory() for block in self.blocks.values())
        return layer_memory + block_memory
        
    def get_flops_breakdown(self):
        """Get breakdown of FLOPs by layer type"""
        breakdown = {}
        
        # Aggregate from layers
        for layer in self.layers.values():
            if layer.layer_type not in breakdown:
                breakdown[layer.layer_type] = 0
            breakdown[layer.layer_type] += layer.flops
            
        # Aggregate from blocks
        for block in self.blocks.values():
            block_breakdown = block.get_flops_breakdown()
            for layer_type, flops in block_breakdown.items():
                if layer_type not in breakdown:
                    breakdown[layer_type] = 0
                breakdown[layer_type] += flops
                
        return breakdown
        
    def get_memory_breakdown(self):
        """Get breakdown of memory usage by layer type"""
        breakdown = {}
        
        # Aggregate from layers
        for layer in self.layers.values():
            if layer.layer_type not in breakdown:
                breakdown[layer.layer_type] = 0
            breakdown[layer.layer_type] += layer.static_memory
            
        # Aggregate from blocks
        for block in self.blocks.values():
            block_breakdown = block.get_memory_breakdown()
            for layer_type, memory in block_breakdown.items():
                if layer_type not in breakdown:
                    breakdown[layer_type] = 0
                breakdown[layer_type] += memory
                
        return breakdown
        
    def get_params_breakdown(self):
        """Get breakdown of parameters by layer type.
        
        Returns:
            Dict mapping layer types to their total parameter count.
        """
        breakdown = {}
        
        # Aggregate from layers
        for layer in self.layers.values():
            if layer.layer_type not in breakdown:
                breakdown[layer.layer_type] = 0
            breakdown[layer.layer_type] += layer.params
        
        # Aggregate from blocks
        for block in self.blocks.values():
            block_breakdown = block.get_params_breakdown()
            for layer_type, params in block_breakdown.items():
                if layer_type not in breakdown:
                    breakdown[layer_type] = 0
                breakdown[layer_type] += params
                
        return breakdown
        
    def summary(self, unit='M'):
        """Print a summary of resource usage"""
        # Determine divisors based on unit
        if unit == 'K':
            param_div, mem_div, flop_div = 1e3, 1024, 1e3
            param_unit, mem_unit, flop_unit = 'K', 'KB', 'KFLOPs'
        elif unit == 'M':
            param_div, mem_div, flop_div = 1e6, 1024**2, 1e6
            param_unit, mem_unit, flop_unit = 'M', 'MB', 'MFLOPs'
        elif unit == 'G':
            param_div, mem_div, flop_div = 1e9, 1024**3, 1e9
            param_unit, mem_unit, flop_unit = 'B', 'GB', 'GFLOPs'
        else:
            raise ValueError(f"Invalid unit: {unit}")
            
        total_params = self.get_total_params()
        total_static_memory = self.get_total_static_memory()
        total_flops = self.get_total_flops()
        total_activation_memory = self.get_total_activation_memory()
        
        print(f"===== {self.model_name} Model Summary =====")
        print(f"Total Parameters: {total_params/param_div:.2f} {param_unit}")
        print(f"Total Static Memory: {total_static_memory/mem_div:.2f} {mem_unit}")
        print(f"Total FLOPs: {total_flops/flop_div:.2f} {flop_unit}")
        print(f"Total Activation Memory: {total_activation_memory/mem_div:.2f} {mem_unit}")
        print("\n--- Breakdown by Layer Type ---")
        print("\n--- Parameters Breakdown by Layer Type ---")
        params_breakdown = self.get_params_breakdown()
        for layer_type, params in sorted(params_breakdown.items(), key=lambda x: x[1], reverse=True):
            print(f"{layer_type}: {params/param_div:.2f} {param_unit} ({params/total_params*100:.1f}%)")

        # FLOPs breakdown
        print("\n--- FLOPs Breakdown by Layer Type ---")
        flops_breakdown = self.get_flops_breakdown()
        for layer_type, flops in sorted(flops_breakdown.items(), key=lambda x: x[1], reverse=True):
            print(f"{layer_type}: {flops/flop_div:.2f} {flop_unit} ({flops/total_flops*100:.1f}%)")
            
        # Static Memory breakdown
        print("\n--- Static Memory Breakdown by Layer Type ---")
        memory_breakdown = self.get_memory_breakdown()
        for layer_type, memory in sorted(memory_breakdown.items(), key=lambda x: x[1], reverse=True):
            print(f"{layer_type}: {memory/mem_div:.2f} {mem_unit} ({memory/total_static_memory*100:.1f}%)")
    
    def to_dict(self):
        """Convert resource tracking to a dictionary (for JSON serialization)"""
        result = {
            "model_name": self.model_name,
            "model_config": self.model_config,
            "position": self.position,
            "total_params": self.get_total_params(),
            "total_static_memory": self.get_total_static_memory(),
            "total_flops": self.get_total_flops(),
            "total_activation_memory": self.get_total_activation_memory(),
            "layers": {},
            "blocks": {}
        }
        
        # Add layers
        for name, layer in self.layers.items():
            result["layers"][name] = {
                "type": layer.layer_type,
                "params": layer.params,
                "static_memory": layer.static_memory,
                "flops": layer.flops,
                "activation_memory": layer.activation_memory,
                "position": layer.position
            }
            
        # Add blocks
        for name, block in self.blocks.items():
            result["blocks"][name] = block.to_dict()
            
        return result

    def print_resources(self, indent_level=0):
        """
        Print the model resources in a human-readable and hierarchical format.
        This method prints the contents of the dictionary produced by to_dict().
        
        Args:
            indent_level (int): Indentation level for printing (used for recursion).
        """
        self._print_dict(self.to_dict(), indent_level)
        
    def _print_dict(self, resources_dict, indent_level=0):
        """
        Recursively prints a resource dictionary with indentation.
        Layers and blocks are printed together in order of their position.
        
        Args:
            resources_dict (dict): The dictionary (from to_dict()) to print.
            indent_level (int): Current indentation level.
        """
        indent = " " * indent_level
        # Print general model information
        print(indent + f"Model Name: {resources_dict.get('model_name', 'N/A')}")
        print(indent + "Model Config:")
        model_config = resources_dict.get("model_config", {})
        if model_config:
            for key, value in model_config.items():
                print(indent + f"  {key}: {value}")
        else:
            print(indent + "  None")
        print(indent + f"Total Params: {resources_dict.get('total_params', 0)}")
        print(indent + f"Total Static Memory: {resources_dict.get('total_static_memory', 0)}")
        print(indent + f"Total FLOPs: {resources_dict.get('total_flops', 0)}")
        print(indent + f"Total Activation Memory: {resources_dict.get('total_activation_memory', 0)}")
        
        # Combine layers and blocks into a single list
        components = []
        
        # Add layers with their positions
        layers = resources_dict.get("layers", {})
        for name, info in layers.items():
            components.append({
                'name': name,
                'type': 'layer',
                'position': info.get('position', float('inf')),
                'info': info
            })
        
        # Add blocks with their positions
        blocks = resources_dict.get("blocks", {})
        for name, info in blocks.items():
            components.append({
                'name': name,
                'type': 'block',
                'position': info.get('position', float('inf')),
                'info': info
            })
        
        # Sort all components by position
        sorted_components = sorted(components, key=lambda x: x['position'])
        
        # Print components in order
        print(indent + "Model Components (in position order):")
        for component in sorted_components:
            name = component['name']
            position = component['position']
            print(indent + f"  {name} (position {position}):")
            
            if component['type'] == 'layer':
                # Print layer metrics
                for metric, value in sorted(component['info'].items()):
                    if metric != 'position':  # Skip position as we already displayed it
                        print(indent + f"    {metric}: {value}")
            else:  # block
                # Recursively print block details
                self._print_dict(component['info'], indent_level=indent_level+4)

    def plot_resources(self):
        """
        Plot model resources (FLOPs, parameters, memory, activations) by layer type.
        Aggregates metrics for similar layer types and treats blocks as single units.
        """
        # Get resource dictionary
        resources_dict = self.to_dict()
        
        # Initialize dictionaries to store metrics by layer type
        metrics = {
            'Parameters': {},
            'Static Memory': {},
            'FLOPs': {},
            'Activation Memory': {}
        }
        
        # Track positions for each layer type
        layer_positions = {}
        
        # Process layers
        for name, info in resources_dict['layers'].items():
            layer_type = info['type']
            position = info.get('position', float('inf'))
            
            # Initialize if layer type not seen before
            for metric_dict in metrics.values():
                if layer_type not in metric_dict:
                    metric_dict[layer_type] = 0
            
            # Aggregate metrics
            metrics['Parameters'][layer_type] += info['params']
            metrics['Static Memory'][layer_type] += info['static_memory']
            metrics['FLOPs'][layer_type] += info['flops']
            metrics['Activation Memory'][layer_type] += info['activation_memory']
            
            # Track earliest position for this layer type
            if layer_type not in layer_positions:
                layer_positions[layer_type] = position
            else:
                layer_positions[layer_type] = min(layer_positions[layer_type], position)
        
        # Process blocks (as single units)
        for name, block_info in resources_dict['blocks'].items():
            block_type = 'attention_block'  # or extract from block_info if available
            position = block_info.get('position', float('inf'))
            
            # Initialize if block type not seen before
            for metric_dict in metrics.values():
                if block_type not in metric_dict:
                    metric_dict[block_type] = 0
            
            # Aggregate block totals
            metrics['Parameters'][block_type] += block_info['total_params']
            metrics['Static Memory'][block_type] += block_info['total_static_memory']
            metrics['FLOPs'][block_type] += block_info['total_flops']
            metrics['Activation Memory'][block_type] += block_info['total_activation_memory']
            
            # Track position
            if block_type not in layer_positions:
                layer_positions[block_type] = position
            else:
                layer_positions[block_type] = min(layer_positions[block_type], position)
        
        # Sort layer types by their positions
        layer_types = sorted(layer_positions.keys(), key=lambda x: layer_positions[x])
        
        # Create the plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Resource Distribution by Layer Type for {resources_dict["model_name"]}', 
                     fontsize=16, y=0.95)
        
        # Color map for different layer types
        colors = plt.cm.Set3(np.linspace(0, 1, len(layer_types)))
        
        # Plot settings
        plot_settings = {
            'Parameters': {'ax': axs[0, 0], 'unit': 'M', 'divisor': 1e6},
            'Static Memory': {'ax': axs[0, 1], 'unit': 'MB', 'divisor': 1024**2},
            'FLOPs': {'ax': axs[1, 0], 'unit': 'G', 'divisor': 1e9},
            'Activation Memory': {'ax': axs[1, 1], 'unit': 'MB', 'divisor': 1024**2}
        }
        
        # Create each subplot
        for metric_name, settings in plot_settings.items():
            ax = settings['ax']
            unit = settings['unit']
            divisor = settings['divisor']
            
            # Prepare data
            values = [metrics[metric_name][lt] / divisor for lt in layer_types]
            
            # Create bar plot
            bars = ax.bar(range(len(layer_types)), values, color=colors)
            
            # Customize plot
            ax.set_title(f'{metric_name} by Layer Type')
            ax.set_xlabel('Layer Type')
            ax.set_ylabel(f'{metric_name} ({unit})')
            ax.set_xticks(range(len(layer_types)))
            ax.set_xticklabels(layer_types, rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
            
            # Add grid
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_path = f"plots/{resources_dict['model_name']}_resource_distribution.png"
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        
        print(f"Resource distribution plot saved as {plot_path}")
        
        return fig, axs
