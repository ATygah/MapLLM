# This file contains the code to compute the NoC resources for a given model.
# It uses the ModelResources class to compute the resources for each layer and block.
# It then uses the NoCResources class to compute the NoC resources for the entire model.
import yaml
import math
import matplotlib.pyplot as plt
'''
IN PROGRESS: Right now I am implementing a PE memory constrained NoC.

TODO:
     I need to implement a PE flops constrained NoC as well.
     I also need to implement a channel bandwidth constrained NoC as well.
     I also need to implement a NoC with In-Network Computing.
'''

class NoCResources:
    """Class to compute the NoC resources for a given model"""
    def __init__(self, noc_config_path, model_resources):
        # Pass a ModelResources object to the NoCResources object.
        self.model_resources = model_resources
        # Pass a NoC config to the NoCResources object.
        self.noc_config = None
        if noc_config_path:
            self.load_noc_config(noc_config_path)
            
    def load_noc_config(self, noc_config_path):
        """Load NoC configuration from YAML file"""
        with open(noc_config_path, 'r') as f:
            self.noc_config = yaml.safe_load(f)
        return self.noc_config

    def compute_noc_resources(self):
        # Get all the parameters from the NoC config.
        num_pes = self.noc_config['max_pes']
        memory_per_pe = self.noc_config['memory_per_pe']
        bandwidth_per_link = self.noc_config['bandwidth_per_link']
        max_flops_per_pe = self.noc_config['max_flops_per_pe']

        # Get the total number of layers in the model.
        num_layers = len(self.model_resources.layers)
        # Get the total number of blocks in the model. Each of these blocks is a ModelResources object.
        # We usually have blocks for transformer's attention and mlp.
        num_blocks = len(self.model_resources.blocks)
        # Get the total number of layer within a block.
        num_layers_per_block = len(list(self.model_resources.blocks.values())[0].layers)


        # Get the total number of parameters in the model.
        total_params = self.model_resources.get_total_params()
        # Get the total number of static memory in the model.
        total_static_memory = self.model_resources.get_total_static_memory()
        # Get the total number of flops in the model.
        total_flops = self.model_resources.get_total_flops()
        # Get the total number of activation memory in the model.
        total_activation_memory = self.model_resources.get_total_activation_memory()


        # Get the total number of PEs required for the model.
        num_pes_required_approx = total_static_memory / memory_per_pe
        num_pes_required_minimum = (total_params + total_activation_memory) / memory_per_pe

        pes_required = []

        # Extracting layer wise size to find the number of PEs required for each layer.
        for layer_info in self.model_resources.layers.values():
            # Extracting the memory, parameters, flops and activation memory for each layer/block according to it's position.
            # the key will be the layer name and the value will be the position in the model and the parameter, memory, flops and activation memory.
            layer_name = layer_info.name
            layer_position = layer_info.position
            layer_params = layer_info.params
            layer_memory = layer_info.static_memory
            layer_flops = layer_info.flops
            layer_activation_memory = layer_info.activation_memory

            pes = math.ceil(layer_memory / memory_per_pe)

            pes_required.append({ 
                "name": layer_name,
                "position": layer_position,
                "sub_name": None,
                "sub_position": None, 
                "pes": pes
                })

        # Extracting block's layer wise size to find the number of PEs required for each layer in a block.
        # Doing this computation for just the first block because the rest are just repetitions of the first block.
        for block_info in self.model_resources.blocks.values():
            block_position = block_info.position
            block_name = block_info.model_name
            for layer_info in block_info.layers.values():
                block_layer_name = layer_info.name
                block_layer_position = layer_info.position
                block_layer_params = layer_info.params
                block_layer_memory = layer_info.static_memory
                block_layer_flops = layer_info.flops
                block_layer_activation_memory = layer_info.activation_memory

                pes = math.ceil(block_layer_memory / memory_per_pe)

                pes_required.append({
                    "name": block_name,
                    "position": block_position, 
                    "sub_name": block_layer_name,
                    "sub_position": block_layer_position, 
                    "pes": pes
                    })
        sorted_pes_required = sorted(pes_required, key=lambda x: x['position'])
        for details in sorted_pes_required:
            print(f"Name: {details['name']}, Position: {details['position']}, Sub Name: {details['sub_name']}, Sub Position: {details['sub_position']}, PEs Required: {details['pes']}")
        #self.plot_pes_required(sorted_pes_required)

    #TODO: Fix this function. It doesn't plot the Block's layers correctly.
    def plot_pes_required(self, pes_required_list):
        """
        Plot the number of PEs required for each layer/block based on their position.

        Args:
            pes_required_list (list): List of dictionaries containing 'name', 'position', 'sub_position', 'pes'.
        """

        names = [item['name'] for item in pes_required_list]
        positions = [item['position'] for item in pes_required_list]
        pes = [item['pes'] for item in pes_required_list]

        # Create bar plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(names, pes, color='skyblue')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom')

        # Customize plot
        plt.title('Processing Elements Required per Layer/Block')
        plt.xlabel('Layer/Block Name')
        plt.ylabel('PEs Required')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save plot
        plot_path = f"plots/{self.model_resources.model_name}_pes_required.png"
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"PE requirements plot saved as {plot_path}")





