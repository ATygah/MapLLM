"""
Example usage of the LLM Resource Calculator with split configuration files
"""

from calResources import LLMResourceCalculator
from nocResources import NoCResources

def main():
    # Initialize calculator with base config
    calculator = LLMResourceCalculator('config/base_config.yaml')
    
    # Method 1: Load all model configurations from directory
    calculator.load_model_configs_from_directory('config/models')
    
    # Method 2: Load specific model configuration
    # calculator.load_model_config('config/models/gpt2_small.yaml')
    
    # Calculate resources for a specific model
    print("\n=== GPT-2 Small Resources ===")
    gpt2_small = calculator.calculate_model_resources('gpt2-small')
    #gpt2_small.summary()
    #gpt2_small.plot_resources()
    noc_resources = NoCResources('config/noc_config.yaml', gpt2_small)
    noc_resources.compute_noc_resources()

    #print("\n=== GPT-3 Resources ===")
    #gpt3 = calculator.calculate_model_resources('gpt3')
    #gpt3.summary()
    # print("\n=== Phi-2 Resources ===")
    # phi2 = calculator.calculate_model_resources('phi-2')
    # phi2.summary()
    # print("\n=== Phi-3-Mini Resources ===")
    # phi3_mini = calculator.calculate_model_resources('phi-3-mini')
    # phi3_mini.summary()
    # Calculate resources for another model
    '''
    print("\n=== ViT-B/16 Resources ===")
    vit = calculator.calculate_model_resources('vit-b16')
    vit.summary()
    # Compare multiple models
    print("\n=== Model Comparison ===")
    calculator.compare_models(['gpt2-small', 'gpt2-medium', 'gpt2-large'], unit='G')
    # Calculate with custom batch size and sequence length
    print("\n=== Custom Batch Size and Sequence Length ===")
    custom_gpt2 = calculator.calculate_model_resources('gpt2-medium', batch_size=16, seq_len=512)
    custom_gpt2.summary()
    '''
    print("\nSuccessful execution with updated configuration structure!")
    
if __name__ == "__main__":
    main() 