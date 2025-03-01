# This module handles all the linear projection and feed forward calculations

from data_structs import dtype_size

def calculate_ff_flops(batch_size: int, seq_len: int, embed_dim: int, vocab_size: int) -> int:
    """
    Calculate FLOPs for the final feed-forward layer (output projection to vocab).
    
    Parameters:
        batch_size (int): Number of samples in the batch.
        seq_len (int): Sequence length.
        embed_dim (int): Input embedding dimension.
        vocab_size (int): Output vocabulary size.
    
    Returns:
        int: Total FLOPs (floating-point operations)
    """
    # FLOPs = 2 * batch_size * seq_len * embed_dim * vocab_size
    return 2 * batch_size * seq_len * embed_dim * vocab_size

def calculate_ff_static_memory(embed_dim: int, vocab_size: int, dtype: str = 'float32') -> int:
    """
    Calculate static memory for the feed-forward layer weights.
    
    Parameters:
        embed_dim (int): Input embedding dimension.
        vocab_size (int): Output vocabulary size.
        dtype (str): Data type of the weights (default: 'float32').
    
    Returns:
        int: Memory usage in bytes
    """
    # Weight matrix shape: (embed_dim, vocab_size)
    num_parameters = embed_dim * vocab_size
    return num_parameters * dtype_size(dtype)

def calculate_ff_activation_memory(batch_size: int, seq_len: int, vocab_size: int, dtype: str = 'float32') -> int:
    """
    Calculate dynamic memory for the feed-forward layer activations.
    
    Parameters:
        batch_size (int): Number of samples in the batch.
        seq_len (int): Sequence length.
        vocab_size (int): Output vocabulary size.
        dtype (str): Data type of the activations (default: 'float32').
    
    Returns:
        int: Memory usage in bytes
    """
    # Activation shape: (batch_size, seq_len, vocab_size)
    num_elements = batch_size * seq_len * vocab_size
    return num_elements * dtype_size(dtype)

def calculate_ff_costs(batch_size: int, seq_len: int, embed_dim: int, vocab_size: int, dtype: str = 'float32') -> tuple:
    """
    Calculate all costs for the final feed-forward layer.
    
    Returns:
        tuple: (flops, static_memory_bytes, activation_memory_bytes)
    """
    flops = calculate_ff_flops(batch_size, seq_len, embed_dim, vocab_size)
    static_memory = calculate_ff_static_memory(embed_dim, vocab_size, dtype)
    activation_memory = calculate_ff_activation_memory(batch_size, seq_len, vocab_size, dtype)
    return flops, static_memory, activation_memory

# Example usage
# if __name__ == "__main__":
#     # Example for GPT-2
#     batch_size = 8
#     seq_len = 1024
#     embed_dim = 768
#     vocab_size = 50257
#     
#     flops, static_mem, activation_mem = calculate_ff_costs(batch_size, seq_len, embed_dim, vocab_size)
#     
#     print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
#     print(f"Static memory: {static_mem / (1024**2):.2f} MB")
#     print(f"Activation memory: {activation_mem / (1024**2):.2f} MB"

