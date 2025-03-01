# Use this file for embedding the input tokens and their positional encodings
# This is the first step of the model

from data_structs import dtype_size
import torch

def calculate_embedding_flops(batch_size: int, seq_len: int, embedding_dim: int) -> int:
    """
    Calculate the FLOPs for the combination of token embeddings and positional embeddings.
    
    Parameters:
        batch_size (int): Number of samples in the batch.
        seq_len (int): Length of the sequence.
        embedding_dim (int): Dimension of the embedding vectors.
    
    Returns:
        int: Total FLOPs for the embedding combination
    """
    # FLOPs come from the element-wise addition of token and positional embeddings
    # Each addition operation counts as 1 FLOP
    return batch_size * seq_len * embedding_dim

def calculate_embedding_memory(vocab_size: int, 
                              embedding_dim: int, 
                              dtype: str = 'float32') -> int:
    """
    Calculate the memory footprint of an embedding table.
    
    Parameters:
        vocab_size (int): Number of tokens in the vocabulary
        embedding_dim (int): Dimension of each embedding vector
        dtype (str): Data type of the embeddings (default: 'float32')
                    Supported types: float32, fp32, float16, fp16, 
                    bfloat16, bf16, int8, uint8
    
    Returns:
        int: Memory usage in bytes
    """
    # Get byte size using the dtype_size function
    bytes_per_element = dtype_size(dtype)
    
    # Calculate total parameters and memory
    total_parameters = vocab_size * embedding_dim
    return total_parameters * bytes_per_element

def calculate_intermediate_size(output_tensor, dtype: str = None):
    """
    Calculate the size of the intermediate output tensor in bytes.
    
    Parameters:
        output_tensor: The intermediate output tensor (PyTorch tensor)
        dtype (str): Optional data type of the tensor. If None, inferred from the tensor.
    
    Returns:
        int: Size of the tensor in bytes
    """
    # Check tensor type
    if isinstance(output_tensor, torch.Tensor):
        if dtype is None:
            dtype = str(output_tensor.dtype).split('.')[-1]  # Extract dtype (e.g., 'float32')
    else:
        raise ValueError("Unsupported tensor type. Must be a PyTorch tensor.")
    
    # Get byte size using the dtype_size function
    bytes_per_element = dtype_size(dtype)
    
    # Calculate total size
    num_elements = output_tensor.numel()
    return num_elements * bytes_per_element

def calculate_intermediate_size_from_spec(batch_size=None, seq_len=None, embedding_dim=None, 
                                          dtype='float32', input_dimensions=None):
    """
    Calculate the size of intermediate results based on input specifications.
    
    Parameters:
        batch_size (int): Number of samples in the batch.
        seq_len (int): Length of the sequence.
        embedding_dim (int): Dimension of the embedding vectors.
        dtype (str): Data type of the intermediate results (default: 'float32').
        input_dimensions: Optional input to parse dimensions from a tuple or dictionary.
                         If provided, overrides batch_size, seq_len, and embedding_dim.
    
    Returns:
        int: Size of the intermediate results in bytes
    """
    # Parse input dimensions if provided
    if input_dimensions is not None:
        if isinstance(input_dimensions, tuple):
            if len(input_dimensions) != 3:
                raise ValueError("Input dimensions tuple must have 3 elements: (batch_size, seq_len, embedding_dim)")
            batch_size, seq_len, embedding_dim = input_dimensions
        elif isinstance(input_dimensions, dict):
            batch_size = input_dimensions.get('batch_size')
            seq_len = input_dimensions.get('seq_len')
            embedding_dim = input_dimensions.get('embedding_dim')
        else:
            raise ValueError("Input dimensions must be a tuple or dictionary.")
    
    # Validate that all dimensions are provided
    if batch_size is None or seq_len is None or embedding_dim is None:
        raise ValueError("Missing dimensions. Provide batch_size, seq_len, and embedding_dim.")
    
    # Get byte size using the dtype_size function
    bytes_per_element = dtype_size(dtype)
    
    # Calculate total size
    num_elements = batch_size * seq_len * embedding_dim
    return num_elements * bytes_per_element

def calculate_embedding_costs(vocab_size: int, embedding_dim: int, max_seq_len: int,
                             batch_size: int, seq_len: int, dtype: str = 'float32'):
    """
    Calculate the FLOPs, static memory footprint, and activation memory for the embedding layer.
    
    Parameters:
        vocab_size (int): Number of tokens in the vocabulary.
        embedding_dim (int): Dimension of the embedding vectors.
        max_seq_len (int): Maximum sequence length (for positional embeddings).
        batch_size (int): Number of samples in the batch.
        seq_len (int): Length of the sequence.
        dtype (str): Data type of the embeddings (default: 'float32').
    
    Returns:
        tuple: (flops, static_memory_bytes, activation_memory_bytes)
    """
    # Get byte size using the dtype_size function
    bytes_per_element = dtype_size(dtype)
    
    # 1. Calculate FLOPs for embedding combination
    # FLOPs come from the element-wise addition of token and positional embeddings
    flops = batch_size * seq_len * embedding_dim
    
    # 2. Calculate static memory footprint
    # Token embeddings: vocab_size * embedding_dim
    # Positional embeddings: max_seq_len * embedding_dim
    static_memory_bytes = (vocab_size * embedding_dim + max_seq_len * embedding_dim) * bytes_per_element
    
    # 3. Calculate activation memory
    # Activations are the sum of token and positional embeddings, with shape (batch_size, seq_len, embedding_dim)
    activation_memory_bytes = batch_size * seq_len * embedding_dim * bytes_per_element
    
    return flops, static_memory_bytes, activation_memory_bytes


