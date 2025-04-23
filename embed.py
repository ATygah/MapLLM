# Use this file for embedding the input tokens and their positional encodings
# This is the first step of the model

from data_structs import dtype_size
import torch
################################################################################
#                               INFO OF FUNCTION                               #
#                                                                              #
# Embedding layers will produce a batch_size x seq_len x embedding_dim tensor. #
# Since positional encodings are added to the input tokens, we set its         #
# activation memory to 0 because the embedding layer already accounts for the  #
# activation memory. The output of the embedding layer is                      #
# batch_size x seq_len x embedding_dim, and the final shape is                #
# batch_size x seq_len x (embedding_dim) after positional encodings are added. #
################################################################################
def calculate_embedding_memory(vocab_size: int, 
                              batch_size: int,
                              seq_len: int,
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
    flops = 0
    static_memory = total_parameters * bytes_per_element
    activation_memory = batch_size * seq_len * embedding_dim * bytes_per_element
    return flops, total_parameters, static_memory, activation_memory

def calculate_positional_embedding_costs(max_seq_len: int, batch_size: int, seq_len: int, embedding_dim: int, dtype: str = 'float32'):
    """
    Calculate the memory footprint of positional embeddings.
    
    Parameters:
        max_seq_len (int): Maximum sequence length.
        embedding_dim (int): Dimension of the embedding vectors.
        dtype (str): Data type of the embeddings (default: 'float32').
    
    Returns:
        int: Memory usage in bytes
    """
    # Get byte size using the dtype_size function
    bytes_per_element = dtype_size(dtype)
    
    # Calculate total parameters and memory
    total_parameters = max_seq_len * embedding_dim
    flops = batch_size * seq_len * embedding_dim
    static_memory = total_parameters * bytes_per_element
    activation_memory = 0
    return flops, total_parameters, static_memory, activation_memory
