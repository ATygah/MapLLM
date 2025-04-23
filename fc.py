# This module handles all the linear projection and feed forward calculations
from data_structs import dtype_size, activation_flops

################################################################################
#                               INFO OF THE LIBRARY                            #
#                                                                              #
# Variables:                                                                   #
# --------------------------------------------                                 #
# - dtype_size                                                                 #
# - activation_flops                                                           #
# - seq_len                                                                    #
# - embed_dim                                                                  #
# - vocab_size                                                                 #
# - batch_size                                                                 #
# - expansion_factor                                                           #
# - activation_type                                                            #
# - dtype                                                                      #
#                                                                              #
# Cost Calculation Functions:                                                  #
# --------------------------------------------                                 #
# - calculate_ff_flops -> int: Total FLOPs                                     #
# - calculate_ff_static_memory -> tuple: (num_parameters, static_memory)       #
# - calculate_ff_activation_memory -> int: Memory usage in bytes               #
# - calculate_ff_costs -> tuple: (flops, parameters, static_memory, activation_memory) #
# - calculate_mlp_costs -> tuple: (flops, parameters, static_memory, activation_memory) #
# - calculate_activation_costs -> tuple: (flops, 0, 0, activation_memory)      #
################################################################################

def calculate_ff_flops(seq_len: int, embed_dim: int, vocab_size: int) -> int:
    # Returns: int: Total FLOPs
    return 2 * seq_len * embed_dim * vocab_size

def calculate_ff_static_memory(embed_dim: int, vocab_size: int, dtype: str = 'float32') -> int:
    # Returns: tuple: (num_parameters, static_memory)
    num_parameters = embed_dim * vocab_size
    return num_parameters, num_parameters * dtype_size(dtype)

def calculate_ff_activation_memory(seq_len: int, vocab_size: int, dtype: str = 'float32') -> int:
    # Returns: int: Memory usage in bytes
    num_elements = seq_len * vocab_size
    return num_elements * dtype_size(dtype)

def calculate_ff_costs(seq_len: int, embed_dim: int, vocab_size: int, dtype: str = 'float32') -> tuple:
    # Returns: tuple: (flops, parameters, static_memory, activation_memory)
    flops = calculate_ff_flops(seq_len, embed_dim, vocab_size)
    parameters, static_memory = calculate_ff_static_memory(embed_dim, vocab_size, dtype)
    activation_memory = calculate_ff_activation_memory(seq_len, vocab_size, dtype)
    return flops, parameters, static_memory, activation_memory

def calculate_mlp_costs(embed_dim: int, batch_size: int, seq_len: int, expansion_factor: int, activation_type: str, dtype: str = 'float32') -> tuple:
    # Returns: tuple: (flops, parameters, static_memory, activation_memory)
    flops_ff1, parameters_ff1, static_memory_ff1, activation_memory_ff1 = calculate_ff_costs(seq_len=seq_len, embed_dim=embed_dim, vocab_size=expansion_factor*embed_dim, dtype=dtype)
    flops_ff2, parameters_ff2, static_memory_ff2, activation_memory_ff2 = calculate_ff_costs(seq_len=seq_len, embed_dim=expansion_factor*embed_dim, vocab_size=embed_dim, dtype=dtype)
    flops_activation, parameters_activation, static_memory_activation, activation_memory_activation = calculate_activation_costs(seq_len, embed_dim, activation_type, dtype)
    flops = (flops_ff1 + flops_ff2 + flops_activation) * batch_size
    parameters = parameters_ff1 + parameters_ff2 + parameters_activation
    static_memory = static_memory_ff1 + static_memory_ff2 + static_memory_activation
    activation_memory = (activation_memory_ff1 + activation_memory_ff2 + activation_memory_activation) * batch_size
    print(f"MLP: {flops}, {parameters}, {static_memory}, {activation_memory}")
    return flops, parameters, static_memory, activation_memory

def calculate_activation_costs(seq_len, embed_dim, activation_type, dtype='float32'):
    # Returns: tuple: (flops, 0, 0, activation_memory)
    data_type_size = dtype_size(dtype)
    flops_per_element = activation_flops(activation_type)
    activation_memory = seq_len * embed_dim * data_type_size
    flops = activation_memory * flops_per_element
    return flops, 0, 0, activation_memory

