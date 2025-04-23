import torch
from data_structs import dtype_size

"""
##############################
#          CALCULATIONS      #
##############################

# Assumptions:
# Let T be the sequence length (number of tokens).
# Let d be the hidden (model) dimension (e.g., 768 in GPT-2 small).
# Let d₍ff₎ be the inner (feedforward) dimension, typically 4d.
# Let L be the number of transformer blocks (e.g., 12 in GPT-2 small).
# Let h be the number of attention heads so that each head has dimension dₕ = d/h.

# Self-Attention Total per Block:
# FLOPs = [6 T d² (projections)] + [2 T² d (score computation)] + 
#          [2 T² d (weighted sum)] + [2 T d² (output projection)]
#        = 8 T d² + 4 T² d
"""

######################################################## 
#          INFO OF THE LIBRARY      #
######################################################## 
# Function Names:                                      #
# linear_projection_flops                              #
# generic_qk_attention_flops                           #
# generic_attention_value_flops                        #
# softmax_flops                                        #
# compute_qkv_flops                                    #
# compute_attention_flops                              #
# attention_flops_per_batch                            #
#                                                      #
# weight_matrix_parameters                             #
# output_projection_parameters                         #
# attention_parameters_memory                          #
#                                                      #
# qkv_activations                                      #
# attention_scores_memory                              #
# context_matrix_memory                                #
# attention_activation_memory_per_batch                #
# calculate_attention_costs                            #    
#                                                      #
# NOMENCLATURE:                                        #
# d_in_q: Input dimension for queries                  #
# d_in_kv: Input dimension for keys/values             #
# d_out: Output dimension                               #
# d_k: Key dimension                                    #
# d_v: Value dimension                                  #
# q_seq_length: Query sequence length                  #
# kv_seq_length: Key/Value sequence length             #
# d_model: Model dimension                              #
# num_heads: Number of attention heads                  #
# dtype: Data type used for activations                #
# batch_size: Number of samples in batch                #
# include_output_layer: Include the output layer in the calculation #
# attention_type: Type of attention ('self' or 'cross')  #
# consider_optimizations: Account for framework optimizations (in-place ops, kernel fusion) #
######################################################## 
def linear_projection_flops(input_dim: int, output_dim: int, seq_length: int) -> int:
    """Calculate FLOPs for a linear projection.
    
    Args:
        input_dim: Input dimension (d_in)
        output_dim: Output dimension (d_out / heads)
        seq_length: Sequence length
        
    Returns:
        FLOP count (2 * input_dim * output_dim * seq_length)
    """
    return 2 * input_dim * output_dim * seq_length

def generic_qk_attention_flops(
    num_heads: int,
    q_seq_length: int,
    kv_seq_length: int,
    d_k: int,
) -> int:
    """Calculate FLOPs for generic QK^T attention matrix computation."""
    return 2 * d_k * q_seq_length * kv_seq_length * num_heads

def generic_attention_value_flops(
    num_heads: int,
    q_seq_length: int,
    kv_seq_length: int, 
    d_v : int,
) -> int:
    return 2 * d_v * q_seq_length * kv_seq_length * num_heads

def softmax_flops(
    q_seq_length: int,
    kv_seq_length: int,
    num_heads: int
) -> int:
    return 3 * q_seq_length * kv_seq_length * num_heads

def compute_qkv_flops(
    attention_type: str,
    d_model: int,
    d_out: int,
    q_seq_length: int,
    kv_seq_length: int = None,
) -> int:
    # INFO: Divide by num_heads to get the FLOPs per head
    # d_out is the dimension of the keys and queries for all heads combined
    # The Q,K,V are computed together for all heads in single matrix operations. 

    if attention_type not in ['self', 'cross']:
        raise ValueError("Invalid attention type. Use 'self' or 'cross'.")
    
    total_flops = 0

    if attention_type == 'self':
        q_flops = linear_projection_flops(input_dim=d_model, output_dim=d_out, seq_length=q_seq_length)
        k_flops = q_flops  # Same as q_flops for self-attention
        v_flops = q_flops  # Same as q_flops for self-attention
        total_flops += q_flops + k_flops + v_flops

    elif attention_type == 'cross':
        if kv_seq_length is None:
            raise ValueError("kv_seq_length must be provided for cross-attention.")
        
        q_flops = linear_projection_flops(input_dim=d_model, output_dim=d_out, seq_length=q_seq_length)
        k_flops = linear_projection_flops(input_dim=d_model, output_dim=d_out, seq_length=kv_seq_length)
        v_flops = linear_projection_flops(input_dim=d_model, output_dim=d_out, seq_length=kv_seq_length)
        total_flops += q_flops + k_flops + v_flops

    return total_flops

def compute_attention_flops(
    attention_type: str,
    q_seq_length: int,
    kv_seq_length: int,
    num_heads: int,
    d_k: int
) -> int:
    # INFO: Divide by num_heads to get the FLOPs per head
    # d_k is the dimension of the keys and queries per head
    # Attention matrixes are computed per head so this function takes head dimension as input
    if attention_type == 'self':
        attn_flops = generic_qk_attention_flops(num_heads=num_heads, q_seq_length=q_seq_length, kv_seq_length=q_seq_length, d_k=d_k)
        value_flops = generic_attention_value_flops(num_heads=num_heads, q_seq_length=q_seq_length, kv_seq_length=q_seq_length, d_v=d_k)
        softmax_flops_count = softmax_flops(q_seq_length=q_seq_length, kv_seq_length=q_seq_length, num_heads=num_heads)
        return attn_flops + value_flops + softmax_flops_count

    elif attention_type == 'cross':
        attn_flops = generic_qk_attention_flops(num_heads=num_heads, q_seq_length=q_seq_length, kv_seq_length=kv_seq_length, d_k=d_k)
        value_flops = generic_attention_value_flops(num_heads=num_heads, q_seq_length=q_seq_length, kv_seq_length=kv_seq_length, d_v=d_k)
        softmax_flops_count = softmax_flops(q_seq_length=q_seq_length, kv_seq_length=kv_seq_length, num_heads=num_heads)
        return attn_flops + value_flops + softmax_flops_count

def attention_flops_per_batch(
    attention_type: str,
    d_model: int,
    d_out: int,
    q_seq_length: int,
    num_heads: int,
    kv_seq_length: int = None,
    include_output_layer: bool = True
) -> int:
    """Calculate total FLOPs for attention mechanisms (self or cross).
    
    Args:
        d_model: Input dimension for queries
    """
    total_flops = compute_qkv_flops(attention_type, d_model, d_out, q_seq_length, kv_seq_length)

    d_k = d_out // num_heads
    total_flops += compute_attention_flops(attention_type, q_seq_length, kv_seq_length, num_heads, d_k)

    if include_output_layer:
        total_flops += linear_projection_flops(d_out, d_out, q_seq_length)  # Output projection

    return total_flops

# ==========================
# Calculation of Parameters and Their Memory Consumption
# ==========================

def weight_matrix_parameters(d_model: int, d_out: int, include_bias: bool = False, dtype: str = 'float32') -> int:
    """Calculate the number of parameters in the weight matrix of the attention mechanism."""
    dtype_bytes = dtype_size(dtype)
    total_params = d_model * d_out
    if include_bias:
        total_params += d_out  # Adding bias parameters
    return total_params, total_params * dtype_bytes

def output_projection_parameters(d_out: int, include_bias: bool = False, dtype: str = 'float32') -> int:
    """Calculate the number of parameters in the output projection layer."""
    dtype_bytes = dtype_size(dtype)
    total_params = d_out * d_out  # Assuming the output layer has d_out parameters
    if include_bias:
        total_params += d_out  # Adding bias parameters
    return total_params, total_params * dtype_bytes

def attention_parameters_memory(
    d_model: int,
    d_out: int,
    include_bias: bool = False,     
    include_output_layer: bool = True,
    dtype: str = 'float32'
) -> int:
    """Calculate the number of parameters in attention mechanisms.
    Returns:
        Total parameter count, static memory.
    """
    # Calculate weight matrices parameters (Q, K, V)
    q_params, q_memory = weight_matrix_parameters(d_model, d_out, include_bias, dtype)
    k_params, k_memory = weight_matrix_parameters(d_model, d_out, include_bias, dtype)
    v_params, v_memory = weight_matrix_parameters(d_model, d_out, include_bias, dtype)
    
    total_params = q_params + k_params + v_params
    static_memory = q_memory + k_memory + v_memory

    # Include output layer parameters if specified
    if include_output_layer:
        output_params, output_memory = output_projection_parameters(d_out, include_bias, dtype)
        total_params += output_params
        static_memory += output_memory

    return total_params, static_memory

# ==========================
# Calculation of Activation Memory
# ==========================

def qkv_activations(seq_length: int, d_model: int, dtype: str = 'float32') -> int:
    """Calculate activation memory for Q, K, or V projections."""
    return seq_length * d_model * dtype_size(dtype)

def attention_scores_memory(num_heads: int, q_seq_length: int, kv_seq_length: int, dtype: str = 'float32') -> int:
    """Calculate activation memory for attention scores."""
    return num_heads * q_seq_length * kv_seq_length * dtype_size(dtype)

def context_matrix_memory(d_model: int, q_seq_length: int, dtype: str = 'float32') -> int:
    """Calculate activation memory for context."""
    return q_seq_length * d_model * dtype_size(dtype)

def attention_activation_memory_per_batch(
    attention_type: str,
    q_seq_length: int,
    d_model: int,
    num_heads: int,
    dtype: str = 'float32',
    kv_seq_length: int = None,  # Only used for cross-attention
    consider_optimizations: bool = True
) -> float:
    """Calculate peak activation memory for attention mechanisms (self or cross).
    
    Args:
        attention_type (str): Type of attention ('self' or 'cross').
        q_seq_length (int): Query sequence length.
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        dtype (str, optional): Data type. Defaults to 'float32'.
        kv_seq_length (int, optional): Key/Value sequence length for cross-attention. Defaults to None.
        consider_optimizations (bool, optional): Account for framework optimizations. Defaults to True.
    
    Returns:
        float: Peak memory usage in bytes.
    """
    if attention_type not in ['self', 'cross']:
        raise ValueError("Invalid attention type. Use 'self' or 'cross'.")
    
    if attention_type == 'self':
        # Self-attention: q_seq_length == kv_seq_length
        kv_seq_length = q_seq_length
    
    # Calculate activation memories without batch_size
    q_proj = qkv_activations(q_seq_length, d_model, dtype)
    k_proj = qkv_activations(kv_seq_length, d_model, dtype)
    v_proj = qkv_activations(kv_seq_length, d_model, dtype)
    
    attn_scores = attention_scores_memory(num_heads, q_seq_length, kv_seq_length, dtype)
    context = context_matrix_memory(d_model, q_seq_length, dtype)
    
    if consider_optimizations:
        # Real-world behavior (PyTorch-like):
        # 1. Q/K/V projections done sequentially, inputs released immediately
        # 2. QK^T computed directly via matmul (no intermediate storage)
        # 3. Softmax done in-place where possible
        peak_memory = max(
            # Phase 1: Projections (Q/K/V)
            q_proj + k_proj + v_proj,
            # Phase 2: Attention computation (Q, K, scores)
            q_proj + k_proj + attn_scores,
            # Phase 3: Context computation (V, scores, context)
            v_proj + attn_scores + context
        )
    else:
        # Theoretical maximum (all intermediates exist simultaneously)
        peak_memory = q_proj + k_proj + v_proj + attn_scores + context
    
    # Multiply by batch_size after computing peak memory
    return peak_memory

# ==========================
# Father function for Calculation of Costs
# ==========================

def calculate_attention_costs(attention_type: str, 
                              batch_size: int, 
                              q_seq_length: int, 
                              kv_seq_length: int, 
                              d_model: int, 
                              num_heads: int, 
                              dtype: str = 'float32') -> tuple:
    """Calculate all costs for the attention layer.
    Returns:
        tuple: (flops, parameters, static_memory, activation_memory)
    """
    if(attention_type == 'self'):
        flops_per_batch = attention_flops_per_batch(attention_type='self', d_model=d_model, d_out=d_model, q_seq_length=q_seq_length, num_heads=num_heads, include_output_layer=True)
        parameters, static_memory = attention_parameters_memory(d_model=d_model, d_out=d_model)
        activation_memory_per_batch = attention_activation_memory_per_batch('self', q_seq_length=q_seq_length, d_model=d_model, num_heads=num_heads, dtype=dtype)
    elif(attention_type == 'cross'):
        flops_per_batch = attention_flops_per_batch(attention_type='cross', d_model=d_model, d_out=d_model, q_seq_length=q_seq_length, kv_seq_length=kv_seq_length, num_heads=num_heads, include_output_layer=True)
        parameters, static_memory = attention_parameters_memory(d_model=d_model, d_out=d_model)
        activation_memory_per_batch = attention_activation_memory_per_batch('cross', q_seq_length=q_seq_length, kv_seq_length=kv_seq_length, d_model=d_model, num_heads=num_heads, dtype=dtype)
    #print(f"Attention: {flops}, {parameters}, {static_memory}, {activation_memory}")
    return flops_per_batch*batch_size, parameters, static_memory, activation_memory_per_batch*batch_size
