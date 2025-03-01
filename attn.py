import torch
from data_structs import dtype_size

def linear_projection_flops(input_dim: int, output_dim: int, batch_size: int, seq_length: int) -> int:
    """Calculate FLOPs for a linear projection.
    
    Args:
        input_dim: Input dimension (d_in)
        output_dim: Output dimension (d_out / heads)
        batch_size: Number of samples in batch
        seq_length: Sequence length
        
    Returns:
        FLOP count (2 * input_dim * output_dim * batch_size * seq_length)
    """
    return 2 * input_dim * output_dim * batch_size * seq_length

def generic_qk_attention_flops(
    batch_size: int,
    num_heads: int,
    q_seq_length: int,
    kv_seq_length: int,
    d_k: int,
) -> int:
    """Calculate FLOPs for generic QK^T attention matrix computation.
    
    Args:
        d_k: Dimension of key vectors
        batch_size: Batch size
        q_seq_length: Query sequence length
        kv_seq_length: Key/Value sequence length
        num_heads: Number of attention heads
        
    Returns:
        FLOP count (2 * d_k * q_seq_length * kv_seq_length * batch_size * num_heads)
    """
    return 2 * d_k * q_seq_length * kv_seq_length * batch_size * num_heads

def generic_attention_value_flops(
    batch_size: int,
    num_heads: int,
    q_seq_length: int,
    kv_seq_length: int, 
    d_v: int,
) -> int:
    """Calculate FLOPs for generic attention value computation.
    
    Args:
        d_v: Dimension of value vectors
        batch_size: Batch size
        q_seq_length: Query sequence length
        kv_seq_length: Key/Value sequence length
        num_heads: Number of attention heads
        
    Returns:
        FLOP count (2 * d_v * q_seq_length * kv_seq_length * batch_size * num_heads)
    """
    return 2 * d_v * q_seq_length * kv_seq_length * batch_size * num_heads

def softmax_flops(
    batch_size: int, 
    q_seq_length: int,
    kv_seq_length: int,
    num_heads: int
) -> int:
    """Calculate FLOPs for attention softmax operation.
    
    Args:
        batch_size: Batch size
        q_seq_length: Query sequence length
        kv_seq_length: Key/Value sequence length
        num_heads: Number of attention heads
        
    Returns:
        FLOP count (3 * q_seq_length * kv_seq_length * batch_size * num_heads)
    """
    return 3 * q_seq_length * kv_seq_length * batch_size * num_heads

def multihead_self_attention_flops(
    d_in: int,
    d_out: int,
    batch_size: int,
    seq_length: int,
    num_heads: int
) -> int:
    """Calculate total FLOPs for multihead self-attention.
    
    Args:
        d_in: Input dimension
        d_out: Output dimension (must be divisible by num_heads)
        batch_size: Batch size
        seq_length: Sequence length
        num_heads: Number of attention heads
        
    Returns:
        Total FLOP count for self-attention operation
    """
    assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
    
    d_k = d_v = d_out // num_heads
    
    # CORRECTED Projections (output_dim should be d_out, not d_k)
    q_flops = linear_projection_flops(d_in, d_out, batch_size, seq_length)
    k_flops = linear_projection_flops(d_in, d_out, batch_size, seq_length)
    v_flops = linear_projection_flops(d_in, d_out, batch_size, seq_length)
    
    # Updated attention computations using generic functions
    attn_flops = generic_qk_attention_flops(d_k, batch_size, seq_length, seq_length, num_heads)
    value_flops = generic_attention_value_flops(d_v, batch_size, seq_length, seq_length, num_heads)
    
    # Softmax after attention computations
    softmax_flops_count = softmax_flops(batch_size, seq_length, seq_length, num_heads)
    
    # Output projection
    out_flops = linear_projection_flops(d_out, d_out, batch_size, seq_length)
    
    return q_flops + k_flops + v_flops + attn_flops + softmax_flops_count + value_flops + out_flops

def cross_attention_flops(
    d_in_q: int,
    d_in_kv: int,
    d_out: int,
    batch_size: int,
    q_seq_length: int,
    kv_seq_length: int,
    num_heads: int
) -> int:
    """Calculate FLOPs for cross-attention between two sequences.
    
    Args:
        d_in_q: Input dimension for queries
        d_in_kv: Input dimension for keys/values
        d_out: Output dimension
        batch_size: Batch size
        q_seq_length: Query sequence length
        kv_seq_length: Key/Value sequence length
        num_heads: Number of attention heads
        
    Returns:
        Total FLOP count for cross-attention operation
    """
    assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
    
    d_k = d_v = d_out // num_heads
    
    # CORRECTED Projections
    q_flops = linear_projection_flops(d_in_q, d_out, batch_size, q_seq_length)
    k_flops = linear_projection_flops(d_in_kv, d_out, batch_size, kv_seq_length)
    v_flops = linear_projection_flops(d_in_kv, d_out, batch_size, kv_seq_length)
    
    # Updated attention computations using generic functions
    attn_flops = generic_qk_attention_flops(d_k, batch_size, q_seq_length, kv_seq_length, num_heads)
    value_flops = generic_attention_value_flops(d_v, batch_size, q_seq_length, kv_seq_length, num_heads)
    
    # Softmax after attention computations
    softmax_flops_count = softmax_flops(batch_size, q_seq_length, kv_seq_length, num_heads)
    
    # Output projection
    out_flops = linear_projection_flops(d_out, d_out, batch_size, q_seq_length)
    
    return q_flops + k_flops + v_flops + attn_flops + softmax_flops_count + value_flops + out_flops

def self_attention_parameters(
    d_in: int,
    d_out: int,
    include_bias: bool = False
) -> int:
    """Calculate the number of parameters in self-attention mechanism.
    
    Args:
        d_in: Input dimension
        d_out: Output dimension
        include_bias: Whether to count bias parameters
        
    Returns:
        Total parameter count: 4*d_in*d_out + 4*d_out (if bias included)
    """
    # Q/K/V projections and output projection
    bias_params = 4 * d_out if include_bias else 0
    return 3 * d_in * d_out + d_out * d_out + bias_params

def cross_attention_parameters(
    d_in_q: int,
    d_in_kv: int,
    d_out: int,
    include_bias: bool = False
) -> int:
    """Calculate the number of parameters in cross-attention mechanism.
    
    Args:
        d_in_q: Input dimension for queries
        d_in_kv: Input dimension for keys/values
        d_out: Output dimension
        include_bias: Whether to count bias parameters
        
    Returns:
        Total parameter count: (d_in_q + 2*d_in_kv + d_out)*d_out + 4*d_out (if bias)
    """
    # Query, Key, Value, and Output projections
    linear_params = (d_in_q + 2*d_in_kv + d_out) * d_out
    bias_params = 4 * d_out if include_bias else 0
    return linear_params + bias_params

def parameters_memory_footprint(
    num_parameters: int, 
    dtype: str = 'float32',
    include_gradients: bool = True
) -> float:
    """Calculate memory footprint for parameters in gigabytes (GB).
    
    Args:
        num_parameters: Number of parameters from *_parameters() functions
        dtype: Data type used for parameters (common: float32, bfloat16, float16)
        include_gradients: Whether to account for gradient storage during training
        
    Returns:
        Memory footprint in bytes
    """
    # Use the existing dtype_size function instead of duplicating the mapping
    dtype_bytes = dtype_size(dtype)
    
    # During training: parameters + gradients + optimizer states (≈2-3x)
    multiplier = 3 if include_gradients else 1
    
    bytes_total = num_parameters * dtype_bytes * multiplier
    return bytes_total

def self_attention_activation_memory(
    batch_size: int,
    seq_length: int,
    d_model: int,
    num_heads: int,
    dtype: str = 'float32',
    consider_optimizations: bool = True
) -> float:
    """Calculate peak activation memory for self-attention with real-world considerations.
    
    Args:
        batch_size: Number of samples in batch
        seq_length: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        dtype: Data type used for activations
        consider_optimizations: Account for framework optimizations (in-place ops, kernel fusion)
        
    Returns:
        Peak memory usage in bytes
    """
    d_head = d_model // num_heads
    elem_size = dtype_size(dtype)
    
    # Key memory components
    q_proj = batch_size * seq_length * d_model * elem_size
    k_proj = q_proj  # Same as Q
    v_proj = q_proj  # Same as Q
    
    attn_scores = batch_size * num_heads * seq_length * seq_length * elem_size
    context = batch_size * seq_length * d_model * elem_size
    
    if consider_optimizations:
        # Real-world behavior (PyTorch-like):
        # 1. Q/K/V projections done sequentially, inputs released immediately
        # 2. QK^T computed directly via matmul (no intermediate storage)
        # 3. Softmax done in-place where possible
        return max(
            # Phase 1: Q/K/V projections (worst case: all three exist)
            3 * q_proj,
            # Phase 2: Attention computation (Q, K, scores)
            q_proj + k_proj + attn_scores,
            # Phase 3: Context computation (V, scores, context)
            v_proj + attn_scores + context
        )
    else:
        # Theoretical maximum (all intermediates exist simultaneously)
        return (3 * q_proj +  # Q/K/V projections
                attn_scores +  # Attention matrix
                context)  # Output context

def cross_attention_activation_memory(
    batch_size: int,
    q_seq_length: int,
    kv_seq_length: int,
    d_model: int,
    num_heads: int,
    dtype: str = 'float32',
    consider_optimizations: bool = True
) -> float:
    """Calculate peak activation memory for cross-attention with real-world considerations.
    
    Args:
        batch_size: Batch size
        q_seq_length: Query sequence length
        kv_seq_length: Key/Value sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        dtype: Data type used for activations
        consider_optimizations: Account for framework optimizations
        
    Returns:
        Peak memory usage in bytes
    """
    d_head = d_model // num_heads
    elem_size = dtype_size(dtype)
    
    # Key memory components
    q_proj = batch_size * q_seq_length * d_model * elem_size
    k_proj = batch_size * kv_seq_length * d_model * elem_size
    v_proj = k_proj  # Same as K
    
    attn_scores = batch_size * num_heads * q_seq_length * kv_seq_length * elem_size
    context = batch_size * q_seq_length * d_model * elem_size

    if consider_optimizations:
        # Real-world framework behavior:
        # 1. Sequential projection computations with input buffer reuse
        # 2. Optimized attention kernel with fused operations
        return max(
            # Phase 1: Projections (Q/K/V)
            q_proj + k_proj + v_proj,
            # Phase 2: Attention computation (Q, K, scores)
            q_proj + k_proj + attn_scores,
            # Phase 3: Context computation (V, scores, context)
            v_proj + attn_scores + context
        )
    else:
        # Theoretical maximum
        return (q_proj + k_proj + v_proj +
                attn_scores + context)

