from data_structs import dtype_size
import torch

def calculate_norm_params_flops(norm_type, input_shape):
    if norm_type == "batchnorm":
        # BatchNorm: 2 params per channel
        # For ConvNets: input_shape is (N, C, H, W)
        # N = batch size, C = channels, H = height, W = width
        C = input_shape[1]  # channels
        params = 2 * C
        # FLOPs: 6 ops per element (2 for mean, 2 for variance, 2 for normalization)
        N, C, H, W = input_shape
        flops = N * C * H * W * 8
    elif norm_type == "layernorm":
        # LayerNorm: 2 params per feature
        d = input_shape[-1]  # hidden dim
        params = 2 * d
        # FLOPs breakdown per element (d = H):
        # 1. Mean: d FLOPs
        # 2. Variance: 3d FLOPs
        # 3. Normalization: 2d + 2 FLOPs
        # 4. Scale/Shift: 2d FLOPs
        # Total: 8d + 2 ≈ 8d FLOPs (for large d)
        b, l, d = input_shape
        flops = b * l * d * 8  # Approximation: 8 FLOPs per element
    else:
        raise ValueError("Unsupported normalization type")
    
    return params, flops

def calculate_norm_storage(norm_type, input_shape, dtype='float32'):
    """
    Calculate the static storage (memory footprint) of a normalization layer.
    
    Parameters:
    -----------
    norm_type : str
        Type of normalization ("batchnorm" or "layernorm").
        
    input_shape : tuple
        For BatchNorm: (N, C, H, W) where N=batch size, C=channels, H=height, W=width.
        For LayerNorm: (B, L, d) where B=batch size, L=sequence length, d=hidden dimension.
        
    dtype_size : int, optional
        Size in bytes of each parameter. Default is 4 (float32).
    
    Returns:
    --------
    params : int
        Number of learnable parameters (scale and shift).
        
    storage : int
        Static storage in bytes required to store the parameters.
    """
    dtype_bytes = dtype_size(dtype)

    if norm_type == "batchnorm":
        # For BatchNorm: 2 parameters per channel (scale and shift),
        # where input_shape is (N, C, H, W)
        C = input_shape[1]
        params = 2 * C
    elif norm_type == "layernorm":
        # For LayerNorm: 2 parameters per feature (scale and shift),
        # where input_shape is (B, L, d)
        d = input_shape[-1]
        params = 2 * d
    else:
        raise ValueError("Unsupported normalization type")
        
    storage = params * dtype_size  # total storage in bytes
    return storage

# # Example usage
# batchnorm_params, batchnorm_flops = calculate_norm_params_flops("batchnorm", (32, 64, 28, 28))
# # Example usage for GPT-2
# # Assuming batch_size=8, seq_len=1024, hidden_size=768
# gpt2_layernorm_params, gpt2_layernorm_flops = calculate_norm_params_flops("layernorm", (8, 1024, 768))

# batchnorm_params, batchnorm_storage = calculate_norm_storage("batchnorm", (32, 64, 28, 28))
# print("BatchNorm: Params = {}, Storage = {} bytes".format(batchnorm_params, batchnorm_storage))

# gpt2_layernorm_params, gpt2_layernorm_storage = calculate_norm_storage("layernorm", (8, 1024, 768))
# print("GPT-2 LayerNorm: Params = {}, Storage = {} bytes".format(gpt2_layernorm_params, gpt2_layernorm_storage))
