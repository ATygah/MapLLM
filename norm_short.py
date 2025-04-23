from data_structs import dtype_size
import torch

def calculate_norm_params_costs(norm_type, batch_size, seq_len, input_dim, dtype='float32'):
    # Ignore batchNorm for now.
    if norm_type == "batchnorm":
        # BatchNorm: 2 params per channel
        # For ConvNets: input_shape is (N, C, H, W)
        # N = batch size, C = channels, H = height, W = width
        C = input_dim  # channels
        params = 2 * C
        # FLOPs: 6 ops per element (2 for mean, 2 for variance, 2 for normalization)
        N, C, H, W = input_shape
        flops = N * C * H * W * 8
    elif norm_type == "layernorm":
        # LayerNorm: 2 params per feature
        d = input_dim  # hidden dim
        params = 2 * d
        # FLOPs breakdown per element (d = H):
        # 1. Mean: d FLOPs
        # 2. Variance: 3d FLOPs
        # 3. Normalization: 2d + 2 FLOPs
        # 4. Scale/Shift: 2d FLOPs
        # Total: 8d + 2 ≈ 8d FLOPs (for large d)
        flops = batch_size * seq_len * input_dim * 8  # Approximation: 8 FLOPs per element
    else:
        raise ValueError("Unsupported normalization type")
    dtype_bytes = dtype_size(dtype)
    storage = params * dtype_bytes
    # TODO: Figure out if you need to add the activation memory.
    return flops, params, storage, 0

def calculate_shortcut_costs(batch_size, seq_len, embed_dim, dtype='float32'):
    dtype_bytes = dtype_size(dtype)
    flops = batch_size * seq_len * embed_dim  # One FLOP per element addition
    activation_memory = batch_size * seq_len * embed_dim * dtype_bytes
    return flops, 0, 0, activation_memory