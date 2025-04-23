def dtype_size(dtype: str) -> int:
    """Get size in bytes for common data types."""
    return {
        'float32': 4, 'fp32': 4,
        'float16': 2, 'fp16': 2,
        'bfloat16': 2, 'bf16': 2,
        'int8': 1, 'uint8': 1,
    }.get(dtype.lower(), 4) 

def activation_flops(activation_type: str) -> int:
    """Get FLOPs for common activation functions."""
    return {
        'gelu': 8,
        'relu': 1,
        'sigmoid': 4,
        'tanh': 5,
        'softmax': 5,
        'softplus': 3,
        'softsign': 3,  # |x|, addition, division
        'leaky_relu': 2,  # comparison, multiplication
        'elu': 4,        # comparison, exponential, subtraction, multiplication
        'selu': 5,       # ELU ops + scaling
        'swish': 5,      # sigmoid (4) + multiplication (1)
    }.get(activation_type.lower(), 0)