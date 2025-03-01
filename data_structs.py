def dtype_size(dtype: str) -> int:
    """Get size in bytes for common data types."""
    return {
        'float32': 4, 'fp32': 4,
        'float16': 2, 'fp16': 2,
        'bfloat16': 2, 'bf16': 2,
        'int8': 1, 'uint8': 1,
    }.get(dtype.lower(), 4) 