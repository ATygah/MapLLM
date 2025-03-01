import torch
from embed import calculate_intermediate_size, calculate_intermediate_size_from_spec

# Example usage
# Example intermediate output tensor (PyTorch)
batch_size = 8
seq_len = 1024
embed_dim = 768
example_tensor = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)

# Calculate size
size_bytes = calculate_intermediate_size(example_tensor)
print(f"Size of intermediate output: {size_bytes / (1024**2):.2f} MB")

# Example usage
# Example 1: Explicit dimensions
size_bytes = calculate_intermediate_size_from_spec(batch_size=8, seq_len=1024, embedding_dim=768)
print(f"Size of intermediate output (explicit): {size_bytes / (1024**2):.2f} MB")

# Example 2: Parse dimensions from a tuple
input_dims = (8, 1024, 768)
size_bytes = calculate_intermediate_size_from_spec(input_dimensions=input_dims)
print(f"Size of intermediate output (tuple): {size_bytes / (1024**2):.2f} MB")

# Example 3: Parse dimensions from a dictionary
input_dims = {'batch_size': 8, 'seq_len': 1024, 'embedding_dim': 768}
size_bytes = calculate_intermediate_size_from_spec(input_dimensions=input_dims)
print(f"Size of intermediate output (dict): {size_bytes / (1024**2):.2f} MB")
