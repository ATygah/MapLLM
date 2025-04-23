import os
import sys

# Add the parent directory of 'optimizer' to the path
# This ensures we can import the optimizer package regardless of working directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get directory of this script
parent_dir = os.path.dirname(os.path.dirname(script_dir))  # Get parent of parent (root directory)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now we can import from optimizer package
from optimizer.peCount import plot_pe_comparison
import matplotlib.pyplot as plt

# Get current working directory for absolute paths
current_dir = os.getcwd()

# Define model parameters for a transformer layer
model_params = {
    'batch_size': 1,             # Batch size (b)
    'seq_length': 2048,           # Sequence length (s)
    'input_dim': 14436,           # Input dimension (I) 
    'output_dim': 14436,          # Output dimension (O)
    'num_heads': 112,             # Number of attention heads (h)
    'head_dim': 128,              # Dimension per head (E = 64)
    'bytes_per_param': 2         # 2 bytes per parameter (FP16)
}

# Define hardware constraints - increased PE memory to ensure all strategies can find valid configurations
hardware_constraints = {
    'pe_memory': 10_000_000,     # 10 MB memory per PE (increased from 1 MB)
    'min_dim_size': 8            # Minimum dimension size after splitting
}

# Plot FC layer comparison
print("Generating FC layer strategy comparison...")
fc_fig = plot_pe_comparison(model_params, hardware_constraints, "fc")
fc_path = os.path.join(current_dir, "fc_strategy_comparison.png")
fc_fig.savefig(fc_path, dpi=300, bbox_inches='tight')
print(f"FC layer comparison saved as '{fc_path}'")

# Plot Attention layer comparison
print("Generating Attention layer strategy comparison...")
attn_fig = plot_pe_comparison(model_params, hardware_constraints, "attn")
attn_path = os.path.join(current_dir, "attn_strategy_comparison.png")
attn_fig.savefig(attn_path, dpi=300, bbox_inches='tight')
print(f"Attention layer comparison saved as '{attn_path}'")

# Try a smaller model
# print("\nTrying with smaller model dimensions...")
# small_model_params = {
#     'batch_size': 1,
#     'seq_length': 128,        # Reduced from 512
#     'input_dim': 256,         # Reduced from 1024
#     'output_dim': 256,        # Reduced from 1024
#     'num_heads': 4,           # Reduced from 16
#     'head_dim': 64,
#     'bytes_per_param': 2
# }

# # Generate plots with smaller model
# fc_fig_small = plot_pe_comparison(small_model_params, hardware_constraints, "fc")
# small_fc_path = os.path.join(current_dir, "fc_strategy_comparison_small.png")
# fc_fig_small.savefig(small_fc_path, dpi=300, bbox_inches='tight')
# print(f"Small FC layer comparison saved as '{small_fc_path}'")

# attn_fig_small = plot_pe_comparison(small_model_params, hardware_constraints, "attn")
# small_attn_path = os.path.join(current_dir, "attn_strategy_comparison_small.png")
# attn_fig_small.savefig(small_attn_path, dpi=300, bbox_inches='tight')
# print(f"Small Attention layer comparison saved as '{small_attn_path}'")

# List all PNG files in the directory
print("\nGenerated PNG files:")
for file in os.listdir(current_dir):
    if file.endswith(".png"):
        print(f"- {file}")

print("\nComparison complete!") 