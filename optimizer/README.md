# NN_Mapper: Neural Network Hardware Mapping Optimizer

NN_Mapper is a toolkit for optimizing neural network models for efficient hardware implementation by analyzing different tensor partitioning strategies and execution models.

## Overview

This project provides tools to:
1. Analyze communication patterns in neural network layers
2. Optimize processing element (PE) requirements for different hardware configurations
3. Visualize tradeoffs between parallel and pipelined execution
4. Compare different tensor partitioning strategies

## Key Components

- **optimizer/time.py**: Analyzes traffic complexity and communication patterns
- **optimizer/peCount.py**: Optimizes and compares PE requirements across strategies
- **test_pe_comparison.py**: Script to generate comparison visualizations

## Supported Layers and Strategies

### Fully Connected Layers
- Row-split
- Column-split
- Hybrid-split
- Sequence-split
- Combined optimization

### Attention Layers
- Head-split
- Query-sequence-split
- Key-sequence-split
- Combined optimization

## Execution Modes

The toolkit supports analysis of two execution paradigms:
1. **Parallel Execution**: All computations run simultaneously on replicated hardware
2. **Pipelined Execution**: Sequential processing with memory reuse between stages

## Usage Examples

### Comparing PE Requirements

```python
from optimizer.peCount import plot_pe_comparison
import matplotlib.pyplot as plt

# Model parameters
model_params = {
    "batch_size": 1,
    "seq_length": 512,
    "input_dim": 1024,
    "output_dim": 1024,
    "num_heads": 16,
    "head_dim": 64,
    "bytes_per_param": 2
}

# Hardware constraints
hw_constraints = {
    "pe_memory": 10000000,  # 10 MB
    "min_dim_size": 8
}

# Generate and save FC layer comparison
plot_pe_comparison(
    layer_type="fc",
    **model_params,
    **hw_constraints,
    save_path="fc_strategy_comparison.png"
)

# Generate and save attention layer comparison
plot_pe_comparison(
    layer_type="attn",
    **model_params,
    **hw_constraints,
    save_path="attn_strategy_comparison.png"
)

plt.show()
```

## Analysis

For detailed analysis of PE requirements across different strategies, see [PE Comparison Analysis](pe_comparison_analysis.md).

## Future Work

- Integration with latency and throughput models
- Support for additional layer types (Conv2D, LayerNorm, etc.)
- Dynamic strategy selection based on workload characteristics
- Hardware-specific optimizations for FPGAs, ASICs, and specialized accelerators 