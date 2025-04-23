# Processing Element (PE) Requirement Analysis

This document provides an analysis of PE requirements across different tensor partitioning strategies for fully connected (FC) and attention layers based on our visualizations.

## Setup

- **Model Parameters** (Large Model):
  - Batch Size: 1
  - Sequence Length: 512
  - Input/Output Dimensions: 1024
  - Attention Heads: 16
  - Head Dimension: 64
  - Parameter Size: 2 bytes (FP16)

- **Small Model Parameters**:
  - Batch Size: 1
  - Sequence Length: 128
  - Input/Output Dimensions: 256
  - Attention Heads: 4
  - Head Dimension: 64
  - Parameter Size: 2 bytes (FP16)

- **Hardware Constraints**:
  - PE Memory: 10MB per PE
  - Minimum Dimension Size: 8

## Key Findings for FC Layer Strategies

The PE comparison visualization for FC layers demonstrates:

1. **Execution Mode Impact**: Pipelined execution consistently requires fewer PEs than parallel execution across all strategies. This is due to memory reuse across pipeline stages.

2. **Strategy Efficiency Ranking**:
   - **Row-Split** and **Column-Split** typically require similar numbers of PEs, as they both split along a single dimension.
   - **Hybrid-Split** generally requires more PEs than single-dimension splits, as it uses r×c PEs, but each PE handles less computation.
   - **Sequence-Split** efficiency depends heavily on sequence length; it's more efficient for long sequences.
   - **Combined** approach often finds the optimal balance but requires the most complex implementation.

3. **PE Reduction in Pipelined Mode**: The reduction percentage annotations show significant savings (often 20-50%) when switching from parallel to pipelined execution.

## Key Findings for Attention Layer Strategies

The PE comparison for attention mechanisms shows:

1. **Critical Dimensions**: Head-Split tends to require fewer PEs than Query/Key-Sequence-Split, especially for models with many heads.

2. **Memory Bottlenecks**: Attention score computation (size proportional to sequence²) dominates memory usage, making sequence splitting particularly effective.

3. **Execution Mode Benefits**: The reduction percentage is typically even higher for attention mechanisms than FC layers, as attention has more distinct computational stages that can be pipelined.

## Scaling Insights

Comparing large and small model visualizations:

1. **Dimension Impact**: PE requirements scale roughly linearly with input/output dimensions for row/column splits, but quadratically for hybrid approaches.

2. **Sequence Length**: Doubling sequence length approximately doubles PE requirements for sequence-based partitioning strategies.

3. **Pipeline Benefits**: Pipelined execution shows proportionally greater benefits as model size increases, making it critical for large-scale deployments.

## Implementation Recommendations

Based on the PE requirement analysis:

1. For FC layers:
   - Use column-split for output-heavy workloads
   - Use row-split for input-heavy workloads
   - Consider hybrid-split only when memory constraints are severe

2. For attention layers:
   - Prefer head-split when possible as it requires fewer PEs
   - Use sequence splitting for very long sequences

3. General guidance:
   - Always implement pipelined execution when possible
   - Combined strategies provide optimal PE efficiency but increase implementation complexity

## Next Steps

- Analyze the performance implications (throughput, latency) of each strategy
- Consider additional hardware constraints such as on-chip bandwidth
- Explore dynamic strategies that adapt based on workload characteristics 