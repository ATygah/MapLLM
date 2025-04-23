import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

# Enable LaTeX-style math rendering
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Define the data from the provided table
strategies = [
    {
        'name': 'Row-Split (FC)', 
        'inputTraffic': 'O(r)', 
        'outputTraffic': 'O(r)', 
        'totalTraffic': 'O(r)',
        'complexity': 1
    },
    { 
        'name': 'Column-Split (FC)', 
        'inputTraffic': 'O(c)', 
        'outputTraffic': 'O(c)', 
        'totalTraffic': 'O(c)',
        'complexity': 1
    },
    { 
        'name': 'Hybrid-Split (FC)', 
        'inputTraffic': 'O(r×c)', 
        'outputTraffic': 'O(r×c)', 
        'totalTraffic': 'O(r×c)',
        'complexity': 2
    },
    { 
        'name': 'Sequence-Split (FC)', 
        'inputTraffic': 'O(s)', 
        'outputTraffic': 'O(s)', 
        'totalTraffic': 'O(s)',
        'complexity': 1
    },
    { 
        'name': 'Embedding-Split (Attention)', 
        'inputTraffic': 'O(r)', 
        'outputTraffic': 'O(r²)', 
        'totalTraffic': 'O(r²)',
        'complexity': 2
    },
    { 
        'name': 'Query-Sequence-Split (Attention)', 
        'inputTraffic': 'O(q_s)', 
        'outputTraffic': 'O(q_s)', 
        'totalTraffic': 'O(q_s)',
        'complexity': 1
    },
    { 
        'name': 'Key-Sequence-Split (Attention)', 
        'inputTraffic': 'O(k_s)', 
        'outputTraffic': 'O(k_s)', 
        'totalTraffic': 'O(k_s)',
        'complexity': 1
    }
]

# Create DataFrames for visualization
df = pd.DataFrame(strategies)

# Set up figure for visualization
plt.figure(figsize=(14, 10))

# Visualization 1: Total Traffic Complexity by Strategy
plt.subplot(2, 1, 1)
colors = ['#3498db', '#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c']  # FC in blue, Attention in red

# X values for the bars
x_pos = np.arange(len(strategies))

# Create bars
plt.bar(x_pos, [1, 1, 4, 1, 4, 1, 1], color=colors, alpha=0.7)

# Customize the plot
plt.xticks(x_pos, [s['name'] for s in strategies], rotation=45, ha='right')
plt.ylabel('Relative Traffic (Arbitrary Units)', fontsize=12)
plt.title('Relative Communication Traffic by Splitting Strategy', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotations to each bar showing the complexity notation
for i, strategy in enumerate(strategies):
    plt.text(i, 0.2, strategy['totalTraffic'], ha='center', color='black', fontweight='bold', fontsize=10)

# Visualization 2: Traffic Growth with Increasing Parallelism
plt.subplot(2, 1, 2)

# Define x-axis (parallelism factor)
parallelism = np.arange(1, 21)

# Define different growth rates
linear_growth = parallelism
quadratic_growth = parallelism**2

# Plot lines for each strategy type
plt.plot(parallelism, linear_growth, 'b-', linewidth=2, 
         label=r'Linear Growth: O(p) - Row, Column, Sequence, $q_s$, $k_s$ Split')
plt.plot(parallelism, quadratic_growth, 'r-', linewidth=2, 
         label=r'Quadratic Growth: O(p²) - Hybrid Split, Embedding Split')

# Add annotations
plt.annotate('Linear strategies scale efficiently', xy=(15, 15), xytext=(15, 30),
             arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=10)
plt.annotate('Quadratic strategies have worse scaling', xy=(15, 225), xytext=(15, 150),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10)

# Customize plot
plt.xlabel('Parallelism Factor (p)', fontsize=12)
plt.ylabel('Communication Traffic', fontsize=12)
plt.title('Traffic Growth with Increasing Parallelism', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')

# Improve layout
plt.tight_layout()
plt.savefig('traffic_complexity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'traffic_complexity_analysis.png'") 