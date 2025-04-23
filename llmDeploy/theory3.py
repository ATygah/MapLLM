import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors

# Parse the task data from the original code
data = """
   task_id        src_pe_with_network       dest_pe_with_network  bytes wait_ids                                                                description
0        1          (0, 0) (external)            (1, 0) (q_proj)     96     None                                 Input distribution (rows 0:12) to PE(1, 0)
1        2            (1, 0) (q_proj)            (1, 0) (q_proj)     96        1                  PE(1, 0) full computation (input:(0, 12), output:(0, 12))
2        3          (0, 0) (external)            (2, 0) (k_proj)     96     None                                 Input distribution (rows 0:12) to PE(2, 0)
3        4            (2, 0) (k_proj)            (2, 0) (k_proj)     96        3                  PE(2, 0) full computation (input:(0, 12), output:(0, 12))
4        5          (0, 0) (external)            (3, 0) (v_proj)     96     None                                 Input distribution (rows 0:12) to PE(3, 0)
5        6            (3, 0) (v_proj)            (3, 0) (v_proj)     96        5                  PE(3, 0) full computation (input:(0, 12), output:(0, 12))
6        7            (1, 0) (q_proj)  (4, 0) (attention_head_0)     48        2                               (1, 0) sends (0:4, 0:6) of input to PE(4, 0)
7        8            (2, 0) (k_proj)  (4, 0) (attention_head_0)     48        4                    (2, 0) sends (0:6, 0:4) of input Transposed to PE(4, 0)
8        9  (4, 0) (attention_head_0)  (4, 0) (attention_head_0)     32     7, 8                        Matrix multiply at PE(4, 0) (output block 0:4, 0:4)
9       10            (3, 0) (v_proj)  (4, 0) (attention_head_0)     48        6                                Send V (0:4, 0:6) from PE(3, 0) to PE(4, 0)
10      11  (4, 0) (attention_head_0)  (4, 0) (attention_head_0)     48    9, 10                   Attention computation ((Q·K^T)·V) at PE(4, 0) (0:4, 0:6)
11      12            (1, 0) (q_proj)  (5, 0) (attention_head_1)     48        2                               (1, 0) sends (0:4, 0:6) of input to PE(5, 0)
12      13            (2, 0) (k_proj)  (5, 0) (attention_head_1)     48        4                    (2, 0) sends (0:6, 0:4) of input Transposed to PE(5, 0)
13      14  (5, 0) (attention_head_1)  (5, 0) (attention_head_1)     32   12, 13                        Matrix multiply at PE(5, 0) (output block 0:4, 0:4)
14      15            (3, 0) (v_proj)  (5, 0) (attention_head_1)     48        6                                Send V (0:4, 0:6) from PE(3, 0) to PE(5, 0)
15      16  (5, 0) (attention_head_1)  (5, 0) (attention_head_1)     48   14, 15                   Attention computation ((Q·K^T)·V) at PE(5, 0) (0:4, 0:6)
16      17  (4, 0) (attention_head_0)       (6, 0) (output_proj)     48       11   Network None input  (0:4,0:6) from previous network PE(4, 0) to PE(6, 0)
17      18  (5, 0) (attention_head_1)       (6, 0) (output_proj)     48       16  Network None input  (0:4,6:12) from previous network PE(5, 0) to PE(6, 0)
18      19       (6, 0) (output_proj)       (6, 0) (output_proj)     96   17, 18                  PE(6, 0) full computation (input:(0, 12), output:(0, 12))
"""

# Parse the data
tasks = []
lines = data.strip().split('\n')
task_descriptions = {}

for i, line in enumerate(lines[1:], 0):
    parts = line.split()
    task_id = int(parts[1])
    
    # Extract dependencies
    wait_ids_str = parts[9]
    dependencies = []
    
    if wait_ids_str != "None":
        # Handle multiple comma-separated dependencies
        for dep_part in parts[9:]:
            if dep_part.endswith(','):
                dep_part = dep_part[:-1]  # Remove trailing comma
            
            # Check if it's a valid task ID (number)
            if dep_part.isdigit():
                dependencies.append(int(dep_part))
            else:
                # Stop when we hit non-numeric content (likely description)
                if not re.match(r'^\d+$', dep_part):
                    break
    
    # Extract src_pe and dest_pe for node label
    src_pe = ' '.join(parts[2:5])
    dest_pe = ' '.join(parts[5:8])
    
    # Determine if task is computation or communication
    is_computation = src_pe == dest_pe and "external" not in src_pe
    task_type = "Computation" if is_computation else "Communication"
    
    # Extract description (after all dependencies)
    desc_index = 10
    while desc_index < len(parts) and (parts[desc_index].isdigit() or parts[desc_index].endswith(',')):
        desc_index += 1
    description = ' '.join(parts[desc_index:]) if desc_index < len(parts) else ""
    
    task_descriptions[task_id] = description
    
    # Add to tasks list
    tasks.append({
        'task_id': task_id,
        'src_pe': src_pe,
        'dest_pe': dest_pe,
        'dependencies': dependencies,
        'is_computation': is_computation,
        'type': task_type,
        'description': description
    })

# Create a directed graph
G = nx.DiGraph()

# Add nodes
for task in tasks:
    task_id = task['task_id']
    # Just add the task ID as the label, without any coordinates
    G.add_node(task_id, label=str(task_id), type=task['type'])

# Add edges
for task in tasks:
    task_id = task['task_id']
    for dep in task['dependencies']:
        G.add_edge(dep, task_id)

# Compute node layers for horizontal positioning
node_layers = {}
root_nodes = [task['task_id'] for task in tasks if not task['dependencies']]

def assign_layers(node, current_layer=0):
    if node in node_layers:
        node_layers[node] = max(node_layers[node], current_layer)
    else:
        node_layers[node] = current_layer
    
    # Process children
    for successor in G.successors(node):
        assign_layers(successor, current_layer + 1)

# Assign initial layers
for root in root_nodes:
    assign_layers(root)

# Find the critical path (longest path through the graph)
critical_path = []
if tasks:
    # Create a topological sort
    topo_sort = list(nx.topological_sort(G))
    
    # Dictionary to store longest path lengths to each node
    longest_paths = {node: 0 for node in G.nodes()}
    predecessors = {node: None for node in G.nodes()}
    
    # Compute longest paths
    for node in topo_sort:
        for successor in G.successors(node):
            if longest_paths[node] + 1 > longest_paths[successor]:
                longest_paths[successor] = longest_paths[node] + 1
                predecessors[successor] = node
    
    # Find node with maximum path length
    end_node = max(longest_paths.items(), key=lambda x: x[1])[0]
    
    # Reconstruct the critical path
    node = end_node
    while node is not None:
        critical_path.append(node)
        node = predecessors[node]
    
    critical_path.reverse()

# Create a hierarchical layout based on node_layers
# Widen the horizontal spacing to make arrows more distinct
pos = {}
for node, layer in node_layers.items():
    pos[node] = (layer * 2.5, 0)  # Initial position with wider horizontal spacing

# Get all nodes at each layer
layer_nodes = {}
for node, layer in node_layers.items():
    if layer not in layer_nodes:
        layer_nodes[layer] = []
    layer_nodes[layer].append(node)

# Sort layers
sorted_layers = sorted(layer_nodes.keys())

# Position nodes in each layer
y_spacing = 1.5  # Increased vertical spacing between nodes
for layer in sorted_layers:
    nodes = layer_nodes[layer]
    
    # Sort nodes within layer to minimize edge crossings
    nodes.sort()
    
    # Assign y positions
    total_height = (len(nodes) - 1) * y_spacing
    for i, node in enumerate(nodes):
        y_pos = -total_height/2 + i * y_spacing
        pos[node] = (layer * 2.5, y_pos)  # Wider horizontal spacing

# Set up the figure with a professional look
plt.figure(figsize=(16, 10))
plt.title("Task Dependency Graph for Attention Layer Processing", fontsize=16, fontweight='bold')

# Define better colors for computation and communication tasks
computation_color = "#009900"  # Green for computation tasks
communication_color = "#0099CC"  # Blue for communication tasks
critical_path_color = "#CC0000"  # Red for critical path

# Assign colors to nodes
node_colors = []
for node in G.nodes():
    if node in critical_path:
        # Highlight critical path
        node_colors.append(critical_path_color)
    elif G.nodes[node]['type'] == 'Computation':
        node_colors.append(computation_color)
    else:
        node_colors.append(communication_color)

# Draw curved edges with custom style to avoid overlap
edge_list = list(G.edges())
edge_colors = []
edge_styles = []
edge_widths = []

for u, v in edge_list:
    if u in critical_path and v in critical_path:
        edge_colors.append(critical_path_color)
        edge_styles.append('solid')
        edge_widths.append(2.0)
    else:
        edge_colors.append('gray')
        edge_styles.append('solid')
        edge_widths.append(1.5)

# FIRST draw all edges before drawing nodes
for i, (u, v) in enumerate(edge_list):
    # Calculate curved path for the edge
    # The higher the rad, the more curved the edge
    rad = 0.3
    
    # Create more curvature for some edges to differentiate
    # Adjust curvature based on source and target nodes
    if u % 2 == 0:  # Even source nodes
        rad = 0.2
    else:  # Odd source nodes
        rad = 0.3
        
    # Special case for the problem edges (9,10 to 11)
    if u == 9 and v == 11:
        rad = 0.4  # Higher curvature for this specific edge
    elif u == 10 and v == 11:
        rad = -0.4  # Negative curvature for opposite direction
    
    # Draw the curved edge
    nx.draw_networkx_edges(G, 
                          pos,
                          edgelist=[(u, v)],
                          width=edge_widths[i],
                          edge_color=[edge_colors[i]],
                          style=edge_styles[i],
                          arrows=True,  # Ensure arrows are drawn
                          arrowstyle='-|>',  # Simple arrow style
                          arrowsize=30,  # Very large arrows
                          connectionstyle=f'arc3, rad={rad}',
                          node_size=800)  # Smaller nodes for arrow visibility

# THEN draw nodes on top of edges, but with a slightly smaller size than used for edge calculation
nx.draw_networkx_nodes(G, 
                      pos, 
                      node_color=node_colors,
                      node_size=1000,  # Smaller size so arrows remain visible
                      alpha=1.0,  # Fully opaque
                      edgecolors='black',  # White border for better contrast
                      linewidths=2)  # Border width

# Use a slightly higher font size for better readability
nx.draw_networkx_labels(G, 
                       pos, 
                       labels={node: G.nodes[node]['label'] for node in G.nodes()},
                       font_size=12,  # Slightly larger font
                       font_color='white',
                       font_weight='bold')

# Add a legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=computation_color, markersize=10, label='Computation Task'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=communication_color, markersize=10, label='Communication Task'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=critical_path_color, markersize=10, label='Critical Path')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Set clean background and axis appearance
plt.grid(False)
plt.axis('off')

# Add a caption with publication-style formatting
caption = (
    "Figure 1: Task dependency graph for a transformer attention layer implementation on NoC. "
    "Tasks flow from left to right, with distinct curved edges showing individual dependencies. "
    "Green nodes represent computation tasks, blue nodes represent communication tasks, "
    "and the critical path is highlighted in red."
)
plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)

# Add publication-friendly metadata
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for caption
plt.savefig('task_dependency_graph.png', dpi=300, bbox_inches='tight')

# Display the graph
plt.show()

print("Task dependency graph created and saved as 'task_dependency_graph.png'")
print(f"Critical path identified: {' → '.join(map(str, critical_path))}")

# Print all task dependencies to verify correct parsing
print("\nTask Dependencies:")
for task in tasks:
    if task['dependencies']:
        dep_str = ', '.join(map(str, task['dependencies']))
        print(f"Task {task['task_id']} depends on: {dep_str}")