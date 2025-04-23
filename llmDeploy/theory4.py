import pandas as pd
import numpy as np
from collections import defaultdict

# Parse the task data
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

# Function to extract PE coordinates from string like "(x, y) (name)"
def extract_pe_coords(pe_str):
    if 'external' in pe_str:
        return (0, 0)  # External source is treated as (0, 0)
    
    # Extract coordinates from format "(x, y) (name)"
    coords_part = pe_str.split('(')[1].split(')')[0]
    x, y = map(int, coords_part.split(','))
    return (x, y)

# Parse the data into a structured format
lines = data.strip().split('\n')
header = lines[0].split()
tasks = []

for line in lines[1:]:
    parts = line.split()
    task_id = int(parts[1])
    src_pe = ' '.join(parts[2:5])
    dest_pe = ' '.join(parts[5:8])
    bytes_count = int(parts[8])
    
    if parts[9] == "None":
        wait_ids = []
    else:
        wait_ids = [int(id_str.rstrip(',')) for id_str in parts[9:] if id_str.rstrip(',').isdigit()]
    
    # Extract the PE coordinates
    src_coords = extract_pe_coords(src_pe)
    dest_coords = extract_pe_coords(dest_pe)
    
    tasks.append({
        'task_id': task_id,
        'src_pe': src_pe,
        'dest_pe': dest_pe,
        'src_coords': src_coords,
        'dest_coords': dest_coords,
        'bytes': bytes_count,
        'wait_ids': wait_ids,
        'is_computation': src_pe == dest_pe and 'external' not in src_pe,
        'start_time': None,
        'end_time': None
    })

# Calculate Manhattan distance (number of hops) between two PEs
def calculate_hops(src_coords, dest_coords):
    return abs(src_coords[0] - dest_coords[0]) + abs(src_coords[1] - dest_coords[1])

# Create a function to simulate task execution with input/output port constraints and network hops
def simulate_execution(tasks, channel_bw=32, cycles_per_hop=1):
    # Create dictionaries to track PE busy times for input and output ports
    pe_input_busy_until = defaultdict(int)   # When a PE's input port is free
    pe_output_busy_until = defaultdict(int)  # When a PE's output port is free
    pe_compute_busy_until = defaultdict(int) # When a PE's computation unit is free
    
    # Track link utilization in the network (simplified model)
    # For each possible pair of adjacent PEs, track when the link is free
    network_links_busy_until = defaultdict(int)
    
    task_completion_times = {}
    
    # Sort tasks by ID to ensure we process them in order
    tasks_sorted = sorted(tasks, key=lambda x: x['task_id'])
    
    # Function to check if a task is eligible for scheduling
    def is_task_eligible(task, current_time):
        # Check if all dependencies are met
        for dep_id in task['wait_ids']:
            if dep_id not in task_completion_times or task_completion_times[dep_id] > current_time:
                return False
        return True
    
    # Function to reserve a path in the network
    def reserve_path(src_coords, dest_coords, start_time, end_time):
        # Calculate the path (simplified as Manhattan path)
        x1, y1 = src_coords
        x2, y2 = dest_coords
        
        # Reserve horizontal links
        x_dir = 1 if x2 > x1 else -1 if x2 < x1 else 0
        if x_dir != 0:  # Only if there's horizontal movement
            for x in range(x1, x2, x_dir):
                link = ((x, y1), (x + x_dir, y1))
                network_links_busy_until[link] = max(network_links_busy_until[link], end_time)
        
        # Reserve vertical links
        y_dir = 1 if y2 > y1 else -1 if y2 < y1 else 0
        if y_dir != 0:  # Only if there's vertical movement
            for y in range(y1, y2, y_dir):
                link = ((x2 if x_dir != 0 else x1, y), (x2 if x_dir != 0 else x1, y + y_dir))
                network_links_busy_until[link] = max(network_links_busy_until[link], end_time)
    
    # Function to check if a path is free
    def is_path_free(src_coords, dest_coords, current_time):
        # Calculate the path
        x1, y1 = src_coords
        x2, y2 = dest_coords
        
        # Check horizontal links
        x_dir = 1 if x2 > x1 else -1 if x2 < x1 else 0
        if x_dir != 0:  # Only if there's horizontal movement
            for x in range(x1, x2, x_dir):
                link = ((x, y1), (x + x_dir, y1))
                if network_links_busy_until[link] > current_time:
                    return False
        
        # Check vertical links
        y_dir = 1 if y2 > y1 else -1 if y2 < y1 else 0
        if y_dir != 0:  # Only if there's vertical movement
            for y in range(y1, y2, y_dir):
                link = ((x2 if x_dir != 0 else x1, y), (x2 if x_dir != 0 else x1, y + y_dir))
                if network_links_busy_until[link] > current_time:
                    return False
        
        return True
    
    # Schedule tasks
    scheduled_tasks = []
    remaining_tasks = tasks_sorted.copy()
    current_time = 0
    
    while remaining_tasks:
        progress_made = False
        
        # Find tasks that can be scheduled at the current time
        tasks_to_remove = []
        
        for task in remaining_tasks:
            if is_task_eligible(task, current_time):
                task_id = task['task_id']
                src_pe = task['src_pe']
                dest_pe = task['dest_pe']
                src_coords = task['src_coords']
                dest_coords = task['dest_coords']
                bytes_count = task['bytes']
                
                if task['is_computation']:
                    # Computation task
                    pe = src_pe  # src_pe and dest_pe are the same
                    
                    # Check if the PE is available for computation
                    if pe_compute_busy_until[pe] <= current_time:
                        start_time = current_time
                        # Computation tasks take 1 clock cycle
                        end_time = start_time + 1
                        pe_compute_busy_until[pe] = end_time
                        
                        task['start_time'] = start_time
                        task['end_time'] = end_time
                        task_completion_times[task_id] = end_time
                        scheduled_tasks.append(task)
                        tasks_to_remove.append(task)
                        progress_made = True
                else:
                    # Communication task
                    # Skip if source is busy sending or destination is busy receiving
                    if 'external' not in src_pe and pe_output_busy_until[src_pe] > current_time:
                        continue
                    if pe_input_busy_until[dest_pe] > current_time:
                        continue
                    
                    # Skip self-communication or check if the path through the network is free
                    if src_coords != dest_coords and not is_path_free(src_coords, dest_coords, current_time):
                        continue
                    
                    start_time = current_time
                    
                    # Calculate number of hops
                    num_hops = calculate_hops(src_coords, dest_coords)
                    
                    # Calculate transmission time in clock cycles
                    # Base transmission time + additional time for hops
                    base_transmission_cycles = (bytes_count + channel_bw - 1) // channel_bw  # Ceiling division
                    hop_delay = num_hops * cycles_per_hop
                    
                    # Total time is base transmission time plus hop delay
                    transmission_cycles = base_transmission_cycles + hop_delay
                    end_time = start_time + transmission_cycles
                    
                    # Update busy time for source output and destination input ports
                    if 'external' not in src_pe:
                        pe_output_busy_until[src_pe] = end_time
                    pe_input_busy_until[dest_pe] = end_time
                    
                    # Reserve the path in the network
                    if src_coords != dest_coords:
                        reserve_path(src_coords, dest_coords, start_time, end_time)
                    
                    task['start_time'] = start_time
                    task['end_time'] = end_time
                    task_completion_times[task_id] = end_time
                    scheduled_tasks.append(task)
                    tasks_to_remove.append(task)
                    progress_made = True
        
        # Remove scheduled tasks from remaining tasks
        for task in tasks_to_remove:
            remaining_tasks.remove(task)
        
        # If no progress was made, advance the time
        if not progress_made:
            # Find the next time when a resource becomes available
            next_time = float('inf')
            
            # Check PE resources
            for time in list(pe_input_busy_until.values()) + list(pe_output_busy_until.values()) + list(pe_compute_busy_until.values()):
                if time > current_time and time < next_time:
                    next_time = time
            
            # Check network links
            for time in network_links_busy_until.values():
                if time > current_time and time < next_time:
                    next_time = time
            
            if next_time == float('inf'):
                # This should not happen with a valid task graph
                print("Error: Cannot make progress. Possible circular dependency.")
                break
            
            current_time = next_time
        
    return scheduled_tasks, max(task_completion_times.values()) if task_completion_times else 0

# Run the simulation
# Try different scenarios to show the impact of hop delay
scenarios = [
    {"name": "No hop delay", "cycles_per_hop": 0},
    {"name": "1 cycle per hop", "cycles_per_hop": 1},
    {"name": "2 cycles per hop", "cycles_per_hop": 2},
]

print("Simulation Results:\n")

for scenario in scenarios:
    updated_tasks, total_cycles = simulate_execution(tasks, channel_bw=32, cycles_per_hop=scenario["cycles_per_hop"])
    
    print(f"\n{scenario['name']} - Total execution time: {total_cycles} clock cycles\n")
    print(f"{'Task ID':<8}{'Start':<8}{'End':<8}{'Duration':<10}{'Bytes':<8}{'Hops':<6}{'Wait IDs':<15}Description")
    print("-" * 105)
    
    for task in sorted(updated_tasks, key=lambda x: x['task_id']):
        duration = task['end_time'] - task['start_time']
        wait_ids_str = ', '.join(map(str, task['wait_ids'])) if task['wait_ids'] else "None"
        
        # Calculate hops for display
        hops = calculate_hops(task['src_coords'], task['dest_coords']) if not task['is_computation'] else 0
        
        print(f"{task['task_id']:<8}{task['start_time']:<8}{task['end_time']:<8}{duration:<10}{task['bytes']:<8}{hops:<6}{wait_ids_str:<15}{task['src_pe']} -> {task['dest_pe']}")

# Additionally, analyze impact of network topology
print("\n\nTask Analysis by Number of Hops:")
print(f"{'Hops':<6}{'# Tasks':<8}{'Total Bytes':<12}{'Avg Duration'}")
print("-" * 40)

hop_stats = defaultdict(lambda: {"count": 0, "bytes": 0, "total_duration": 0})

for task in updated_tasks:
    if not task['is_computation']:
        hops = calculate_hops(task['src_coords'], task['dest_coords'])
        duration = task['end_time'] - task['start_time']
        
        hop_stats[hops]["count"] += 1
        hop_stats[hops]["bytes"] += task['bytes']
        hop_stats[hops]["total_duration"] += duration

for hops, stats in sorted(hop_stats.items()):
    avg_duration = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
    print(f"{hops:<6}{stats['count']:<8}{stats['bytes']:<12}{avg_duration:.2f}")