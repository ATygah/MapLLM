import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd

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

# Parse the data into a structured format
lines = data.strip().split('\n')
header = lines[0].split()
tasks = []

for line in lines[1:]:
    parts = line.split()
    task_id = int(parts[1])
    src_pe = parts[2:5]
    dest_pe = parts[5:8]
    bytes_count = int(parts[8])
    
    if parts[9] == "None":
        wait_ids = []
    else:
        wait_ids = [int(id_str.rstrip(',')) for id_str in parts[9:] if id_str.rstrip(',').isdigit()]
    
    # Extract the src_pe and dest_pe coordinates
    src_pe_str = ' '.join(src_pe)
    dest_pe_str = ' '.join(dest_pe)
    
    tasks.append({
        'task_id': task_id,
        'src_pe': src_pe_str,
        'dest_pe': dest_pe_str,
        'bytes': bytes_count,
        'wait_ids': wait_ids,
        'is_computation': src_pe_str == dest_pe_str and 'external' not in src_pe_str,
        'start_time': None,
        'end_time': None
    })

# Create a function to simulate task execution with input/output port constraints
def simulate_execution(tasks, channel_bw=32):
    # Create dictionaries to track PE busy times for input and output ports
    pe_input_busy_until = defaultdict(int)   # When a PE's input port is free
    pe_output_busy_until = defaultdict(int)  # When a PE's output port is free
    pe_compute_busy_until = defaultdict(int) # When a PE's computation unit is free
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
                    
                    start_time = current_time
                    
                    # Calculate transmission time in clock cycles
                    transmission_cycles = (bytes_count + channel_bw - 1) // channel_bw  # Ceiling division
                    end_time = start_time + transmission_cycles
                    
                    # Update busy time for source output and destination input ports
                    if 'external' not in src_pe:
                        pe_output_busy_until[src_pe] = end_time
                    pe_input_busy_until[dest_pe] = end_time
                    
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
            for time in list(pe_input_busy_until.values()) + list(pe_output_busy_until.values()) + list(pe_compute_busy_until.values()):
                if time > current_time and time < next_time:
                    next_time = time
            
            if next_time == float('inf'):
                # This should not happen with a valid task graph
                print("Error: Cannot make progress. Possible circular dependency.")
                break
            
            current_time = next_time
        
    return scheduled_tasks, max(task_completion_times.values()) if task_completion_times else 0

# Run the simulation
updated_tasks, total_cycles = simulate_execution(tasks)

# Create visual representation similar to the examples
def create_visual_timeline(tasks, channel_bw=32):
    # Create dictionaries to track PE busy times for input and output ports
    pe_input_busy_until = defaultdict(int)   # When a PE's input port is free
    pe_output_busy_until = defaultdict(int)  # When a PE's output port is free
    pe_compute_busy_until = defaultdict(int) # When a PE's computation unit is free
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
                    
                    start_time = current_time
                    
                    # Calculate transmission time in clock cycles
                    transmission_cycles = (bytes_count + channel_bw - 1) // channel_bw  # Ceiling division
                    end_time = start_time + transmission_cycles
                    
                    # Update busy time for source output and destination input ports
                    if 'external' not in src_pe:
                        pe_output_busy_until[src_pe] = end_time
                    pe_input_busy_until[dest_pe] = end_time
                    
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
            for time in list(pe_input_busy_until.values()) + list(pe_output_busy_until.values()) + list(pe_compute_busy_until.values()):
                if time > current_time and time < next_time:
                    next_time = time
            
            if next_time == float('inf'):
                # This should not happen with a valid task graph
                print("Error: Cannot make progress. Possible circular dependency.")
                break
            
            current_time = next_time
        
    return scheduled_tasks, max(task_completion_times.values()) if task_completion_times else 0

# Create a function to create a visual timeline
def create_visual_timeline(tasks, max_cycles):
    # Get all unique PEs
    pe_list = set()
    for task in tasks:
        if 'external' not in task['src_pe']:
            pe_list.add(task['src_pe'])
        if 'external' not in task['dest_pe']:
            pe_list.add(task['dest_pe'])
    
    pe_list = sorted(list(pe_list))
    
    # Create a timeline for each PE
    pe_timeline = {}
    for pe in pe_list:
        pe_timeline[pe] = [None] * (max_cycles + 1)
    
    # Populate the timelines with task information
    # For each task, we'll track:
    # 1. When a PE is computing (computation tasks)
    # 2. When a PE is receiving data (destination of network tasks)
    # 3. When a PE is sending data (source of network tasks)
    
    # First, mark the computation tasks
    for task in tasks:
        if task['is_computation']:
            pe = task['src_pe']
            for t in range(task['start_time'], task['end_time']):
                pe_timeline[pe][t] = {'type': 'compute', 'task_id': task['task_id']}
    
    # Then mark the network tasks (data receiving)
    for task in tasks:
        if not task['is_computation'] and 'external' not in task['dest_pe']:
            pe = task['dest_pe']
            for t in range(task['start_time'], task['end_time']):
                if pe_timeline[pe][t] is None:  # Don't overwrite computation tasks
                    pe_timeline[pe][t] = {'type': 'receive', 'task_id': task['task_id']}
    
    # Then mark the network tasks (data sending)
    for task in tasks:
        if not task['is_computation'] and 'external' not in task['src_pe']:
            pe = task['src_pe']
            for t in range(task['start_time'], task['end_time']):
                if pe_timeline[pe][t] is None:  # Don't overwrite computation or receiving
                    pe_timeline[pe][t] = {'type': 'send', 'task_id': task['task_id']}
    
    return pe_list, pe_timeline

# Create color mapping for tasks
def get_task_colors():
    # Create a colormap for tasks
    colors = plt.cm.get_cmap('tab20', 20)
    task_colors = {}
    for i in range(1, 20):
        task_colors[i] = colors(i-1)
    return task_colors

# Create the visualization
def plot_timeline(pe_list, pe_timeline, max_cycles, task_colors):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for different task types
    type_colors = {
        'compute': 'green',
        'receive': 'blue',
        'send': 'orange',
        None: 'lightgray'
    }
    
    # Create the grid
    for i, pe in enumerate(pe_list):
        # Draw PE labels
        ax.text(-1, i+0.5, pe, ha='right', va='center', fontsize=10)
        
        # Draw timeline for this PE
        for t in range(max_cycles + 1):
            cell_info = pe_timeline[pe][t]
            
            if cell_info is None:
                # Idle time
                rect = plt.Rectangle((t, i), 1, 1, facecolor='lightgray', edgecolor='white', alpha=0.5)
            else:
                # Active time
                task_id = cell_info['task_id']
                activity_type = cell_info['type']
                
                # Use task color but vary by activity type
                base_color = task_colors[task_id]
                
                if activity_type == 'compute':
                    # Computation is darker
                    facecolor = base_color
                    text_color = 'white'
                    alpha = 1.0
                elif activity_type == 'receive':
                    # Receiving is medium
                    facecolor = base_color
                    text_color = 'white'
                    alpha = 0.8
                else:  # send
                    # Sending is lighter
                    facecolor = base_color
                    text_color = 'black'
                    alpha = 0.6
                
                rect = plt.Rectangle((t, i), 1, 1, facecolor=facecolor, edgecolor='white', alpha=alpha)
                
                # Add task ID text
                ax.text(t+0.5, i+0.5, str(task_id), ha='center', va='center', fontsize=8, 
                        color=text_color, fontweight='bold')
            
            ax.add_patch(rect)
    
    # Configure the axes
    ax.set_xlim(-1, max_cycles + 1)
    ax.set_ylim(0, len(pe_list))
    ax.set_xlabel('Clock Cycles')
    ax.set_title('NoC Task Execution Timeline')
    
    # Add legend for task types
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='white', label='Computation'),
        plt.Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='white', label='Receiving Data'),
        plt.Rectangle((0, 0), 1, 1, facecolor='orange', edgecolor='white', label='Sending Data'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='white', label='Idle')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    ax.grid(True, linestyle='-', alpha=0.3, color='black')
    ax.set_xticks(np.arange(0, max_cycles + 1, 1))
    ax.set_yticks(np.arange(0.5, len(pe_list) + 0.5, 1))
    ax.set_yticklabels([])
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig, ax

# Create a second visualization showing the task flow
def plot_task_flow(tasks, max_cycles):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a colormap for tasks
    colors = plt.cm.get_cmap('tab20', 20)
    
    # Group tasks by their ID
    for i, task in enumerate(sorted(tasks, key=lambda x: x['task_id'])):
        task_id = task['task_id']
        start = task['start_time']
        end = task['end_time']
        src = task['src_pe'] if 'external' not in task['src_pe'] else 'External'
        dest = task['dest_pe']
        
        # Determine color based on task type
        if task['is_computation']:
            color = 'green'
            label = f"Task {task_id}: Compute at {dest}"
        else:
            color = colors(task_id % 20)
            label = f"Task {task_id}: {src} → {dest}"
        
        # Draw the task bar
        rect = plt.Rectangle((start, i+0.1), end-start, 0.8, facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        
        # Add task label
        ax.text(start + (end-start)/2, i+0.5, str(task_id), ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        
        # Add task description
        ax.text(end + 0.5, i+0.5, label, ha='left', va='center', fontsize=8)
    
    # Configure the axes
    ax.set_xlim(0, max_cycles + 5)  # Add some space for labels
    ax.set_ylim(0, len(tasks))
    ax.set_xlabel('Clock Cycles')
    ax.set_ylabel('Task')
    ax.set_title('Task Execution Flow')
    
    # Add grid lines
    ax.grid(True, axis='both', linestyle='-', alpha=0.3, color='black')
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(0, max_cycles + 1, 1))
    
    # Set y-ticks to task IDs
    ax.set_yticks([i+0.5 for i in range(len(tasks))])
    ax.set_yticklabels([f"Task {task['task_id']}" for task in sorted(tasks, key=lambda x: x['task_id'])])
    
    plt.tight_layout()
    return fig, ax

# Create a Gantt chart showing the critical path
def plot_critical_path(tasks, max_cycles):
    # Calculate earliest finish time for each task
    earliest_finish = {}
    
    # Function to calculate earliest finish time for a task
    def get_earliest_finish(task_id):
        if task_id in earliest_finish:
            return earliest_finish[task_id]
        
        task = next(t for t in tasks if t['task_id'] == task_id)
        if not task['wait_ids']:
            earliest_finish[task_id] = task['end_time']
            return task['end_time']
        
        # Calculate earliest finish time based on dependencies
        max_pred_finish = max([get_earliest_finish(pred_id) for pred_id in task['wait_ids']])
        earliest_finish[task_id] = max(max_pred_finish, task['start_time']) + (task['end_time'] - task['start_time'])
        return earliest_finish[task_id]
    
    # Calculate earliest finish time for all tasks
    for task in tasks:
        get_earliest_finish(task['task_id'])
    
    # Find critical path (tasks that if delayed, delay the entire schedule)
    final_tasks = [t for t in tasks if t['end_time'] == max_cycles]
    critical_task_ids = []
    
    if final_tasks:
        current_task = final_tasks[0]
        while current_task:
            critical_task_ids.append(current_task['task_id'])
            
            # Find the predecessor that finishes latest
            if not current_task['wait_ids']:
                current_task = None
            else:
                pred_id = max(current_task['wait_ids'], key=lambda pid: earliest_finish.get(pid, 0))
                current_task = next((t for t in tasks if t['task_id'] == pred_id), None)
    
    # Create the critical path visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw PE labels and execution blocks
    pe_set = set()
    for task in tasks:
        if 'external' not in task['src_pe']:
            pe_set.add(task['src_pe'])
        if 'external' not in task['dest_pe']:
            pe_set.add(task['dest_pe'])
    
    pe_list = sorted(list(pe_set))
    
    # Plot all tasks
    for i, pe in enumerate(pe_list):
        # Draw PE label
        ax.text(-1, i+0.5, pe, ha='right', va='center', fontsize=10)
        
        # Get tasks for this PE
        pe_tasks = [t for t in tasks if 
                    (t['dest_pe'] == pe or 
                     (t['src_pe'] == pe and 'external' not in t['src_pe']))]
        
        for task in pe_tasks:
            task_id = task['task_id']
            start = task['start_time']
            end = task['end_time']
            
            # Determine if this task is on the critical path
            is_critical = task_id in critical_task_ids
            
            # Determine color and style based on task type and critical path
            if task['is_computation']:
                color = 'darkgreen' if is_critical else 'green'
                alpha = 1.0 if is_critical else 0.7
                ec = 'red' if is_critical else 'black'
                lw = 2 if is_critical else 1
            elif task['dest_pe'] == pe and task['src_pe'] != pe:
                # Receiving data
                color = 'darkblue' if is_critical else 'blue'
                alpha = 1.0 if is_critical else 0.7
                ec = 'red' if is_critical else 'black'
                lw = 2 if is_critical else 1
            elif task['src_pe'] == pe and task['dest_pe'] != pe:
                # Sending data
                color = 'darkorange' if is_critical else 'orange'
                alpha = 1.0 if is_critical else 0.7
                ec = 'red' if is_critical else 'black'
                lw = 2 if is_critical else 1
            
            # Draw the task block
            rect = plt.Rectangle((start, i+0.1), end-start, 0.8, 
                                facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha)
            ax.add_patch(rect)
            
            # Add task ID label
            ax.text(start + (end-start)/2, i+0.5, str(task_id), 
                   ha='center', va='center', fontsize=9, 
                   color='white', fontweight='bold')
    
    # Configure the axes
    ax.set_xlim(-1, max_cycles + 1)
    ax.set_ylim(0, len(pe_list))
    ax.set_xlabel('Clock Cycles')
    ax.set_title('NoC Task Execution Timeline with Critical Path')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black', label='Computation'),
        plt.Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='black', label='Receiving Data'),
        plt.Rectangle((0, 0), 1, 1, facecolor='orange', edgecolor='black', label='Sending Data'),
        plt.Rectangle((0, 0), 1, 1, facecolor='darkgreen', edgecolor='red', linewidth=2, label='Critical Path')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    ax.grid(True, linestyle='-', alpha=0.3, color='black')
    ax.set_xticks(np.arange(0, max_cycles + 1, 1))
    ax.set_yticks(np.arange(0.5, len(pe_list) + 0.5, 1))
    ax.set_yticklabels([])
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig, ax

# Generate the visualizations
pe_list, pe_timeline = create_visual_timeline(updated_tasks, total_cycles)
task_colors = get_task_colors()

# Create the visualization
fig1, ax1 = plot_timeline(pe_list, pe_timeline, total_cycles, task_colors)
fig1.savefig('plots/tasks/noc_timeline.png', dpi=150, bbox_inches='tight')

# Create the task flow visualization
fig2, ax2 = plot_task_flow(updated_tasks, total_cycles)
fig2.savefig('plots/tasks/task_flow.png', dpi=150, bbox_inches='tight')

# Create the critical path visualization
fig3, ax3 = plot_critical_path(updated_tasks, total_cycles)
fig3.savefig('plots/tasks/critical_path.png', dpi=150, bbox_inches='tight')

print(f"Total execution time: {total_cycles} clock cycles")
print("Visualizations saved in 'plots/tasks/' directory")

# Display summary of results
print("\nTask Execution Summary:")
print(f"{'Task ID':<8}{'Start':<8}{'End':<8}{'Duration':<10}{'Type':<15}{'Source PE':<20}{'Destination PE':<20}")
print("-" * 90)

for task in sorted(updated_tasks, key=lambda x: x['task_id']):
    task_id = task['task_id']
    start = task['start_time']
    end = task['end_time']
    duration = end - start
    task_type = "Computation" if task['is_computation'] else "Communication"
    src_pe = task['src_pe']
    dest_pe = task['dest_pe']
    
    print(f"{task_id:<8}{start:<8}{end:<8}{duration:<10}{task_type:<15}{src_pe:<20}{dest_pe:<20}")