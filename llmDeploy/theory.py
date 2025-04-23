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
6        7            (1, 0) (q_proj)  (2, 2) (attention_head_0)     48        2                               (1, 0) sends (0:4, 0:6) of input to PE(4, 0)
7        8            (2, 0) (k_proj)  (0, 5) (attention_head_0)     48        4                    (2, 0) sends (0:6, 0:4) of input Transposed to PE(4, 0)
8        9            (1, 0) (q_proj)  (5, 0) (attention_head_1)     48        2                               (1, 0) sends (0:4, 0:6) of input to PE(5, 0)
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

# Function to read and process demo.tsv file
def read_task_data_from_tsv(file_path):
    """
    Read task data from a TSV file and process it into a structured format.
    
    Args:
        file_path (str): Path to the TSV file
        
    Returns:
        list: List of dictionaries containing task information
    """
    import pandas as pd
    
    # Read the TSV file
    df = pd.read_csv(file_path, sep='\t', dtype={'wait_ids': str})
    
    # Process the data into the same format as the hardcoded data
    tasks = []
    
    count = 0
    for _, row in df.iterrows():
        # Parse wait_ids
        print(f"Iteration {count}")
        count += 1
        if row['wait_ids'] == 'None':
            wait_ids = []
        else:
            wait_ids = [int(id_str.strip()) for id_str in str(row['wait_ids']).split(',') if id_str.strip().isdigit()]
        
        # Create task entry
        tasks.append({
            'task_id': int(row['task_id']),
            'src_pe': row['src_pe'],
            'dest_pe': row['dest_pe'],
            'bytes': int(row['bytes']),
            'wait_ids': wait_ids,
            'is_computation': row['src_pe'] == row['dest_pe'] and 'external' not in row['src_pe'],
            'start_time': None,
            'end_time': None
        })
    
    return tasks

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
    debug_print = False  # Set to False for cleaner output
    #count = 0

    while remaining_tasks:
        #print(f"Simulation Iteration {count}")
        #count += 1
        progress_made = False
        
        if debug_print:
            print(f"Current time: {current_time}")
            print(f"Remaining tasks: {[t['task_id'] for t in remaining_tasks]}")
            print(f"PE output busy until: {dict(pe_output_busy_until)}")
            print(f"PE input busy until: {dict(pe_input_busy_until)}")
            print(f"Task completion times: {task_completion_times}")
            print()
        
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
                    # Skip if source output port is busy or destination input port is busy
                    if 'external' not in src_pe and pe_output_busy_until[src_pe] > current_time:
                        if debug_print:
                            print(f"Task {task_id} source port busy: {src_pe} until {pe_output_busy_until[src_pe]}")
                        continue
                    if pe_input_busy_until[dest_pe] > current_time:
                        if debug_print:
                            print(f"Task {task_id} destination port busy: {dest_pe} until {pe_input_busy_until[dest_pe]}")
                        continue
                    
                    start_time = current_time
                    
                    # Extract coordinates for Manhattan distance
                    src_coords = None
                    dest_coords = None
                    if 'external' not in src_pe:
                        src_coord_part = src_pe.split('(')[1].split(')')[0]
                        src_coords = tuple(map(int, src_coord_part.split(',')))
                    dest_coord_part = dest_pe.split('(')[1].split(')')[0]
                    dest_coords = tuple(map(int, dest_coord_part.split(',')))
                    
                    # Calculate Manhattan distance
                    manhattan_distance = 0
                    if src_coords and dest_coords:
                        manhattan_distance = abs(src_coords[0] - dest_coords[0]) + abs(src_coords[1] - dest_coords[1])
                    
                    # Calculate packet count and send time
                    packet_count = (bytes_count + channel_bw - 1) // channel_bw  # Ceiling division
                    source_send_cycles = packet_count  # Time to send all packets from source
                    
                    # Total transmission time includes network latency
                    #This is not the accurate way to calculate the transmission time but is a decent approximation
                    transmission_cycles = source_send_cycles + manhattan_distance  # Send time + hops
                    
                    # Set times
                    source_free_time = start_time + source_send_cycles  # When source is free
                    end_time = start_time + transmission_cycles         # When destination receives
                    
                    # Update busy times
                    if 'external' not in src_pe:
                        pe_output_busy_until[src_pe] = source_free_time  # Source output port is free
                    pe_input_busy_until[dest_pe] = end_time              # Destination busy until receipt
                    
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
print("Starting simulation...\n")
tasks_from_tsv = read_task_data_from_tsv('llmDeploy/final_traces/gpt2_small/paper_disc/gpt2_small_traffic_enhanced_full_compact_opt1_20250412_035102.tsv')
updated_tasks, total_cycles = simulate_execution(tasks_from_tsv)
#updated_tasks, total_cycles = simulate_execution(tasks)

# Create a nice output
print(f"Total execution time: {total_cycles} clock cycles\n")
print("Task Schedule:")
print(f"{'Task ID':<8}{'Start':<8}{'End':<8}{'Duration':<10}{'Bytes':<8}{'Wait IDs':<15}Description")
print("-" * 100)

for task in sorted(updated_tasks, key=lambda x: x['task_id']):
    duration = task['end_time'] - task['start_time']
    wait_ids_str = ', '.join(map(str, task['wait_ids'])) if task['wait_ids'] else "None"
    #print(f"{task['task_id']:<8}{task['start_time']:<8}{task['end_time']:<8}{duration:<10}{task['bytes']:<8}{wait_ids_str:<15}{task['src_pe']} -> {task['dest_pe']}")
    if not task['is_computation'] and 'external' not in task['src_pe']:
        packet_count = (task['bytes'] + 32 - 1) // 32  # Use channel_bw
    #    print(f"    Source sending packets: {task['start_time']} to {task['start_time'] + packet_count} (packets: {packet_count})")

# Function to detect conflicts between tasks in the NoC
def detect_transmission_conflicts(tasks):
    # Filter out computation tasks and tasks from external sources
    comm_tasks = [t for t in tasks if not t['is_computation'] and 'external' not in t['src_pe']]
    
    conflicts = []
    count = 0
    # Check each pair of communication tasks
    for i, task1 in enumerate(comm_tasks):
        for j, task2 in enumerate(comm_tasks):
            print(f"Conflicts detection iteration: {count}")
            count += 1
            if i >= j:  # Skip comparing a task with itself or repeating comparisons
                continue
                
            # Extract task information
            task1_id = task1['task_id']
            task2_id = task2['task_id']
            
            # Extract coordinates
            src1_coords = tuple(map(int, task1['src_pe'].split('(')[1].split(')')[0].split(',')))
            dest1_coords = tuple(map(int, task1['dest_pe'].split('(')[1].split(')')[0].split(',')))
            src2_coords = tuple(map(int, task2['src_pe'].split('(')[1].split(')')[0].split(',')))
            dest2_coords = tuple(map(int, task2['dest_pe'].split('(')[1].split(')')[0].split(',')))
            
            # Check for time overlap
            # Calculate actual transmission periods (when packets are in the network)
            channel_bw = 32
            
            # For task1
            task1_packet_count = (task1['bytes'] + channel_bw - 1) // channel_bw
            task1_start_sending = task1['start_time']
            task1_end_sending = task1_start_sending + task1_packet_count
            task1_end_receiving = task1['end_time']
            
            # For task2
            task2_packet_count = (task2['bytes'] + channel_bw - 1) // channel_bw
            task2_start_sending = task2['start_time']
            task2_end_sending = task2_start_sending + task2_packet_count
            task2_end_receiving = task2['end_time']
            
            # Check if there's any time overlap in network occupation
            # We consider overlap if any packet of task1 is in the network at the same time as any packet of task2
            time_overlap = (task1_start_sending < task2_end_receiving and 
                           task2_start_sending < task1_end_receiving)
                           
            if not time_overlap:
                continue
                
            # Check for path intersection using XY routing
            
            # In XY routing:
            # - First move in X direction from source to dest_x
            # - Then move in Y direction to reach destination
            
            # For task1 - XY routing creates two segments:
            # Segment 1: (src1_x, src1_y) -> (dest1_x, src1_y)  # X movement
            # Segment 2: (dest1_x, src1_y) -> (dest1_x, dest1_y)  # Y movement
            
            # For task2 - XY routing creates two segments:
            # Segment 1: (src2_x, src2_y) -> (dest2_x, src2_y)  # X movement
            # Segment 2: (dest2_x, src2_y) -> (dest2_x, dest2_y)  # Y movement
            
            # Create intermediate points for XY routing
            mid1 = (dest1_coords[0], src1_coords[1])  # (dest1_x, src1_y)
            mid2 = (dest2_coords[0], src2_coords[1])  # (dest2_x, src2_y)
            
            # Path segments for task1
            task1_segment1 = (src1_coords, mid1)  # X direction
            task1_segment2 = (mid1, dest1_coords)  # Y direction
            
            # Path segments for task2
            task2_segment1 = (src2_coords, mid2)  # X direction
            task2_segment2 = (mid2, dest2_coords)
            
            # Function to check if two line segments share any point
            def segments_share_point(seg1, seg2):
                # For horizontal segments (x varies, y fixed)
                if seg1[0][1] == seg1[1][1] and seg2[0][1] == seg2[1][1]:
                    # Both segments are horizontal - check if they're on the same y-coordinate
                    if seg1[0][1] != seg2[0][1]:
                        return False
                    
                    # Check x-range overlap
                    seg1_x_min = min(seg1[0][0], seg1[1][0])
                    seg1_x_max = max(seg1[0][0], seg1[1][0])
                    seg2_x_min = min(seg2[0][0], seg2[1][0])
                    seg2_x_max = max(seg2[0][0], seg2[1][0])
                    
                    # Check if x ranges overlap
                    return not (seg1_x_max < seg2_x_min or seg1_x_min > seg2_x_max)
                
                # For vertical segments (y varies, x fixed)
                elif seg1[0][0] == seg1[1][0] and seg2[0][0] == seg2[1][0]:
                    # Both segments are vertical - check if they're on the same x-coordinate
                    if seg1[0][0] != seg2[0][0]:
                        return False
                    
                    # Check y-range overlap
                    seg1_y_min = min(seg1[0][1], seg1[1][1])
                    seg1_y_max = max(seg1[0][1], seg1[1][1])
                    seg2_y_min = min(seg2[0][1], seg2[1][1])
                    seg2_y_max = max(seg2[0][1], seg2[1][1])
                    
                    # Check if y ranges overlap
                    return not (seg1_y_max < seg2_y_min or seg1_y_min > seg2_y_max)
                
                # One is horizontal, one is vertical - check for intersection
                else:
                    # Determine which is horizontal and which is vertical
                    if seg1[0][1] == seg1[1][1]:  # seg1 is horizontal
                        horiz_seg = seg1
                        vert_seg = seg2
                    else:  # seg2 is horizontal
                        horiz_seg = seg2
                        vert_seg = seg1
                    
                    # Check if the vertical line's x is within the horizontal line's x range
                    horiz_x_min = min(horiz_seg[0][0], horiz_seg[1][0])
                    horiz_x_max = max(horiz_seg[0][0], horiz_seg[1][0])
                    vert_x = vert_seg[0][0]
                    
                    if vert_x < horiz_x_min or vert_x > horiz_x_max:
                        return False
                    
                    # Check if the horizontal line's y is within the vertical line's y range
                    vert_y_min = min(vert_seg[0][1], vert_seg[1][1])
                    vert_y_max = max(vert_seg[0][1], vert_seg[1][1])
                    horiz_y = horiz_seg[0][1]
                    
                    return vert_y_min <= horiz_y <= vert_y_max
            
            # Check for path conflicts
            path_intersect = (
                segments_share_point(task1_segment1, task2_segment1) or
                segments_share_point(task1_segment1, task2_segment2) or
                segments_share_point(task1_segment2, task2_segment1) or
                segments_share_point(task1_segment2, task2_segment2)
            )
            
            # Check for destination PE conflicts
            dest_pe_conflict = (dest1_coords == dest2_coords and time_overlap)
            
            if time_overlap and (path_intersect or dest_pe_conflict):
                # Get the specific network nodes where conflict occurs
                path1 = [src1_coords, mid1, dest1_coords]
                path2 = [src2_coords, mid2, dest2_coords]
                
                conflict_points = []
                conflict_type = []
                
                # Check for shared points in the paths, excluding source PEs
                for p1 in [mid1, dest1_coords]:  # Exclude src1_coords
                    for p2 in [mid2, dest2_coords]:  # Exclude src2_coords
                        if p1 == p2:
                            conflict_points.append(p1)
                
                # Determine conflict type
                if dest_pe_conflict:
                    conflict_type.append("destination_pe")
                if path_intersect:  # Report path intersection even if no conflict points found
                    conflict_type.append("path_intersection")
                
                # Create conflict info if there's either a path intersection or destination PE conflict
                conflict_info = {
                    'task1_id': task1_id,
                    'task2_id': task2_id,
                    'task1_path': f"{src1_coords} -> {mid1} -> {dest1_coords}",
                    'task2_path': f"{src2_coords} -> {mid2} -> {dest2_coords}",
                    'task1_time': f"{task1_start_sending}-{task1_end_receiving}",
                    'task2_time': f"{task2_start_sending}-{task2_end_receiving}",
                    'overlap_period': (max(task1_start_sending, task2_start_sending), 
                                      min(task1_end_receiving, task2_end_receiving)),
                    'conflict_points': conflict_points,
                    'conflict_type': conflict_type
                }
                conflicts.append(conflict_info)
    
    return conflicts

# Print the transmission conflicts
print("\nDetecting Transmission Conflicts in the Network-on-Chip...")
#conflicts = detect_transmission_conflicts(updated_tasks)

# if conflicts:
#     print(f"Found {len(conflicts)} potential conflicts:")
#     for i, conflict in enumerate(conflicts):
#         print(f"\nConflict {i+1}:")
#         print(f"  Task {conflict['task1_id']} vs Task {conflict['task2_id']}")
#         print(f"  Task {conflict['task1_id']} path: {conflict['task1_path']}")
#         print(f"  Task {conflict['task2_id']} path: {conflict['task2_path']}")
#         print(f"  Task {conflict['task1_id']} time: {conflict['task1_time']}")
#         print(f"  Task {conflict['task2_id']} time: {conflict['task2_time']}")
#         print(f"  Overlap period: {conflict['overlap_period']}")
#         print(f"  Conflict type(s): {', '.join(conflict['conflict_type'])}")
#         if conflict['conflict_points']:
#             print(f"  Conflict points: {', '.join(str(p) for p in conflict['conflict_points'])}")
#         else:
#             print("  Note: Segments overlap but no common nodes detected")
# else:
#     print("No conflicts detected between tasks.")

# data = """
#    task_id        src_pe_with_network       dest_pe_with_network  bytes wait_ids                                                                description
# 0        1          (0, 0) (external)            (1, 0) (q_proj)     96     None                                 Input distribution (rows 0:12) to PE(1, 0)
# 1        2            (1, 0) (q_proj)            (1, 0) (q_proj)     96        1                  PE(1, 0) full computation (input:(0, 12), output:(0, 12))
# 2        3          (0, 0) (external)            (2, 0) (k_proj)     96     None                                 Input distribution (rows 0:12) to PE(2, 0)
# 3        4            (2, 0) (k_proj)            (2, 0) (k_proj)     96        3                  PE(2, 0) full computation (input:(0, 12), output:(0, 12))
# 4        5          (0, 0) (external)            (3, 0) (v_proj)     96     None                                 Input distribution (rows 0:12) to PE(3, 0)
# 5        6            (3, 0) (v_proj)            (3, 0) (v_proj)     96        5                  PE(3, 0) full computation (input:(0, 12), output:(0, 12))
# 6        7            (1, 0) (q_proj)  (4, 0) (attention_head_0)     48        2                               (1, 0) sends (0:4, 0:6) of input to PE(4, 0)
# 7        8            (2, 0) (k_proj)  (4, 0) (attention_head_0)     48        4                    (2, 0) sends (0:6, 0:4) of input Transposed to PE(4, 0)
# 8        9  (4, 0) (attention_head_0)  (4, 0) (attention_head_0)     32     7, 8                        Matrix multiply at PE(4, 0) (output block 0:4, 0:4)
# 9       10            (3, 0) (v_proj)  (4, 0) (attention_head_0)     48        6                                Send V (0:4, 0:6) from PE(3, 0) to PE(4, 0)
# 10      11  (4, 0) (attention_head_0)  (4, 0) (attention_head_0)     48    9, 10                   Attention computation ((Q·K^T)·V) at PE(4, 0) (0:4, 0:6)
# 11      12            (1, 0) (q_proj)  (5, 0) (attention_head_1)     48        2                               (1, 0) sends (0:4, 0:6) of input to PE(5, 0)
# 12      13            (2, 0) (k_proj)  (5, 0) (attention_head_1)     48        4                    (2, 0) sends (0:6, 0:4) of input Transposed to PE(5, 0)
# 13      14  (5, 0) (attention_head_1)  (5, 0) (attention_head_1)     32   12, 13                        Matrix multiply at PE(5, 0) (output block 0:4, 0:4)
# 14      15            (3, 0) (v_proj)  (5, 0) (attention_head_1)     48        6                                Send V (0:4, 0:6) from PE(3, 0) to PE(5, 0)
# 15      16  (5, 0) (attention_head_1)  (5, 0) (attention_head_1)     48   14, 15                   Attention computation ((Q·K^T)·V) at PE(5, 0) (0:4, 0:6)
# 16      17  (4, 0) (attention_head_0)       (6, 0) (output_proj)     48       11   Network None input  (0:4,0:6) from previous network PE(4, 0) to PE(6, 0)
# 17      18  (5, 0) (attention_head_1)       (6, 0) (output_proj)     48       16  Network None input  (0:4,6:12) from previous network PE(5, 0) to PE(6, 0)
# 18      19       (6, 0) (output_proj)       (6, 0) (output_proj)     96   17, 18                  PE(6, 0) full computation (input:(0, 12), output:(0, 12))
# """