import uuid
import math
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class TrafficTask:
    """Class representing a communication task between PEs."""
    task_id: str
    src_pe: Tuple[int, int]
    dest_pe: Tuple[int, int]
    tensor_shape: Tuple[int, ...]
    bytes_count: int
    cycle_count: int
    wait_ids: List[str] = field(default_factory=list)  # Changed from wait_id to wait_ids
    description: str = ""
    network_id: Optional[int] = None  # Identifies which network this task belongs to
    
    def __str__(self):
        wait_str = ", ".join(self.wait_ids) if self.wait_ids else "None"
        network_str = f"Network {self.network_id}" if self.network_id is not None else "Unknown Network"
        return (f"Task {self.task_id} ({network_str}): {self.src_pe} -> {self.dest_pe}, "
                f"Shape: {self.tensor_shape}, Bytes: {self.bytes_count}, "
                f"Cycles: {self.cycle_count}, Wait on: {wait_str}")

class TaskScheduler:
    """Manages and schedules communication tasks between PEs."""
    def __init__(self, channel_bandwidth: float = 32.0, data_type_bytes: int = 2):  # B/cycle
        self.tasks = {}
        self.traffic_table = []
        self.channel_bandwidth = channel_bandwidth  # B/cycle
        self.task_dependencies = {}
        self.data_type_bytes = data_type_bytes  # Size of data type in bytes
        self.next_task_id = 1  # Counter for sequential task IDs
    
    def create_task(self, 
                   src_pe: Tuple[int, int],
                   dest_pe: Tuple[int, int],
                   tensor_shape: Tuple[int, ...],
                   wait_ids: Optional[List[str]] = None,  # Changed from wait_id
                   description: str = "",
                   network_id: Optional[int] = None) -> str:
        """Create a new communication task and return its ID."""
        # Generate a sequential task ID
        task_id = str(self.next_task_id)
        self.next_task_id += 1
        
        # Handle single wait_id for backward compatibility
        if not isinstance(wait_ids, list) and wait_ids is not None:
            wait_ids = [wait_ids]
        elif wait_ids is None:
            wait_ids = []
        
        # Calculate bytes and cycles - now using the parameterized data type size
        bytes_count = math.prod(tensor_shape) * self.data_type_bytes
        cycle_count = math.ceil(bytes_count / self.channel_bandwidth)
        
        task = TrafficTask(
            task_id=task_id,
            src_pe=src_pe,
            dest_pe=dest_pe,
            tensor_shape=tensor_shape,
            bytes_count=bytes_count,
            cycle_count=cycle_count,
            wait_ids=wait_ids,
            description=description,
            network_id=network_id
        )
        
        # TODO: What is the difference between tasks and traffic_table?     
        self.tasks[task_id] = task
        self.traffic_table.append(task)
        return task_id
    
    def get_traffic_table(self) -> pd.DataFrame:
        """Get the traffic table as a pandas DataFrame."""
        data = []
        for task in self.traffic_table:
            data.append({
                'task_id': task.task_id,
                'network_id': task.network_id if task.network_id is not None else "N/A",
                'src_pe': f"({task.src_pe[0]}, {task.src_pe[1]})",
                'dest_pe': f"({task.dest_pe[0]}, {task.dest_pe[1]})" if task.dest_pe else "None",
                'tensor_shape': str(task.tensor_shape),
                'bytes': task.bytes_count,
                'cycles': task.cycle_count,
                'wait_ids': ", ".join(task.wait_ids) if task.wait_ids else "None",
                'description': task.description,
            })
        return pd.DataFrame(data) 