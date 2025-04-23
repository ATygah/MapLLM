import os
from datetime import datetime

def simple_traffic_export(filename=None, directory='llmDeploy/traces'):
    """
    Export a sample traffic table to a text file with tab separation.
    
    Args:
        filename: Name of output file (default: traffic_simple_{timestamp}.txt)
        directory: Directory to save the file (default: 'llmDeploy/traces')
    
    Returns:
        Path to the exported file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'traffic_simple_{timestamp}.txt'
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Full path to output file
    output_path = os.path.join(directory, filename)
    
    # Sample data - order: task_id, source_pe, dest_pe, data_size, wait_ids
    rows = [
        ('task1', 0, 100, 96, 'None'),
        ('task2', 100, 100, 96, '1'),
        ('task3', 0, 200, 96, '1,2'),
        ('task4', 200, 200, 96, 'None')
    ]
    
    # Write to file with explicit UTF-8 encoding and LF line endings
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        # Write header as comments
        f.write(f"# Traffic table exported example\n")
        f.write(f"# NoC dimensions: 100x100\n")
        f.write(f"# Columns: task_id\tsource_pe\tdest_pe\tdata_size\twait_ids\n\n")
        
        # Write data rows with tab separation
        for task_id, src_pe, dest_pe, data_size, wait_ids in rows:
            line = f"{task_id}\t{src_pe}\t{dest_pe}\t{data_size}\t{wait_ids}\n"
            f.write(line)
    
    # Verify the file was written correctly
    with open(output_path, 'rb') as f:
        content = f.read()
        tab_count = content.count(b'\t')
        print(f"File written with {tab_count} tab characters")
        
        # Print first few bytes for debugging
        print("First 100 bytes:", ' '.join(f'{b:02x}' for b in content[:100]))
    
    print(f"Sample traffic table exported to {output_path}")
    return output_path

if __name__ == '__main__':
    simple_traffic_export() 