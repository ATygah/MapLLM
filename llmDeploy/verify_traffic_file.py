"""
This script verifies the contents of a traffic table file with tab separators.
"""
import os
import sys

def verify_traffic_file(filename):
    """Verify and display the contents of a traffic file with tab separators."""
    with open(filename, 'rb') as f:
        binary_content = f.read()
        
    # Count and show tab characters
    tab_count = binary_content.count(b'\t')
    print(f"File contains {tab_count} tab characters (0x09)")
    
    # Read as text and display with visible tabs
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("\nContents with tabs shown as '\\t':")
    for i, line in enumerate(lines[:20]):  # Show first 20 lines
        # Replace tabs with visible marker
        visible_tabs = line.replace('\t', '\\t')
        print(f"{i+1:02d}: {visible_tabs}", end='')
    
    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")
    
    print("\nContents with tab alignment (columns):")
    for i, line in enumerate(lines[:20]):  # Show first 20 lines
        if line.startswith('#') or not line.strip():
            # Print comments and empty lines as-is
            print(f"{i+1:02d}: {line}", end='')
            continue
            
        # Split by tabs and display aligned
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            print(f"{i+1:02d}: {parts[0]:15} | {parts[1]:8} | {parts[2]:8} | {parts[3]:8} | {parts[4]}")
        else:
            print(f"{i+1:02d}: {line}", end='')

if __name__ == "__main__":
    # Get the latest traffic file in the traces directory
    traces_dir = os.path.join('llmDeploy', 'traces')
    files = [f for f in os.listdir(traces_dir) if f.endswith('.txt')]
    
    if not files:
        print("No traffic files found in", traces_dir)
        sys.exit(1)
        
    # Sort by modification time (newest first)
    latest_file = sorted(files, key=lambda f: os.path.getmtime(os.path.join(traces_dir, f)), reverse=True)[0]
    
    file_path = os.path.join(traces_dir, latest_file)
    print(f"Verifying latest traffic file: {file_path}")
    verify_traffic_file(file_path) 