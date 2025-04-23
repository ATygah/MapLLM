import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Union, Any

# Add matplotlib import for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')  # Use non-interactive backend


def analyze_network_traffic(traffic_table: pd.DataFrame, pe_to_network: Dict[str, str]) -> Dict[str, Any]:
    """
    Analyze network traffic table to calculate communication metrics.
    
    Args:
        traffic_table: DataFrame containing traffic data
        pe_to_network: Dictionary mapping PE coordinates to network names
        
    Returns:
        Dictionary containing calculated metrics
    """
    if traffic_table.empty:
        return {
            "total_bytes": 0,
            "total_tasks": 0,
            "metrics_available": False
        }
    
    # Calculate basic metrics
    total_bytes = traffic_table['bytes'].sum()
    total_tasks = len(traffic_table)
    
    # Filter out external inputs
    internal_traffic = traffic_table[traffic_table['src_pe'] != '(0, 0)']
    internal_bytes = internal_traffic['bytes'].sum() if not internal_traffic.empty else 0
    
    # Add network information to traffic table
    traffic_with_networks = enrich_traffic_with_networks(traffic_table, pe_to_network)
    
    # Calculate distance metrics
    distance_metrics = calculate_distance_metrics(internal_traffic)
    
    # Separate traffic by type (cross-network vs intra-network)
    network_traffic = separate_network_traffic(traffic_with_networks, pe_to_network)
    
    # Calculate cross-network metrics
    cross_network_metrics = calculate_network_type_metrics(
        network_traffic["cross_network"], 
        "cross_network"
    ) if network_traffic["cross_network"] is not None else {}
    
    # Calculate intra-network metrics
    intra_network_metrics = calculate_network_type_metrics(
        network_traffic["intra_network"], 
        "intra_network"
    ) if network_traffic["intra_network"] is not None else {}
    
    # Calculate per-network metrics
    per_network_metrics = calculate_per_network_metrics(
        network_traffic["intra_network"], 
        pe_to_network
    ) if network_traffic["intra_network"] is not None else {}
    
    # Combine all metrics
    results = {
        "total_bytes": total_bytes,
        "total_tasks": total_tasks,
        "internal_bytes": internal_bytes,
        "metrics_available": True,
        **distance_metrics,
        **cross_network_metrics,
        **intra_network_metrics,
        "per_network_metrics": per_network_metrics
    }
    
    return results


def enrich_traffic_with_networks(traffic_table: pd.DataFrame, pe_to_network: Dict[str, str]) -> pd.DataFrame:
    """
    Add source and destination network information to traffic table.
    
    Args:
        traffic_table: DataFrame containing traffic data
        pe_to_network: Dictionary mapping PE coordinates to network names
        
    Returns:
        DataFrame with added network information
    """
    # Create a copy to avoid modifying the original
    enriched_table = traffic_table.copy()
    
    # Add source and destination network columns
    src_networks = []
    dest_networks = []
    
    for _, row in traffic_table.iterrows():
        # Extract PE coordinates from string like "(0, 0)"
        src_pe_str = row['src_pe'].strip('()')
        if src_pe_str and ',' in src_pe_str:
            src_x, src_y = map(int, src_pe_str.split(','))
            src_pe = (src_x, src_y)
            src_network = pe_to_network.get(str(src_pe), "external")
        else:
            src_network = "external"
        src_networks.append(src_network)
        
        dest_pe_str = row['dest_pe'].strip('()') if row['dest_pe'] != "None" else ""
        if dest_pe_str and ',' in dest_pe_str:
            dest_x, dest_y = map(int, dest_pe_str.split(','))
            dest_pe = (dest_x, dest_y)
            dest_network = pe_to_network.get(str(dest_pe), "external")
        else:
            dest_network = "external"
        dest_networks.append(dest_network)
    
    # Add the network names to the traffic table
    enriched_table['src_network'] = src_networks
    enriched_table['dest_network'] = dest_networks
    
    # Create enhanced source and destination PE columns that include network names
    enriched_table['src_pe_with_network'] = enriched_table.apply(
        lambda row: f"{row['src_pe']} ({row['src_network']})", axis=1
    )
    enriched_table['dest_pe_with_network'] = enriched_table.apply(
        lambda row: f"{row['dest_pe']} ({row['dest_network']})", axis=1
    )
    
    return enriched_table


def calculate_distance_metrics(traffic_df: pd.DataFrame) -> Dict[str, Union[float, int]]:
    """
    Calculate distance metrics from traffic data.
    
    Args:
        traffic_df: DataFrame containing traffic data
        
    Returns:
        Dictionary of distance metrics
    """
    if traffic_df.empty:
        return {
            "avg_manhattan_distance": 0,
            "weighted_distance": 0,
            "avg_hops": 0,
            "max_hops": 0,
            "total_hops": 0
        }
    
    manhattan_distances = []
    weighted_distances = []
    hop_counts = []
    
    for _, row in traffic_df.iterrows():
        src_pe_str = row['src_pe'].strip('()')
        dest_pe_str = row['dest_pe'].strip('()')
        
        if src_pe_str and dest_pe_str and ',' in src_pe_str and ',' in dest_pe_str:
            src_x, src_y = map(int, src_pe_str.split(','))
            dest_x, dest_y = map(int, dest_pe_str.split(','))
            
            # Calculate Manhattan distance
            distance = abs(dest_x - src_x) + abs(dest_y - src_y)
            manhattan_distances.append(distance)
            weighted_distances.append(distance * row['bytes'])
            hop_counts.append(distance)
    
    avg_manhattan = sum(manhattan_distances) / len(manhattan_distances) if manhattan_distances else 0
    total_weighted_distance = sum(weighted_distances)
    total_hops = sum(hop_counts)
    avg_hops = sum(hop_counts) / len(hop_counts) if hop_counts else 0
    max_hops = max(hop_counts) if hop_counts else 0
    
    return {
        "avg_manhattan_distance": avg_manhattan,
        "weighted_distance": total_weighted_distance,
        "avg_hops": avg_hops,
        "max_hops": max_hops,
        "total_hops": total_hops
    }


def separate_network_traffic(traffic_df: pd.DataFrame, pe_to_network: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Separate traffic into cross-network and intra-network categories.
    
    Args:
        traffic_df: DataFrame containing traffic data with network information
        pe_to_network: Dictionary mapping PE coordinates to network names
        
    Returns:
        Dictionary with separated traffic DataFrames
    """
    if traffic_df.empty:
        return {
            "cross_network": None,
            "intra_network": None
        }
    
    cross_network_traffic = []
    intra_network_traffic = []
    
    for _, row in traffic_df.iterrows():
        src_network = row.get('src_network', "external")
        dest_network = row.get('dest_network', "external")
        
        if src_network != dest_network and src_network != "external" and dest_network != "external":
            # This is cross-network traffic
            cross_network_traffic.append(row)
        elif src_network == dest_network and src_network != "external":
            # This is intra-network traffic (within the same network)
            intra_network_traffic.append(row)
    
    cross_network_df = pd.DataFrame(cross_network_traffic) if cross_network_traffic else None
    intra_network_df = pd.DataFrame(intra_network_traffic) if intra_network_traffic else None
    
    return {
        "cross_network": cross_network_df,
        "intra_network": intra_network_df
    }


def calculate_network_type_metrics(traffic_df: pd.DataFrame, prefix: str) -> Dict[str, Union[float, int]]:
    """
    Calculate metrics for a specific type of network traffic (cross or intra).
    
    Args:
        traffic_df: DataFrame containing traffic data
        prefix: Prefix for the metric names (e.g., "cross_network" or "intra_network")
        
    Returns:
        Dictionary of metrics with prefixed keys
    """
    if traffic_df is None or traffic_df.empty:
        return {
            f"{prefix}_bytes": 0,
            f"{prefix}_tasks": 0,
            f"{prefix}_avg_manhattan": 0,
            f"{prefix}_total_hops": 0,
            f"{prefix}_avg_hops": 0,
            f"{prefix}_max_hops": 0,
            f"{prefix}_weighted_distance_avg": 0,
            f"{prefix}_weighted_distance_max": 0
        }
    
    bytes_transferred = traffic_df['bytes'].sum()
    tasks = len(traffic_df)
    
    manhattan_distances = []
    hop_counts = []
    weighted_distances = []
    
    for _, row in traffic_df.iterrows():
        src_pe_str = row['src_pe'].strip('()')
        dest_pe_str = row['dest_pe'].strip('()')
        
        if src_pe_str and dest_pe_str and ',' in src_pe_str and ',' in dest_pe_str:
            src_x, src_y = map(int, src_pe_str.split(','))
            dest_x, dest_y = map(int, dest_pe_str.split(','))
            
            # Calculate Manhattan distance
            distance = abs(dest_x - src_x) + abs(dest_y - src_y)
            manhattan_distances.append(distance)
            hop_counts.append(distance)
            
            # Calculate weighted distance (distance * bytes)
            weighted_dist = distance * row['bytes']
            weighted_distances.append(weighted_dist)
    
    avg_manhattan = sum(manhattan_distances) / len(manhattan_distances) if manhattan_distances else 0
    total_hops = sum(hop_counts)
    avg_hops = sum(hop_counts) / len(hop_counts) if hop_counts else 0
    max_hops = max(hop_counts) if hop_counts else 0
    
    # Calculate weighted distance metrics
    weighted_distance_avg = sum(weighted_distances) / bytes_transferred if bytes_transferred > 0 else 0
    weighted_distance_max = max(weighted_distances) if weighted_distances else 0
    
    return {
        f"{prefix}_bytes": bytes_transferred,
        f"{prefix}_tasks": tasks,
        f"{prefix}_avg_manhattan": avg_manhattan,
        f"{prefix}_total_hops": total_hops,
        f"{prefix}_avg_hops": avg_hops,
        f"{prefix}_max_hops": max_hops,
        f"{prefix}_weighted_distance_avg": weighted_distance_avg,
        f"{prefix}_weighted_distance_max": weighted_distance_max
    }


def calculate_per_network_metrics(traffic_df: pd.DataFrame, pe_to_network: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate metrics for each individual network.
    
    Args:
        traffic_df: DataFrame containing intra-network traffic data
        pe_to_network: Dictionary mapping PE coordinates to network names
        
    Returns:
        Dictionary of per-network metrics
    """
    if traffic_df is None or traffic_df.empty:
        return {}
    
    network_specific_metrics = {}
    
    for _, row in traffic_df.iterrows():
        src_pe_str = row['src_pe'].strip('()')
        dest_pe_str = row['dest_pe'].strip('()')
        
        if src_pe_str and dest_pe_str and ',' in src_pe_str and ',' in dest_pe_str:
            src_x, src_y = map(int, src_pe_str.split(','))
            dest_x, dest_y = map(int, dest_pe_str.split(','))
            
            # Calculate Manhattan distance
            distance = abs(dest_x - src_x) + abs(dest_y - src_y)
            
            # Calculate weighted distance
            weighted_distance = distance * row['bytes']
            
            # Track which network this communication belongs to
            network_name = pe_to_network.get(f"({src_x}, {src_y})", "unknown")
            if network_name not in network_specific_metrics:
                network_specific_metrics[network_name] = {
                    'distances': [],
                    'weighted_distances': [],
                    'bytes': 0,
                    'tasks': 0
                }
            
            network_specific_metrics[network_name]['distances'].append(distance)
            network_specific_metrics[network_name]['weighted_distances'].append(weighted_distance)
            network_specific_metrics[network_name]['bytes'] += row['bytes']
            network_specific_metrics[network_name]['tasks'] += 1
    
    # Calculate metrics for each network
    for network_name, metrics in network_specific_metrics.items():
        # Calculate average and max distances
        metrics['avg_manhattan'] = sum(metrics['distances']) / len(metrics['distances']) if metrics['distances'] else 0
        metrics['max_hops'] = max(metrics['distances']) if metrics['distances'] else 0
        metrics['avg_hops'] = metrics['avg_manhattan']  # They're the same for grid-based NoCs
        
        # Calculate weighted distance metrics
        total_weighted_distance = sum(metrics['weighted_distances'])
        metrics['weighted_distance_total'] = total_weighted_distance
        metrics['weighted_distance_avg'] = total_weighted_distance / metrics['bytes'] if metrics['bytes'] > 0 else 0
        metrics['weighted_distance_max'] = max(metrics['weighted_distances']) if metrics['weighted_distances'] else 0
        
        # Clean up temporary data
        del metrics['distances']
        del metrics['weighted_distances']
    
    return network_specific_metrics


def write_network_metrics_to_log(metrics: Dict[str, Any], f) -> None:
    """
    Write network metrics to a log file.
    
    Args:
        metrics: Dictionary of metrics to write
        f: File object to write to
    """
    if metrics.get("metrics_available", False):
        f.write("Network Traffic Metrics:\n")
        f.write(f"Total Bytes Transferred: {metrics['total_bytes']:,} bytes\n")
        f.write(f"Total Tasks: {metrics['total_tasks']:,}\n")
        f.write(f"Internal Bytes (excluding external): {metrics.get('internal_bytes', 0):,} bytes\n\n")
        
        f.write("Distance Metrics:\n")
        f.write(f"Average Manhattan Distance: {metrics['avg_manhattan_distance']:.2f} units\n")
        f.write(f"Total Weighted Distance: {metrics['weighted_distance']:,} byte-units\n")
        if 'weighted_distance_avg' in metrics:
            f.write(f"Average Weighted Distance: {metrics['weighted_distance_avg']:.2f} units per byte\n")
        if 'weighted_distance_max' in metrics:
            f.write(f"Maximum Weighted Distance: {metrics['weighted_distance_max']:,} byte-units\n")
        f.write(f"Average Hops: {metrics['avg_hops']:.2f}\n")
        f.write(f"Maximum Hops: {metrics['max_hops']}\n")
        f.write(f"Total Hops: {metrics['total_hops']:,}\n\n")
        
        # Cross-network metrics
        if "cross_network_bytes" in metrics:
            f.write("Cross-Network Traffic Metrics:\n")
            f.write(f"Cross-Network Bytes: {metrics['cross_network_bytes']:,} bytes\n")
            f.write(f"Cross-Network Tasks: {metrics['cross_network_tasks']:,}\n")
            f.write(f"Cross-Network Avg Manhattan: {metrics['cross_network_avg_manhattan']:.2f} units\n")
            if 'cross_network_weighted_distance_avg' in metrics:
                f.write(f"Cross-Network Avg Weighted Distance: {metrics['cross_network_weighted_distance_avg']:.2f} units per byte\n")
            if 'cross_network_weighted_distance_max' in metrics:
                f.write(f"Cross-Network Max Weighted Distance: {metrics['cross_network_weighted_distance_max']:,} byte-units\n")
            f.write(f"Cross-Network Avg Hops: {metrics['cross_network_avg_hops']:.2f}\n")
            f.write(f"Cross-Network Max Hops: {metrics['cross_network_max_hops']}\n")
            f.write(f"Cross-Network Total Hops: {metrics['cross_network_total_hops']:,}\n\n")
        
        # Intra-network metrics
        if "intra_network_bytes" in metrics:
            f.write("Intra-Network Traffic Metrics:\n")
            f.write(f"Intra-Network Bytes: {metrics['intra_network_bytes']:,} bytes\n")
            f.write(f"Intra-Network Tasks: {metrics['intra_network_tasks']:,}\n")
            f.write(f"Intra-Network Avg Manhattan: {metrics['intra_network_avg_manhattan']:.2f} units\n")
            if 'intra_network_weighted_distance_avg' in metrics:
                f.write(f"Intra-Network Avg Weighted Distance: {metrics['intra_network_weighted_distance_avg']:.2f} units per byte\n")
            if 'intra_network_weighted_distance_max' in metrics:
                f.write(f"Intra-Network Max Weighted Distance: {metrics['intra_network_weighted_distance_max']:,} byte-units\n")
            f.write(f"Intra-Network Avg Hops: {metrics['intra_network_avg_hops']:.2f}\n")
            f.write(f"Intra-Network Max Hops: {metrics['intra_network_max_hops']}\n")
            f.write(f"Intra-Network Total Hops: {metrics['intra_network_total_hops']:,}\n\n")
        
        # Per-network metrics
        if "per_network_metrics" in metrics and metrics["per_network_metrics"]:
            f.write("Per-Network Metrics:\n")
            for network, network_metrics in metrics["per_network_metrics"].items():
                f.write(f"\n{network}:\n")
                f.write(f"  Bytes: {network_metrics['bytes']:,}\n")
                f.write(f"  Tasks: {network_metrics['tasks']:,}\n")
                f.write(f"  Avg Manhattan Distance: {network_metrics['avg_manhattan']:.2f}\n")
                f.write(f"  Avg Hops: {network_metrics['avg_hops']:.2f}\n")
                f.write(f"  Max Hops: {network_metrics['max_hops']}\n")
                
                if 'weighted_distance_avg' in network_metrics:
                    f.write(f"  Avg Weighted Distance: {network_metrics['weighted_distance_avg']:.2f} units per byte\n")
                if 'weighted_distance_max' in network_metrics:
                    f.write(f"  Max Weighted Distance: {network_metrics['weighted_distance_max']:,} byte-units\n")
    else:
        f.write("Network traffic metrics not available.\n")


def create_pe_to_network_mapping(llm) -> Dict[str, str]:
    """
    Create a mapping from PE coordinates to network names.
    
    Args:
        llm: LLM instance containing networks
        
    Returns:
        Dictionary mapping PE coordinate strings to network names
    """
    pe_to_network = {}
    for name, network in llm.networks.items():
        for pe_coord in network.active_pes:
            pe_to_network[str(pe_coord)] = name
    return pe_to_network


def calculate_network_bounds(active_pes: Set[Tuple[int, int]]) -> Dict[str, Any]:
    """
    Calculate the bounding box of a set of active PEs.
    
    Args:
        active_pes: Set of (x, y) coordinates representing active PEs
        
    Returns:
        Dictionary containing boundary information or None if no active PEs
    """
    if not active_pes:
        return None
    
    # Extract x and y coordinates
    x_coords = [x for x, y in active_pes]
    y_coords = [y for x, y in active_pes]
    
    # Calculate min/max coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Calculate dimensions
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = width * height
    
    return {
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'width': width,
        'height': height,
        'area': area,
        'coordinates': (x_min, x_max, y_min, y_max)
    }


def calculate_pe_density(active_pes: Set[Tuple[int, int]], bounds: Dict[str, Any]) -> float:
    """
    Calculate the PE density within the bounding box.
    
    Args:
        active_pes: Set of active PEs
        bounds: Dictionary containing boundary information
        
    Returns:
        Density as a percentage
    """
    if not bounds or bounds['area'] == 0:
        return 0.0
    
    return (len(active_pes) / bounds['area']) * 100


def generate_network_layout_visualization(llm, f, max_display_size: int = 20) -> None:
    """
    Generate a visualization of the NoC layout showing network assignments.
    
    Args:
        llm: LLM instance containing networks
        f: File object to write to
        max_display_size: Maximum size of the visualization grid
    """
    f.write("\n--- Network Layout Visualization ---\n")
    
    # Create a grid representing PE allocation
    layout_grid = {}
    for name, network in llm.networks.items():
        for pe in network.active_pes:
            layout_grid[pe] = name
    
    # If there are active PEs, create a visual representation
    if layout_grid:
        # Determine boundaries of used area
        pe_coords = list(layout_grid.keys())
        min_x = max(0, min(x for x, _ in pe_coords))
        max_x = min(min_x + max_display_size, max(x for x, _ in pe_coords))
        min_y = max(0, min(y for _, y in pe_coords))
        max_y = min(min_y + max_display_size, max(y for _, y in pe_coords))
        
        # Write header
        f.write(f"PE layout (showing coordinates {min_x},{min_y} to {max_x},{max_y}):\n")
        
        # Write column headers
        f.write("     | ")
        for x in range(min_x, max_x + 1):
            f.write(f"{x:3d} | ")
        f.write("\n")
        f.write("-" * (7 + 6 * (max_x - min_x + 1)) + "\n")
        
        # Write grid
        for y in range(min_y, max_y + 1):
            f.write(f"{y:3d}  | ")
            for x in range(min_x, max_x + 1):
                if (x, y) in layout_grid:
                    net_name = layout_grid[(x, y)]
                    f.write(f" {net_name[:3]} | ")
                else:
                    f.write("     | ")
            f.write("\n")
            f.write("-" * (7 + 6 * (max_x - min_x + 1)) + "\n")


def compare_mapping_strategies_metrics(results: Dict[str, Dict[str, Any]], f) -> None:
    """
    Generate a comparison table of metrics across different mapping strategies.
    
    Args:
        results: Dictionary of results keyed by strategy name
        f: File object to write to
    """
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("COMPARATIVE ANALYSIS OF MAPPING STRATEGIES\n")
    f.write("=" * 80 + "\n\n")
    
    # Create a comparison table
    f.write("Metrics comparison:\n\n")
    
    mapping_strategies = list(results.keys())
    metrics_header = ["Metric"] + mapping_strategies
    f.write(" | ".join(metrics_header) + "\n")
    f.write("-" * (sum(len(h) for h in metrics_header) + len(metrics_header) - 1) + "\n")
    
    # Common metrics to compare
    metrics = [
        {"name": "Effective Area (positions)", "key": "effective_area", "format": "{:,d}"},
        {"name": "PE Count", "key": "pe_count", "format": "{:d}"},
        {"name": "Area Utilization (%)", "key": "utilization", "format": "{:.2f}%"},
        {"name": "Avg Manhattan Distance", "key": "avg_manhattan_distance", "format": "{:.2f}"},
        {"name": "Cross-Network Avg Distance", "key": "cross_network_avg_manhattan", "format": "{:.2f}"},
        {"name": "Intra-Network Avg Distance", "key": "intra_network_avg_manhattan", "format": "{:.2f}"},
        {"name": "Total Communication (bytes)", "key": "total_bytes", "format": "{:,d}"},
        {"name": "Intra-Network Bytes", "key": "intra_network_bytes", "format": "{:,d}"},
        {"name": "Weighted Distance (bytes×dist)", "key": "weighted_distance", "format": "{:,d}"},
        {"name": "Average Hops", "key": "avg_hops", "format": "{:.2f}"},
        {"name": "Intra-Network Avg Hops", "key": "intra_network_avg_hops", "format": "{:.2f}"},
        {"name": "Cross-Network Avg Hops", "key": "cross_network_avg_hops", "format": "{:.2f}"},
        {"name": "Maximum Hops", "key": "max_hops", "format": "{:d}"},
        {"name": "Intra-Network Max Hops", "key": "intra_network_max_hops", "format": "{:d}"},
        {"name": "Cross-Network Max Hops", "key": "cross_network_max_hops", "format": "{:d}"},
        {"name": "Total Hops", "key": "total_hops", "format": "{:,d}"},
        {"name": "Intra-Network Total Hops", "key": "intra_network_total_hops", "format": "{:,d}"},
        {"name": "Cross-Network Total Hops", "key": "cross_network_total_hops", "format": "{:,d}"}
    ]
    
    # Output metrics in comparison table
    for metric in metrics:
        row = [metric["name"]]
        
        for strategy in mapping_strategies:
            if strategy in results:
                value = results[strategy].get(metric["key"], 0)
                formatted = metric["format"].format(value)
                row.append(formatted)
            else:
                row.append("N/A")
        
        f.write(" | ".join(row) + "\n")


def rank_mapping_strategies(results: Dict[str, Dict[str, Any]], f) -> None:
    """
    Generate rankings of mapping strategies based on key metrics.
    
    Args:
        results: Dictionary of results keyed by strategy name
        f: File object to write to
    """
    mapping_strategies = list(results.keys())
    
    # Strategy rankings
    f.write("\nStrategy Rankings:\n\n")
    
    ranking_metrics = [
        {"name": "Area Efficiency (higher is better)", "key": "utilization", "higher_is_better": True},
        {"name": "Communication Efficiency (lower is better)", "key": "weighted_distance", "higher_is_better": False},
        {"name": "Network Separation (lower is better)", "key": "cross_network_avg_manhattan", "higher_is_better": False},
        {"name": "Intra-Network Efficiency (lower is better)", "key": "intra_network_avg_manhattan", "higher_is_better": False},
        {"name": "Hop Efficiency (lower is better)", "key": "avg_hops", "higher_is_better": False},
        {"name": "Intra-Network Hop Efficiency (lower is better)", "key": "intra_network_avg_hops", "higher_is_better": False},
        {"name": "Cross-Network Hop Efficiency (lower is better)", "key": "cross_network_avg_hops", "higher_is_better": False}
    ]
    
    for r_metric in ranking_metrics:
        f.write(f"{r_metric['name']}:\n")
        
        # Get values for each strategy
        values = []
        for strategy in mapping_strategies:
            if strategy in results:
                value = results[strategy].get(r_metric["key"], 0)
                values.append((strategy, value))
        
        # Sort by value
        if r_metric["higher_is_better"]:
            values.sort(key=lambda x: x[1], reverse=True)
        else:
            values.sort(key=lambda x: x[1])
        
        # Output rankings
        for i, (strategy, value) in enumerate(values):
            formatted = r_metric["name"].split(" ")[0] + ": "
            if "Efficiency" in r_metric["name"]:
                formatted += f"{value:.2f}%"
            elif "Distance" in r_metric["name"]:
                formatted += f"{value:.2f}"
            else:
                formatted += f"{value:,}"
            
            f.write(f"{i+1}. {strategy} ({formatted})\n")
        
        f.write("\n")


def get_strategy_qualitative_assessment() -> Dict[str, Dict[str, List[str]]]:
    """
    Get qualitative assessments of different mapping strategies.
    
    Returns:
        Dictionary of pros and cons for each strategy
    """
    return {
        "column_wise": {
            "pros": ["Simple to implement", "Predictable layout", "Good for networks with few PEs per layer"],
            "cons": ["Poor utilization for large networks", "May lead to long inter-network distances", 
                     "Doesn't optimize for communication patterns"]
        },
        "row_wise": {
            "pros": ["Simple to implement", "Predictable layout", "Good for networks with few PEs per layer"],
            "cons": ["Poor utilization for large networks", "May lead to long inter-network distances", 
                     "Doesn't optimize for communication patterns"]
        },
        "grid_wise": {
            "pros": ["Better area utilization", "More flexible arrangement", "Works well with hybrid split strategy"],
            "cons": ["More complex implementation", "May still have suboptimal communication patterns", 
                     "Can be unpredictable for different network sizes"]
        },
        "compact": {
            "pros": ["Optimized for area efficiency", "Keeps related PEs close together", "Predictable layout shape"],
            "cons": ["Higher intra-network distances", "May create irregular shapes", 
                     "Trade-off between internal and cross-network efficiency"]
        },
        "proximity": {
            "pros": ["Optimized for communication patterns", "Reduces overall hop count", "Efficient for complex networks"],
            "cons": ["More complex implementation", "May have lower density", "Requires explicit communication pattern analysis"]
        }
    }

def calculate_model_utilization(model):
    """Calculate model utilization based on active processing elements."""
    all_pes = set()
    for name, network in model.networks.items():
        if hasattr(network, 'active_pes'):
            all_pes.update(network.active_pes)
        else:
            print(f"Warning: Network '{name}' does not have an 'active_pes' attribute")
            
    if not all_pes:
        print("Warning: No active PEs found. Cannot calculate utilization.")
        return 0.0
        
    x_coords = [x for x, y in all_pes]
    y_coords = [y for x, y in all_pes]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    effective_width = x_max - x_min + 1
    effective_height = y_max - y_min + 1
    effective_area = effective_width * effective_height
    pe_count = len(all_pes)
    utilization = (pe_count / effective_area) * 100
    return utilization


def plot_mapping_strategy_metrics(results: Dict[str, Dict[str, Any]], output_dir: str, prefix: str = "") -> Dict[str, str]:
    """
    Generate plots comparing different metrics across mapping strategies.
    
    Args:
        results: Dictionary of results for each mapping strategy
        output_dir: Directory where plots will be saved
        prefix: Optional prefix for filenames
        
    Returns:
        Dictionary mapping metric names to saved file paths
    """
    if not results:
        return {}
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine available strategies and metrics
    strategies = list(results.keys())
    
    # Group metrics by type
    metric_groups = {
        "utilization": ["utilization", "pe_count", "effective_area"],
        "distance": ["avg_manhattan_distance", "weighted_distance", "avg_hops", "max_hops", "total_hops", 
                    "weighted_distance_avg", "weighted_distance_max"],
        "cross_network": [
            "cross_network_avg_manhattan", "cross_network_avg_hops", 
            "cross_network_max_hops", "cross_network_total_hops", "cross_network_bytes",
            "cross_network_weighted_distance_avg", "cross_network_weighted_distance_max"
        ],
        "intra_network": [
            "intra_network_avg_manhattan", "intra_network_avg_hops", 
            "intra_network_max_hops", "intra_network_total_hops", "intra_network_bytes",
            "intra_network_weighted_distance_avg", "intra_network_weighted_distance_max"
        ],
        "bytes": ["total_bytes"]
    }
    
    # Nicer display names for metrics
    metric_display_names = {
        "utilization": "PE Util(%)",
        "pe_count": "Total PEs Used",
        "effective_area": "Effective Area",
        "avg_manhattan_distance": "Avg Manhattan Distance",
        "weighted_distance": "Weighted Distance",
        "weighted_distance_avg": "Avg Weighted Distance",
        "weighted_distance_max": "Max Weighted Distance",
        "avg_hops": "Average Hops",
        "max_hops": "Max Hops",
        "total_hops": "Total Hops",
        "cross_network_avg_manhattan": "Cross-Network Avg Manhattan",
        "cross_network_avg_hops": "Cross-Network Avg Hops",
        "cross_network_max_hops": "Cross-Network Max Hops",
        "cross_network_total_hops": "Cross-Network Total Hops",
        "cross_network_bytes": "Cross-Network Bytes",
        "cross_network_weighted_distance_avg": "Cross-Network Avg Weighted Distance",
        "cross_network_weighted_distance_max": "Cross-Network Max Weighted Distance",
        "intra_network_avg_manhattan": "Intra-Network Avg Manhattan",
        "intra_network_avg_hops": "Intra-Network Avg Hops",
        "intra_network_max_hops": "Intra-Network Max Hops",
        "intra_network_total_hops": "Intra-Network Total Hops",
        "intra_network_bytes": "Intra-Network Bytes",
        "intra_network_weighted_distance_avg": "Intra-Network Avg Weighted Distance",
        "intra_network_weighted_distance_max": "Intra-Network Max Weighted Distance",
        "total_bytes": "Total Bytes Transferred"
    }
    
    # Save paths for generated plots
    plot_files = {}
    
    # Generate plot for each metric group
    for group_name, metrics in metric_groups.items():
        # Filter to metrics that are actually present in at least one strategy
        available_metrics = [m for m in metrics if any(m in results[s] for s in strategies)]
        
        if not available_metrics:
            continue
            
        # Create figure for this group
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), constrained_layout=True)
        
        # Handle case of single metric
        if n_metrics == 1:
            axes = [axes]
            
        # Plot each metric
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Extract data for each strategy
            data = {}
            for strategy in strategies:
                if metric in results[strategy]:
                    data[strategy] = results[strategy][metric]
            
            if not data:
                continue
                
            # Create bar plot
            x = list(range(len(data)))
            strategy_names = list(data.keys())
            values = [data[s] for s in strategy_names]
            
            # Set colors for different strategies
            colors = plt.cm.viridis(np.linspace(0, 1, len(strategy_names)))
            
            bars = ax.bar(x, values, color=colors, width=0.6)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                value_text = f"{height:.1f}" if height < 1000 else f"{height/1000:.1f}k"
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height * 1.01,
                    value_text,
                    ha='center', va='bottom', 
                    fontsize=10
                )
            
            # Set labels
            ax.set_title(metric_display_names.get(metric, metric))
            ax.set_xticks(x)
            ax.set_xticklabels(strategy_names, rotation=45, ha='right')
            ax.set_ylabel("Value")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plot_filename = f"{prefix}_{group_name}_metrics_{timestamp}.png" if prefix else f"{group_name}_metrics_{timestamp}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        
        plot_files[group_name] = plot_path
    
    # Create comparison plot for PE utilization, the most important metric
    if "utilization" in metric_groups and any("utilization" in results[s] for s in strategies):
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        
        # Extract utilization data
        utilization_data = {}
        for strategy in strategies:
            if "utilization" in results[strategy]:
                utilization_data[strategy] = results[strategy]["utilization"]
        
        if utilization_data:
            # Sort strategies by utilization value (descending)
            sorted_strategies = sorted(utilization_data.keys(), key=lambda s: utilization_data[s], reverse=True)
            x = list(range(len(sorted_strategies)))
            values = [utilization_data[s] for s in sorted_strategies]
            
            # Create color gradient from red (worst) to green (best)
            colors = plt.cm.RdYlGn(np.linspace(0, 1, len(sorted_strategies)))
            
            bars = ax.bar(x, values, color=colors, width=0.6)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + 1,
                    f"{height:.1f}%",
                    ha='center', va='bottom', 
                    fontsize=12, fontweight='bold'
                )
            
            # Set labels
            ax.set_xticks(x)
            ax.set_xticklabels(sorted_strategies, rotation=45, ha='right', fontsize=12)
            ax.set_ylabel("Utilization (%)", fontsize=14)
            ax.set_ylim(0, max(values) * 1.15)  # Add some headroom
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add a horizontal line for average utilization
            avg_utilization = sum(values) / len(values)
            ax.axhline(avg_utilization, color='black', linestyle='--', alpha=0.7, 
                      label=f"Average: {avg_utilization:.1f}%")
            
            # Save plot
            plot_filename = f"{prefix}_utilization_comparison_{timestamp}.png" if prefix else f"utilization_comparison_{timestamp}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            
            plot_files["utilization_comparison"] = plot_path
    
    # Create radar chart comparing key metrics across strategies
    key_metrics = ["utilization", "avg_manhattan_distance", "max_hops", "weighted_distance"]
    key_metrics = [m for m in key_metrics if any(m in results[s] for s in strategies)]
    
    if len(key_metrics) >= 3:  # Need at least 3 metrics for a meaningful radar chart
        fig = plt.figure(figsize=(16, 14), constrained_layout=True)
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics = number of axes
        num_metrics = len(key_metrics)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Set the angle labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric_display_names.get(m, m) for m in key_metrics], fontsize=30, fontweight='bold')
        
        # Normalize the data for each metric to be between 0 and 1
        max_values = {}
        for metric in key_metrics:
            values = [results[s].get(metric, 0) for s in strategies if metric in results[s]]
            if values:
                # For utilization, higher is better, for others lower is better
                if metric == "utilization":
                    max_values[metric] = max(values)
                else:
                    max_values[metric] = max(values) if max(values) > 0 else 1
        
        # Draw the radar chart for each strategy
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
        
        for i, strategy in enumerate(strategies):
            values = []
            for metric in key_metrics:
                if metric in results[strategy] and metric in max_values:
                    # For utilization, higher is better (keep as is)
                    if metric == "utilization":
                        values.append(results[strategy][metric] / max_values[metric])
                    # For other metrics, lower is better (invert)
                    else:
                        val = results[strategy][metric]
                        max_val = max_values[metric]
                        # Invert and normalize to 0-1
                        values.append(1 - (val / max_val) if max_val > 0 else 0)
                else:
                    values.append(0)
            
            # Close the polygon
            values += values[:1]
            
            # Get display name for strategy (change strategy names to more readable versions)
            if strategy == "row_wise":
                display_strategy = "Linear"
            elif strategy == "grid_wise":
                display_strategy = "Grid"
            elif strategy == "proximity":
                display_strategy = "Proximity"
            elif strategy == "compact":
                display_strategy = "Compact"
            elif strategy == "column_wise":
                display_strategy = "Column"
            else:
                display_strategy = strategy
            
            # Plot the strategy
            ax.plot(angles, values, linewidth=2, color=colors[i], label=display_strategy)
            ax.fill(angles, values, color=colors[i], alpha=0.25)
        
        # No legend - removed
        
        # Save plot
        plot_filename = f"{prefix}_radar_comparison_{timestamp}.png" if prefix else f"radar_comparison_{timestamp}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        
        plot_files["radar_comparison"] = plot_path
    
    return plot_files 