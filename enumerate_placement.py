import json
from typing import List, Tuple, Dict, Any

def load_config(json_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def enumerate_placements(
    layers_remaining: int, 
    gpu_usage: List[int], 
    current_placement: List[Tuple[int, int]], 
    gpu_config: Dict[str, Any]
) -> List[List[Tuple[int, int]]]:
    """
    Enumerate all valid placements for heterogeneous GPU cluster
    
    Args:
        layers_remaining: Number of layers still to be placed
        gpu_usage: List of currently used GPUs for each type [used_0, used_1, ...]
        current_placement: Current placement sequence [(gpu_type, segment_size), ...]
        gpu_config: Configuration loaded from JSON
    
    Returns:
        List of valid placements, each placement is [(gpu_type, segment_size), ...]
    """
    if layers_remaining == 0:
        return [current_placement.copy()]  # Found valid complete placement
    
    valid_placements = []
    n_gpu_types = len(gpu_config['gpu_types'])
    
    # Try each GPU type
    for gpu_type_idx in range(n_gpu_types):
        gpu_type_info = gpu_config['gpu_types'][gpu_type_idx]
        
        # Calculate max layers this GPU type can handle
        max_layers_by_memory = gpu_type_info['max_layers']
        max_layers_available = min(max_layers_by_memory, layers_remaining)
        
        # Try different segment sizes (1 to max possible)
        for segment_size in range(1, max_layers_available + 1):
            # Check if we have available GPU of this type
            if gpu_usage[gpu_type_idx] < gpu_type_info['count']:
                # Make recursive call
                new_gpu_usage = gpu_usage.copy()
                new_gpu_usage[gpu_type_idx] += 1
                new_placement = current_placement + [(gpu_type_idx, segment_size)]
                
                valid_placements.extend(
                    enumerate_placements(
                        layers_remaining - segment_size,
                        new_gpu_usage,
                        new_placement,
                        gpu_config
                    )
                )
    
    return valid_placements

def calculate_throughput(placement: List[Tuple[int, int]], gpu_config: Dict[str, Any]) -> float:
    """Calculate end-to-end throughput for a given placement"""
    min_throughput = float('inf')
    
    for gpu_type_idx, segment_size in placement:
        gpu_type_info = gpu_config['gpu_types'][gpu_type_idx]
        
        # Get throughput for this GPU type with this segment size
        throughput_map = gpu_type_info['throughput']
        gpu_throughput = throughput_map.get(str(segment_size), 0)
        
        min_throughput = min(min_throughput, gpu_throughput)
    
    return min_throughput if min_throughput != float('inf') else 0

def solve_placement_problem(config_file: str) -> Dict[str, Any]:
    """Main function to solve the placement problem"""
    
    # Load configuration
    config = load_config(config_file)
    
    # Extract problem parameters
    total_layers = config['model']['total_layers']
    n_gpu_types = len(config['gpu_types'])
    
    # Initialize state
    initial_gpu_usage = [0] * n_gpu_types
    initial_placement = []
    
    print(f"Solving placement problem for {total_layers} layers...")
    print(f"GPU cluster: {n_gpu_types} different types")
    
    # Enumerate all valid placements
    all_placements = enumerate_placements(
        total_layers, 
        initial_gpu_usage, 
        initial_placement, 
        config
    )
    
    print(f"Found {len(all_placements)} valid placements")
    
    # Calculate throughput for each placement
    placement_results = []
    best_throughput = 0
    best_placement = None
    
    for i, placement in enumerate(all_placements):
        throughput = calculate_throughput(placement, config)
        placement_results.append({
            'placement_id': i,
            'assignment': placement,
            'throughput': throughput
        })
        
        if throughput > best_throughput:
            best_throughput = throughput
            best_placement = placement
    
    # Sort by throughput (descending)
    placement_results.sort(key=lambda x: x['throughput'], reverse=True)
    
    return {
        'total_valid_placements': len(all_placements),
        'best_placement': {
            'assignment': best_placement,
            'throughput': best_throughput
        },
        'all_placements': placement_results
    }

def print_placement_summary(results: Dict[str, Any], config: Dict[str, Any]):
    """Print a summary of the placement results"""
    print("\n" + "="*60)
    print("PLACEMENT PROBLEM RESULTS")
    print("="*60)
    
    print(f"Total valid placements: {results['total_valid_placements']}")
    print(f"Best throughput: {results['best_placement']['throughput']:.2f}")
    
    print("\nBest placement details:")
    best = results['best_placement']['assignment']
    for i, (gpu_type, segment_size) in enumerate(best):
        gpu_name = config['gpu_types'][gpu_type]['name']
        print(f"  Segment {i+1}: {segment_size} layers on {gpu_name}")
    
    print("\nTop 5 placements by throughput:")
    for i, result in enumerate(results['all_placements'][:5]):
        print(f"  {i+1}. Throughput: {result['throughput']:.2f}, Assignment: {result['assignment']}")

# Example usage
if __name__ == "__main__":
    # Solve the problem
    input_json = 'input.json'
    config = load_config(input_json)
    results = solve_placement_problem(input_json)
    
    # Print results
    print_placement_summary(results, config)
    
    # Save results
    with open('placement_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'placement_results.json'")