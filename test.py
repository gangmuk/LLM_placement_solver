#!/usr/bin/env python3
"""
Test script for LLM Placement Solver
Demonstrates usage with the example configuration files
"""

import sys
import os
import tempfile
from solver import LLMPlacementSolver, main
import logging

def create_test_files():
    """Create test configuration files"""
    
    # GPU Pool CSV
    gpu_pool_content = """gpu_type,count,memory_gb
A100,2,80
V100,2,32"""
    
    # Network Bandwidth Matrix (simplified for testing - 4 GPUs total)
    network_content = """,gpu_0,gpu_1,gpu_2,gpu_3
gpu_0,400,400,200,200
gpu_1,400,400,200,200
gpu_2,200,200,300,300
gpu_3,200,200,300,300"""
    
    # Runtime Configuration
    config_content = """parameter,value
sequence_length,2048
batch_size,16
model_name,llama-7b
num_decoder_layers,4
d_model,4096
d_hidden,11008
vocab_size,32000
num_attention_heads,32
layer_weight_memory_gb,0.5
time_limit_seconds,60
optimality_gap,0.01"""
    
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    
    gpu_pool_file = os.path.join(temp_dir, 'gpu_pool.csv')
    network_file = os.path.join(temp_dir, 'network.csv')
    config_file = os.path.join(temp_dir, 'config.csv')
    
    with open(gpu_pool_file, 'w') as f:
        f.write(gpu_pool_content)
    
    with open(network_file, 'w') as f:
        f.write(network_content)
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return gpu_pool_file, network_file, config_file, temp_dir

def test_solver():
    """Test the solver with example configuration"""
    print("Creating test configuration files...")
    gpu_pool_file, network_file, config_file, temp_dir = create_test_files()
    
    try:
        print("Initializing solver...")
        solver = LLMPlacementSolver(gpu_pool_file, network_file, config_file)
        
        print("Building optimization model...")
        solver.build_model()
        
        print("Solving optimization problem...")
        if solver.solve():
            print("\nSolution found!")
            solver.print_solution()
            
            # Save solution
            output_file = os.path.join(temp_dir, 'solution.json')
            solver.save_solution(output_file)
            print(f"\nSolution saved to: {output_file}")
            
        else:
            print("No solution found!")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    return True

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("LLM PLACEMENT SOLVER TEST")
    print("=" * 60)
    
    if test_solver():
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Test failed!")
        sys.exit(1)