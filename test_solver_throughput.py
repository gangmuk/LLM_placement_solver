#!/usr/bin/env python3
"""
Test the actual solver's gpu_throughput_with_tp function with debug logging.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the solver (need to handle the dash in filename)
import importlib.util
spec = importlib.util.spec_from_file_location("solver", "solver_constrained_with_tp-2.py")
solver_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver_module)

ThroughputFunctions = solver_module.ThroughputFunctions

# Exact configuration from hal/config.csv
gpu_type = "V100"
seq_len = 4096
batch_size = 32
num_layers = 10  # Per segment (80 / 8 = 10)
d_model = 8192
d_hidden = 22016
bytes_per_element = 2
tp_degree = 4
nvlink_bw_gbps = 300.0

print("="*80)
print("TESTING SOLVER'S gpu_throughput_with_tp FUNCTION")
print("="*80)
print(f"Config: {num_layers} layers, TP={tp_degree}, batch={batch_size}, seq={seq_len}")
print(f"GPU: {gpu_type}")
print()

# Call with debug=True
throughput = ThroughputFunctions.gpu_throughput_with_tp(
    gpu_type, seq_len, batch_size,
    num_layers, d_model, bytes_per_element,
    tp_degree, d_hidden, nvlink_bw_gbps,
    debug=True  # Enable debug logging
)

print()
print("="*80)
print("RESULT")
print("="*80)
print(f"  Returned throughput: {throughput:,.0f} tokens/sec")
print()
print(f"  Expected: ~11,862 tokens/sec")
print(f"  Actual: {throughput:,.0f} tokens/sec")
if throughput > 0:
    print(f"  Ratio: {throughput / 11862:.1f}×")
print("="*80)

