#!/usr/bin/env python3
"""
Test throughput calculation to find the 88× bug.
Expected: ~12,000 tokens/sec
Actual: ~1,045,000 tokens/sec
"""

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

# V100 specs
tflops = 125
mem_bw_gbps = 900
efficiency = 0.42

print("="*80)
print("THROUGHPUT CALCULATION DEBUG")
print("="*80)
print(f"Config: {num_layers} layers, TP={tp_degree}, batch={batch_size}, seq={seq_len}")
print(f"GPU: {gpu_type} ({tflops} TFLOPS, {mem_bw_gbps} GB/s, {efficiency:.0%} efficiency)")
print()

# === Step 1: FLOPs Calculation (per GPU with TP) ===
print("STEP 1: FLOPs Calculation (per GPU)")
print("-" * 80)
attn_proj_flops = 2 * 4 * batch_size * seq_len * d_model * (d_model / tp_degree)
attn_score_flops = 4 * batch_size * seq_len * seq_len * (d_model / tp_degree)
ffn_flops = 2 * batch_size * seq_len * (
    d_model * (d_hidden / tp_degree) +
    d_model * (d_hidden / tp_degree) +
    (d_hidden / tp_degree) * d_model
)

flops_per_layer = attn_proj_flops + attn_score_flops + ffn_flops
total_flops_per_gpu = num_layers * flops_per_layer

print(f"  Attention projections per layer: {attn_proj_flops:.2e} FLOPs")
print(f"  Attention scores per layer:      {attn_score_flops:.2e} FLOPs")
print(f"  FFN per layer:                   {ffn_flops:.2e} FLOPs")
print(f"  Total per layer:                 {flops_per_layer:.2e} FLOPs")
print(f"  Total for {num_layers} layers:   {total_flops_per_gpu:.2e} FLOPs")
print()

# === Step 2: Memory Access Calculation (per GPU with TP) ===
print("STEP 2: Memory Access (per GPU)")
print("-" * 80)
weight_bytes = num_layers * (4 * d_model * (d_model / tp_degree) + 
                             3 * d_model * (d_hidden / tp_degree)) * bytes_per_element
activation_bytes = batch_size * seq_len * d_model * bytes_per_element
kv_cache_bytes = num_layers * 2 * batch_size * seq_len * (d_model / tp_degree) * bytes_per_element
total_bytes_per_gpu = weight_bytes + activation_bytes + kv_cache_bytes

print(f"  Weights (sharded):     {weight_bytes / 1e9:.2f} GB")
print(f"  Activations:           {activation_bytes / 1e9:.2f} GB")
print(f"  KV cache (sharded):    {kv_cache_bytes / 1e9:.2f} GB")
print(f"  Total:                 {total_bytes_per_gpu / 1e9:.2f} GB")
print()

# === Step 3: Arithmetic Intensity & Regime ===
print("STEP 3: Roofline Analysis")
print("-" * 80)
arithmetic_intensity = total_flops_per_gpu / total_bytes_per_gpu
ridge_point = (tflops * 1e12) / (mem_bw_gbps * 1e9)

print(f"  Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPs/byte")
print(f"  Ridge Point:          {ridge_point:.2f} FLOPs/byte")
print(f"  Regime:               {'COMPUTE-BOUND' if arithmetic_intensity > ridge_point else 'MEMORY-BOUND'}")
print()

# === Step 4: Time Calculation ===
print("STEP 4: Time per Batch")
print("-" * 80)
compute_time = total_flops_per_gpu / (tflops * 1e12 * efficiency)
memory_time = total_bytes_per_gpu / (mem_bw_gbps * 1e9 * efficiency)

print(f"  Compute time:  {compute_time:.4f} sec")
print(f"  Memory time:   {memory_time:.4f} sec")
print(f"  Base time:     {max(compute_time, memory_time):.4f} sec (bottleneck)")

time_per_batch = max(compute_time, memory_time)

# === Step 5: Communication Overhead ===
print()
print("STEP 5: Communication Overhead")
print("-" * 80)
activation_size_bytes = batch_size * seq_len * d_model * bytes_per_element
comm_time_per_layer = 2 * activation_size_bytes / (nvlink_bw_gbps * 1e9) * (tp_degree - 1) / tp_degree
total_comm_time = num_layers * comm_time_per_layer

print(f"  Activation size:       {activation_size_bytes / 1e9:.4f} GB")
print(f"  Comm time per layer:   {comm_time_per_layer * 1000:.4f} ms")
print(f"  Total comm time:       {total_comm_time:.4f} sec")
print()

# === Step 6: Total Time & Throughput ===
print("STEP 6: Final Throughput")
print("-" * 80)
total_time = time_per_batch + total_comm_time
tokens_per_batch = batch_size * seq_len
base_throughput = tokens_per_batch / total_time

# Batch efficiency
batch_eff = 1.0  # For batch_size >= 32
final_throughput = base_throughput * batch_eff

print(f"  Total time per batch:  {total_time:.4f} sec")
print(f"  Tokens per batch:      {tokens_per_batch:,}")
print(f"  Base throughput:       {base_throughput:,.0f} tokens/sec")
print(f"  Batch efficiency:      {batch_eff:.0%}")
print(f"  FINAL THROUGHPUT:      {final_throughput:,.0f} tokens/sec")
print()

print("="*80)
print("COMPARISON")
print("="*80)
print(f"  Expected (from manual calc):  ~12,000 tokens/sec")
print(f"  Calculated here:              {final_throughput:,.0f} tokens/sec")
print(f"  Solver reported:              1,045,854 tokens/sec")
print(f"  Discrepancy (solver/correct): {1045854 / final_throughput:.1f}×")
print()

# === Step 7: Check for units errors ===
print("="*80)
print("POTENTIAL BUG SOURCES")
print("="*80)
print(f"  88 ÷ 8 (num stages) = {88 / 8:.1f}")
print(f"  88 ÷ 10 (num layers) = {88 / 10:.1f}")
print(f"  88 ÷ 80 (total layers) = {88 / 80:.2f}")
print()
print("Could the bug be:")
print("  - Multiplying by num_layers instead of dividing?")
print("  - Missing a division somewhere?")
print("  - Units mismatch (GB vs bytes)?")

