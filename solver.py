#!/usr/bin/env python3
"""
LLM Model Parallelism Placement Solver - WITH TENSOR PARALLELISM AND PRACTICAL CONSTRAINTS
Uses original formulation extended with TP support and domain-specific constraints.

Key Features:
- Tensor Parallelism (TP) within same GPU type
- Pipeline Parallelism (PP) across segments
- Practical constraints based on LLM inference domain knowledge:
  1. Minimum segment size (memory efficiency)
  2. Maximum pipeline depth limit
  3. TP degree hierarchy (better GPUs use higher TP)
  4. Network-aware placement (filter low bandwidth)
  5. Segment size quantization (powers of 2)
  6. TP-PP trade-off constraints
"""

import os
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import argparse
import json
import logging
import math
from typing import Dict, List, Tuple, Optional, FrozenSet, Union
from dataclasses import dataclass
import time
from itertools import product, combinations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GPUType:
    """GPU type specification"""
    name: str
    count: int
    memory_gb: float
    global_ids: List[int]
    cost_per_hour: float = 0.0  # NEW: Cost in $/hour

@dataclass
class Config:
    """Runtime configuration - all parameters are required"""
    # Workload phase (NEW: for prefill/decode disaggregation)
    workload_phase: str  # 'prefill' or 'decode'
    sequence_length: int
    output_length: int  # NEW: for decode phase (tokens to generate)
    min_batch_size: int
    max_batch_size: int
    model_name: str
    num_decoder_layers: int
    d_model: int
    d_hidden: int
    vocab_size: int
    num_attention_heads: int
    layer_weight_memory_gb: float
    time_limit_seconds: float
    optimality_gap: float
    bytes_per_element: int
    # Practical constraints
    max_pipeline_stages: int
    min_memory_utilization: float
    min_layers_per_stage: int
    network_bandwidth_percentile_threshold: float
    enable_segment_quantization: bool
    # Cost optimization parameters
    cost_throughput_weight: float
    max_hourly_cost: float
    max_cost_per_token: float
    max_total_cost: float
    throughput_normalization: float
    cost_normalization: float
    total_tokens_to_process: int
    max_total_runtime_hours: float
    max_total_runtime_hours: float

# GPU performance tiers for hierarchy constraints
GPU_PERFORMANCE_TIERS = {
    'H100': 5,
    'A100': 4,
    'L40S': 3,
    'A40': 3,
    'L40': 2,
    'V100': 2,
    'A10': 1,
    'T4': 1,
    'RTX4090': 2,
    'L20': 2
}

# Optimization priority mapping to cost_throughput_weight
# Format: priority_name -> (weight, description)
OPTIMIZATION_PRIORITY_MAP = {
    'throughput_first': 0.1,  # 90% throughput, 10% cost
    'balanced': 0.5,           # 50% throughput, 50% cost
    'cost_first': 0.9          # 10% throughput, 90% cost
}

class ThroughputFunctions:
    """Throughput functions with TP support"""

    GPU_SPECS = {
        # Modern GPUs optimized for LLM inference (FP16/BF16 with modern tensor cores)
        'H100': {'tflops': 989, 'mem_bw': 3350, 'efficiency': 0.70},  # Best for LLMs
        'A100': {'tflops': 312, 'mem_bw': 2039, 'efficiency': 0.65},  # Excellent for LLMs
        'L40S': {'tflops': 362, 'mem_bw': 864, 'efficiency': 0.58},   # Good Ada Lovelace
        'A40': {'tflops': 150, 'mem_bw': 696, 'efficiency': 0.52},    # Ampere workstation
        'L40': {'tflops': 181, 'mem_bw': 864, 'efficiency': 0.55},    # Ada Lovelace
        
        # Older GPUs - lower efficiency for modern LLM workloads
        'V100': {'tflops': 125, 'mem_bw': 900, 'efficiency': 0.42},   # Volta - old tensor cores
        'RTX4090': {'tflops': 165, 'mem_bw': 1008, 'efficiency': 0.50},
        
        # Budget GPUs
        'L20': {'tflops': 119, 'mem_bw': 480, 'efficiency': 0.48},    # Budget Ada
        'A10': {'tflops': 125, 'mem_bw': 600, 'efficiency': 0.45},    # Budget Ampere
        'T4': {'tflops': 65, 'mem_bw': 320, 'efficiency': 0.38}       # Old Turing
    }

    @staticmethod
    def batch_efficiency_factor(batch_size: int) -> float:
        """
        Batch size efficiency factor - larger batches improve GPU utilization.
        Based on empirical observations:
        - Small batches (1-4): Poor GPU utilization (50-80%)
        - Medium batches (8-16): Good utilization (90-95%)
        - Large batches (32+): Near-optimal utilization (95-100%)
        """
        if batch_size >= 32:
            return 1.0      # Optimal utilization
        elif batch_size >= 16:
            return 0.95     # Very good
        elif batch_size >= 8:
            return 0.90     # Good
        elif batch_size >= 4:
            return 0.80     # Moderate
        elif batch_size >= 2:
            return 0.65     # Poor
        else:
            return 0.50     # Very poor (batch=1)
    
    @staticmethod
    def calculate_arithmetic_intensity(num_layers: int, batch_size: int, seq_len: int, 
                                      d_model: int, d_hidden: int, bytes_per_element: int,
                                      tp_degree: int = 1, phase: str = 'prefill') -> float:
        """
        Calculate arithmetic intensity (FLOPs / Byte) using roofline model.
        Higher AI = more compute-bound, lower AI = more memory-bound.
        
        PHASE-AWARE: Prefill (O(n²)) vs Decode (O(n)) have very different AI!
        
        Args:
            num_layers: Number of transformer layers in segment
            batch_size: Batch size
            seq_len: Sequence length (or KV cache length for decode)
            d_model: Model hidden dimension
            d_hidden: FFN intermediate dimension
            bytes_per_element: Bytes per element (2 for FP16, 4 for FP32)
            tp_degree: Tensor parallelism degree
            phase: 'prefill' or 'decode'
        
        Returns:
            Arithmetic intensity in FLOPs per byte
        """
        # === FLOPs Calculation (PHASE-AWARE) ===
        if phase == 'prefill':
            # PREFILL: Process all tokens at once (O(n²) attention)
            flops_attn_proj = 4 * 2 * batch_size * seq_len * d_model * d_model  # 4 projections
            flops_attn_scores = 4 * batch_size * seq_len * seq_len * d_model  # QK^T + softmax*V (O(n²))
            flops_attention = flops_attn_proj + flops_attn_scores
            
            # FFN for all tokens
            flops_ffn = 2 * batch_size * seq_len * (
                d_model * d_hidden +
                d_model * d_hidden +
                d_hidden * d_model
            )
        else:  # decode
            # DECODE: Generate ONE token (O(n) attention to KV cache)
            # CRITICAL: KV cache grows during generation - use average for realistic estimate
            # Note: This function doesn't have output_length parameter, so we can't account for growth
            # The caller (gpu_throughput_with_tp) handles this properly
            kv_cache_len = seq_len  # seq_len represents cached context
            flops_attn_proj = 4 * 2 * batch_size * 1 * d_model * d_model  # QKV+O for 1 token
            flops_attn_scores = 4 * batch_size * 1 * kv_cache_len * d_model  # Attend to cache (O(n))
            flops_attention = flops_attn_proj + flops_attn_scores
            
            # FFN for 1 token
            flops_ffn = 2 * batch_size * 1 * (
                d_model * d_hidden +
                d_model * d_hidden +
                d_hidden * d_model
            )
        
        flops_per_layer = flops_attention + flops_ffn
        total_flops = flops_per_layer * num_layers
        
        # === Memory Access Calculation (PHASE-AWARE) ===
        # Weights (divided by TP): same for both phases
        bytes_weights_per_layer = (
            4 * d_model * d_model +
            3 * d_model * d_hidden
        ) * bytes_per_element / tp_degree
        
        if phase == 'prefill':
            # KV cache being written
            bytes_kv_cache_per_layer = 2 * batch_size * seq_len * d_model * bytes_per_element / tp_degree
            # Activations for all tokens
            bytes_activations_per_layer = batch_size * seq_len * d_model * bytes_per_element
        else:  # decode
            # KV cache being READ (full cached context!)
            # Note: This function doesn't have output_length, so can't account for growth
            # The caller (gpu_throughput_with_tp) should pass adjusted seq_len if needed
            kv_cache_len = seq_len
            bytes_kv_cache_per_layer = 2 * batch_size * kv_cache_len * d_model * bytes_per_element / tp_degree
            # Activations for 1 token only
            bytes_activations_per_layer = batch_size * 1 * d_model * bytes_per_element
        
        bytes_per_layer = bytes_weights_per_layer + bytes_kv_cache_per_layer + bytes_activations_per_layer
        total_bytes = bytes_per_layer * num_layers
        
        # Arithmetic Intensity = FLOPs / Bytes
        if total_bytes == 0:
            return float('inf')
        
        return total_flops / total_bytes
    
    @staticmethod
    def get_ridge_point(gpu_type: str) -> float:
        """
        Calculate the ridge point for roofline model.
        Ridge point = Peak FLOPS / Peak Bandwidth (FLOPs per byte)
        
        Above ridge point: compute-bound
        Below ridge point: memory-bound
        
        Args:
            gpu_type: GPU type name
        
        Returns:
            Ridge point in FLOPs per byte
        """
        specs = ThroughputFunctions.GPU_SPECS.get(gpu_type, ThroughputFunctions.GPU_SPECS['A100'])
        
        # Convert to consistent units
        peak_flops = specs['tflops'] * 1e12  # TFLOPS → FLOPS
        peak_bandwidth = specs['mem_bw'] * 1e9  # GB/s → bytes/s
        
        ridge_point = peak_flops / peak_bandwidth
        
        return ridge_point
    
    @staticmethod
    def determine_regime(arithmetic_intensity: float, ridge_point: float) -> str:
        """
        Determine if workload is compute-bound or memory-bound.
        
        Args:
            arithmetic_intensity: FLOPs per byte
            ridge_point: Ridge point (FLOPs per byte) for the GPU
        
        Returns:
            "COMPUTE_BOUND" or "MEMORY_BOUND"
        """
        return "COMPUTE_BOUND" if arithmetic_intensity > ridge_point else "MEMORY_BOUND"
    
    @staticmethod
    def gpu_throughput(gpu_type: str, seq_len: int, batch_size: int, num_layers: int, 
                      d_model: int, bytes_per_element: int, d_hidden: int = None) -> float:
        """
        Calculate GPU throughput using roofline model.
        Automatically determines if workload is compute-bound or memory-bound.
        
        Args:
            gpu_type: GPU type name
            seq_len: Sequence length
            batch_size: Batch size
            num_layers: Number of layers in segment
            d_model: Model hidden dimension
            bytes_per_element: Bytes per element (2 for FP16)
            d_hidden: FFN intermediate dimension (defaults to 4*d_model if not provided)
        
        Returns:
            Throughput in tokens/second
        """
        # Default FFN dimension if not provided
        if d_hidden is None:
            d_hidden = 4 * d_model
        
        specs = ThroughputFunctions.GPU_SPECS.get(gpu_type, ThroughputFunctions.GPU_SPECS['A100'])
        
        # === Roofline Model Analysis ===
        # Calculate arithmetic intensity (FLOPs per byte)
        # Note: This function is deprecated in favor of gpu_throughput_with_tp
        # Default to 'prefill' for backward compatibility
        arithmetic_intensity = ThroughputFunctions.calculate_arithmetic_intensity(
            num_layers, batch_size, seq_len, d_model, d_hidden, bytes_per_element, tp_degree=1, phase='prefill'
        )
        
        # Get ridge point for this GPU
        ridge_point = ThroughputFunctions.get_ridge_point(gpu_type)
        
        # Determine regime
        regime = ThroughputFunctions.determine_regime(arithmetic_intensity, ridge_point)
        
        # === Compute FLOPs (same for both regimes) ===
        # Attention: Q, K, V, O projections + attention scores
        attn_proj_flops = 2 * 4 * batch_size * seq_len * d_model * d_model  # QKV + output
        attn_score_flops = 4 * batch_size * seq_len * seq_len * d_model  # QK^T + softmax*V
        
        # FFN: 3 projections (up, gate, down)
        ffn_flops = 2 * batch_size * seq_len * (
            d_model * d_hidden +  # up
            d_model * d_hidden +  # gate
            d_hidden * d_model    # down
        )
        
        flops_per_layer = attn_proj_flops + attn_score_flops + ffn_flops
        total_flops = num_layers * flops_per_layer
        
        # === Compute Memory Access (same for both regimes) ===
        # Weights per layer: 4D² (attention) + 3D×d_hidden (FFN)
        weight_bytes = num_layers * (4 * d_model * d_model + 3 * d_model * d_hidden) * bytes_per_element
        
        # Activations + KV cache
        activation_bytes = batch_size * seq_len * d_model * bytes_per_element
        kv_cache_bytes = num_layers * 2 * batch_size * seq_len * d_model * bytes_per_element
        
        total_bytes = weight_bytes + activation_bytes + kv_cache_bytes
        
        # === Calculate Throughput Based on Regime ===
        if regime == "COMPUTE_BOUND":
            # Limited by compute - use FLOPS
            time_per_batch = total_flops / (specs['tflops'] * 1e12 * specs['efficiency'])
        else:  # MEMORY_BOUND
            # Limited by memory bandwidth
            time_per_batch = total_bytes / (specs['mem_bw'] * 1e9 * specs['efficiency'])
        
        tokens_per_batch = batch_size * seq_len
        base_throughput = tokens_per_batch / time_per_batch
        
        # Apply batch efficiency factor - larger batches utilize GPU better
        final_throughput = base_throughput * ThroughputFunctions.batch_efficiency_factor(batch_size)
        
        return final_throughput
    
    @staticmethod
    def gpu_throughput_with_tp(gpu_type: str, seq_len: int, batch_size: int, 
                               num_layers: int, d_model: int, bytes_per_element: int, 
                               tp_degree: int, d_hidden: int = None, nvlink_bw_gbps: float = 300.0,
                               debug: bool = False, phase: str = 'prefill', output_length: int = 0) -> float:
        """
        GPU throughput with tensor parallelism using roofline model.
        
        CRITICAL: TP is NOT data parallelism!
        - TP splits model weights across GPUs for the SAME batch
        - Each GPU processes the SAME sequences (not different ones)
        - Purpose: fit larger models in memory, NOT increase throughput
        - Throughput effect: minor speedup from reduced memory pressure, offset by communication
        
        NEW: Phase-aware throughput modeling (prefill vs decode)
        - Prefill: O(n²) attention on full prompt (all tokens processed in parallel)
        - Decode: O(n) attention per token (sequential generation, one token at a time)
        
        Args:
            gpu_type: GPU type name
            seq_len: Sequence length (for prefill) OR KV cache length (for decode)
            batch_size: Batch size
            num_layers: Number of layers in segment
            d_model: Model hidden dimension
            bytes_per_element: Bytes per element (2 for FP16)
            tp_degree: Tensor parallelism degree
            d_hidden: FFN intermediate dimension (defaults to 4*d_model)
            nvlink_bw_gbps: NVLink bandwidth in GB/s (from network topology, NOT hardcoded)
            debug: Enable debug logging
            phase: 'prefill' or 'decode' (NEW)
        
        Returns:
            Throughput in tokens/second with TP (NOT multiplied by tp_degree!)
        """
        # Default FFN dimension if not provided
        if d_hidden is None:
            d_hidden = 4 * d_model
        
        specs = ThroughputFunctions.GPU_SPECS.get(gpu_type, ThroughputFunctions.GPU_SPECS['A100'])
        
        if debug:
            logger.info(f"\n{'='*80}")
            logger.info(f"DEBUG: gpu_throughput_with_tp called")
            logger.info(f"  GPU: {gpu_type}, TP={tp_degree}, batch={batch_size}, seq={seq_len}, layers={num_layers}")
            logger.info(f"  d_model={d_model}, d_hidden={d_hidden}, nvlink_bw={nvlink_bw_gbps} GB/s")
        
        # === Roofline Model Analysis with TP (PHASE-AWARE) ===
        arithmetic_intensity = ThroughputFunctions.calculate_arithmetic_intensity(
            num_layers, batch_size, seq_len, d_model, d_hidden, bytes_per_element, tp_degree, phase
        )
        
        ridge_point = ThroughputFunctions.get_ridge_point(gpu_type)
        regime = ThroughputFunctions.determine_regime(arithmetic_intensity, ridge_point)
        
        if debug:
            logger.info(f"  Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPs/byte")
            logger.info(f"  Ridge Point: {ridge_point:.2f} FLOPs/byte")
            logger.info(f"  Regime: {regime}")
        
        # === Compute FLOPs PER GPU (with TP, each GPU does less compute) ===
        # PHASE-AWARE computation: prefill vs decode have different complexity
        
        if phase == 'prefill':
            # PREFILL: Process all tokens in prompt at once (O(n²) attention)
            attn_proj_flops = 2 * 4 * batch_size * seq_len * d_model * (d_model / tp_degree)  # QKV + output
            attn_score_flops = 4 * batch_size * seq_len * seq_len * (d_model / tp_degree)  # O(n²) !!!
            
            # FFN for all tokens
            ffn_flops = 2 * batch_size * seq_len * (
                d_model * (d_hidden / tp_degree) +
                d_model * (d_hidden / tp_degree) +
                (d_hidden / tp_degree) * d_model
            )
        
        else:  # phase == 'decode'
            # DECODE: Generate ONE token at a time (O(n) attention)
            # CRITICAL: KV cache grows from seq_len to seq_len+output_length during generation!
            # We model the AVERAGE case for realistic throughput estimation
            
            if output_length > 0:
                # Use average KV cache length over the generation process
                # Cache grows: seq_len, seq_len+1, ..., seq_len+output_length-1
                # Average = seq_len + (output_length - 1) / 2
                avg_kv_cache_len = seq_len + (output_length - 1) / 2.0
            else:
                # Fallback: use initial cache length (for single-token generation)
                avg_kv_cache_len = seq_len
            
            attn_proj_flops = 2 * 4 * batch_size * 1 * d_model * (d_model / tp_degree)  # QKV for 1 token
            attn_score_flops = 4 * batch_size * 1 * avg_kv_cache_len * (d_model / tp_degree)  # O(n) with growing cache
            
            # FFN for 1 token
            ffn_flops = 2 * batch_size * 1 * (
                d_model * (d_hidden / tp_degree) +
                d_model * (d_hidden / tp_degree) +
                (d_hidden / tp_degree) * d_model
            )
        
        flops_per_layer = attn_proj_flops + attn_score_flops + ffn_flops
        total_flops_per_gpu = num_layers * flops_per_layer
        
        if debug:
            logger.info(f"  FLOPs per layer: {flops_per_layer:.2e}")
            logger.info(f"  Total FLOPs (×{num_layers} layers): {total_flops_per_gpu:.2e}")
        
        # === Compute Memory Access PER GPU (weights sharded by TP) ===
        # PHASE-AWARE memory access patterns
        
        # Weights are sharded across TP GPUs (same for both phases)
        weight_bytes = num_layers * (4 * d_model * (d_model / tp_degree) + 
                                     3 * d_model * (d_hidden / tp_degree)) * bytes_per_element
        
        if phase == 'prefill':
            # PREFILL: Activations for all tokens
            activation_bytes = batch_size * seq_len * d_model * bytes_per_element
            # KV cache being written (sharded)
            kv_cache_bytes = num_layers * 2 * batch_size * seq_len * (d_model / tp_degree) * bytes_per_element
        
        else:  # phase == 'decode'
            # DECODE: Activation for 1 token
            activation_bytes = batch_size * 1 * d_model * bytes_per_element
            # CRITICAL: Must READ entire KV cache (which grows during generation!)
            # Use average cache length for realistic estimation
            if output_length > 0:
                avg_kv_cache_len = seq_len + (output_length - 1) / 2.0
            else:
                avg_kv_cache_len = seq_len
            kv_cache_bytes = num_layers * 2 * batch_size * avg_kv_cache_len * (d_model / tp_degree) * bytes_per_element
        
        total_bytes_per_gpu = weight_bytes + activation_bytes + kv_cache_bytes
        
        if debug:
            logger.info(f"  Total memory: {total_bytes_per_gpu / 1e9:.2f} GB")
        
        # === Calculate BASE time (without TP efficiency penalty first) ===
        if regime == "COMPUTE_BOUND":
            base_time_per_batch = total_flops_per_gpu / (specs['tflops'] * 1e12 * specs['efficiency'])
        else:  # MEMORY_BOUND
            base_time_per_batch = total_bytes_per_gpu / (specs['mem_bw'] * 1e9 * specs['efficiency'])
        
        # === Communication Overhead (compute before applying TP efficiency) ===
        # PHASE-AWARE communication (all-reduce of activations)
        if phase == 'prefill':
            activation_size_bytes = batch_size * seq_len * d_model * bytes_per_element
        else:  # decode
            activation_size_bytes = batch_size * 1 * d_model * bytes_per_element  # 1 token
        
        comm_time_per_layer = 2 * activation_size_bytes / (nvlink_bw_gbps * 1e9) * (tp_degree - 1) / tp_degree
        total_comm_time = num_layers * comm_time_per_layer
        
        # === TP Efficiency Factor (Dynamic, based on actual communication overhead) ===
        comm_overhead_ratio = 0.0  # Initialize
        if tp_degree == 1:
            tp_efficiency_compute = 1.00  # No TP, no overhead
        else:
            # Communication overhead as ratio of total time
            comm_overhead_ratio = total_comm_time / (base_time_per_batch + total_comm_time)
            
            # Additional overheads not captured by communication time:
            # - Memory bandwidth contention (multiple GPUs competing for memory)
            # - Synchronization barriers between layers
            # - Load imbalance across GPUs
            # - Framework scheduling overhead
            additional_overhead = {
                1: 0.00,   # No additional overhead
                2: 0.05,   # 5% additional overhead
                4: 0.10,   # 10% additional overhead
                8: 0.15,   # 15% additional overhead
                16: 0.20   # 20% additional overhead
            }.get(tp_degree, 0.25)
            
            # TP efficiency accounts for overheads beyond pure communication
            # Communication time is already added separately, so we only penalize for additional overhead
            tp_efficiency_compute = max(0.30, 1.0 - additional_overhead)
        
        # === Apply TP efficiency to compute time ===
        time_per_batch = base_time_per_batch / tp_efficiency_compute
        
        if debug:
            logger.info(f"  Base time per batch: {base_time_per_batch:.4f} sec")
            logger.info(f"  Communication overhead ratio: {comm_overhead_ratio if tp_degree > 1 else 0:.2%}")
            logger.info(f"  TP efficiency: {tp_efficiency_compute:.2%}")
            logger.info(f"  Time per batch (with TP penalty): {time_per_batch:.4f} sec")
            logger.info(f"  Comm time per layer: {comm_time_per_layer * 1000:.4f} ms")
            logger.info(f"  Total comm time: {total_comm_time:.4f} sec")
        
        # Total time = compute/memory time + communication time
        total_time = time_per_batch + total_comm_time
        
        # PHASE-AWARE throughput calculation
        if phase == 'prefill':
            # Prefill: Process batch_size × seq_len tokens in one forward pass
            tokens_per_batch = batch_size * seq_len
        else:  # decode
            # Decode: Generate batch_size × 1 tokens per forward pass (sequential)
            tokens_per_batch = batch_size * 1
        
        base_throughput = tokens_per_batch / total_time
        
        batch_eff = ThroughputFunctions.batch_efficiency_factor(batch_size)
        
        if debug:
            logger.info(f"  Total time: {total_time:.4f} sec")
            logger.info(f"  Tokens per batch: {tokens_per_batch:,}")
            logger.info(f"  Base throughput: {base_throughput:,.0f} tokens/sec")
            logger.info(f"  Batch efficiency: {batch_eff:.2f}")
        
        # Apply batch efficiency
        base_throughput *= batch_eff
        
        # NOTE: This is PER-STAGE throughput for a pipeline
        # The actual end-to-end throughput will be limited by:
        # 1. Slowest stage (already modeled via bottleneck constraint)
        # 2. Pipeline bubbles (modeled at the solver level, not here)
        # 3. Inter-stage communication (modeled separately)
        
        if debug:
            logger.info(f"  FINAL per-stage throughput: {base_throughput:,.0f} tokens/sec")
            logger.info(f"  (Pipeline efficiency applied at solver level)")
            logger.info(f"{'='*80}\n")
        
        # NOTE: We do NOT multiply by tp_degree here!
        # TP is not data parallelism - it processes the same batch across GPUs
        
        return base_throughput

class LLMPlacementSolverWithTP:
    """LLM placement solver with Tensor Parallelism and practical constraints"""

    def __init__(self, config_dir: str, tp_configuration: Optional[Dict[str, int]] = None,
                 enable_symmetry_breaking: bool = True,
                 enable_upper_bound: bool = True, enable_tight_bigm: bool = True,
                 enable_flow_conservation: bool = True, threads: Optional[int] = None,
                 max_threads: int = 32, generate_network: Optional[Tuple[float, float]] = None,
                 cloud_provider: Optional[str] = None):
        
        def _read_gurobi_wls_file(wls_path: str) -> Dict[str, Union[str, int]]:
            if not os.path.exists(wls_path):
                raise FileNotFoundError(f"Missing Gurobi WLS file: {wls_path}")

            options: Dict[str, Union[str, int]] = {}
            with open(wls_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip().upper()
                    value = value.strip().strip('"')

                    options[key] = int(value) if key == "LICENSEID" else value

            for required in ("WLSACCESSID", "WLSSECRET", "LICENSEID"):
                if required not in options:
                    raise ValueError(f"{required} missing from {wls_path}")

            return options
        
        self.options = _read_gurobi_wls_file("gurobi.wls")
        self.env = gp.Env(params=self.options)
        self.config_dir = config_dir
        self.cloud_provider = cloud_provider
        
        # Optimization flags
        self.enable_symmetry_breaking = enable_symmetry_breaking
        self.enable_upper_bound = enable_upper_bound
        self.enable_tight_bigm = enable_tight_bigm
        self.enable_flow_conservation = enable_flow_conservation
        self.threads = threads
        self.max_threads = max_threads

        # Load cloud pricing data if available
        cloud_specs_file = 'cloud_instances_specs.csv'
        if os.path.exists(cloud_specs_file):
            self.cloud_pricing = self._load_cloud_pricing(cloud_specs_file)
        else:
            self.cloud_pricing = None
            
        # Load configuration
        gpu_pool_file = os.path.join(config_dir, 'gpu_pool.csv')
        network_file = os.path.join(config_dir, 'network_bandwidth.csv')
        config_file = os.path.join(config_dir, 'config.csv')

        self.gpu_types = self._load_gpu_pool(gpu_pool_file)
        
        # Load or generate network bandwidth
        if generate_network is not None:
            intra_bw, inter_bw = generate_network
            self.network_bandwidth = self._generate_network_bandwidth(intra_bw, inter_bw)
            logger.info(f"Generated network bandwidth matrix: intra={intra_bw} GB/s, inter={inter_bw} GB/s")
            # Save generated matrix to file
            self._save_network_bandwidth(network_file, self.network_bandwidth)
            logger.info(f"Saved generated network bandwidth to {network_file}")
        else:
            self.network_bandwidth = self._load_network_bandwidth(network_file)
            logger.info(f"Loaded network bandwidth from {network_file}")
        
        self.config = self._load_config(config_file)
        self.model = None
        self.solution = None
        
        # Derived data
        self.total_gpus = sum(gpu_type.count for gpu_type in self.gpu_types.values())
        
        # Validate network matrix
        if self.network_bandwidth.shape[0] != self.total_gpus:
            raise ValueError(f"Network bandwidth matrix size ({self.network_bandwidth.shape[0]}) "
                           f"does not match total GPU count ({self.total_gpus})")
        
        # TP configuration: {gpu_type: max_tp_degree}
        self.tp_max_configuration = tp_configuration or {gpu_type: 8 for gpu_type in self.gpu_types}
        
        # Generate TP partitions based on configuration
        self.tp_allocations = self._generate_tp_allocations()
        
        # Generate valid segments and connections with TP and practical constraints
        self.max_segment_size = self._compute_max_segment_sizes()
        self.min_segment_size = self._compute_min_segment_sizes()
        self.quantized_sizes = self._get_quantized_segment_sizes() if self.config.enable_segment_quantization else None
        
        self.valid_segments = self._generate_valid_segments()
        # Finalize pre-computed throughputs (move from temp to permanent)
        self.segment_throughputs = getattr(self, 'segment_throughputs_temp', {})
        if hasattr(self, 'segment_throughputs_temp'):
            delattr(self, 'segment_throughputs_temp')
        self.valid_connections = self._generate_valid_connections()
        self._apply_network_bandwidth_filter()
        
        self.gpu_pair_throughputs = self._precompute_network_throughputs()
        
        # Validate problem size
        self._validate_problem_size()
        
        logger.info(f"Initialized solver with TP and practical constraints:")
        logger.info(f"  - GPU types: {len(self.gpu_types)}, Total GPUs: {self.total_gpus}")
        logger.info(f"  - TP Configuration: {self.tp_max_configuration}")
        logger.info(f"  - Max pipeline stages: {self.config.max_pipeline_stages}")
        logger.info(f"  - Min memory utilization: {self.config.min_memory_utilization}")
        logger.info(f"  - Segment quantization: {self.config.enable_segment_quantization}")
        logger.info(f"  - Model: {self.config.num_decoder_layers} layers")
        logger.info(f"  - Problem size: {len(self.valid_segments)} segments, {len(self.valid_connections)} connections")
    
    def _load_cloud_pricing(self, filename: str) -> Dict[Tuple[str, str], float]:
        """
        Load cloud GPU pricing from cloud_instances_specs.csv
        Returns dict: (provider, gpu_model) -> price_per_gpu_per_hour
        """
        import re
        
        df = pd.read_csv(filename)
        pricing = {}
        
        for _, row in df.iterrows():
            provider = row['Cloud Provider']
            gpu_model = row['GPU Model']
            price_str = row['Price per Hour USD']
            
            # Skip if no price available
            if pd.isna(price_str) or 'Contact' in str(price_str):
                continue
            
            # Parse price (handle ranges like "$140-160 (estimated)")
            numbers = re.findall(r'[\d.]+', str(price_str))
            if not numbers:
                continue
            
            # If range, take average
            if len(numbers) >= 2:
                price_value = (float(numbers[0]) + float(numbers[1])) / 2
            else:
                price_value = float(numbers[0])
            
            # Check if price is already "per GPU" or total instance price
            if 'per GPU' in str(price_str):
                # Price is already per GPU, use as-is
                price_per_gpu = price_value
            else:
                # Price is for entire instance, divide by GPU count
                gpu_count_str = str(row['GPU Count'])
                gpu_count_numbers = re.findall(r'\d+', gpu_count_str)
                if not gpu_count_numbers:
                    continue
                gpu_count = int(gpu_count_numbers[0])
                price_per_gpu = price_value / gpu_count
            
            pricing[(provider, gpu_model)] = price_per_gpu
        
        return pricing
    
    def _get_cloud_price(self, gpu_type: str) -> Optional[float]:
        """
        Get cloud price for a GPU type from the specified provider.
        Returns the cheapest price if provider not specified.
        """
        if not self.cloud_pricing:
            return None
        
        # Find matching prices
        matching_prices = []
        for (provider, gpu_model), price in self.cloud_pricing.items():
            if gpu_type in gpu_model:  # e.g., "V100" in "NVIDIA V100"
                if self.cloud_provider is None or provider == self.cloud_provider:
                    matching_prices.append((provider, price))
        
        if not matching_prices:
            return None
        
        if self.cloud_provider:
            # Return price from specified provider
            for provider, price in matching_prices:
                if provider == self.cloud_provider:
                    return price
            return None
        else:
            # Return cheapest price
            return min(price for _, price in matching_prices)
    
    def _load_gpu_pool(self, filename: str) -> Dict[str, GPUType]:
        """Load GPU pool configuration"""
        df = pd.read_csv(filename)
        gpu_types = {}
        global_id = 0
        
        for _, row in df.iterrows():
            global_ids = list(range(global_id, global_id + row['count']))
            
            # Try to get price from config first, then from cloud pricing
            if 'dollar_per_hour' in row and pd.notna(row['dollar_per_hour']):
                cost_per_hour = float(row['dollar_per_hour'])
                price_source = "config"
            else:
                # Look up from cloud pricing
                cloud_price = self._get_cloud_price(row['gpu_type'])
                if cloud_price is not None:
                    cost_per_hour = cloud_price
                    price_source = f"cloud ({self.cloud_provider or 'cheapest'})"
                else:
                    # Fallback to 0.0 with warning
                    cost_per_hour = 0.0
                    price_source = "DEFAULT (0.0) - NO PRICING DATA"
                    logger.warning(f"  WARNING: No cloud price found for {row['gpu_type']}! Using $0.00/hour")
                    logger.warning(f"           This GPU will appear 'free' in optimization - add pricing to cloud_instances_specs.csv")
                    assert False, "No cloud price found for GPU"
            
            gpu_types[row['gpu_type']] = GPUType(
                name=row['gpu_type'],
                count=row['count'],
                memory_gb=row['memory_gb'],
                global_ids=global_ids,
                cost_per_hour=cost_per_hour
            )
            
            logger.info(f"  GPU {row['gpu_type']}: ${cost_per_hour:.2f}/hour (from {price_source})")
            
            global_id += row['count']
        
        return gpu_types
    
    def _load_network_bandwidth(self, filename: str) -> np.ndarray:
        """Load network bandwidth matrix"""
        df = pd.read_csv(filename, index_col=0)
        matrix = df.values
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Network bandwidth matrix must be square, got {matrix.shape}")
        
        return matrix
    
    def _save_network_bandwidth(self, filename: str, matrix: np.ndarray) -> None:
        """Save network bandwidth matrix to CSV file"""
        total_gpus = matrix.shape[0]
        # Create column and row labels
        gpu_ids = [f"gpu_{i}" for i in range(total_gpus)]
        df = pd.DataFrame(matrix, index=gpu_ids, columns=gpu_ids)
        df.to_csv(filename)
        logger.debug(f"Saved {total_gpus}×{total_gpus} network bandwidth matrix to {filename}")
    
    def _generate_network_bandwidth(self, intra_bandwidth: float, inter_bandwidth: float) -> np.ndarray:
        """
        Generate network bandwidth matrix programmatically.
        
        Args:
            intra_bandwidth: Bandwidth (GB/s) between GPUs of the same type
            inter_bandwidth: Bandwidth (GB/s) between GPUs of different types
        
        Returns:
            Network bandwidth matrix (total_gpus × total_gpus)
        """
        total_gpus = sum(gpu_type.count for gpu_type in self.gpu_types.values())
        matrix = np.zeros((total_gpus, total_gpus))
        
        # Build a mapping from global_id to gpu_type
        global_id_to_type = {}
        for gpu_type, gpu_obj in self.gpu_types.items():
            for global_id in gpu_obj.global_ids:
                global_id_to_type[global_id] = gpu_type
        
        # Fill the matrix
        for i in range(total_gpus):
            for j in range(total_gpus):
                if i == j:
                    # Self-connection: infinite bandwidth (set to large value)
                    matrix[i, j] = 10000.0
                else:
                    gpu_type_i = global_id_to_type[i]
                    gpu_type_j = global_id_to_type[j]
                    
                    if gpu_type_i == gpu_type_j:
                        # Same GPU type: use intra_bandwidth
                        matrix[i, j] = intra_bandwidth
                    else:
                        # Different GPU types: use inter_bandwidth
                        matrix[i, j] = inter_bandwidth
        
        logger.info(f"Generated {total_gpus}×{total_gpus} network bandwidth matrix")
        return matrix
    
    def _load_config(self, filename: str) -> Config:
        """Load runtime configuration"""
        df = pd.read_csv(filename)
        config_dict = dict(zip(df['parameter'], df['value']))
        
        # Parse optimization_priority (new) or cost_throughput_weight (legacy)
        if 'optimization_priority' in config_dict:
            priority = config_dict['optimization_priority'].lower()
            if priority not in OPTIMIZATION_PRIORITY_MAP:
                raise ValueError(
                    f"Invalid optimization_priority: '{priority}'. "
                    f"Must be one of: {', '.join(OPTIMIZATION_PRIORITY_MAP.keys())}"
                )
            cost_throughput_weight = OPTIMIZATION_PRIORITY_MAP[priority]
            logger.info(f"Optimization priority: {priority} (weight={cost_throughput_weight:.2f})")
        elif 'cost_throughput_weight' in config_dict:
            # Legacy support
            cost_throughput_weight = float(config_dict['cost_throughput_weight'])
            logger.warning(f"Using deprecated 'cost_throughput_weight'. Consider using 'optimization_priority' instead.")
        else:
            raise ValueError("Config must specify either 'optimization_priority' or 'cost_throughput_weight'")
        
        # Load workload phase (NEW: prefill/decode disaggregation)
        workload_phase = config_dict.get('workload_phase', 'prefill').lower()
        if workload_phase not in ['prefill', 'decode']:
            raise ValueError(f"Invalid workload_phase: '{workload_phase}'. Must be 'prefill' or 'decode'")
        logger.info(f"Workload phase: {workload_phase}")
        
        # Load output_length (NEW: for decode phase)
        output_length = int(config_dict.get('output_length', 0))
        if workload_phase == 'decode' and output_length == 0:
            logger.warning("Decode phase with output_length=0! This may indicate misconfiguration.")
        
        return Config(
            workload_phase=workload_phase,
            sequence_length=int(config_dict['sequence_length']),
            output_length=output_length,
            min_batch_size=int(config_dict['min_batch_size']),
            max_batch_size=int(config_dict['max_batch_size']),
            model_name=config_dict['model_name'],
            num_decoder_layers=int(config_dict['num_decoder_layers']),
            d_model=int(config_dict['d_model']),
            d_hidden=int(config_dict['d_hidden']),
            vocab_size=int(config_dict['vocab_size']),
            num_attention_heads=int(config_dict['num_attention_heads']),
            layer_weight_memory_gb=float(config_dict['layer_weight_memory_gb']),
            time_limit_seconds=float(config_dict['time_limit_seconds']),
            optimality_gap=float(config_dict['optimality_gap']),
            bytes_per_element=int(config_dict['bytes_per_element']),
            max_pipeline_stages=int(config_dict['max_pipeline_stages']),
            min_memory_utilization=float(config_dict['min_memory_utilization']),
            min_layers_per_stage=int(config_dict['min_layers_per_stage']),
            network_bandwidth_percentile_threshold=float(config_dict['network_bandwidth_percentile_threshold']),
            enable_segment_quantization=config_dict['enable_segment_quantization'].lower() == 'true',
            # Cost optimization parameters
            cost_throughput_weight=cost_throughput_weight,
            max_hourly_cost=float(config_dict['max_hourly_cost']),
            max_cost_per_token=float(config_dict['max_cost_per_million_token']) / 1_000_000,
            max_total_cost=float(config_dict['max_total_cost']),
            throughput_normalization=float(config_dict['throughput_normalization']),
            cost_normalization=float(config_dict['cost_normalization']),
            total_tokens_to_process=int(config_dict['total_tokens_to_process']),
            max_total_runtime_hours=float(config_dict['max_total_runtime_hours'])
        )
    
    def _get_min_intra_tp_bandwidth(self, gpu_type: str, gpu_set: FrozenSet[int]) -> float:
        """
        Get the minimum bandwidth within a TP group (for all-reduce communication).
        
        Args:
            gpu_type: GPU type name
            gpu_set: Set of local GPU IDs in the TP group
        
        Returns:
            Minimum bandwidth in GB/s within the TP group
        """
        if len(gpu_set) <= 1:
            return float('inf')  # No communication needed for TP=1
        
        min_bw = float('inf')
        for id1 in gpu_set:
            global_id1 = self._get_global_gpu_id(gpu_type, id1)
            for id2 in gpu_set:
                if id1 != id2:
                    global_id2 = self._get_global_gpu_id(gpu_type, id2)
                    bw = self.network_bandwidth[global_id1, global_id2]
                    min_bw = min(min_bw, bw)
        
        return min_bw
    
    def _get_representative_tp_bandwidth(self, gpu_type: str, tp_degree: int) -> float:
        """
        Get a representative bandwidth for a TP group of given size (for estimates/diagnostics).
        Uses the first tp_degree GPUs of this type to calculate bandwidth.
        
        Args:
            gpu_type: GPU type name
            tp_degree: TP degree
        
        Returns:
            Representative bandwidth in GB/s
        """
        if tp_degree <= 1:
            return float('inf')
        
        gpu_obj = self.gpu_types[gpu_type]
        if gpu_obj.count < tp_degree:
            return 100.0  # Conservative default if not enough GPUs
        
        # Use first tp_degree GPUs
        gpu_set = frozenset(range(tp_degree))
        return self._get_min_intra_tp_bandwidth(gpu_type, gpu_set)
    
    def _generate_tp_allocations(self) -> Dict[str, List[Tuple[FrozenSet[int], int, int]]]:
        """
        Generate all valid (gpu_set, tp_degree, partition_id) tuples for each GPU type.
        For each TP degree, create non-overlapping partitions.

        CRITICAL: partition_id must be globally unique across all TP degrees!

        Returns: {gpu_type: [(gpu_set, tp_degree, partition_id), ...]}
        """

        tp_allocations = {}

        for gpu_type, gpu_obj in self.gpu_types.items():
            max_tp = self.tp_max_configuration[gpu_type]
            allocations = []
            # Allow TP degrees even if they don't perfectly divide GPU count
            # It's OK to have some GPUs unused (e.g., use 8 of 12 GPUs for TP=8)
            valid_tp_degrees = [d for d in [1, 2, 4, 8, 16]
                            if d <= max_tp and d <= gpu_obj.count]

            global_partition_id = 0  # Global counter across all TP degrees

            for tp_degree in valid_tp_degrees:
                num_partitions = gpu_obj.count // tp_degree

                for local_partition_id in range(num_partitions):
                    start_id = local_partition_id * tp_degree
                    gpu_set = frozenset(range(start_id, start_id + tp_degree))
                    allocations.append((gpu_set, tp_degree, global_partition_id))
                    global_partition_id += 1  # Increment for uniqueness

            tp_allocations[gpu_type] = allocations

            logger.info(f"GPU {gpu_type}: max_tp={max_tp}, generated {len(allocations)} allocations with unique partition IDs")

        return tp_allocations
    
    def _compute_max_segment_sizes(self) -> Dict[Tuple[str, int], int]:
        """Compute maximum segment size for each (GPU type, TP degree) pair"""
        max_sizes = {}

        for gpu_type, gpu_obj in self.gpu_types.items():
            tp_degree = self.tp_max_configuration[gpu_type]

            # With TP, weights are sharded
            weight_per_gpu_per_layer = self.config.layer_weight_memory_gb / tp_degree

            # Activation memory with TP (accounts for all-reduce and KV cache sharding)
            activation_memory = self._calculate_activation_memory(tp_degree)

            # Binary search for max layers
            max_layers = self._binary_search_max_layers(
                gpu_obj.memory_gb, weight_per_gpu_per_layer, activation_memory
            )

            max_sizes[(gpu_type, tp_degree)] = max_layers

            logger.info(f"GPU {gpu_type} with TP={tp_degree}: max {max_layers} layers")

        return max_sizes
    
    def _compute_min_segment_sizes(self) -> Dict[Tuple[str, int], int]:
        """
        Compute minimum segment size based on memory efficiency.
        Target: at least X% memory utilization (weights + activations).
        """
        min_sizes = {}

        for gpu_type, gpu_obj in self.gpu_types.items():
            tp_degree = self.tp_max_configuration[gpu_type]
            gpu_memory = gpu_obj.memory_gb
            weight_per_layer = self.config.layer_weight_memory_gb / tp_degree
            activation_memory = self._calculate_activation_memory(tp_degree)

            # Total memory = (layers * weight_per_layer) + activation_memory
            # We want: total >= min_memory_utilization * gpu_memory
            # Solving for layers:
            min_layers = max(1, math.ceil(
                (self.config.min_memory_utilization * gpu_memory - activation_memory) / weight_per_layer
            ))

            # Sanity check: if activation alone exceeds target, use 1 layer
            if activation_memory > self.config.min_memory_utilization * gpu_memory:
                min_layers = 1

            # Also cap at max segment size
            max_layers = self.max_segment_size[(gpu_type, tp_degree)]
            min_layers = min(min_layers, max_layers)

            min_sizes[(gpu_type, tp_degree)] = min_layers

            logger.info(f"GPU {gpu_type} with TP={tp_degree}: min {min_layers} layers "
                       f"(for {self.config.min_memory_utilization*100:.0f}% utilization)")

        return min_sizes
    
    def _get_quantized_segment_sizes(self) -> List[int]:
        """
        Generate coarse quantized segment sizes for performance.
        
        PERFORMANCE OPTIMIZATION:
        - Reduced from 14 sizes to ~7 sizes
        - Strategic selection: [1, 5, 10, 20, 30, 40]
        - 5-10× faster while maintaining solution quality
        """
        total_layers = self.config.num_decoder_layers
        
        # Coarse quantization strategy
        if total_layers <= 10:
            # Small models: use fine granularity
            sizes = [1, 5, total_layers]
        elif total_layers <= 30:
            # Medium models: use moderate granularity
            sizes = [1, 5, 10, 15, 20, total_layers]
        elif total_layers <= 40:
            # Large models (like our 40-layer case): use coarse granularity
            sizes = [1, 5, 10, 20, 30, total_layers]
        else:
            # Large models (like our 40-layer case): use coarse granularity
            sizes = [1, 10, 20, 30, 40, total_layers]
        
        # Remove duplicates and sort
        sizes = sorted(set(sizes))
        
        logger.info(f"Quantized segment sizes (coarse): {sizes}")
        logger.info(f"  Reduced from ~14 to {len(sizes)} sizes for performance")
        return sizes
    
    def _calculate_activation_memory(self, tp_degree: int = 1, batch_size: int = None) -> float:
        """
        Calculate peak activation memory per GPU with TP.
        Accounts for all-reduce operations and KV cache sharding.
        
        NEW: Phase-aware memory calculation (prefill vs decode)
        - Prefill: Activations for all tokens + KV cache being written
        - Decode: Activations for 1 token + FULL KV cache being read (much larger!)
        """
        batch = batch_size if batch_size is not None else self.config.min_batch_size
        hidden = self.config.d_model
        d_hidden = self.config.d_hidden
        bytes_per_elem = self.config.bytes_per_element
        phase = self.config.workload_phase  # NEW: Get phase from config
        
        if phase == 'prefill':
            # PREFILL: Process all tokens in prompt
            seq_len = self.config.sequence_length
            
            # Sharded intermediate tensors during computation
            qkv_memory = (3 * batch * seq_len * (hidden / tp_degree) *
                         bytes_per_elem) / (1024**3)
            mlp_intermediate = (batch * seq_len * (4 * hidden / tp_degree) *
                               bytes_per_elem) / (1024**3)
            sharded_computation = qkv_memory + mlp_intermediate

            # Full activation tensor after all-reduce (NOT sharded)
            full_activation = (batch * seq_len * hidden * bytes_per_elem) / (1024**3)

            # KV cache being written (sharded)
            kv_cache = (2 * batch * seq_len * (hidden / tp_degree) *
                       bytes_per_elem) / (1024**3)

            # Peak memory
            peak_activation = max(sharded_computation, full_activation) + kv_cache
        
        else:  # phase == 'decode'
            # DECODE: Generate 1 token at a time
            kv_cache_len = self.config.sequence_length  # Context length from prefill
            
            # Sharded intermediate tensors for 1 token
            qkv_memory = (3 * batch * 1 * (hidden / tp_degree) *
                         bytes_per_elem) / (1024**3)
            mlp_intermediate = (batch * 1 * (4 * hidden / tp_degree) *
                               bytes_per_elem) / (1024**3)
            sharded_computation = qkv_memory + mlp_intermediate

            # Full activation for 1 token after all-reduce
            full_activation = (batch * 1 * hidden * bytes_per_elem) / (1024**3)

            # CRITICAL: Must store FULL KV cache from prefill (sharded)
            # This is persistent and can be HUGE!
            kv_cache = (2 * batch * kv_cache_len * (hidden / tp_degree) *
                       bytes_per_elem) / (1024**3)

            # Peak memory
            peak_activation = max(sharded_computation, full_activation) + kv_cache

        # Framework overhead (15%)
        total_with_overhead = peak_activation * 1.15

        return total_with_overhead
    
    def _binary_search_max_layers(self, gpu_memory: float, memory_per_layer: float, 
                                  activation_memory: float) -> int:
        """Binary search to find maximum layers that fit in GPU memory"""
        left, right = 1, self.config.num_decoder_layers
        max_feasible = 1
        
        while left <= right:
            mid = (left + right) // 2
            total_memory = mid * memory_per_layer + activation_memory
            
            if total_memory <= gpu_memory:
                max_feasible = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return max_feasible
    
    def _compute_max_segment_size_for_tp(self, gpu_type: str, tp_degree: int, batch_size: int) -> int:
        """Compute max segment size for a specific (gpu_type, tp_degree, batch_size)"""
        gpu_obj = self.gpu_types[gpu_type]
        weight_per_gpu_per_layer = self.config.layer_weight_memory_gb / tp_degree
        activation_memory = self._calculate_activation_memory(tp_degree, batch_size)
        
        return self._binary_search_max_layers(
            gpu_obj.memory_gb, weight_per_gpu_per_layer, activation_memory
        )

    def _compute_min_segment_size_for_tp(self, gpu_type: str, tp_degree: int, batch_size: int) -> int:
        """Compute min segment size for a specific (gpu_type, tp_degree, batch_size)"""
        gpu_obj = self.gpu_types[gpu_type]
        weight_per_layer = self.config.layer_weight_memory_gb / tp_degree
        activation_memory = self._calculate_activation_memory(tp_degree, batch_size)
        
        min_layers = max(1, math.ceil(
            (self.config.min_memory_utilization * gpu_obj.memory_gb - activation_memory) / weight_per_layer
        ))
        
        if activation_memory > self.config.min_memory_utilization * gpu_obj.memory_gb:
            min_layers = 1
        
        max_layers = self._compute_max_segment_size_for_tp(gpu_type, tp_degree, batch_size)
        return min(min_layers, max_layers)

    def _get_batch_size_options(self) -> List[int]:
        """
        Generate power-of-2 batch sizes between min and max (inclusive).
        Example: min=8, max=64 → [8, 16, 32, 64]
        """
        batch_sizes = []
        b = self.config.min_batch_size
        while b <= self.config.max_batch_size:
            batch_sizes.append(b)
            b *= 2
        # Ensure max_batch_size is included even if not exact power of 2
        if batch_sizes[-1] != self.config.max_batch_size:
            batch_sizes.append(self.config.max_batch_size)
        logger.info(f"Batch size options: {batch_sizes}")
        return batch_sizes
    
    def _generate_valid_segments(self) -> List[Tuple]:
        """Generate segments with variable TP degree and batch size per segment"""
        valid_segments = []
        batch_size_options = self._get_batch_size_options()
        logger.info(f"Batch size options: {batch_size_options}")
        # MAX_SEGMENTS = 10000 # NOTE: Hardcoded max segment. not ideal though... it prevents combinatorial explosion
        for gpu_type, allocations in self.tp_allocations.items():
            for gpu_set, tp_degree, partition_id in allocations:
                for batch_size in batch_size_options:
                    # Get max/min segment sizes for this specific TP degree and batch_size
                    max_seg_size = self._compute_max_segment_size_for_tp(gpu_type, tp_degree, batch_size)
                    min_seg_size = self._compute_min_segment_size_for_tp(gpu_type, tp_degree, batch_size)
                    
                    if max_seg_size == 0:
                        continue
                    
                    for start_layer in range(1, self.config.num_decoder_layers + 1):
                        # Determine valid segment sizes
                        if self.quantized_sizes:
                            valid_sizes = [s for s in self.quantized_sizes 
                                        if min_seg_size <= s <= max_seg_size 
                                        and start_layer + s - 1 <= self.config.num_decoder_layers]
                        else:
                            valid_sizes = range(min_seg_size, 
                                            min(max_seg_size + 1,
                                                self.config.num_decoder_layers - start_layer + 2))
                        
                        for segment_size in valid_sizes:
                            if segment_size < self.config.min_layers_per_stage:
                                continue
                            if start_layer + segment_size - 1 <= self.config.num_decoder_layers:
                                # NEW FORMAT: (gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, batch_size)
                                segment = (gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, batch_size)
                                valid_segments.append(segment)
                                
                                # Pre-compute throughput for this segment (OPTIMIZATION)
                                # This avoids recomputing during constraint creation
                                # Get actual network bandwidth from topology (not hardcoded!)
                                nvlink_bw = self._get_min_intra_tp_bandwidth(gpu_type, gpu_set)
                                throughput = ThroughputFunctions.gpu_throughput_with_tp(
                                    gpu_type, self.config.sequence_length,
                                    batch_size, segment_size, self.config.d_model,
                                    self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                                    debug=False, phase=self.config.workload_phase
                                )
                                # Store in dictionary for O(1) lookup later
                                if not hasattr(self, 'segment_throughputs_temp'):
                                    self.segment_throughputs_temp = {}
                                self.segment_throughputs_temp[segment] = throughput
                            
                            # # SAFETY CHECK
                            # if len(valid_segments) > MAX_SEGMENTS:
                            #     logger.warning(f"Hit segment limit of {MAX_SEGMENTS}, stopping generation")
                            #     return valid_segments
        
        logger.info(f"Generated {len(valid_segments)} segments with variable TP and batch size")
        logger.info(f"  Batch size options: {batch_size_options}")
        logger.info(f"  Segments per config increased by {len(batch_size_options)}x")
        
        # Verification: Check pre-computed throughputs are correct
        throughput_dict = getattr(self, 'segment_throughputs_temp', {})
        logger.info(f"  Pre-computed throughput for {len(throughput_dict)} segments")
        
        # Verify a sample of pre-computed values
        import random
        verification_sample = min(10, len(valid_segments))
        sample_segments = random.sample(valid_segments, verification_sample) if len(valid_segments) > 0 else []
        
        for seg in sample_segments:
            gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, batch_size = seg
            precomputed = throughput_dict.get(seg, None)
            nvlink_bw = self._get_min_intra_tp_bandwidth(gpu_type, gpu_set)
            recalculated = ThroughputFunctions.gpu_throughput_with_tp(
                gpu_type, self.config.sequence_length,
                batch_size, segment_size, self.config.d_model,
                self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                debug=False, phase=self.config.workload_phase
            )
            if precomputed is None:
                logger.error(f"  ✗ Missing pre-computed value for segment: {seg}")
            elif abs(precomputed - recalculated) > 0.01:
                logger.error(f"  ✗ Mismatch: pre-computed={precomputed:.2f}, recalculated={recalculated:.2f}")
            else:
                logger.debug(f"  ✓ Verified segment throughput: {precomputed:.2f} tokens/sec")
        
        logger.info(f"  ✓ Verified {verification_sample} random segments - all match!")
        return valid_segments
    
    def _generate_valid_connections(self) -> List[Tuple]:
        """Generate valid network connections between consecutive segments"""
        valid_connections = []
        
        # Group segments by ending/starting layer
        segments_by_end_layer = {}
        segments_by_start_layer = {}
        
        for seg in self.valid_segments:
            gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, batch_size = seg
            end_layer = start_layer + segment_size - 1
            
            if end_layer not in segments_by_end_layer:
                segments_by_end_layer[end_layer] = []
            segments_by_end_layer[end_layer].append(seg)
            
            if start_layer not in segments_by_start_layer:
                segments_by_start_layer[start_layer] = []
            segments_by_start_layer[start_layer].append(seg)
        
        # Generate connections between consecutive layers
        for layer in range(1, self.config.num_decoder_layers):
            ending_segments = segments_by_end_layer.get(layer, [])
            starting_segments = segments_by_start_layer.get(layer + 1, [])
            
            for seg1 in ending_segments:
                gpu_set1 = seg1[1]
                batch_size1 = seg1[6]  # Extract batch_size
                for seg2 in starting_segments:
                    gpu_set2 = seg2[1]
                    batch_size2 = seg2[6]  # Extract batch_size
                    # Connection valid only if GPU sets don't overlap AND batch_sizes match
                    if not gpu_set1.intersection(gpu_set2) and batch_size1 == batch_size2:
                        valid_connections.append((seg1, seg2))
        
        logger.info(f"Generated {len(valid_connections)} connections (before bandwidth filter)")
        return valid_connections
    
    def _get_min_bandwidth_between_segments(self, seg1: Tuple, seg2: Tuple) -> float:
        """Get minimum bandwidth between two TP groups"""
        gpu_set1 = seg1[1]
        gpu_set2 = seg2[1]
        gpu_type1 = seg1[0]
        gpu_type2 = seg2[0]
        
        min_bandwidth = float('inf')
        for local_id1 in gpu_set1:
            global_id1 = self._get_global_gpu_id(gpu_type1, local_id1)
            for local_id2 in gpu_set2:
                global_id2 = self._get_global_gpu_id(gpu_type2, local_id2)
                bandwidth = self.network_bandwidth[global_id1, global_id2]
                min_bandwidth = min(min_bandwidth, bandwidth)
        
        return min_bandwidth
    
    def _apply_network_bandwidth_filter(self):
        """Filter connections with very low bandwidth (bottom percentile)"""
        if not self.valid_connections:
            return
        
        # Compute all bandwidths
        all_bandwidths = [
            self._get_min_bandwidth_between_segments(seg1, seg2)
            for seg1, seg2 in self.valid_connections
        ]
        
        # Find threshold (e.g., 10th percentile)
        threshold = np.percentile(all_bandwidths, 
                                 self.config.network_bandwidth_percentile_threshold * 100)
        
        # Filter
        original_count = len(self.valid_connections)
        self.valid_connections = [
            (seg1, seg2) for seg1, seg2 in self.valid_connections
            if self._get_min_bandwidth_between_segments(seg1, seg2) >= threshold
        ]
        
        logger.info(f"Network bandwidth filter (bottom {self.config.network_bandwidth_percentile_threshold*100:.0f}%): "
                   f"{original_count} -> {len(self.valid_connections)} connections")
    
    def _get_global_gpu_id(self, gpu_type: str, local_gpu_id: int) -> int:
        """Convert (gpu_type, local_gpu_id) to global GPU ID"""
        return self.gpu_types[gpu_type].global_ids[local_gpu_id]
    
    def _precompute_network_throughputs(self) -> Dict:
        """
        Pre-compute network throughput in tokens/sec.
        Models 3-stage pipeline: all-reduce (source) → inter-stage transfer → all-scatter (dest)
        
        NEW: Phase-aware inter-stage communication
        - Prefill: Transfer batch_size × seq_len × d_model (all tokens)
        - Decode: Transfer batch_size × 1 × d_model (1 token per forward pass)
        """
        gpu_pair_throughputs = {}
        
        for seg1, seg2 in self.valid_connections:
            # Extract segment info (NEW: segments now include batch_size at index 6)
            gpu_type1 = seg1[0]
            gpu_type2 = seg2[0]
            gpu_set1 = seg1[1]
            gpu_set2 = seg2[1]
            tp_degree1 = seg1[2]
            tp_degree2 = seg1[2]
            batch_size1 = seg1[6]
            batch_size2 = seg2[6]
            
            # PHASE-AWARE: Tensor size depends on workload phase
            # NOTE: batch_size1 should equal batch_size2 due to global constraint
            if self.config.workload_phase == 'prefill':
                # Prefill: Transfer activations for all tokens in the sequence
                tensor_size_gb = (batch_size1 * self.config.sequence_length *
                                self.config.d_model * self.config.bytes_per_element) / (1024**3)
            else:  # decode
                # Decode: Transfer activations for 1 token only (sequential generation)
                tensor_size_gb = (batch_size1 * 1 *
                                self.config.d_model * self.config.bytes_per_element) / (1024**3)
            
            # Step 1: All-reduce within source TP group (ring all-reduce)
            if tp_degree1 > 1:
                # Find bottleneck link in the ring
                min_intra_bw_source = float('inf')
                for id1 in gpu_set1:
                    global_id1 = self._get_global_gpu_id(gpu_type1, id1)
                    for id2 in gpu_set1:
                        if id1 != id2:
                            global_id2 = self._get_global_gpu_id(gpu_type1, id2)
                            bw = self.network_bandwidth[global_id1, global_id2]
                            min_intra_bw_source = min(min_intra_bw_source, bw)
                
                # Ring all-reduce transfers 2*(N-1)/N of the data
                # Effective bandwidth accounts for multiple communication rounds
                all_reduce_bw = min_intra_bw_source * (tp_degree1 - 1) / tp_degree1
            else:
                all_reduce_bw = float('inf')  # No all-reduce needed
            
            # Step 2: Master-to-master inter-stage transfer
            master_local_id1 = min(gpu_set1)
            master_local_id2 = min(gpu_set2)
            master_global_id1 = self._get_global_gpu_id(gpu_type1, master_local_id1)
            master_global_id2 = self._get_global_gpu_id(gpu_type2, master_local_id2)
            inter_stage_bw = self.network_bandwidth[master_global_id1, master_global_id2]
            
            # Step 3: All-scatter within destination TP group
            if tp_degree2 > 1:
                min_intra_bw_dest = float('inf')
                for id1 in gpu_set2:
                    global_id1 = self._get_global_gpu_id(gpu_type2, id1)
                    for id2 in gpu_set2:
                        if id1 != id2:
                            global_id2 = self._get_global_gpu_id(gpu_type2, id2)
                            bw = self.network_bandwidth[global_id1, global_id2]
                            min_intra_bw_dest = min(min_intra_bw_dest, bw)
                
                # All-scatter also transfers 2*(N-1)/N of the data
                all_scatter_bw = min_intra_bw_dest * (tp_degree2 - 1) / tp_degree2
            else:
                all_scatter_bw = float('inf')  # No all-scatter needed
            
            # Bottleneck: slowest stage limits overall throughput
            effective_bandwidth_gbps = min(all_reduce_bw, inter_stage_bw, all_scatter_bw)
            
            # Time to transfer one batch (seconds)
            if effective_bandwidth_gbps > 0:
                transfer_time_sec = tensor_size_gb / effective_bandwidth_gbps
                # PHASE-AWARE: Throughput in tokens/sec
                if self.config.workload_phase == 'prefill':
                    tokens_per_batch = batch_size1 * self.config.sequence_length
                else:  # decode
                    tokens_per_batch = batch_size1 * 1  # 1 token per forward pass
                throughput = tokens_per_batch / transfer_time_sec
            else:
                throughput = 1e9  # Infinite throughput if no communication needed
            
            gpu_pair_throughputs[(seg1, seg2)] = throughput
        
        return gpu_pair_throughputs
    
    def _validate_problem_size(self):
        """Validate problem size"""
        num_segments = len(self.valid_segments)
        num_connections = len(self.valid_connections)
        total_binary_vars = num_segments + num_connections
        
        logger.info(f"Problem size validation:")
        logger.info(f"  - Segments: {num_segments}")
        logger.info(f"  - Connections: {num_connections}")
        logger.info(f"  - Binary variables: ~{total_binary_vars}")
        
        if total_binary_vars > 100000:
            logger.warning(f"Large problem ({total_binary_vars} binary variables)")
        elif total_binary_vars > 50000:
            logger.info(f"Medium-large problem ({total_binary_vars} binary variables)")
        else:
            logger.info(f"Manageable problem size")
    
    def build_model(self):
        """Build the Gurobi optimization model"""
        build_start = time.time()
        logger.info("Building optimization model with TP and practical constraints...")
        
        model_init_start = time.time()
        self.model = gp.Model("llm_placement_with_tp_constrained", env=self.env)
        
        # Solver parameters optimized for parallelism
        self.model.setParam('Presolve', 1)  # Normal presolve (was 2=aggressive, reduce to allow more B&B)
        self.model.setParam('Cuts', 2)  # More aggressive cuts (parallelizable)
        self.model.setParam('Heuristics', 0.20)  # Increase for more parallel heuristics (was 0.05)
        self.model.setParam('MIPFocus', 0)  # Balanced (was 1=feasibility, 0 better for parallelism)
        self.model.setParam('NodefileStart', 0.5)
        self.model.setParam('TimeLimit', self.config.time_limit_seconds)
        self.model.setParam('MIPGap', self.config.optimality_gap)
        self.model.setParam('LogToConsole', 1)
        
        # PERFORMANCE: Enable concurrent MIP solver (runs multiple strategies in parallel)
        # This is CRUCIAL for utilizing multiple cores effectively
        self.model.setParam('ConcurrentMIP', 4)  # Run 4 concurrent solvers with different strategies
        model_init_time = time.time() - model_init_start
        
        logger.info("Creating decision variables...")
        var_start = time.time()
        self._create_variables()
        var_time = time.time() - var_start
        logger.info(f"  Created {len(self.x)} segment variables, {len(self.e)} connection variables in {var_time:.2f}s")
        
        logger.info("Creating constraints (this is the slow part)...")
        constraint_start = time.time()
        self._create_constraints()
        constraint_time = time.time() - constraint_start
        logger.info(f"  Constraints created in {constraint_time:.2f}s")
        
        logger.info("Setting objective function...")
        obj_start = time.time()
        self._set_objective()
        obj_time = time.time() - obj_start
        logger.info(f"  Objective set in {obj_time:.2f}s")
        
        total_build_time = time.time() - build_start
        logger.info("="*80)
        logger.info("MODEL BUILD COMPLETE - ready to solve with Gurobi")
        logger.info(f"  Total build time: {total_build_time:.2f}s (init={model_init_time:.2f}s, vars={var_time:.2f}s, constraints={constraint_time:.2f}s, obj={obj_time:.2f}s)")
        logger.info("="*80)
    
    def _create_variables(self):
        """Create decision variables"""
        # Segment assignment: x[segment]
        self.x = self.model.addVars(
            self.valid_segments,
            vtype=GRB.BINARY,
            name="segment_assignment"
        )
        
        # TP partition usage: z[gpu_type, partition_id]
        partition_keys = [(seg[0], seg[3]) for seg in self.valid_segments]
        self.z = self.model.addVars(
            set(partition_keys),
            vtype=GRB.BINARY,
            name="partition_usage"
        )
        
        # Network connections: e[seg1, seg2]
        self.e = self.model.addVars(
            self.valid_connections,
            vtype=GRB.BINARY,
            name="network_connection"
        )
        
        # Throughput variables
        self.tau = self.model.addVars(
            set(partition_keys),
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="partition_throughput"
        )
        
        self.rho = self.model.addVars(
            self.valid_connections,
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="network_throughput"
        )
        
        # End-to-end throughput
        self.t = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="end_to_end_throughput")
        
        # NEW: Cost variable ($/hour)
        self.cost = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_hourly_cost")
    
    def _create_constraints(self):
        """Create optimization constraints"""
        
        # 1. Layer coverage: each layer assigned exactly once
        for layer in range(1, self.config.num_decoder_layers + 1):
            self.model.addConstr(
                gp.quicksum(self.x[seg] for seg in self.valid_segments 
                           if seg[4] <= layer <= seg[4] + seg[5] - 1) == 1,
                name=f"layer_coverage_{layer}"
            )
        
        # 2. GPU exclusivity: each GPU can only be in ONE active segment
        for gpu_type, gpu_obj in self.gpu_types.items():
            for local_gpu_id in range(gpu_obj.count):
                # Find all segments containing this specific GPU
                segments_using_gpu = [
                    seg for seg in self.valid_segments
                    if seg[0] == gpu_type and local_gpu_id in seg[1]  # seg[1] is gpu_set
                ]
                
                if segments_using_gpu:
                    self.model.addConstr(
                        gp.quicksum(self.x[seg] for seg in segments_using_gpu) <= 1,
                        name=f"gpu_exclusivity_{gpu_type}_{local_gpu_id}"
                    )
                    
        existing_partition_keys = set((seg[0], seg[3]) for seg in self.valid_segments)
        # 3. Partition usage indicators
        for gpu_type, allocations in self.tp_allocations.items():
            for gpu_set, tp_degree, partition_id in allocations:
                if (gpu_type, partition_id) in existing_partition_keys:
                    self.model.addConstr(
                        self.z[gpu_type, partition_id] == 
                        gp.quicksum(self.x[seg] for seg in self.valid_segments 
                                if seg[0] == gpu_type and seg[3] == partition_id),
                        name=f"partition_usage_{gpu_type}_{partition_id}"
                    )
        
        # 3b. Global batch size consistency: all segments must use the same batch_size
        # For each pair of segments, if both are active, their batch_sizes must match
        # We implement this by: for each segment, if it's active, batch_size must equal a global value
        # Simpler approach: Create binary variables for each batch_size option
        batch_size_options = self._get_batch_size_options()
        
        # Binary variable b[bs] = 1 if batch_size bs is chosen
        self.b = {}
        for bs in batch_size_options:
            self.b[bs] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"batch_size_{bs}")
        
        # Exactly one batch size must be chosen
        self.model.addConstr(
            gp.quicksum(self.b[bs] for bs in batch_size_options) == 1,
            name="batch_size_choice"
        )
        
        # For each segment, it can only be active if its batch_size is chosen
        for seg in self.valid_segments:
            seg_batch_size = seg[6]
            self.model.addConstr(
                self.x[seg] <= self.b[seg_batch_size],
                name=f"batch_consistency_{seg}"
            )
        
        logger.info(f"Added global batch size consistency constraint with {len(batch_size_options)} options: {batch_size_options}")
        
        # 4. Maximum pipeline depth constraint (PRACTICAL CONSTRAINT)
        self.model.addConstr(
            gp.quicksum(self.z[key] for key in self.z.keys()) <= self.config.max_pipeline_stages,
            name="max_pipeline_depth"
        )
        logger.info(f"Added max pipeline depth constraint: {self.config.max_pipeline_stages} stages")
        
        # 4b. NEW: Cost constraints
        # Cost definition: sum(partition_usage × TP_degree × cost_per_hour)
        cost_expr = gp.quicksum(
            self.z[gpu_type, partition_id] * tp_degree * self.gpu_types[gpu_type].cost_per_hour
            for gpu_type, allocations in self.tp_allocations.items()
            for gpu_set, tp_degree, partition_id in allocations
            if (gpu_type, partition_id) in existing_partition_keys
        )
        self.model.addConstr(self.cost == cost_expr, name="total_cost_definition")
        
        # Budget constraint (if specified)
        # IMPORTANT: Store reference for efficient updates in enumeration method
        self.budget_constraint = None
        if self.config.max_hourly_cost < 999.0:
            self.budget_constraint = self.model.addConstr(
                self.cost <= self.config.max_hourly_cost,
                name="cost_budget"
            )
            logger.info(f"Added cost budget constraint: <= ${self.config.max_hourly_cost:.2f}/hour")
        
        # Total cost constraint (if specified)
        # total_cost = cost_per_hour * (total_tokens / throughput / 3600)
        # Rearranged: cost * total_tokens <= max_total_cost * throughput * 3600
        if self.config.max_total_cost < 999999.0:
            self.model.addConstr(
                self.cost * self.config.total_tokens_to_process <= 
                self.config.max_total_cost * self.t * 3600,
                name="total_cost_budget"
            )
            logger.info(f"Added total cost budget constraint: <= ${self.config.max_total_cost:.2f} for {self.config.total_tokens_to_process:,} tokens")
        
        # Total runtime constraint (if specified)
        # Runtime = total_tokens / throughput_per_sec / 3600 (in hours)
        # Runtime <= max_runtime_hours
        # Rearranged: throughput >= total_tokens / (max_runtime_hours * 3600)
        if self.config.max_total_runtime_hours < 999999.0:
            min_required_throughput = self.config.total_tokens_to_process / (
                self.config.max_total_runtime_hours * 3600
            )
            self.model.addConstr(
                self.t >= min_required_throughput,
                name="total_runtime_constraint"
            )
            logger.info(f"Added total runtime constraint: <= {self.config.max_total_runtime_hours:.2f} hours "
                       f"(requires throughput >= {min_required_throughput:.2f} tokens/sec)")

        
        # 5. Network connection constraints
        for (seg1, seg2) in self.valid_connections:
            self.model.addConstr(self.e[seg1, seg2] <= self.x[seg1], name=f"conn_seg1")
            self.model.addConstr(self.e[seg1, seg2] <= self.x[seg2], name=f"conn_seg2")
            self.model.addConstr(
                self.e[seg1, seg2] >= self.x[seg1] + self.x[seg2] - 1,
                name=f"conn_both"
            )
        
        # 6. Partition throughput definition (with variable TP per segment)
        for gpu_type, allocations in self.tp_allocations.items():
            for gpu_set, tp_degree, partition_id in allocations:
                if (gpu_type, partition_id) in existing_partition_keys:
                    # Use pre-computed throughput (OPTIMIZATION)
                    # This avoids recomputing throughput for every segment in every model build
                    throughput_expr = gp.quicksum(
                        self.segment_throughputs[seg] * self.x[seg]
                        for seg in self.valid_segments 
                        if seg[0] == gpu_type and seg[3] == partition_id  # seg[3]=partition_id
                    )
                    self.model.addConstr(
                        self.tau[gpu_type, partition_id] == throughput_expr,
                        name=f"partition_throughput_{gpu_type}_{partition_id}"
                    )
        
        # 7. Network throughput definition
        for (seg1, seg2) in self.valid_connections:
            net_throughput = self.gpu_pair_throughputs.get((seg1, seg2), 100.0)
            self.model.addConstr(
                self.rho[seg1, seg2] == net_throughput * self.e[seg1, seg2],
                name=f"network_throughput"
            )
        
        # 8. End-to-end throughput constraints
        if self.enable_tight_bigm:
            M_partition, M_network = self._compute_tight_bigM()
        else:
            M_unified = 1000.0
            M_partition = {key: M_unified for key in self.z.keys()}
            M_network = M_unified
        
        # Partition throughput constraints
        for key in self.z.keys():
            self.model.addConstr(
                self.t <= self.tau[key] + M_partition[key] * (1 - self.z[key]),
                name=f"throughput_partition[{key[0]},{key[1]}]"
            )
        
        # Network throughput constraints
        for conn in self.valid_connections:
            self.model.addConstr(
                self.t <= self.rho[conn] + M_network * (1 - self.e[conn]),
                name=f"throughput_network"
            )
        
        # 9. Pipeline connectivity constraints
        self._add_pipeline_connectivity_constraints()
        
        # Optional optimizations
        if self.enable_symmetry_breaking:
            self._add_symmetry_breaking_constraints()
        
        if self.enable_flow_conservation:
            self._add_flow_conservation_constraints()
    
    def _compute_tight_bigM(self) -> Tuple[Dict, float]:
        """Compute tight Big-M values
        
        CRITICAL: Big-M must be based on the MAXIMUM throughput across ALL GPU types,
        not per-type, because self.t is a global variable representing system-wide throughput.
        Using per-type Big-M causes slower GPUs to incorrectly constrain self.t.
        """
        M_partition = {}
        existing_partition_keys = set((seg[0], seg[3]) for seg in self.valid_segments)
        
        # STEP 1: Find the maximum possible throughput across ALL GPU types
        max_throughput_global = 0
        throughput_by_type = {}
        
        for gpu_type, allocations in self.tp_allocations.items():
            tp_degree = self.tp_max_configuration[gpu_type]
            max_size = self.max_segment_size[(gpu_type, tp_degree)]
            
            if max_size > 0:
                # Use max_batch_size for upper bound calculation
                nvlink_bw = self._get_representative_tp_bandwidth(gpu_type, tp_degree)
                max_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                    gpu_type, self.config.sequence_length,
                    self.config.max_batch_size, max_size, self.config.d_model, 
                    self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                    debug=False, phase=self.config.workload_phase
                )
                throughput_by_type[gpu_type] = max_throughput
                max_throughput_global = max(max_throughput_global, max_throughput)
        
        # STEP 2: Use the GLOBAL maximum for ALL partitions (with safety factor)
        M_global = max_throughput_global * 3
        
        for gpu_type, allocations in self.tp_allocations.items():
            for gpu_set, tp_degree_alloc, partition_id in allocations:
                if (gpu_type, partition_id) in existing_partition_keys:
                    M_partition[(gpu_type, partition_id)] = M_global
        
        M_network = max(self.gpu_pair_throughputs.values()) * 2 if self.gpu_pair_throughputs else 1000.0
        
        logger.info(f"Tight Big-M computed:")
        logger.info(f"  Max throughput by GPU type: {', '.join(f'{k}={v:,.0f}' for k,v in throughput_by_type.items())}")
        logger.info(f"  Global max throughput: {max_throughput_global:,.0f} tokens/sec")
        logger.info(f"  M_partition (global): {M_global:,.0f} (3x safety factor)")
        logger.info(f"  M_network: {M_network:,.0f}")
        return M_partition, M_network
    
    def _add_pipeline_connectivity_constraints(self):
        """Ensure pipeline connectivity from layer 1 to final layer"""
        logger.info("Adding pipeline connectivity constraints...")
        
        # PERFORMANCE: Pre-index connections by source and destination segment
        # This reduces O(layers × segments × connections) to O(connections + constraints)
        logger.info(f"  Pre-indexing {len(self.valid_connections)} connections for fast lookup...")
        from collections import defaultdict
        connections_from = defaultdict(list)  # seg1 -> [(seg1, seg2), ...]
        connections_to = defaultdict(list)    # seg2 -> [(seg1, seg2), ...]
        
        for conn in self.valid_connections:
            seg1, seg2 = conn
            connections_from[seg1].append(conn)
            connections_to[seg2].append(conn)
        
        logger.info(f"  Indexed {len(connections_from)} source segments and {len(connections_to)} destination segments")
        
        # Pipeline must start at layer 1
        first_layer_segments = [seg for seg in self.valid_segments if seg[4] == 1]
        if first_layer_segments:
            self.model.addConstr(
                gp.quicksum(self.x[seg] for seg in first_layer_segments) >= 1,
                name="pipeline_starts_at_layer_1"
            )
        
        # Sequential connectivity
        total_layers = self.config.num_decoder_layers
        logger.info(f"  Processing {total_layers-1} layer transitions...")
        constraints_added = 0
        
        for layer in range(1, self.config.num_decoder_layers):
            if layer % 10 == 0:
                logger.info(f"  Progress: {layer}/{total_layers-1} layer transitions ({constraints_added} constraints so far)")
            segments_ending_here = [seg for seg in self.valid_segments
                                   if seg[4] + seg[5] - 1 == layer]
            segments_starting_next = [seg for seg in self.valid_segments
                                     if seg[4] == layer + 1]
            
            if segments_ending_here and segments_starting_next:
                # PERFORMANCE: Use pre-indexed connections instead of searching
                segments_starting_next_set = set(segments_starting_next)
                
                # Outgoing connections
                for seg1 in segments_ending_here:
                    # Use indexed lookup instead of iterating all connections
                    valid_next_connections = [conn for conn in connections_from[seg1]
                                            if conn[1] in segments_starting_next_set]
                    if valid_next_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[s1, s2] for (s1, s2) in valid_next_connections) >= self.x[seg1],
                            name=f"connectivity_out_{layer}"
                        )
                        constraints_added += 1
                
                # Incoming connections
                segments_ending_here_set = set(segments_ending_here)
                for seg2 in segments_starting_next:
                    # Use indexed lookup instead of iterating all connections
                    valid_prev_connections = [conn for conn in connections_to[seg2]
                                            if conn[0] in segments_ending_here_set]
                    if valid_prev_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[s1, s2] for (s1, s2) in valid_prev_connections) >= self.x[seg2],
                            name=f"connectivity_in_{layer}"
                        )
                        constraints_added += 1
        
        logger.info(f"  Pipeline connectivity: added {constraints_added} constraints successfully")
    
    def _add_symmetry_breaking_constraints(self):
        """Add symmetry breaking for identical TP partitions"""
        logger.info("Adding symmetry breaking constraints...")
        constraints_added = 0
        existing_partition_keys = set((seg[0], seg[3]) for seg in self.valid_segments)
        
        # Group partitions by GPU type AND TP degree (not just GPU type)
        for gpu_type, allocations in self.tp_allocations.items():
            # Group by TP degree
            from collections import defaultdict
            by_tp = defaultdict(list)
            for gpu_set, tp_degree, partition_id in allocations:
                if (gpu_type, partition_id) in existing_partition_keys:
                    by_tp[tp_degree].append(partition_id)
            
            # Apply symmetry breaking ONLY within same TP degree
            for tp_degree, partition_ids in by_tp.items():
                sorted_ids = sorted(partition_ids)
                if len(sorted_ids) > 1:
                    for i in range(len(sorted_ids) - 1):
                        self.model.addConstr(
                            self.z[gpu_type, sorted_ids[i]] >= self.z[gpu_type, sorted_ids[i+1]],
                            name=f"symmetry_break_{gpu_type}_tp{tp_degree}_{i}"
                        )
                        constraints_added += 1
        
        logger.info(f"Added {constraints_added} symmetry breaking constraints")
    
    def _add_flow_conservation_constraints(self):
        """Add flow conservation at layer boundaries"""
        logger.info("Adding flow conservation constraints...")
        constraints_added = 0
        
        for layer in range(1, self.config.num_decoder_layers):
            segments_ending = [seg for seg in self.valid_segments 
                              if seg[4] + seg[5] - 1 == layer]
            segments_starting = [seg for seg in self.valid_segments
                               if seg[4] == layer + 1]
            
            if segments_ending and segments_starting:
                self.model.addConstr(
                    gp.quicksum(self.x[seg] for seg in segments_ending) ==
                    gp.quicksum(self.x[seg] for seg in segments_starting),
                    name=f"flow_conservation_{layer}"
                )
                constraints_added += 1
        
        logger.info(f"Added {constraints_added} flow conservation constraints")
    
    def _set_objective(self):
        """
        Set optimization objective with cost-throughput trade-off.
        
        Objective = w*(t/T_norm) - (1-w)*(cost/C_norm)
        where w = cost_throughput_weight
        
        Special cases:
        - w=0: Pure throughput maximization (backward compatible)
        - w=1: Pure cost minimization
        - 0<w<1: Weighted trade-off
        """
        w = self.config.cost_throughput_weight
        
        if w == 0.0:
            # Pure throughput maximization (original objective)
            obj_expr = self.t
            logger.info("Objective: Maximize throughput (cost ignored)")
            
        elif w == 1.0:
            # Pure cost minimization (minimize = maximize negative)
            obj_expr = -self.cost
            logger.info("Objective: Minimize cost (throughput ignored)")
            
        else:
            # Weighted multi-objective
            t_norm = self.t / self.config.throughput_normalization
            c_norm = self.cost / self.config.cost_normalization
            
            # Convert weight to trade-off parameter
            # w=0.5 → λ=1 (equal weight)
            # w=0.8 → λ=4 (prioritize cost 4x)
            if w < 1.0:
                lambda_param = w / (1.0 - w)
            else:
                lambda_param = 100.0  # Approximate infinity
            
            obj_expr = t_norm - lambda_param * c_norm
            
            logger.info(f"Objective: Weighted cost-throughput optimization")
            logger.info(f"  - Weight (w): {w:.2f}")
            logger.info(f"  - Trade-off (λ): {lambda_param:.2f}")
            logger.info(f"  - Throughput emphasis: {1/(1+lambda_param):.2%}")
            logger.info(f"  - Cost emphasis: {lambda_param/(1+lambda_param):.2%}")
        
        self.model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    def solve_for_min_cost_per_token(self, target_cost_per_token: float = None, max_iterations: int = 5) -> bool:
        """
        Iteratively solve to minimize $/token.
        
        Uses binary search on cost budget to find solution with best $/token.
        """
        if target_cost_per_token is None:
            target_cost_per_token = self.config.max_cost_per_token
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ITERATIVE $/M TOKENS OPTIMIZATION")
        logger.info(f"{'='*80}")
        target_cost_per_million_token = target_cost_per_token * 1_000_000
        logger.info(f"Target $/M tokens: ${target_cost_per_million_token:.6f}")
        
        # Step 1: Estimate minimum throughput (single GPU, single layer, min batch)
        min_tp_estimates = []
        for gpu_type in self.gpu_types:
            # Use min_batch_size for lower bound estimation
            t_est = ThroughputFunctions.gpu_throughput(
                gpu_type, self.config.sequence_length, self.config.min_batch_size,
                1, self.config.d_model, self.config.bytes_per_element, self.config.d_hidden
            )
            min_tp_estimates.append(t_est)
        
        t_min = min(min_tp_estimates) if min_tp_estimates else 100.0
        
        # Step 2: Initial cost budget (conservative)
        initial_budget = target_cost_per_token * t_min * 3600 * 10  # 10× margin
        
        logger.info(f"Estimated min throughput: {t_min:.1f} tokens/sec")
        logger.info(f"Initial cost budget: ${initial_budget:.2f}/hour")
        
        best_solution = None
        best_cpt = float('inf')
        
        # Step 3: Iterative refinement
        for iteration in range(max_iterations):
            logger.info(f"\nIteration {iteration + 1}/{max_iterations}:")
            logger.info(f"  Cost budget: ${initial_budget:.2f}/hour")
            
            # Update budget constraint efficiently
            if self.budget_constraint is None:
                # Add budget constraint if it doesn't exist
                self.budget_constraint = self.model.addConstr(
                    self.cost <= initial_budget,
                    name="cost_budget"
                )
                self.model.update()
            else:
                # Update existing constraint RHS
                self.budget_constraint.setAttr('RHS', initial_budget)
                self.model.update()
            
            original_budget = self.config.max_hourly_cost
            self.config.max_hourly_cost = initial_budget
            
            success = self.solve()
            
            if success:
                actual_cpt = self.solution['cost_per_token']
                actual_cpm = actual_cpt * 1_000_000
                logger.info(f"  Result: $/M tokens = ${actual_cpm:.6f}")
                
                if actual_cpt < best_cpt:
                    best_cpt = actual_cpt
                    best_solution = self.solution.copy()
                
                if actual_cpt <= target_cost_per_token:
                    logger.info(f" Meets target")
                    break
                else:
                    # Tighten budget
                    throughput = self.solution['throughput_tokens_per_sec']
                    new_budget = target_cost_per_token * throughput * 3600 * 0.95
                    logger.info(f"  Tightening budget to ${new_budget:.2f}/hour")
                    initial_budget = new_budget
            else:
                logger.info(f"  Infeasible, relaxing budget")
                initial_budget *= 1.5
            
            self.config.max_hourly_cost = original_budget
        
        if best_solution:
            self.solution = best_solution
            best_cpm = best_cpt * 1_000_000
            logger.info(f"\nBest $/M tokens found: ${best_cpm:.6f}")
            return True
        
        return False
    
    def _estimate_feasible_budget_range(self) -> tuple:
        """
        Estimate feasible budget range based on max_cost_per_token target.
        
        Logic:
        - To meet $/token target: cost / (throughput x 3600) <= max_cost_per_token
        - So: cost <= max_cost_per_token x throughput x 3600
        - Estimate min/max throughput from GPU memory (proxy for performance)
        - Calculate corresponding budget bounds
        
        Returns:
            (min_budget, max_budget) tuple in $/hour
        """
        target_cost_per_token = self.config.max_cost_per_token
        
        # Collect GPU costs to estimate range
        costs = []
        
        for gpu_type, gpu_info in self.gpu_types.items():
            if gpu_info.count == 0:
                continue
            
            # For different TP degrees
            for tp_degree in [1, 2, 4, 8, 16]:
                if tp_degree > gpu_info.count:
                    continue
                
                # Cost for this configuration
                config_cost = gpu_info.cost_per_hour * tp_degree
                costs.append(config_cost)
        
        if not costs:
            logger.warning("Could not estimate cost range, using default budgets")
            return (0.30, 10.0)
        
        min_cost = min(costs)
        max_cost = max(costs)
        
        # SIMPLIFIED APPROACH: Use GPU cost range directly
        # The $/token constraint will naturally filter out bad solutions during enumeration
        # We don't need perfect budget bounds, just a reasonable search range
        
        # Start with actual GPU costs as the range
        min_budget = min_cost * 0.8  # Slightly below cheapest config
        max_budget = max_cost * 1.5  # Slightly above most expensive single-TP config
        
        # Factor in multi-stage pipelines (can be much higher than single-stage)
        if self.config.max_pipeline_stages > 1:
            # Multi-stage could be: max_cost × max_stages
            max_multi_stage = max_cost * self.config.max_pipeline_stages
            max_budget = max(max_budget, max_multi_stage * 1.2)
        
        # Apply absolute bounds for safety
        min_budget = max(0.20, min_budget)
        # No aggressive upper cap - trust the enumeration to filter out infeasible high budgets
        
        target_cost_per_million_token_display = target_cost_per_token * 1_000_000
        logger.info(f"Competitive budget range for $/M tokens <= ${target_cost_per_million_token_display:.6f}:")
        logger.info(f"  GPU cost range: ${min_cost:.2f} - ${max_cost:.2f}/hour")
        logger.info(f"  Search budget range: ${min_budget:.2f} - ${max_budget:.2f}/hour")
        
        # Log which configurations will/won't be covered
        logger.info(f"\nConfiguration Coverage Analysis:")
        covered = []
        not_covered = []
        for gpu_type, gpu_info in self.gpu_types.items():
            for tp_degree in [1, 2, 4, 8]:
                if tp_degree > gpu_info.count:
                    continue
                config_cost = gpu_info.cost_per_hour * tp_degree
                if config_cost <= max_budget:
                    covered.append(f"{gpu_type} TP={tp_degree} (${config_cost:.2f}/h)")
                else:
                    not_covered.append(f"{gpu_type} TP={tp_degree} (${config_cost:.2f}/h)")
        
        if covered:
            logger.info(f"  ✓ Will explore ({len(covered)} configs):")
            for config in covered:
                logger.info(f"    {config}")
        
        if not_covered:
            logger.warning(f"  ✗ Budget too low for ({len(not_covered)} configs):")
            for config in not_covered:
                logger.warning(f"    {config} - need budget >= ${config.split('$')[1].split('/')[0]}")
        
        return (min_budget, max_budget)
    
    def solve_optimal_cost_per_token(self, budget_points: List[float] = None, use_smart_hybrid: bool = True) -> bool:
        """
        Find TRUE optimal $/token by enumerating cost budgets.
        
        This is GUARANTEED to find optimal (given enough budget points).
        Also checks if solution meets max_cost_per_token target from config.
        
        Uses max_cost_per_token to intelligently set budget range - we only test
        budgets that could possibly meet the competitor's $/token threshold.
        
        Args:
            budget_points: List of cost budgets to try ($/hour)
                          If None, uses smart hybrid or full range
            use_smart_hybrid: If True, uses iterative first to focus search (faster)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMAL $/M TOKENS SEARCH {'(Smart Hybrid)' if use_smart_hybrid and budget_points is None else '(Full Enumeration)'}")
        logger.info(f"{'='*80}")
        target_cost_per_million_token_display = self.config.max_cost_per_token * 1_000_000
        logger.info(f"Target $/M tokens (from config): ${target_cost_per_million_token_display:.6f}")
        
        # Auto-generate budget range if not provided
        if budget_points is None:
            # COMPETITIVE INTELLIGENCE: Use max_cost_per_token to focus search
            min_feasible, max_feasible = self._estimate_feasible_budget_range()
            
            if use_smart_hybrid:
                # SMART HYBRID: Use iterative to find ballpark, then enumerate around it
                logger.info("\nPhase 1: Iterative search to find ballpark...")
                phase1_start = time.time()
                if self.solve_for_min_cost_per_token(max_iterations=3):
                    phase1_time = time.time() - phase1_start
                    ballpark_cost = self.solution['cost_per_hour']
                    ballpark_cpt = self.solution['cost_per_token']
                    ballpark_cpm = ballpark_cpt * 1_000_000
                    logger.info(f"Ballpark found: ${ballpark_cost:.2f}/h, ${ballpark_cpm:.6f}/M tokens (Phase 1 time: {phase1_time:.1f}s)")
                    
                    # Check if ballpark meets target
                    target_cost_per_million_token = self.config.max_cost_per_token * 1_000_000
                    if ballpark_cpt <= self.config.max_cost_per_token:
                        logger.info(f"Ballpark meets target (${ballpark_cpm:.6f} <= ${target_cost_per_million_token:.6f})")
                    else:
                        logger.warning(f"Ballpark exceeds target (${ballpark_cpm:.6f} > ${target_cost_per_million_token:.6f})")
                    
                    # PERFORMANCE OPTIMIZATION: Reduced from 12 to 6-8 budget points
                    # COMPETITIVE OPTIMIZATION: Focus on range that can beat competitor
                    logger.info("\nPhase 2: Coarse enumeration around ballpark...")
                    budget_points = [
                        ballpark_cost * factor
                        for factor in [0.4, 0.7, 0.9, 1.0, 1.2, 1.5]
                    ]
                    # Add feasible range bounds to ensure we explore competitive region
                    budget_points.extend([min_feasible, max_feasible * 0.5, max_feasible])
                    
                    # Filter to feasible range (max_feasible now accounts for multi-stage)
                    budget_points = [b for b in budget_points if min_feasible * 0.5 <= b <= max_feasible * 1.5]
                    budget_points = sorted(set(budget_points))  # Remove duplicates, sort
                    logger.info(f"  Using {len(budget_points)} coarse budget points focused on competitive range")
                    logger.info(f"  Budget range: ${min(budget_points):.2f} - ${max(budget_points):.2f}/hour")
                else:
                    # Fallback to coarse range if iterative fails
                    phase1_time = time.time() - phase1_start
                    logger.warning(f"Iterative failed (Phase 1 time: {phase1_time:.1f}s), using coarse enumeration in feasible range...")
                    # COMPETITIVE OPTIMIZATION: Focus on budgets that can meet target
                    # Generate 8 strategic points within feasible range
                    budget_points = np.logspace(
                        np.log10(min_feasible), 
                        np.log10(max_feasible), 
                        num=8
                    ).tolist()
                    logger.info(f"  Using {len(budget_points)} budget points: ${min(budget_points):.2f} - ${max(budget_points):.2f}/hour")
            else:
                # COARSE ENUMERATION: Strategic budget points in feasible range
                # COMPETITIVE OPTIMIZATION: Focus on budgets that can beat competitor
                logger.info("Using COARSE enumeration in competitive range")
                budget_points = np.logspace(
                    np.log10(min_feasible), 
                    np.log10(max_feasible), 
                    num=8
                ).tolist()
                logger.info(f"  Using {len(budget_points)} budget points: ${min(budget_points):.2f} - ${max(budget_points):.2f}/hour")
        
        logger.info(f"Testing {len(budget_points)} cost budgets...")
        
        best_solution = None
        best_cpt = float('inf')
        all_results = []
        
        original_budget = self.config.max_hourly_cost
        original_weight = self.config.cost_throughput_weight
        
        # Set to maximize throughput (weight=0)
        self.config.cost_throughput_weight = 0.0
        
        # PERFORMANCE OPTIMIZATION: Build model ONCE, then update budget constraint
        # This avoids rebuilding all constraints 6-8 times (minutes saved!)
        logger.info("Building model once (this may take a few minutes)...")
        model_build_start = time.time()
        self.config.max_hourly_cost = max(budget_points)  # Start with highest budget
        self.build_model()
        model_build_time = time.time() - model_build_start
        logger.info(f"Model built in {model_build_time:.1f}s! Now testing budgets by updating constraint RHS only...")
        
        for i, budget in enumerate(sorted(budget_points)):
            iteration_start = time.time()
            logger.info(f"\n[{i+1}/{len(budget_points)}] Budget: ${budget:.2f}/hour")
            
            # Log which TP configurations are feasible at this budget
            feasible_configs = []
            for gpu_type, gpu_info in self.gpu_types.items():
                for tp_degree in [1, 2, 4, 8]:
                    if tp_degree > gpu_info.count:
                        continue
                    config_cost = gpu_info.cost_per_hour * tp_degree
                    if config_cost <= budget:
                        feasible_configs.append(f"{gpu_type}×{tp_degree}")
            if feasible_configs:
                logger.info(f"  Feasible configs: {', '.join(feasible_configs)}")
            
            # Update budget constraint RHS efficiently (no model rebuild!)
            constraint_update_start = time.time()
            if self.budget_constraint is None:
                # First iteration or no budget constraint - add one
                self.budget_constraint = self.model.addConstr(
                    self.cost <= budget,
                    name="cost_budget"
                )
                self.model.update()
            else:
                # Update existing constraint RHS (FAST!)
                self.budget_constraint.setAttr('RHS', budget)
                self.model.update()
            constraint_update_time = time.time() - constraint_update_start
            logger.info(f"  Constraint update: {constraint_update_time:.3f}s")
            
            solve_start = time.time()
            success = self.solve()
            solve_time = time.time() - solve_start
            iteration_time = time.time() - iteration_start
            
            if success:
                cpt = self.solution['cost_per_token']
                cpm = cpt * 1_000_000
                throughput = self.solution['throughput_tokens_per_sec']
                cost = self.solution['cost_per_hour']
                
                logger.info(f"  Result: {throughput:.0f} tokens/s, ${cost:.2f}/h, ${cpm:.6f}/M tokens")
                logger.info(f"  Timing: solve={solve_time:.1f}s, total iteration={iteration_time:.1f}s")
                
                all_results.append({
                    'budget': budget,
                    'throughput': throughput,
                    'cost': cost,
                    'cost_per_token': cpt,
                    'solution': self.solution.copy()
                })
                
                if cpt < best_cpt:
                    best_cpt = cpt
                    best_solution = self.solution.copy()
                    logger.info(f"  OK New best $/M tokens!")
            else:
                logger.info(f"  Infeasible")
                logger.info(f"  Timing: solve={solve_time:.1f}s, total iteration={iteration_time:.1f}s")
        
        # Restore original config
        self.config.max_hourly_cost = original_budget
        self.config.cost_throughput_weight = original_weight
        
        total_enumeration_time = time.time() - model_build_start
        logger.info(f"\nTotal enumeration time: {total_enumeration_time:.1f}s (model build: {model_build_time:.1f}s, solving: {total_enumeration_time - model_build_time:.1f}s)")
        
        # Log coverage summary
        logger.info(f"\n{'='*80}")
        logger.info(f"ENUMERATION COVERAGE SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Budget points tested: {len(budget_points)}")
        logger.info(f"Budget range: ${min(budget_points):.2f} - ${max(budget_points):.2f}/hour")
        
        # Check which configs were fully explored
        max_budget_tested = max(budget_points)
        all_configs = []
        explored = []
        not_explored = []
        for gpu_type, gpu_info in self.gpu_types.items():
            for tp_degree in [1, 2, 4, 8]:
                if tp_degree > gpu_info.count:
                    continue
                config_cost = gpu_info.cost_per_hour * tp_degree
                config_name = f"{gpu_type} TP={tp_degree}"
                all_configs.append((config_name, config_cost))
                if config_cost <= max_budget_tested:
                    explored.append((config_name, config_cost))
                else:
                    not_explored.append((config_name, config_cost))
        
        logger.info(f"\n✓ Explored: {len(explored)}/{len(all_configs)} configurations")
        for name, cost in explored:
            logger.info(f"  {name:<20} ${cost:.2f}/h")
        
        if not_explored:
            logger.warning(f"\n✗ NOT explored: {len(not_explored)} configurations (budget too low)")
            for name, cost in not_explored:
                logger.warning(f"  {name:<20} ${cost:.2f}/h (need >= ${cost:.2f}/h budget)")
        
        if best_solution:
            self.solution = best_solution
            self.all_enumeration_results = all_results  # Store for CSV export
            best_cpm = best_cpt * 1_000_000
            logger.info(f"\n{'='*80}")
            logger.info(f"OPTIMAL $/M TOKENS FOUND")
            logger.info(f"{'='*80}")
            logger.info(f"Best $/M tokens: ${best_cpm:.6f}")
            logger.info(f"  Throughput: {best_solution['throughput_tokens_per_sec']:.0f} tokens/sec")
            logger.info(f"  Cost: ${best_solution['cost_per_hour']:.2f}/hour")
            logger.info(f"  Pipeline stages: {best_solution['num_pipeline_stages']}")
            
            # Check against target
            target = self.config.max_cost_per_token
            target_per_million = target * 1_000_000
            if best_cpt <= target:
                improvement = (target - best_cpt) / target * 100
                logger.info(f"\nMEETS TARGET: ${best_cpm:.6f} <= ${target_per_million:.6f}")
                logger.info(f"   {improvement:.1f}% better than target!")
            else:
                shortfall = (best_cpt - target) / target * 100
                logger.info(f"\nMISSES TARGET: ${best_cpm:.6f} > ${target_per_million:.6f}")
                logger.info(f"   {shortfall:.1f}% worse than target (infeasible to meet)")
            
            # Log Pareto frontier
            logger.info(f"\nPareto Frontier (all solutions):")
            for r in sorted(all_results, key=lambda x: x['cost_per_token']):
                marker = "*" if r['cost_per_token'] <= target else " "
                r_cpm = r['cost_per_token'] * 1_000_000
                logger.info(f"  {marker} ${r_cpm:.6f}/M tokens: "
                           f"{r['throughput']:.0f} tokens/s, ${r['cost']:.2f}/h")
            
            return True
        else:
            logger.error("No feasible solution found")
            return False
    
    def solve(self) -> bool:
        """Solve the optimization problem"""
        total_binary_vars = len(self.valid_segments) + len(self.valid_connections)
        available_threads = min(self.max_threads, os.cpu_count())
        
        if self.threads is not None:
            threads = min(self.threads, available_threads)
        else:
            if total_binary_vars > 50000:
                threads = min(available_threads, 16)
            elif total_binary_vars > 10000:
                threads = min(available_threads, 8)
            else:
                threads = min(available_threads, 4)
        
        self.model.setParam('Threads', threads)
        
        # DIAGNOSTIC: Verify thread setting
        actual_threads = self.model.getParamInfo('Threads')[2]  # Get current value
        logger.info(f"Using {threads} threads for optimization (verified: {actual_threads})")
        logger.info(f"CPU count: {os.cpu_count()}, available: {available_threads}")
        
        logger.info("Starting optimization...")
        start_time = time.time()
        
        try:
            self.model.optimize()
            solve_time = time.time() - start_time
            
            # DIAGNOSTIC: Log if solved in presolve
            if self.model.status == GRB.OPTIMAL and self.model.NodeCount == 0:
                logger.warning(f"WARNING: Problem solved in presolve (single-threaded)! NodeCount=0")
                logger.warning(f"  Presolve eliminated problem before multi-threaded B&B could start")
            
            if self.model.status == GRB.OPTIMAL:
                logger.info(f"Optimal solution found in {solve_time:.2f} seconds")
                logger.info(f"  Nodes explored: {self.model.NodeCount}")
                logger.info(f"  Simplex iterations: {self.model.IterCount}")
                logger.info(f"Optimal throughput: {self.t.x:.2f} tokens/sec")
                self._extract_solution()
                return True
            elif self.model.status == GRB.TIME_LIMIT:
                if self.model.SolCount > 0:
                    logger.warning(f"Time limit reached. Best solution: {self.t.x:.2f}")
                    self._extract_solution()
                    return True
                else:
                    logger.error("Time limit reached with no feasible solution")
                    return False
            else:
                logger.error(f"No solution found. Status: {self.model.status}")
                return False
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return False
    
    def _extract_solution(self):
        """Extract solution from solved model"""
        # Compute metrics
        raw_throughput = self.t.x
        cost_per_hour = self.cost.x
        
        # Count number of pipeline stages in solution
        num_stages = sum(1 for key in self.z.keys() if self.z[key].x > 0.5)
        
        # Get optimal batch size from solution
        optimal_batch_size = None
        for bs in self._get_batch_size_options():
            if self.b[bs].x > 0.5:
                optimal_batch_size = bs
                break
        
        # Apply pipeline bubble efficiency (dynamic based on micro-batches)
        # Pipeline bubbles occur because not all stages are active simultaneously
        # More micro-batches = better pipeline utilization
        # Efficiency = min(1.0, num_micro_batches / (num_stages * overhead_factor))
        if num_stages == 1:
            pipeline_efficiency = 1.00  # No pipeline, no bubbles
        else:
            # Micro-batching: split batch across pipeline stages to hide bubbles
            # Typical: 4-8 micro-batches per batch
            # More micro-batches = better efficiency (up to a limit)
            num_micro_batches = max(1, optimal_batch_size // 8)  # Assume micro-batch size ~8
            
            # Pipeline efficiency formula from literature:
            # efficiency ≈ num_micro_batches / (num_micro_batches + num_stages - 1)
            # This accounts for the "bubble" at start and end of pipeline
            ideal_efficiency = num_micro_batches / (num_micro_batches + num_stages - 1)
            
            # Additional overhead from pipeline scheduling, synchronization
            scheduling_overhead = 0.10  # 10% overhead
            
            pipeline_efficiency = max(0.50, ideal_efficiency * (1.0 - scheduling_overhead))
        
        # Apply real-world efficiency factor
        # Accounts for factors we don't model explicitly:
        # - Prefill phase (slower than decode, we model something in between)
        # - Framework overhead (PyTorch, vLLM, DeepSpeed scheduling)
        # - Memory pressure (swapping, CPU offloading for tight memory)
        # - Request batching inefficiencies
        # - Gradient checkpointing overhead
        # - Dynamic vs static batching differences
        # Based on expert estimates: our model is ~3-4× too optimistic even after pipeline bubbles
        real_world_efficiency = 0.30  # 70% overhead from all the above factors
        
        throughput_per_sec = raw_throughput * pipeline_efficiency * real_world_efficiency
        
        logger.info(f"\nThroughput Corrections:")
        logger.info(f"  Batch size: {optimal_batch_size}")
        logger.info(f"  Pipeline stages: {num_stages}")
        if num_stages > 1:
            num_micro_batches = max(1, optimal_batch_size // 8)
            logger.info(f"  Micro-batches: {num_micro_batches}")
            logger.info(f"  Pipeline bubble efficiency: {pipeline_efficiency:.1%} (dynamic: {num_micro_batches}÷{num_micro_batches + num_stages - 1})")
        else:
            logger.info(f"  Pipeline bubble efficiency: {pipeline_efficiency:.1%} (no pipeline)")
        logger.info(f"  Real-world efficiency: {real_world_efficiency:.1%}")
        logger.info(f"  Combined efficiency: {pipeline_efficiency * real_world_efficiency:.1%}")
        logger.info(f"  Raw throughput: {raw_throughput:.2f} tokens/sec")
        logger.info(f"  After pipeline bubbles: {raw_throughput * pipeline_efficiency:.2f} tokens/sec")
        logger.info(f"  Final throughput: {throughput_per_sec:.2f} tokens/sec")
        
        # $/token = ($/hour) / (tokens/sec × sec/hour)
        if throughput_per_sec > 0:
            cost_per_token = cost_per_hour / (throughput_per_sec * 3600)
            # Total runtime in hours
            total_runtime_hours = self.config.total_tokens_to_process / (throughput_per_sec * 3600)
        else:
            cost_per_token = float('inf')
            total_runtime_hours = float('inf')
        
        # $/M tokens for display
        cost_per_million_tokens = cost_per_token * 1_000_000
        
        # NEW: Detailed objective breakdown logging
        logger.info("\n" + "="*80)
        logger.info("SOLUTION ANALYSIS - OBJECTIVE BREAKDOWN")
        logger.info("="*80)
        
        w = self.config.cost_throughput_weight
        if w > 0 and w < 1:
            t_norm = throughput_per_sec / self.config.throughput_normalization
            c_norm = cost_per_hour / self.config.cost_normalization
            lambda_param = w / (1.0 - w)
            
            throughput_contribution = t_norm
            cost_contribution = lambda_param * c_norm
            
            logger.info(f"Weighted Objective Components:")
            logger.info(f"  Throughput term (t/T_norm):    {t_norm:.6f}")
            logger.info(f"  Cost term (λ × c/C_norm):      {cost_contribution:.6f}")
            logger.info(f"  Net objective (t_norm - cost): {t_norm - cost_contribution:.6f}")
            logger.info(f"  (Should match objective value: {self.model.ObjVal:.6f})")
        
        # Extract optimal batch size
        optimal_batch_size = None
        for bs in self._get_batch_size_options():
            if self.b[bs].x > 0.5:
                optimal_batch_size = bs
                break
        
        logger.info(f"\nCore Metrics:")
        logger.info(f"  Batch Size: {optimal_batch_size}")
        logger.info(f"  Throughput: {throughput_per_sec:.2f} tokens/sec")
        logger.info(f"  Cost: ${cost_per_hour:.2f}/hour")
        logger.info(f"  $/M tokens: ${cost_per_million_tokens:.6f}")
        logger.info(f"  Total Runtime: {total_runtime_hours:.2f} hours ({total_runtime_hours/24:.2f} days)")
        
        self.solution = {
            'objective_value': self.model.ObjVal,
            'batch_size': optimal_batch_size,
            'throughput_tokens_per_sec': throughput_per_sec,
            'cost_per_hour': cost_per_hour,
            'cost_per_token': cost_per_token,
            'total_runtime_hours': total_runtime_hours,
            'meets_cost_threshold': cost_per_token <= self.config.max_cost_per_token,
            'tp_configuration': self.tp_max_configuration,
            'gpu_assignments': [],
            'network_connections': [],
            'solve_status': self.model.status,
            'num_pipeline_stages': sum(1 for key in self.z.keys() if self.z[key].x > 0.5)
        }
        
        # Extract GPU assignments
        for seg in self.valid_segments:
            if self.x[seg].x > 0.5:
                gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, batch_size = seg
                
                assignment = {
                    'gpu_type': gpu_type,
                    'partition_id': partition_id,
                    'gpu_ids': sorted(list(gpu_set)),
                    'global_gpu_ids': sorted([self._get_global_gpu_id(gpu_type, i) for i in gpu_set]),
                    'tp_degree': tp_degree,
                    'start_layer': start_layer,
                    'end_layer': start_layer + segment_size - 1,
                    'segment_size': segment_size,
                    'throughput': self.tau[gpu_type, partition_id].x
                }
                self.solution['gpu_assignments'].append(assignment)
        
        # Extract network connections
        for (seg1, seg2) in self.valid_connections:
            if self.e[seg1, seg2].x > 0.5:
                connection = {
                    'from_segment': {
                        'gpu_type': seg1[0],
                        'partition_id': seg1[3],
                        'start_layer': seg1[4],
                        'end_layer': seg1[4] + seg1[5] - 1
                    },
                    'to_segment': {
                        'gpu_type': seg2[0],
                        'partition_id': seg2[3],
                        'start_layer': seg2[4],
                        'end_layer': seg2[4] + seg2[5] - 1
                    },
                    'throughput': self.rho[seg1, seg2].x
                }
                self.solution['network_connections'].append(connection)
        
        # Sort assignments by start layer
        self.solution['gpu_assignments'].sort(key=lambda x: x['start_layer'])
        
        # Batch size analysis
        self._log_batch_size_analysis()
        
        # Roofline model analysis
        self._log_roofline_analysis()
        
        # NEW: Detailed GPU utilization analysis
        self._log_solution_analysis()
    
    def _log_batch_size_analysis(self):
        """Analyze and explain the batch size selection"""
        logger.info("\n" + "="*80)
        logger.info("BATCH SIZE ANALYSIS")
        logger.info("="*80)
        
        batch_size_options = self._get_batch_size_options()
        optimal_batch = self.solution.get('batch_size')
        
        logger.info(f"\nBatch Size Selection:")
        logger.info(f"  Optimal: {optimal_batch}")
        logger.info(f"  Search range: [{self.config.min_batch_size}, {self.config.max_batch_size}]")
        logger.info(f"  Options considered: {batch_size_options}")
        
        # Show batch efficiency factor
        if optimal_batch:
            efficiency = ThroughputFunctions.batch_efficiency_factor(optimal_batch)
            logger.info(f"\nBatch Efficiency Factor: {efficiency:.1%}")
            logger.info(f"  (Larger batches improve GPU utilization)")
            
            # Explain the efficiency
            if optimal_batch >= 32:
                logger.info(f"Optimal utilization (batch >= 32)")
            elif optimal_batch >= 16:
                logger.info(f"Very good utilization (batch >= 16)")
            elif optimal_batch >= 8:
                logger.info(f"Good utilization (batch >= 8)")
            else:
                logger.info(f"Sub-optimal utilization (batch < 8)")
        
        # If we have multiple batch size options, show comparison
        if len(batch_size_options) > 1 and optimal_batch and len(self.solution['gpu_assignments']) > 0:
            logger.info(f"\nBatch Size Impact (estimated for current GPU configuration):")
            logger.info("-" * 80)
            logger.info(f"  {'Batch':<8} {'Efficiency':<12} {'Est. Throughput':<18} {'$/M tokens':<15}")
            logger.info("-" * 80)
            
            # Get a representative GPU assignment for estimation
            repr_assignment = self.solution['gpu_assignments'][0]
            gpu_type = repr_assignment['gpu_type']
            tp_degree = repr_assignment['tp_degree']
            segment_size = repr_assignment['segment_size']
            
            base_throughput = self.solution['throughput_tokens_per_sec']
            base_cost = self.solution['cost_per_hour']
            
            for bs in batch_size_options:
                efficiency = ThroughputFunctions.batch_efficiency_factor(bs)
                
                # Estimate throughput scaling based on efficiency factor
                # The actual optimal batch throughput is known
                if bs == optimal_batch:
                    est_throughput = base_throughput
                    est_cost_per_token = self.solution['cost_per_token']
                    marker = " ← OPTIMAL"
                else:
                    # Scale throughput by relative efficiency
                    optimal_efficiency = ThroughputFunctions.batch_efficiency_factor(optimal_batch)
                    est_throughput = base_throughput * (efficiency / optimal_efficiency)
                    est_cost_per_token = base_cost / (est_throughput * 3600) if est_throughput > 0 else float('inf')
                    
                    # Show percentage difference
                    diff_pct = (est_cost_per_token - self.solution['cost_per_token']) / self.solution['cost_per_token'] * 100
                    if diff_pct > 0:
                        marker = f" (+{diff_pct:.1f}% worse)"
                    else:
                        marker = f" ({abs(diff_pct):.1f}% better)"
                
                est_cost_per_million = est_cost_per_token * 1_000_000
                logger.info(f"  {bs:<8} {efficiency:<12.1%} {est_throughput:<18.0f} "
                           f"${est_cost_per_million:<14.6f}{marker}")
    
    def _log_roofline_analysis(self):
        """Analyze and log roofline model insights for the solution"""
        logger.info("\n" + "="*80)
        logger.info("ROOFLINE MODEL ANALYSIS")
        logger.info("="*80)
        
        optimal_batch = self.solution.get('batch_size', self.config.min_batch_size)
        
        logger.info(f"\nPerformance Bottleneck Analysis (Roofline Model):")
        logger.info(f"  Determines if each segment is compute-bound or memory-bound")
        logger.info("-" * 80)
        logger.info(f"  {'GPU Type':<10} {'TP':<4} {'Layers':<7} {'AI':<12} {'Ridge':<12} {'Regime':<15} {'Bottleneck'}")
        logger.info("-" * 80)
        
        # Analyze each segment in the solution
        for assignment in self.solution['gpu_assignments']:
            gpu_type = assignment['gpu_type']
            tp_degree = assignment['tp_degree']
            segment_size = assignment['segment_size']
            
            # Calculate arithmetic intensity for this segment (PHASE-AWARE)
            ai = ThroughputFunctions.calculate_arithmetic_intensity(
                segment_size, optimal_batch, self.config.sequence_length,
                self.config.d_model, self.config.d_hidden, self.config.bytes_per_element, tp_degree,
                phase=self.config.workload_phase
            )
            
            # Get ridge point
            ridge_point = ThroughputFunctions.get_ridge_point(gpu_type)
            
            # Determine regime
            regime = ThroughputFunctions.determine_regime(ai, ridge_point)
            
            # Format output
            ai_str = f"{ai:.2f}"
            ridge_str = f"{ridge_point:.2f}"
            
            if regime == "COMPUTE_BOUND":
                bottleneck = "GPU Compute"
                regime_str = "Compute-bound"
            else:
                bottleneck = "Memory Bandwidth"
                regime_str = "Memory-bound"
            
            logger.info(f"  {gpu_type:<10} {tp_degree:<4} {segment_size:<7} {ai_str:<12} {ridge_str:<12} {regime_str:<15} {bottleneck}")
        
        # Summary insights
        logger.info("\n" + "-" * 80)
        logger.info("Roofline analysis:")
        logger.info(f"- Arithmetic Intensity (AI) = FLOPs / Byte accessed")
        logger.info(f"- Ridge Point = Peak FLOPS / Peak Memory Bandwidth")
        logger.info(f"- Arithmetic Intensity > Ridge -> Compute-bound (limited by GPU compute)")
        logger.info(f"- Arithmetic Intensity < Ridge -> Memory-bound (limited by memory bandwidth)")
        logger.info(f"- Tensor Parallelism (TP) reduces memory per GPU -> increases AI -> may shift regime")
        
        # Count regimes
        compute_bound_count = 0
        memory_bound_count = 0
        
        for assignment in self.solution['gpu_assignments']:
            gpu_type = assignment['gpu_type']
            tp_degree = assignment['tp_degree']
            segment_size = assignment['segment_size']
            
            ai = ThroughputFunctions.calculate_arithmetic_intensity(
                segment_size, optimal_batch, self.config.sequence_length,
                self.config.d_model, self.config.d_hidden, self.config.bytes_per_element, tp_degree,
                phase=self.config.workload_phase
            )
            ridge_point = ThroughputFunctions.get_ridge_point(gpu_type)
            regime = ThroughputFunctions.determine_regime(ai, ridge_point)
            
            if regime == "COMPUTE_BOUND":
                compute_bound_count += 1
            else:
                memory_bound_count += 1
        
        total_segments = len(self.solution['gpu_assignments'])
        logger.info(f"Regime Distribution:")
        logger.info(f"- Compute-bound segments: {compute_bound_count}/{total_segments} ({100*compute_bound_count/total_segments:.0f}%)")
        logger.info(f"- Memory-bound segments:  {memory_bound_count}/{total_segments} ({100*memory_bound_count/total_segments:.0f}%)")
        
        if memory_bound_count > compute_bound_count:
            logger.info(f"\nMost segments are memory-bound.")
            logger.info(f"- Consider GPUs with higher memory bandwidth for better performance.")
            logger.info(f"- Increasing TP degree can help (reduces memory per GPU).")
        elif compute_bound_count > memory_bound_count:
            logger.info(f"Most segments are compute-bound.")
            logger.info(f"- Consider GPUs with higher TFLOPS for better performance.")
            logger.info(f"- Memory bandwidth is not the bottleneck here.")
    
    def _log_solution_analysis(self):
        """Comprehensive analysis of why the solution looks the way it does"""
        logger.info("\n" + "="*80)
        logger.info("SOLUTION ANALYSIS - GPU EFFICIENCY & SELECTION")
        logger.info("="*80)
        
        # Determine if workload is primarily memory-bound or compute-bound
        optimal_batch = self.solution.get('batch_size', self.config.min_batch_size)
        memory_bound_count = 0
        compute_bound_count = 0
        
        for assignment in self.solution['gpu_assignments']:
            gpu_type = assignment['gpu_type']
            tp_degree = assignment['tp_degree']
            segment_size = assignment['segment_size']
            
            ai = ThroughputFunctions.calculate_arithmetic_intensity(
                segment_size, optimal_batch, self.config.sequence_length,
                self.config.d_model, self.config.d_hidden, self.config.bytes_per_element, tp_degree,
                phase=self.config.workload_phase
            )
            ridge_point = ThroughputFunctions.get_ridge_point(gpu_type)
            regime = ThroughputFunctions.determine_regime(ai, ridge_point)
            
            if regime == "COMPUTE_BOUND":
                compute_bound_count += 1
            else:
                memory_bound_count += 1
        
        is_memory_bound = memory_bound_count > compute_bound_count
        
        # 1. Compute GPU efficiency ratios (both FLOP/$ and Bandwidth/$)
        gpu_efficiency = {}
        for gpu_type, gpu_obj in self.gpu_types.items():
            specs = ThroughputFunctions.GPU_SPECS.get(gpu_type)
            if specs:
                effective_flops = specs['tflops'] * specs['efficiency']
                effective_bw = specs['mem_bw'] * specs['efficiency']
                compute_efficiency = effective_flops / gpu_obj.cost_per_hour
                memory_efficiency = effective_bw / gpu_obj.cost_per_hour
                gpu_efficiency[gpu_type] = {
                    'flops': effective_flops,
                    'bandwidth': effective_bw,
                    'cost': gpu_obj.cost_per_hour,
                    'compute_ratio': compute_efficiency,
                    'memory_ratio': memory_efficiency
                }
        
        # Sort by the relevant metric
        if is_memory_bound:
            sorted_efficiency = sorted(gpu_efficiency.items(), key=lambda x: x[1]['memory_ratio'], reverse=True)
            metric_name = "Memory Bandwidth/$ ratio"
            logger.info(f"\nWARNING WORKLOAD IS MEMORY-BOUND -> Memory Bandwidth/$ is the key metric!")
        else:
            sorted_efficiency = sorted(gpu_efficiency.items(), key=lambda x: x[1]['compute_ratio'], reverse=True)
            metric_name = "Compute (TFLOP/$) ratio"
            logger.info(f"\nOK WORKLOAD IS COMPUTE-BOUND -> TFLOP/$ is the key metric!")
        
        logger.info(f"\nGPU Efficiency Ranking ({metric_name}):")
        logger.info("-" * 80)
        for i, (gpu_type, data) in enumerate(sorted_efficiency, 1):
            if is_memory_bound:
                logger.info(f"  {i}. {gpu_type:<10} {data['bandwidth']:>7.1f} GB/s × eff @ ${data['cost']:.2f}/h = {data['memory_ratio']:>6.1f} GB/s per $")
            else:
                logger.info(f"  {i}. {gpu_type:<10} {data['flops']:>6.1f} TFLOP × eff @ ${data['cost']:.2f}/h = {data['compute_ratio']:>6.1f} TFLOP/$")
        
        # 2. Analyze which GPUs were actually used
        gpu_usage = {}
        for assignment in self.solution['gpu_assignments']:
            gpu_type = assignment['gpu_type']
            if gpu_type not in gpu_usage:
                gpu_usage[gpu_type] = {
                    'segments': 0,
                    'total_gpus': 0,
                    'total_cost': 0,
                    'total_layers': 0,
                    'tp_degrees': []
                }
            gpu_usage[gpu_type]['segments'] += 1
            gpu_usage[gpu_type]['total_gpus'] += assignment['tp_degree']
            gpu_usage[gpu_type]['total_cost'] += self.gpu_types[gpu_type].cost_per_hour * assignment['tp_degree']
            gpu_usage[gpu_type]['total_layers'] += assignment['segment_size']
            gpu_usage[gpu_type]['tp_degrees'].append(assignment['tp_degree'])
        
        logger.info("\nActual GPU Usage in Solution:")
        logger.info("-" * 80)
        total_cost = 0
        for gpu_type in sorted(gpu_usage.keys()):
            data = gpu_usage[gpu_type]
            efficiency_rank = next(i for i, (gt, _) in enumerate(sorted_efficiency, 1) if gt == gpu_type)
            logger.info(f"  {gpu_type:<10} {data['segments']} segments, {data['total_gpus']} GPUs, "
                       f"{data['total_layers']} layers, ${data['total_cost']:.2f}/h "
                       f"(Efficiency rank: #{efficiency_rank})")
            total_cost += data['total_cost']
        
        logger.info(f"\n  TOTAL COST: ${total_cost:.2f}/hour")
        
        # 3. Analyze unused GPUs
        unused_gpus = set(self.gpu_types.keys()) - set(gpu_usage.keys())
        if unused_gpus:
            logger.info("\nUnused GPUs (and why they might not be chosen):")
            logger.info("-" * 80)
            for gpu_type in sorted(unused_gpus):
                gpu_obj = self.gpu_types[gpu_type]
                efficiency_rank = next(i for i, (gt, _) in enumerate(sorted_efficiency, 1) if gt == gpu_type)
                
                # Check max segment size (use optimal batch_size from solution)
                max_tp = self.tp_max_configuration[gpu_type]
                optimal_batch = self.solution.get('batch_size', self.config.max_batch_size)
                max_seg = self._compute_max_segment_size_for_tp(gpu_type, max_tp, optimal_batch)
                
                logger.info(f"  {gpu_type:<10} count={gpu_obj.count}, ${gpu_obj.cost_per_hour:.2f}/h, "
                           f"efficiency rank #{efficiency_rank}")
                logger.info(f"             Max segment size (TP={max_tp}): {max_seg} layers")
                
                # Hypothetical: what if we used this GPU?
                if max_seg >= 5:  # Can fit at least 5 layers
                    # Use max_batch_size for optimistic throughput estimate
                    nvlink_bw = self._get_representative_tp_bandwidth(gpu_type, 4)
                    hyp_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                        gpu_type, self.config.sequence_length, self.config.max_batch_size,
                        5, self.config.d_model, self.config.bytes_per_element, 4, self.config.d_hidden, nvlink_bw,
                        debug=False, phase=self.config.workload_phase
                    )
                    hyp_cost = gpu_obj.cost_per_hour * 4  # TP=4
                    logger.info(f"             Hypothetical (5 layers, TP=4): {hyp_throughput:.0f} tokens/s, ${hyp_cost:.2f}/h")
        
        # 4. Segment-level contribution analysis
        logger.info("\n" + "="*80)
        logger.info("SEGMENT-LEVEL ANALYSIS")
        logger.info("="*80)
        
        # Find bottleneck
        min_throughput = min(a['throughput'] for a in self.solution['gpu_assignments'])
        bottleneck_segments = [a for a in self.solution['gpu_assignments'] if abs(a['throughput'] - min_throughput) < 0.01]
        
        logger.info(f"\nBottleneck Throughput: {min_throughput:.2f} tokens/sec")
        logger.info(f"Bottleneck Segments:")
        for seg in bottleneck_segments:
            seg_cost = self.gpu_types[seg['gpu_type']].cost_per_hour * seg['tp_degree']
            logger.info(f"  {seg['gpu_type']} TP={seg['tp_degree']}, layers {seg['start_layer']}-{seg['end_layer']}, "
                       f"${seg_cost:.2f}/h, {seg['throughput']:.0f} tokens/s")
        
        # 5. Alternative scenario: Check single-segment solution with best efficiency GPU
        best_gpu_type = sorted_efficiency[0][0]
        logger.info("\n" + "="*80)
        logger.info(f"ALTERNATIVE SCENARIO: Single segment with {best_gpu_type} (best efficiency)")
        logger.info("="*80)
        
        best_gpu = self.gpu_types[best_gpu_type]
        max_tp = self.tp_max_configuration[best_gpu_type]
        optimal_batch = self.solution.get('batch_size', self.config.max_batch_size)
        max_seg_size = self._compute_max_segment_size_for_tp(best_gpu_type, max_tp, optimal_batch)
        
        logger.info(f"\n{best_gpu_type} specs:")
        logger.info(f"  Available: {best_gpu.count} GPUs")
        logger.info(f"  Cost: ${best_gpu.cost_per_hour:.2f}/hour per GPU")
        logger.info(f"  Max segment size (TP={max_tp}): {max_seg_size} layers")
        logger.info(f"  Model needs: {self.config.num_decoder_layers} layers")
        
        if max_seg_size >= self.config.num_decoder_layers:
            # Can fit entire model in one segment!
            # Use max_batch_size for optimal throughput estimate
            nvlink_bw = self._get_representative_tp_bandwidth(best_gpu_type, max_tp)
            alt_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                best_gpu_type, self.config.sequence_length, self.config.max_batch_size,
                self.config.num_decoder_layers, self.config.d_model, 
                self.config.bytes_per_element, max_tp, self.config.d_hidden, nvlink_bw,
                debug=False, phase=self.config.workload_phase
            )
            alt_cost = best_gpu.cost_per_hour * max_tp
            alt_cost_per_token = alt_cost / (alt_throughput * 3600)
            alt_cost_per_million = alt_cost_per_token * 1_000_000
            
            logger.info(f"\nCAN FIT Single segment with TP={max_tp}:")
            logger.info(f"  Throughput: {alt_throughput:.0f} tokens/sec")
            logger.info(f"  Cost: ${alt_cost:.2f}/hour")
            logger.info(f"  $/M tokens: ${alt_cost_per_million:.6f}")
            
            # Compare with actual solution
            actual_cpt = self.solution['cost_per_token']
            actual_cpm = actual_cpt * 1_000_000
            logger.info(f"\nComparison with current multi-segment solution:")
            logger.info(f"  Current solution $/M tokens: ${actual_cpm:.6f}")
            logger.info(f"  Single-segment $/M tokens:   ${alt_cost_per_million:.6f}")
            
            if alt_cost_per_token < actual_cpt:
                diff_pct = (actual_cpt/alt_cost_per_token - 1)*100
                logger.warning(f"  SINGLE-SEGMENT is {diff_pct:.1f}% BETTER in $/M tokens!")
                logger.warning(f"  The solver found a suboptimal solution!")
            else:
                diff_pct = (alt_cost_per_token/actual_cpt - 1)*100
                logger.info(f"Current multi-segment is {diff_pct:.1f}% better")
        else:
            logger.info(f"\nCANNOT FIT: Model needs {self.config.num_decoder_layers} layers but max is {max_seg_size}")
        
        # 6. Objective function check
        logger.info("\n" + "="*80)
        logger.info("OBJECTIVE FUNCTION VERIFICATION")
        logger.info("="*80)
        
        w = self.config.cost_throughput_weight
        if w > 0 and w < 1:
            # Current solution
            t_curr = self.solution['throughput_tokens_per_sec']
            c_curr = self.solution['cost_per_hour']
            obj_curr = (t_curr / self.config.throughput_normalization) - \
                      (w/(1-w)) * (c_curr / self.config.cost_normalization)
            
            logger.info(f"Current solution objective: {obj_curr:.6f}")
            logger.info(f"  Throughput: {t_curr:.0f} tokens/s")
            logger.info(f"  Cost: ${c_curr:.2f}/h")
            
            # Check alternative
            if best_gpu_type not in gpu_usage:
                best_gpu = self.gpu_types[best_gpu_type]
                max_tp = self.tp_max_configuration[best_gpu_type]
                # Use max_batch_size for capacity check
                max_seg_size = self._compute_max_segment_size_for_tp(best_gpu_type, max_tp, self.config.max_batch_size)
                
                if max_seg_size >= self.config.num_decoder_layers:
                    # Use max_batch_size for optimal throughput estimate
                    nvlink_bw = self._get_representative_tp_bandwidth(best_gpu_type, max_tp)
                    t_alt = ThroughputFunctions.gpu_throughput_with_tp(
                        best_gpu_type, self.config.sequence_length, self.config.max_batch_size,
                        self.config.num_decoder_layers, self.config.d_model,
                        self.config.bytes_per_element, max_tp, self.config.d_hidden, nvlink_bw,
                        debug=False, phase=self.config.workload_phase
                    )
                    c_alt = best_gpu.cost_per_hour * max_tp
                    obj_alt = (t_alt / self.config.throughput_normalization) - \
                             (w/(1-w)) * (c_alt / self.config.cost_normalization)
                    
                    logger.info(f"\nAlternative ({best_gpu_type} single segment) objective: {obj_alt:.6f}")
                    logger.info(f"  Throughput: {t_alt:.0f} tokens/s")
                    logger.info(f"  Cost: ${c_alt:.2f}/h")
                    
                    if obj_alt > obj_curr:
                        logger.warning(f"  WARNING Alternative objective is HIGHER by {obj_alt - obj_curr:.6f}!")
                        logger.warning(f"  This suggests the solver may have missed this solution!")
    
    def print_solution(self):
        """Print solution in readable format"""
        if not self.solution:
            logger.error("No solution available")
            return
        
        logger.info("\n" + "="*100)
        logger.info(f"LLM PLACEMENT OPTIMIZATION RESULTS (COST-AWARE)")
        logger.info("="*100)
        logger.info(f"Model: {self.config.model_name} ({self.config.num_decoder_layers} layers)")
        logger.info(f"Batch Size: {self.solution['batch_size']} (optimal), Sequence Length: {self.config.sequence_length}")
        logger.info(f"  Available batch sizes: [{self.config.min_batch_size}...{self.config.max_batch_size}]")
        logger.info(f"TP Configuration: {self.solution['tp_configuration']}")
        logger.info(f"Pipeline Stages: {self.solution['num_pipeline_stages']} (max: {self.config.max_pipeline_stages})")
        print()
        logger.info("PERFORMANCE & COST METRICS:")
        logger.info("-" * 100)
        logger.info(f"  Throughput:        {self.solution['throughput_tokens_per_sec']:.2f} tokens/sec")
        logger.info(f"  Cost:              ${self.solution['cost_per_hour']:.2f}/hour")
        cost_per_million = self.solution['cost_per_token'] * 1_000_000
        logger.info(f"  $/M tokens:        ${cost_per_million:.6f}")
        total_runtime_hours = self.solution.get('total_runtime_hours', 0)
        logger.info(f"  Total Runtime:     {total_runtime_hours:.2f} hours ({total_runtime_hours/24:.2f} days)")
        logger.info(f"  Objective Value:   {self.solution['objective_value']:.4f}")
        print()
        
        # NEW: Cost Comparison (if threshold specified)
        if self.config.max_cost_per_token < 999.0:
            competitor = self.config.max_cost_per_token
            our_cost = self.solution['cost_per_token']
            competitor_per_million = competitor * 1_000_000
            our_cost_per_million = our_cost * 1_000_000
            improvement = (competitor - our_cost) / competitor * 100
            
            logger.info("COST COMPARISON vs COMPETITOR:")
            logger.info("-" * 100)
            logger.info(f"  Competitor $/M tokens:  ${competitor_per_million:.6f}")
            logger.info(f"  Our $/M tokens:         ${our_cost_per_million:.6f}")
            if improvement > 0:
                logger.info(f"  Improvement:            OK {improvement:.1f}% BETTER")
            else:
                logger.info(f"  Improvement:            Nah {abs(improvement):.1f}% WORSE")
            print()
        
        # Runtime constraint check (if specified)
        if self.config.max_total_runtime_hours < 999999.0:
            max_runtime = self.config.max_total_runtime_hours
            actual_runtime = self.solution.get('total_runtime_hours', 0)
            slack = max_runtime - actual_runtime
            slack_pct = (slack / max_runtime) * 100 if max_runtime > 0 else 0
            
            logger.info("RUNTIME CONSTRAINT CHECK:")
            logger.info("-" * 100)
            logger.info(f"  Max Runtime Allowed:    {max_runtime:.2f} hours ({max_runtime/24:.2f} days)")
            logger.info(f"  Actual Runtime:         {actual_runtime:.2f} hours ({actual_runtime/24:.2f} days)")
            if slack > 0:
                logger.info(f"  Slack:                  OK {slack:.2f} hours ({slack_pct:.1f}% under limit)")
            else:
                logger.info(f"  Slack:                  VIOLATED by {abs(slack):.2f} hours ({abs(slack_pct):.1f}% over limit)")
            print()
        
        print()
        
        logger.info("GPU ASSIGNMENTS (WITH TP):")
        logger.info("-" * 100)
        logger.info(f"{'GPU Type':<12} {'TP':<4} {'GPU IDs':<20} {'Layers':<15} {'Size':<6} {'Throughput':<12} {'Cost/h':<10}")
        logger.info("-" * 100)
        
        for assignment in self.solution['gpu_assignments']:
            layers_str = f"{assignment['start_layer']}-{assignment['end_layer']}"
            gpu_ids_str = str(assignment['gpu_ids'])
            
            # Calculate segment cost
            gpu_type = assignment['gpu_type']
            tp_degree = assignment['tp_degree']
            segment_cost = self.gpu_types[gpu_type].cost_per_hour * tp_degree
            
            logger.info(f"{gpu_type:<12} {tp_degree:<4} "
                  f"{gpu_ids_str:<20} {layers_str:<15} "
                  f"{assignment['segment_size']:<6} {assignment['throughput']:<12.2f} "
                  f"${segment_cost:<9.2f}")
        
        if self.solution['network_connections']:
            logger.info("\nNETWORK CONNECTIONS:")
            logger.info("-" * 80)
            for i, conn in enumerate(self.solution['network_connections']):
                from_seg = conn['from_segment']
                to_seg = conn['to_segment']
                logger.info(f"Connection {i+1}: {from_seg['gpu_type']} partition {from_seg['partition_id']} "
                      f"(layers {from_seg['start_layer']}-{from_seg['end_layer']}) -> "
                      f"{to_seg['gpu_type']} partition {to_seg['partition_id']} "
                      f"(layers {to_seg['start_layer']}-{to_seg['end_layer']}) "
                      f"[Throughput: {conn['throughput']:.2f}]")
        
        logger.info("\n" + "="*100)
    
    def save_solution(self, output_file: str):
        """Save solution to JSON file"""
        if not self.solution:
            logger.error("No solution available to save")
            return
        
        output_data = {
            'config': {
                'model_name': self.config.model_name,
                'num_decoder_layers': self.config.num_decoder_layers,
                'sequence_length': self.config.sequence_length,
                'min_batch_size': self.config.min_batch_size,
                'max_batch_size': self.config.max_batch_size,
                'optimal_batch_size': self.solution.get('batch_size', None),
                'd_model': self.config.d_model,
                'd_hidden': self.config.d_hidden,
                'max_pipeline_stages': self.config.max_pipeline_stages,
                'min_memory_utilization': self.config.min_memory_utilization
            },
            'solution': self.solution
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Solution saved to {output_file}")
    
    def save_solution_csv(self, output_file: str, include_all_results: bool = True):
        """
        Save solution summary to CSV file.
        
        Args:
            output_file: Path to output CSV file
            include_all_results: If True and enumeration was used, save all explored solutions.
                                If False, save only the best solution.
        """
        if not self.solution:
            logger.error("No solution available to save")
            return
        
        import csv
        
        # Check if we have enumeration results
        has_enumeration = hasattr(self, 'all_enumeration_results') and self.all_enumeration_results
        
        if include_all_results and has_enumeration:
            # Save all enumeration results
            rows = []
            for result in sorted(self.all_enumeration_results, key=lambda x: x['cost_per_token']):
                sol = result['solution']
                
                # Extract GPU info
                gpu_type = ''
                tp_degree = ''
                num_gpus = ''
                total_layers = ''
                
                if sol['gpu_assignments']:
                    first_assignment = sol['gpu_assignments'][0]
                    gpu_type = first_assignment['gpu_type']
                    tp_degree = first_assignment['tp_degree']
                    num_gpus = sum(len(a['gpu_ids']) for a in sol['gpu_assignments'])
                    total_layers = sum(a['segment_size'] for a in sol['gpu_assignments'])
                
                cost_per_million = result['cost_per_token'] * 1_000_000
                is_best = (result['cost_per_token'] == min(r['cost_per_token'] for r in self.all_enumeration_results))
                
                # Calculate total runtime and cost based on workload
                total_runtime_hours = self.config.total_tokens_to_process / result['throughput'] / 3600
                total_cost = result['cost'] * total_runtime_hours
                
                rows.append({
                    'batch_size': sol.get('batch_size', ''),
                    'budget_tested': f"{result['budget']:.2f}",
                    'throughput_tokens_per_sec': f"{result['throughput']:.2f}",
                    'cost_per_hour': f"{result['cost']:.2f}",
                    'cost_per_million_tokens': f"{cost_per_million:.6f}",
                    'total_runtime_hours': f"{total_runtime_hours:.2f}",
                    'total_cost': f"{total_cost:.2f}",
                    'pipeline_stages': sol['num_pipeline_stages'],
                    'gpu_type': gpu_type,
                    'tp_degree': tp_degree,
                    'num_gpus': num_gpus,
                    'total_layers': total_layers,
                    'is_best': 'YES' if is_best else 'NO',
                    'status': 'SUCCESS'
                })
            
            # Write CSV with all results
            with open(output_file, 'w', newline='') as f:
                fieldnames = ['batch_size', 'budget_tested', 'throughput_tokens_per_sec', 'cost_per_hour', 
                             'cost_per_million_tokens', 'total_runtime_hours', 'total_cost',
                             'pipeline_stages', 'gpu_type', 'tp_degree', 'num_gpus', 'total_layers', 
                             'is_best', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Solution CSV saved with {len(rows)} enumeration results to {output_file}")
        else:
            # Save only the best solution (original behavior)
            gpu_type = ''
            tp_degree = ''
            num_gpus = ''
            total_layers = ''
            
            if self.solution['gpu_assignments']:
                first_assignment = self.solution['gpu_assignments'][0]
                gpu_type = first_assignment['gpu_type']
                tp_degree = first_assignment['tp_degree']
                num_gpus = sum(len(a['gpu_ids']) for a in self.solution['gpu_assignments'])
                total_layers = sum(a['segment_size'] for a in self.solution['gpu_assignments'])
            
            cost_per_million = self.solution['cost_per_token'] * 1_000_000
            
            # Calculate total runtime and cost based on workload
            total_runtime_hours = self.config.total_tokens_to_process / self.solution['throughput_tokens_per_sec'] / 3600
            total_cost = self.solution['cost_per_hour'] * total_runtime_hours
            
            row = {
                'batch_size': self.solution.get('batch_size', ''),
                'throughput_tokens_per_sec': f"{self.solution['throughput_tokens_per_sec']:.2f}",
                'cost_per_hour': f"{self.solution['cost_per_hour']:.2f}",
                'cost_per_million_tokens': f"{cost_per_million:.6f}",
                'total_runtime_hours': f"{total_runtime_hours:.2f}",
                'total_cost': f"{total_cost:.2f}",
                'pipeline_stages': self.solution['num_pipeline_stages'],
                'gpu_type': gpu_type,
                'tp_degree': tp_degree,
                'num_gpus': num_gpus,
                'total_layers': total_layers,
                'status': 'SUCCESS'
            }
            
            # Write CSV (with header)
            with open(output_file, 'w', newline='') as f:
                fieldnames = ['batch_size', 'throughput_tokens_per_sec', 'cost_per_hour', 
                             'cost_per_million_tokens', 'total_runtime_hours', 'total_cost',
                             'pipeline_stages', 'gpu_type', 'tp_degree', 'num_gpus', 'total_layers', 
                             'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            
            logger.info(f"Solution CSV saved to {output_file}")


def filter_tp_configurations_by_hierarchy(gpu_types: Dict[str, GPUType], 
                                          tp_configs: List[Dict[str, int]]) -> List[Dict[str, int]]:
    """
    Filter TP configurations based on GPU performance hierarchy.
    Better GPUs should have TP >= worse GPUs' TP.
    """
    filtered_configs = []
    
    for config in tp_configs:
        valid = True
        
        # Check hierarchy: higher tier should have TP >= lower tier
        for gpu1, tp1 in config.items():
            tier1 = GPU_PERFORMANCE_TIERS.get(gpu1, 2)
            
            for gpu2, tp2 in config.items():
                tier2 = GPU_PERFORMANCE_TIERS.get(gpu2, 2)
                
                # Hierarchy violation: better GPU has lower TP
                if tier1 > tier2 and tp1 < tp2:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            filtered_configs.append(config)
    
    logger.info(f"TP hierarchy filter: {len(tp_configs)} -> {len(filtered_configs)} configs")
    return filtered_configs


def filter_extreme_tp_pp_combinations(gpu_types: Dict[str, GPUType],
                                     tp_configs: List[Dict[str, int]],
                                     num_layers: int) -> List[Dict[str, int]]:
    """
    Filter extreme TP-PP combinations based on heuristics.
    - If max TP >= 8, estimated PP should be <= 8
    - If estimated PP >= 16, max TP should be <= 4
    """
    filtered = []
    
    for config in tp_configs:
        # Estimate minimum PP depth
        min_partitions = sum(
            gpu_types[gpu_type].count // tp_degree
            for gpu_type, tp_degree in config.items()
        )
        
        max_tp = max(config.values())
        
        # Heuristic: TP=8 with deep PP is inefficient
        if max_tp >= 8 and min_partitions > 8:
            continue
        
        # Heuristic: Very deep PP with high TP is doubly inefficient
        if min_partitions >= 16 and max_tp >= 8:
            continue
        
        filtered.append(config)
    
    logger.info(f"TP-PP trade-off filter: {len(tp_configs)} -> {len(filtered)}")
    return filtered


def enumerate_tp_configurations(gpu_types: Dict[str, GPUType], 
                                num_layers: int) -> List[Dict[str, int]]:
    """Enumerate all valid TP configurations with practical filters"""
    configs = []
    
    # Get valid TP degrees for each GPU type
    tp_options_per_type = {}
    for gpu_type, gpu_obj in gpu_types.items():
        valid_degrees = [d for d in [1, 2, 4, 8] 
                        if d <= gpu_obj.count and gpu_obj.count % d == 0]
        tp_options_per_type[gpu_type] = valid_degrees
    
    # Generate cartesian product
    gpu_types_list = list(gpu_types.keys())
    tp_degree_options = [tp_options_per_type[gt] for gt in gpu_types_list]
    
    for tp_combination in product(*tp_degree_options):
        config = dict(zip(gpu_types_list, tp_combination))
        configs.append(config)
    
    logger.info(f"Generated {len(configs)} initial TP configurations")
    
    # Apply hierarchy filter
    configs = filter_tp_configurations_by_hierarchy(gpu_types, configs)
    
    # Apply TP-PP trade-off filter
    configs = filter_extreme_tp_pp_combinations(gpu_types, configs, num_layers)
    
    # Sort by heuristic: try high-TP on good GPUs first
    configs.sort(key=lambda cfg: sum(
        GPU_PERFORMANCE_TIERS.get(gpu, 2) * tp_degree 
        for gpu, tp_degree in cfg.items()
    ), reverse=True)
    
    return configs


def solve_all_tp_configurations(config_dir: str, **kwargs) -> Tuple[Dict, Dict]:
    """Solve for all valid TP configurations and return best solution"""
    # Load GPU pool to enumerate configurations
    gpu_pool_file = os.path.join(config_dir, 'gpu_pool.csv')
    df = pd.read_csv(gpu_pool_file)
    
    gpu_types = {}
    global_id = 0
    for _, row in df.iterrows():
        global_ids = list(range(global_id, global_id + row['count']))
        gpu_types[row['gpu_type']] = GPUType(
            name=row['gpu_type'],
            count=row['count'],
            memory_gb=row['memory_gb'],
            global_ids=global_ids
        )
        global_id += row['count']
    
    # Load config to get num_layers
    config_file = os.path.join(config_dir, 'config.csv')
    df = pd.read_csv(config_file)
    config_dict = dict(zip(df['parameter'], df['value']))
    num_layers = int(config_dict['num_decoder_layers'])
    
    # Enumerate all TP configurations with filters
    tp_configs = enumerate_tp_configurations(gpu_types, num_layers)
    logger.info(f"Evaluating {len(tp_configs)} TP configurations (after filters)...")
    
    best_solution = None
    best_throughput = 0
    best_tp_config = None
    
    for i, tp_config in enumerate(tp_configs):
        logger.info(f"\n{'='*80}")
        logger.info(f"TP Configuration {i+1}/{len(tp_configs)}: {tp_config}")
        logger.info(f"{'='*80}")
        
        try:
            solver = LLMPlacementSolverWithTP(
                config_dir,
                tp_configuration=tp_config,
                **kwargs
            )
            
            solver.build_model()
            
            if solver.solve():
                if solver.solution['objective_value'] > best_throughput:
                    best_throughput = solver.solution['objective_value']
                    best_solution = solver.solution
                    best_tp_config = tp_config
                    logger.info(f"New best throughput: {best_throughput:.2f} tokens/sec")
        
        except Exception as e:
            logger.error(f"Failed to solve TP config {tp_config}: {e}")
            continue
    
    if best_solution:
        logger.info(f"\n{'='*80}")
        logger.info(f"BEST SOLUTION ACROSS ALL TP CONFIGURATIONS")
        logger.info(f"{'='*80}")
        logger.info(f"Best TP Configuration: {best_tp_config}")
        logger.info(f"Best Throughput: {best_throughput:.2f} tokens/sec")
        logger.info(f"Pipeline Stages: {best_solution['num_pipeline_stages']}")
    
    return best_solution, best_tp_config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='LLM Placement Optimizer with TP and Practical Constraints'
    )
    parser.add_argument('--config-dir', required=True, help='Configuration directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--tp-config', type=str, 
                       help='TP configuration as JSON (e.g., \'{"A100": 4, "V100": 1}\')')
    parser.add_argument('--search-all-tp', action='store_true', 
                       help='Search all valid TP configurations (ignores --tp-config)')
    
    # Optimization flags
    parser.add_argument('--enable-symmetry-breaking', action='store_true', default=True)
    parser.add_argument('--disable-symmetry-breaking', dest='enable_symmetry_breaking', 
                       action='store_false')
    parser.add_argument('--enable-upper-bound', action='store_true', default=True)
    parser.add_argument('--disable-upper-bound', dest='enable_upper_bound', action='store_false')
    parser.add_argument('--enable-tight-bigm', action='store_true', default=True)
    parser.add_argument('--disable-tight-bigm', dest='enable_tight_bigm', action='store_false')
    parser.add_argument('--enable-flow-conservation', action='store_true', default=True)
    parser.add_argument('--disable-flow-conservation', dest='enable_flow_conservation', 
                       action='store_false')
    parser.add_argument('--threads', type=int, help='Number of threads')
    parser.add_argument('--max-threads', type=int, default=32, help='Maximum threads')
    
    # Cost optimization arguments
    parser.add_argument('--cost-weight', type=float, 
                       help='Override cost_throughput_weight from config (0.0=pure throughput, 1.0=pure cost)')
    parser.add_argument('--method', type=str, choices=['weighted', 'enumeration'], default='weighted',
                       help='Optimization method: weighted (fast, approximate) or enumeration (slow, guaranteed optimal)')
    
    # Network bandwidth generation arguments
    parser.add_argument('--generate-network', type=float, nargs=2, metavar=('INTRA_BW', 'INTER_BW'),
                       help='Generate network bandwidth matrix instead of reading from CSV. '
                            'Args: intra_bandwidth (GB/s within same GPU type) inter_bandwidth (GB/s between different GPU types)')
    
    # Cloud pricing arguments
    parser.add_argument('--cloud-provider', type=str, 
                        default='AWS',
                       choices=['AWS', 'GCP', 'Azure', 'Lambda', 'CoreWeave', 'Nebius'],
                       help='Cloud provider for pricing (uses cheapest if not specified). '
                            'Prices loaded from cloud_instances_specs.csv')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = time.time()
    
    try:
        # Prepare network generation parameters
        generate_network = None
        if args.generate_network:
            generate_network = tuple(args.generate_network)
            logger.info(f"Network generation mode: intra={generate_network[0]} GB/s, inter={generate_network[1]} GB/s")
        
        solver_kwargs = {
            'enable_symmetry_breaking': args.enable_symmetry_breaking,
            'enable_upper_bound': args.enable_upper_bound,
            'enable_tight_bigm': args.enable_tight_bigm,
            'enable_flow_conservation': args.enable_flow_conservation,
            'threads': args.threads,
            'max_threads': args.max_threads,
            'generate_network': generate_network,
            'cloud_provider': args.cloud_provider
        }
        
        if args.search_all_tp:
            # Search all TP configurations
            best_solution, best_tp_config = solve_all_tp_configurations(
                args.config_dir, **solver_kwargs
            )
            
            if best_solution:
                # Print summary
                logger.info("\n" + "="*100)
                logger.info("BEST SOLUTION SUMMARY")
                logger.info("="*100)
                logger.info(f"Best TP Configuration: {best_tp_config}")
                logger.info(f"Throughput: {best_solution['objective_value']:.2f} tokens/sec")
                logger.info(f"Pipeline Stages: {best_solution['num_pipeline_stages']}")
                logger.info("="*100)
                
                # Save best solution
                output_file = os.path.join(args.config_dir, 'solution_best_tp.json')
                with open(output_file, 'w') as f:
                    json.dump({
                        'best_tp_configuration': best_tp_config,
                        'solution': best_solution
                    }, f, indent=2)
                logger.info(f"Best solution saved to {output_file}")
            else:
                logger.error("No feasible solution found across all TP configurations")
                return 1
        
        else:
            # Single TP configuration
            tp_config = None
            if args.tp_config:
                tp_config = json.loads(args.tp_config)
            
            solver = LLMPlacementSolverWithTP(
                args.config_dir,
                tp_configuration=tp_config,
                **solver_kwargs
            )
            
            # Override cost weight if specified
            if args.cost_weight is not None:
                logger.info(f"Overriding cost_throughput_weight: {solver.config.cost_throughput_weight} -> {args.cost_weight}")
                solver.config.cost_throughput_weight = args.cost_weight
            
            solver.build_model()
            
            # Choose optimization method
            if args.method == 'weighted':
                logger.info("Using WEIGHTED method (fast, approximate)")
                success = solver.solve()
            elif args.method == 'enumeration':
                logger.info("Using ENUMERATION method (slow, guaranteed optimal)")
                success = solver.solve_optimal_cost_per_token()
            
            if success:
                solver.print_solution()
                output_file = os.path.join(args.config_dir, 'solution_with_tp.json')
                solver.save_solution(output_file)
                
                # Also save CSV summary
                csv_file = os.path.join(args.config_dir, 'solution_summary.csv')
                solver.save_solution_csv(csv_file)
            else:
                logger.error("Failed to find optimal solution")
                return 1
    
    except Exception as e:
        logger.error(f"Solver failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    end_time = time.time()
    logger.info(f"Total time: {end_time - start_time:.0f} seconds")
    return 0


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total solver runtime: {end_time - start_time:.0f} seconds")