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
import io
import json
import logging
import math
import shutil
from typing import Dict, List, Tuple, Optional, FrozenSet, Union
from dataclasses import dataclass, field
import time
from itertools import product, combinations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class _LogCapture(logging.Handler):
    """Handler that captures log records into a list of strings."""
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: List[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


def _json_safe(value):
    """Convert numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _resolve_eval_output_dir(args, config_dir: str, result_dir_name: str) -> str:
    """Build eval output path as {root}/{config}/{result_dir}."""
    config_name = os.path.basename(os.path.normpath(config_dir))
    root = args.eval_output_root or args.output_dir or config_dir
    return os.path.join(root, config_name, result_dir_name)



@dataclass
class GPUType:
    """GPU type specification"""
    name: str
    count: int
    memory_gb: float
    global_ids: List[int]
    cost_per_hour: float = 0.0  # Cost in $/hour per instance
    gpu_model: str = ""         # Underlying GPU model (e.g., A100, H100)

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
    num_kv_heads: int
    max_position_embeddings: int  # Model's max context length (caps max_model_len)
    head_dim: int  # Per-head dimension (may differ from d_model/num_attention_heads, e.g. Qwen3)
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
    real_world_efficiency: float
    micro_batch_size: int
    # Max token limits for feasibility checking (0 = use avg as fallback)
    max_input_tokens: int = 0
    max_output_tokens: int = 0
    # vLLM scheduler parameter: max concurrent sequences (decode batch size)
    # This is the actual decode concurrency used by the engine, NOT the KV-pool capacity.
    max_num_seqs: int = 256
    # vLLM max_num_batched_tokens: max tokens per prefill iteration (0 = fall back to max_model_len)
    max_num_batched_tokens: int = 16384
    # vLLM gpu_memory_utilization: fraction of GPU memory available for model/KV cache
    gpu_memory_utilization: float = 0.90
    # NEW: Penalty for cross-instance PP connections (encourages intra-node placement)
    # Higher values more strongly prefer same-instance connections
    cross_instance_penalty: float = 0.01  # Default: small penalty per cross-instance edge
    # NEW: Tight memory filtering - exclude configurations with low memory headroom
    # Prevents OOM from unmodeled overhead (CUDA graphs, sampler, fragmentation, etc.)
    filter_tight_memory: bool = True  # Default: filter out tight memory configs
    tight_memory_threshold_pct: float = 10.0  # Default: 10% headroom threshold
    # vLLM PagedAttention KV block size (tokens per block)
    kv_block_size: int = 16
    # enforce_eager mode penalty (disables CUDA graphs → decode overhead)
    enforce_eager: bool = False
    enforce_eager_penalty: float = 0.80
    # Cross-node PP latency (microseconds per pipeline stage boundary)
    cross_node_pp_latency_us: float = 0.0

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
        # HBM GPUs achieve higher memory_efficiency (0.65-0.72) due to HBM streaming access.
        # GDDR6X GPUs have lower memory_efficiency (0.50-0.58) due to GDDR latency/bank conflicts.
        #
        # generation_factor: multiplier on real_world_efficiency to account for
        # software-level differences across GPU architectures. vLLM kernels
        # (PagedAttention, fused RMSNorm, etc.) are optimized for Ampere+;
        # older architectures (Volta, Turing) lack BF16, FlashAttention v2,
        # and have less efficient tensor core micro-ops. We are currently applying ~0.5x factor on V100.
        # real-world throughput than roofline predicts (Lambda A100-vs-V100
        # benchmarks show 1.6-3.4x gap for transformers).
        # Baseline 1.0 = Ampere/Ada (the generation the roofline model is calibrated for).
        'H100': {'tflops': 989, 'mem_bw': 3350, 'compute_efficiency': 0.80, 'memory_efficiency': 0.72, 'generation_factor': 1.15},
        'A100': {'tflops': 312, 'mem_bw': 2039, 'compute_efficiency': 0.78, 'memory_efficiency': 0.70, 'generation_factor': 1.0},
        'L40S': {'tflops': 362, 'mem_bw': 864,  'compute_efficiency': 0.70, 'memory_efficiency': 0.55, 'generation_factor': 1.0},
        'A40':  {'tflops': 150, 'mem_bw': 696,  'compute_efficiency': 0.70, 'memory_efficiency': 0.55, 'generation_factor': 1.0},
        'L40':  {'tflops': 181, 'mem_bw': 864,  'compute_efficiency': 0.70, 'memory_efficiency': 0.55, 'generation_factor': 1.0},

        # Older GPUs — penalized for missing BF16, FlashAttention v2, unoptimized kernels
        'V100': {'tflops': 125, 'mem_bw': 900,  'compute_efficiency': 0.68, 'memory_efficiency': 0.65, 'generation_factor': 0.50},
        'RTX4090': {'tflops': 165, 'mem_bw': 1008, 'compute_efficiency': 0.72, 'memory_efficiency': 0.58, 'generation_factor': 1.0},

        # Budget GPUs
        'L20':  {'tflops': 119, 'mem_bw': 480,  'compute_efficiency': 0.68, 'memory_efficiency': 0.55, 'generation_factor': 1.0},
        'L4':   {'tflops': 121, 'mem_bw': 300,  'compute_efficiency': 0.65, 'memory_efficiency': 0.50, 'generation_factor': 1.0},
        'A10':  {'tflops': 62.5,'mem_bw': 600,  'compute_efficiency': 0.65, 'memory_efficiency': 0.40, 'generation_factor': 1.0},  # Dense FP16 (not sparse 125), GDDR6
        'T4':   {'tflops': 65,  'mem_bw': 320,  'compute_efficiency': 0.60, 'memory_efficiency': 0.50, 'generation_factor': 0.55},
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
                                      tp_degree: int = 1, phase: str = 'prefill',
                                      num_attention_heads: Optional[int] = None,
                                      num_kv_heads: Optional[int] = None,
                                      head_dim: Optional[int] = None,
                                      vocab_size: int = 0,
                                      total_num_layers: int = 0) -> float:
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
        if num_attention_heads is None:
            num_attention_heads = max(1, d_model // 128)
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads
        d_head = head_dim if head_dim is not None else d_model / num_attention_heads
        kv_dim = num_kv_heads * d_head
        q_o_dim = num_attention_heads * d_head  # Q/O projection output dim (may differ from d_model)

        if phase == 'prefill':
            # PREFILL: Process all tokens at once (O(n²) attention)
            # TP sharding splits compute across GPUs
            flops_q = 2 * batch_size * seq_len * d_model * (q_o_dim / tp_degree)
            flops_k = 2 * batch_size * seq_len * d_model * (kv_dim / tp_degree)
            flops_v = 2 * batch_size * seq_len * d_model * (kv_dim / tp_degree)
            flops_o = 2 * batch_size * seq_len * q_o_dim * (d_model / tp_degree)
            flops_attn_proj = flops_q + flops_k + flops_v + flops_o
            flops_attn_scores = 4 * batch_size * seq_len * seq_len * (q_o_dim / tp_degree)  # O(n²)
            flops_attention = flops_attn_proj + flops_attn_scores

            # FFN for all tokens
            flops_ffn = 2 * batch_size * seq_len * (
                d_model * (d_hidden / tp_degree) +
                d_model * (d_hidden / tp_degree) +
                (d_hidden / tp_degree) * d_model
            )
        else:  # decode
            # DECODE: Generate ONE token (O(n) attention to KV cache)
            # CRITICAL: KV cache grows during generation - use average for realistic estimate
            # Note: This function doesn't have output_length parameter, so we can't account for growth
            # The caller (gpu_throughput_with_tp) handles this properly
            kv_cache_len = seq_len  # seq_len represents cached context
            flops_q = 2 * batch_size * 1 * d_model * (q_o_dim / tp_degree)
            flops_k = 2 * batch_size * 1 * d_model * (kv_dim / tp_degree)
            flops_v = 2 * batch_size * 1 * d_model * (kv_dim / tp_degree)
            flops_o = 2 * batch_size * 1 * q_o_dim * (d_model / tp_degree)
            flops_attn_proj = flops_q + flops_k + flops_v + flops_o  # QKV+O for 1 token
            flops_attn_scores = 4 * batch_size * 1 * kv_cache_len * (q_o_dim / tp_degree)  # O(n)
            flops_attention = flops_attn_proj + flops_attn_scores

            # FFN for 1 token
            flops_ffn = 2 * batch_size * 1 * (
                d_model * (d_hidden / tp_degree) +
                d_model * (d_hidden / tp_degree) +
                (d_hidden / tp_degree) * d_model
            )

        # RMSNorm FLOPs: 2 norms per layer (pre-attention + pre-FFN), ~5 ops per element
        tokens = seq_len if phase == 'prefill' else 1
        rmsnorm_flops_per_layer = 2 * 5 * batch_size * tokens * d_model

        flops_per_layer = flops_attention + flops_ffn + rmsnorm_flops_per_layer
        total_flops = flops_per_layer * num_layers

        # LM head (amortized across pipeline stages proportionally)
        if vocab_size > 0 and total_num_layers > 0:
            layer_fraction = num_layers / total_num_layers
            lm_head_flops = 2 * batch_size * tokens * d_model * vocab_size * layer_fraction
        else:
            lm_head_flops = 0
        total_flops += lm_head_flops

        # === Memory Access Calculation (PHASE-AWARE) ===
        # Weights (divided by TP): same for both phases
        bytes_weights_per_layer = (
            (d_model * q_o_dim + q_o_dim * d_model) +  # W_q, W_o
            (2 * d_model * kv_dim) +                    # W_k, W_v (GQA/MQA aware)
            (3 * d_model * d_hidden)                    # W_gate, W_up, W_down
        ) * bytes_per_element / tp_degree

        if phase == 'prefill':
            # KV cache being written
            bytes_kv_cache_per_layer = 2 * batch_size * seq_len * kv_dim * bytes_per_element / tp_degree
            # Activations for all tokens
            bytes_activations_per_layer = batch_size * seq_len * d_model * bytes_per_element
            # Attention score intermediate tensors: QK^T write + softmax read + V-multiply read
            attn_score_bytes_per_layer = (batch_size * (num_attention_heads / tp_degree)
                                          * seq_len * seq_len * bytes_per_element * 3)
        else:  # decode
            # KV cache being READ (full cached context!)
            kv_cache_len = seq_len
            bytes_kv_cache_per_layer = 2 * batch_size * kv_cache_len * kv_dim * bytes_per_element / tp_degree
            # PagedAttention scatter/gather overhead (~15% extra on KV cache reads)
            bytes_kv_cache_per_layer *= 1.15
            # Activations for 1 token only
            bytes_activations_per_layer = batch_size * 1 * d_model * bytes_per_element
            # Attention score intermediate tensors for decode (1 query token × kv_cache_len)
            attn_score_bytes_per_layer = (batch_size * (num_attention_heads / tp_degree)
                                          * kv_cache_len * bytes_per_element * 3)

        # RMSNorm bytes: 2 norms × (read input + write output + read norm weight)
        # Norm weights are replicated (not sharded by TP)
        rmsnorm_bytes_per_layer = 2 * (2 * batch_size * tokens * d_model + d_model) * bytes_per_element

        # Residual connections: 2 per layer × (read + write) × d_model per token
        residual_bytes_per_layer = 2 * 2 * batch_size * tokens * d_model * bytes_per_element

        bytes_per_layer = (bytes_weights_per_layer + bytes_kv_cache_per_layer +
                           bytes_activations_per_layer + rmsnorm_bytes_per_layer +
                           residual_bytes_per_layer + attn_score_bytes_per_layer)
        total_bytes = bytes_per_layer * num_layers

        # LM head bytes (amortized)
        if vocab_size > 0 and total_num_layers > 0:
            layer_fraction = num_layers / total_num_layers
            # Weight read per GPU (column-parallel, sharded by TP)
            lm_head_weight_bytes = d_model * vocab_size * bytes_per_element / tp_degree * layer_fraction
            # Output activation per GPU (before all-gather)
            lm_head_act_bytes = batch_size * tokens * vocab_size * bytes_per_element / tp_degree * layer_fraction
            total_bytes += lm_head_weight_bytes + lm_head_act_bytes

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
        specs = ThroughputFunctions.GPU_SPECS.get(gpu_type) or ThroughputFunctions.GPU_SPECS['A100']
        
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
                      d_model: int, bytes_per_element: int, d_hidden: int = None,
                      num_attention_heads: Optional[int] = None,
                      num_kv_heads: Optional[int] = None,
                      head_dim: Optional[int] = None) -> float:
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

        specs = ThroughputFunctions.GPU_SPECS.get(gpu_type) or ThroughputFunctions.GPU_SPECS['A100']

        # === Roofline Model Analysis ===
        # Calculate arithmetic intensity (FLOPs per byte)
        # Note: This function is deprecated in favor of gpu_throughput_with_tp
        # Default to 'prefill' for backward compatibility
        arithmetic_intensity = ThroughputFunctions.calculate_arithmetic_intensity(
            num_layers, batch_size, seq_len, d_model, d_hidden, bytes_per_element,
            tp_degree=1, phase='prefill',
            num_attention_heads=num_attention_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, vocab_size=0, total_num_layers=0
        )

        # Get ridge point for this GPU
        ridge_point = ThroughputFunctions.get_ridge_point(gpu_type)

        # Determine regime
        regime = ThroughputFunctions.determine_regime(arithmetic_intensity, ridge_point)

        # === Compute FLOPs (same for both regimes) ===
        if num_attention_heads is None:
            num_attention_heads = max(1, d_model // 128)
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads
        d_head = head_dim if head_dim is not None else d_model / num_attention_heads
        kv_dim = num_kv_heads * d_head
        q_o_dim = num_attention_heads * d_head

        # Attention: Q, K, V, O projections + attention scores (GQA/MQA aware)
        flops_q = 2 * batch_size * seq_len * d_model * q_o_dim
        flops_k = 2 * batch_size * seq_len * d_model * kv_dim
        flops_v = 2 * batch_size * seq_len * d_model * kv_dim
        flops_o = 2 * batch_size * seq_len * q_o_dim * d_model
        attn_proj_flops = flops_q + flops_k + flops_v + flops_o
        attn_score_flops = 4 * batch_size * seq_len * seq_len * q_o_dim  # QK^T + softmax*V

        # FFN: 3 projections (up, gate, down)
        ffn_flops = 2 * batch_size * seq_len * (
            d_model * d_hidden +  # up
            d_model * d_hidden +  # gate
            d_hidden * d_model    # down
        )

        flops_per_layer = attn_proj_flops + attn_score_flops + ffn_flops
        total_flops = num_layers * flops_per_layer

        # === Compute Memory Access (same for both regimes) ===
        weight_bytes = num_layers * (
            (d_model * q_o_dim + q_o_dim * d_model) +  # W_q, W_o
            (2 * d_model * kv_dim) +                    # W_k, W_v
            (3 * d_model * d_hidden)
        ) * bytes_per_element

        # Activations + KV cache
        activation_bytes = batch_size * seq_len * d_model * bytes_per_element
        kv_cache_bytes = num_layers * 2 * batch_size * seq_len * kv_dim * bytes_per_element

        total_bytes = weight_bytes + activation_bytes + kv_cache_bytes
        
        # === Calculate Throughput Based on Regime ===
        if regime == "COMPUTE_BOUND":
            # Limited by compute - use FLOPS
            time_per_batch = total_flops / (specs['tflops'] * 1e12 * specs['compute_efficiency'])
        else:  # MEMORY_BOUND
            # Limited by memory bandwidth
            time_per_batch = total_bytes / (specs['mem_bw'] * 1e9 * specs['memory_efficiency'])
        
        tokens_per_batch = batch_size * seq_len
        base_throughput = tokens_per_batch / time_per_batch
        
        # Apply batch efficiency factor - larger batches utilize GPU better
        final_throughput = base_throughput * ThroughputFunctions.batch_efficiency_factor(batch_size)
        
        return final_throughput
    
    @staticmethod
    def gpu_throughput_with_tp(gpu_type: str, seq_len: int, batch_size: int,
                               num_layers: int, d_model: int, bytes_per_element: int,
                               tp_degree: int, d_hidden: int = None, nvlink_bw_gbps: float = 300.0,
                               debug: bool = False, phase: str = 'prefill', output_length: int = 0,
                               num_attention_heads: Optional[int] = None,
                               num_kv_heads: Optional[int] = None,
                               head_dim: Optional[int] = None,
                               prefill_batch: int = 1,
                               vocab_size: int = 0,
                               total_num_layers: int = 0) -> float:
        """
        GPU throughput with tensor parallelism using roofline model.

        CRITICAL: TP is NOT data parallelism!
        - TP splits model weights across GPUs for the SAME batch
        - Each GPU processes the SAME sequences (not different ones)
        - Purpose: fit larger models in memory, NOT increase throughput
        - Throughput effect: minor speedup from reduced memory pressure, offset by communication

        NEW: Phase-aware throughput modeling (prefill vs decode vs aggregated)
        - Prefill: O(n²) attention on full prompt (all tokens processed in parallel)
        - Decode: O(n) attention per token (sequential generation, one token at a time)
        - Aggregated: Combined prefill→decode throughput using wave model

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
            phase: 'prefill', 'decode', or 'aggregated'

        Returns:
            Throughput in tokens/second with TP (NOT multiplied by tp_degree!)
        """
        # Handle aggregated phase: compute prefill and decode separately, then combine
        if phase == 'aggregated':
            # Default FFN dimension if not provided (needed for recursive calls)
            if d_hidden is None:
                d_hidden = 4 * d_model

            # Compute prefill throughput at batch=prefill_batch (not 1)
            prefill_tp = ThroughputFunctions.gpu_throughput_with_tp(
                gpu_type=gpu_type, seq_len=seq_len, batch_size=prefill_batch,
                num_layers=num_layers, d_model=d_model, bytes_per_element=bytes_per_element,
                tp_degree=tp_degree, d_hidden=d_hidden, nvlink_bw_gbps=nvlink_bw_gbps,
                debug=debug, phase='prefill', output_length=0,
                num_attention_heads=num_attention_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, vocab_size=vocab_size, total_num_layers=total_num_layers
            )
            # Compute decode throughput at batch=batch_size (decode concurrency)
            decode_tp = ThroughputFunctions.gpu_throughput_with_tp(
                gpu_type=gpu_type, seq_len=seq_len, batch_size=batch_size,
                num_layers=num_layers, d_model=d_model, bytes_per_element=bytes_per_element,
                tp_degree=tp_degree, d_hidden=d_hidden, nvlink_bw_gbps=nvlink_bw_gbps,
                debug=debug, phase='decode', output_length=output_length,
                num_attention_heads=num_attention_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, vocab_size=vocab_size, total_num_layers=total_num_layers
            )
            # Combine using wave model (replaces Little's Law + interference_factor)
            aggregated_tp = ThroughputFunctions.calculate_aggregated_throughput(
                prefill_throughput=prefill_tp,
                decode_throughput=decode_tp,
                seq_len=seq_len,
                output_len=output_length,
                max_num_seqs=batch_size,  # caller's batch_size is the decode concurrency
                prefill_batch=prefill_batch,
            )
            if debug:
                logger.info(f"AGGREGATED: prefill={prefill_tp:.0f} tok/s (batch={prefill_batch}), "
                           f"decode={decode_tp:.0f} tok/s, combined={aggregated_tp:.0f} tok/s")
            return aggregated_tp

        # Default FFN dimension if not provided
        if d_hidden is None:
            d_hidden = 4 * d_model

        specs = ThroughputFunctions.GPU_SPECS.get(gpu_type)
        if specs is None:
            logger.warning(f"GPU '{gpu_type}' not in GPU_SPECS — falling back to A100 specs! "
                           f"Throughput estimates will be WRONG.")
            specs = ThroughputFunctions.GPU_SPECS['A100']

        if debug:
            logger.info(f'='*80)
            logger.info(f"DEBUG: gpu_throughput_with_tp called")
            logger.info(f"  GPU: {gpu_type}, TP={tp_degree}, batch={batch_size}, seq={seq_len}, layers={num_layers}")
            logger.info(f"  d_model={d_model}, d_hidden={d_hidden}, nvlink_bw={nvlink_bw_gbps} GB/s")
        
        # === Roofline Model Analysis with TP (PHASE-AWARE) ===
        arithmetic_intensity = ThroughputFunctions.calculate_arithmetic_intensity(
            num_layers, batch_size, seq_len, d_model, d_hidden, bytes_per_element, tp_degree, phase,
            num_attention_heads=num_attention_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, vocab_size=vocab_size, total_num_layers=total_num_layers
        )
        
        ridge_point = ThroughputFunctions.get_ridge_point(gpu_type)
        regime = ThroughputFunctions.determine_regime(arithmetic_intensity, ridge_point)
        
        if debug:
            logger.info(f"  Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPs/byte")
            logger.info(f"  Ridge Point: {ridge_point:.2f} FLOPs/byte")
            logger.info(f"  Regime: {regime}")
        
        # === Compute FLOPs PER GPU (with TP, each GPU does less compute) ===
        # PHASE-AWARE computation: prefill vs decode have different complexity

        if num_attention_heads is None:
            num_attention_heads = max(1, d_model // 128)
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads
        d_head = head_dim if head_dim is not None else d_model / num_attention_heads
        kv_dim = num_kv_heads * d_head
        q_o_dim = num_attention_heads * d_head  # Q/O projection output dim (may differ from d_model)

        if phase == 'prefill':
            # PREFILL: Process all tokens in prompt at once (O(n²) attention)
            flops_q = 2 * batch_size * seq_len * d_model * (q_o_dim / tp_degree)
            flops_k = 2 * batch_size * seq_len * d_model * (kv_dim / tp_degree)
            flops_v = 2 * batch_size * seq_len * d_model * (kv_dim / tp_degree)
            flops_o = 2 * batch_size * seq_len * q_o_dim * (d_model / tp_degree)
            attn_proj_flops = flops_q + flops_k + flops_v + flops_o  # QKV + output
            attn_score_flops = 4 * batch_size * seq_len * seq_len * (q_o_dim / tp_degree)  # O(n²) !!!

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

            flops_q = 2 * batch_size * 1 * d_model * (q_o_dim / tp_degree)
            flops_k = 2 * batch_size * 1 * d_model * (kv_dim / tp_degree)
            flops_v = 2 * batch_size * 1 * d_model * (kv_dim / tp_degree)
            flops_o = 2 * batch_size * 1 * q_o_dim * (d_model / tp_degree)
            attn_proj_flops = flops_q + flops_k + flops_v + flops_o  # QKV for 1 token
            attn_score_flops = 4 * batch_size * 1 * avg_kv_cache_len * (q_o_dim / tp_degree)  # O(n) with growing cache

            # FFN for 1 token
            ffn_flops = 2 * batch_size * 1 * (
                d_model * (d_hidden / tp_degree) +
                d_model * (d_hidden / tp_degree) +
                (d_hidden / tp_degree) * d_model
            )

        # RMSNorm FLOPs: 2 norms per layer (pre-attention + pre-FFN), ~5 ops per element
        tokens = seq_len if phase == 'prefill' else 1
        rmsnorm_flops_per_layer = 2 * 5 * batch_size * tokens * d_model

        flops_per_layer = attn_proj_flops + attn_score_flops + ffn_flops + rmsnorm_flops_per_layer
        total_flops_per_gpu = num_layers * flops_per_layer

        # LM head FLOPs (amortized across pipeline stages)
        lm_head_flops = 0
        if vocab_size > 0 and total_num_layers > 0:
            layer_fraction = num_layers / total_num_layers
            lm_head_flops = 2 * batch_size * tokens * d_model * vocab_size * layer_fraction
            total_flops_per_gpu += lm_head_flops

        if debug:
            logger.info(f"  FLOPs per layer: {flops_per_layer:.2e} (rmsnorm={rmsnorm_flops_per_layer:.2e})")
            logger.info(f"  Total FLOPs (×{num_layers} layers + lm_head={lm_head_flops:.2e}): {total_flops_per_gpu:.2e}")

        # === Compute Memory Access PER GPU (weights sharded by TP) ===
        # PHASE-AWARE memory access patterns

        # Weights are sharded across TP GPUs (same for both phases)
        weight_bytes = num_layers * (
            (d_model * (q_o_dim / tp_degree) + q_o_dim * (d_model / tp_degree)) +  # W_q, W_o
            (2 * d_model * (kv_dim / tp_degree)) +                                  # W_k, W_v
            (3 * d_model * (d_hidden / tp_degree))
        ) * bytes_per_element

        if phase == 'prefill':
            # PREFILL: Activations for all tokens per layer
            activation_bytes = num_layers * batch_size * seq_len * d_model * bytes_per_element
            # KV cache being written (sharded)
            kv_cache_bytes = num_layers * 2 * batch_size * seq_len * (kv_dim / tp_degree) * bytes_per_element
            # Attention score intermediate tensors: QK^T write + softmax read + V-multiply read
            attn_score_bytes = num_layers * batch_size * (num_attention_heads / tp_degree) * seq_len * seq_len * bytes_per_element * 3

        else:  # phase == 'decode'
            # DECODE: Activation for 1 token per layer
            activation_bytes = num_layers * batch_size * 1 * d_model * bytes_per_element
            # CRITICAL: Must READ entire KV cache (which grows during generation!)
            if output_length > 0:
                avg_kv_cache_len = seq_len + (output_length - 1) / 2.0
            else:
                avg_kv_cache_len = seq_len
            kv_cache_bytes = num_layers * 2 * batch_size * avg_kv_cache_len * (kv_dim / tp_degree) * bytes_per_element
            # PagedAttention scatter/gather overhead (~15% extra on KV cache reads)
            kv_cache_bytes *= 1.15
            # Attention score intermediate tensors for decode (1 query × kv_cache_len)
            attn_score_bytes = num_layers * batch_size * (num_attention_heads / tp_degree) * avg_kv_cache_len * bytes_per_element * 3

        # RMSNorm bytes: 2 norms × (read input + write output + read norm weight)
        # Norm weights are replicated (not sharded by TP)
        rmsnorm_bytes = num_layers * 2 * (2 * batch_size * tokens * d_model + d_model) * bytes_per_element

        # Residual connections: 2 per layer × (read + write) × d_model per token
        residual_bytes = num_layers * 2 * 2 * batch_size * tokens * d_model * bytes_per_element

        # LM head bytes (amortized)
        lm_head_bytes = 0
        if vocab_size > 0 and total_num_layers > 0:
            layer_fraction = num_layers / total_num_layers
            lm_head_wt_bytes = d_model * vocab_size * bytes_per_element / tp_degree * layer_fraction
            lm_head_act_bytes = batch_size * tokens * vocab_size * bytes_per_element / tp_degree * layer_fraction
            lm_head_bytes = lm_head_wt_bytes + lm_head_act_bytes

        total_bytes_per_gpu = (weight_bytes + activation_bytes + kv_cache_bytes +
                               attn_score_bytes + rmsnorm_bytes + residual_bytes + lm_head_bytes)

        if debug:
            logger.info(f"  Memory: weights={weight_bytes/1e9:.3f}GB, act={activation_bytes/1e9:.3f}GB, "
                        f"kv={kv_cache_bytes/1e9:.3f}GB, attn_score={attn_score_bytes/1e9:.3f}GB, "
                        f"rmsnorm={rmsnorm_bytes/1e9:.3f}GB, residual={residual_bytes/1e9:.3f}GB, "
                        f"lm_head={lm_head_bytes/1e9:.3f}GB")
            logger.info(f"  Total memory: {total_bytes_per_gpu / 1e9:.2f} GB")
        
        # === Calculate BASE time (without TP efficiency penalty first) ===
        if regime == "COMPUTE_BOUND":
            base_time_per_batch = total_flops_per_gpu / (specs['tflops'] * 1e12 * specs['compute_efficiency'])
        else:  # MEMORY_BOUND
            base_time_per_batch = total_bytes_per_gpu / (specs['mem_bw'] * 1e9 * specs['memory_efficiency'])
        
        # === Communication Overhead (compute before applying TP efficiency) ===
        # Megatron-style TP has 2 all-reduces per transformer layer:
        #   1. After attention (row-parallel output projection)
        #   2. After FFN (row-parallel down projection)
        # Ring all-reduce transfers 2*(N-1)/N * msg_size bytes per operation.
        # Total per layer: 2 ops × 2*(N-1)/N * msg_size / bandwidth
        if phase == 'prefill':
            activation_size_bytes = batch_size * seq_len * d_model * bytes_per_element
        else:  # decode
            activation_size_bytes = batch_size * 1 * d_model * bytes_per_element  # 1 token

        num_allreduce_per_layer = 2  # post-attention + post-FFN
        # Per-op latency and effective BW depend on interconnect type
        if nvlink_bw_gbps >= 100:  # NVLink
            allreduce_latency_s = 5e-6
            effective_comm_bw = nvlink_bw_gbps
        else:  # PCIe
            allreduce_latency_s = 200e-6
            # PCIe effective BW degrades with more GPUs due to switch/root-complex contention.
            # Nominal PCIe 4.0 x16 = 32 GB/s bidirectional, but ring all-reduce across
            # multiple GPUs shares switch bandwidth and suffers from topology bottlenecks.
            pcie_tp_bw_scale = {1: 1.0, 2: 0.75, 4: 0.375, 8: 0.125}.get(
                tp_degree, max(0.10, 0.5 / tp_degree))
            effective_comm_bw = nvlink_bw_gbps * pcie_tp_bw_scale
        comm_time_per_layer = num_allreduce_per_layer * (
            2 * activation_size_bytes / (effective_comm_bw * 1e9) * (tp_degree - 1) / tp_degree
            + allreduce_latency_s
        )
        total_comm_time = num_layers * comm_time_per_layer
        
        # === TP Efficiency Factor ===
        # Residual overhead after explicit comm_time: CUDA stream sync, BW contention
        # Per-op latency is now captured in comm_time (Change 4), so this is minimal.
        comm_overhead_ratio = 0.0  # Initialize
        if tp_degree == 1:
            tp_efficiency_compute = 1.00  # No TP, no overhead
        else:
            comm_overhead_ratio = total_comm_time / (base_time_per_batch + total_comm_time)

            residual_overhead = {2: 0.01, 4: 0.02, 8: 0.03, 16: 0.05}.get(tp_degree, 0.06)
            tp_efficiency_compute = max(0.50, 1.0 - residual_overhead)

            logger.debug(
                f"  [tp_overhead] TP={tp_degree}, bw={nvlink_bw_gbps:.1f}GB/s, "
                f"residual_overhead={residual_overhead:.3f}, "
                f"comm_overhead_ratio={comm_overhead_ratio:.4f}, "
                f"tp_eff={tp_efficiency_compute:.4f}"
            )

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

        # Phase-aware batch efficiency:
        # - Prefill: a single long prefill already saturates the GPU via O(n²) attention,
        #   so batch_efficiency_factor(1)=0.50 is wrong. Use seq_len-based saturation.
        # - Decode: use standard batch_efficiency_factor (more concurrent seqs = better utilization)
        if phase == 'prefill':
            batch_eff = min(1.0, 0.70 + 0.30 * min(1.0, seq_len / 512))
        else:
            # Decode is memory-bound; even batch=1 achieves high BW utilization with CUDA graphs
            if batch_size >= 16:
                batch_eff = 1.0
            elif batch_size >= 4:
                batch_eff = 0.95
            elif batch_size >= 2:
                batch_eff = 0.90
            else:
                batch_eff = 0.85
        
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
            logger.info('='*80)
        
        # NOTE: We do NOT multiply by tp_degree here!
        # TP is not data parallelism - it processes the same batch across GPUs

        return base_throughput

    @staticmethod
    def calculate_aggregated_throughput(
        prefill_throughput: float,
        decode_throughput: float,
        seq_len: int,
        output_len: int,
        max_num_seqs: int = 256,
        prefill_batch: int = 1,
    ) -> float:
        """
        Wave-based system throughput model matching vLLM batch scheduling.

        vLLM processes requests in waves of N=max_num_seqs:
        1. Prefill phase: ceil(N/P) iterations, each prefilling P sequences
           (P = prefill_batch = min(max_num_batched_tokens // seq_len, N))
        2. Decode phase: output_len iterations, each decoding N sequences

        wave_time = ceil(N/P) * (P*S / prefill_tp) + O * (N / decode_tp)
        system_tp = N * (S + O) / wave_time

        Args:
            prefill_throughput: Prefill throughput at batch=P (tokens/sec)
            decode_throughput: Decode throughput at batch=N (system-wide tokens/sec)
            seq_len: Input sequence length (prefill tokens)
            output_len: Output length (decode tokens to generate)
            max_num_seqs: vLLM max_num_seqs (actual decode concurrency = N)
            prefill_batch: Sequences prefilled per iteration (P)

        Returns:
            System-wide throughput in tokens/second
        """
        if prefill_throughput <= 0 or decode_throughput <= 0:
            return 0.0
        if seq_len <= 0 and output_len <= 0:
            return 0.0

        N = max(1, max_num_seqs)
        P = max(1, min(prefill_batch, N))
        S = max(1, seq_len)
        O = max(1, output_len)

        prefill_iters = math.ceil(N / P)
        t_prefill = prefill_iters * (P * S / prefill_throughput)
        t_decode = O * (N / decode_throughput)
        wave_time = t_prefill + t_decode

        system_tp = N * (S + O) / wave_time if wave_time > 0 else 0.0

        logger.debug(
            f"  [wave_model] N={N}, P={P}, S={S}, O={O}, "
            f"prefill_iters={prefill_iters}, t_prefill={t_prefill:.4f}s, "
            f"t_decode={t_decode:.4f}s, wave_time={wave_time:.4f}s, "
            f"system_tp={system_tp:.0f} tok/s"
        )

        return system_tp

class LLMPlacementSolverWithTP:
    """LLM placement solver with Tensor Parallelism and practical constraints"""

    def __init__(self, config_dir: str, tp_configuration: Optional[Dict[str, int]] = None,
                 enable_symmetry_breaking: bool = True,
                 enable_upper_bound: bool = True, enable_tight_bigm: bool = True,
                 enable_flow_conservation: bool = True, threads: Optional[int] = None,
                 max_threads: int = 32, generate_network: Optional[Tuple[float, float]] = None,
                 cloud_provider: Optional[str] = None, throughput_debug_samples: int = 0,
                 workload_phase: Optional[str] = None, sequence_length: Optional[int] = None,
                 output_length: Optional[int] = None, min_batch_size: Optional[int] = None,
                 max_batch_size: Optional[int] = None, skip_gurobi: bool = False,
                 gpu_pool_file: Optional[str] = None,
                 network_bandwidth_file: Optional[str] = None,
                 cloud_specs_file: Optional[str] = None,
                 skip_solve_init: bool = False,
                 max_input_tokens: Optional[int] = None,
                 max_output_tokens: Optional[int] = None,
                 max_num_seqs: Optional[int] = None,
                 max_num_batched_tokens: Optional[int] = None,
                 gpu_memory_utilization: Optional[float] = None,
                 total_tokens_to_process: Optional[int] = None,
                 max_total_runtime_hours: Optional[float] = None):

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
        
        if skip_gurobi:
            self.options = {}
            self.env = None
        else:
            self.options = _read_gurobi_wls_file("gurobi.wls")
            self.env = gp.Env(params=self.options)
        self.config_dir = config_dir
        self.cloud_provider = cloud_provider
        self.throughput_debug_samples = max(0, int(throughput_debug_samples))
        
        # Optimization flags
        self.enable_symmetry_breaking = enable_symmetry_breaking
        self.enable_upper_bound = enable_upper_bound
        self.enable_tight_bigm = enable_tight_bigm
        self.enable_flow_conservation = enable_flow_conservation
        self.threads = threads
        self.max_threads = max_threads

        # Load cloud pricing data if available
        # Use provided path or fall back to default relative path
        _cloud_specs_file = cloud_specs_file or 'config/cloud_instances_specs.csv'
        if os.path.exists(_cloud_specs_file):
            self.cloud_instances = self._load_cloud_instances(_cloud_specs_file)
        else:
            self.cloud_instances = None

        # Load configuration
        config_file = os.path.join(config_dir, 'config.csv')
        base_config_dir = os.path.abspath(os.path.join(config_dir, os.pardir))

        # Use provided paths or fall back to defaults (relative to config_dir parent)
        _gpu_pool_file = gpu_pool_file or os.path.join(base_config_dir, 'gpu_pool.csv')
        _network_file = network_bandwidth_file or os.path.join(base_config_dir, 'network_bandwidth.csv')

        self.gpu_types = self._load_gpu_pool(_gpu_pool_file)
        
        # Load or generate network bandwidth
        if generate_network is not None:
            intra_bw, inter_bw = generate_network
            generated = self._generate_network_bandwidth(intra_bw, inter_bw)
            logger.info(f"Generated network bandwidth matrix: intra={intra_bw} GB/s, inter={inter_bw} GB/s")
            # Save to the specified network file path
            self._save_network_bandwidth(_network_file, generated)
            logger.info(f"Saved generated network bandwidth to {_network_file}")

        self.network_bandwidth = self._load_network_bandwidth(_network_file)
        logger.info(f"Loaded network bandwidth from {_network_file}")
        
        self.config = self._load_config(
            config_file,
            workload_phase=workload_phase,
            sequence_length=sequence_length,
            output_length=output_length,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            total_tokens_to_process=total_tokens_to_process,
            max_total_runtime_hours=max_total_runtime_hours,
        )
        # Allow runtime overrides for SLO-driven placement
        if total_tokens_to_process is not None:
            self.config.total_tokens_to_process = total_tokens_to_process
        if max_total_runtime_hours is not None:
            self.config.max_total_runtime_hours = max_total_runtime_hours
        self.model = None
        self.solution = None
        self.solve_log = ""
        
        # Derived data
        self.total_gpus = sum(gpu_type.count for gpu_type in self.gpu_types.values())
        
        # Validate network matrix
        if self.network_bandwidth.shape[0] != self.total_gpus:
            raise ValueError(f"Network bandwidth matrix size ({self.network_bandwidth.shape[0]}) "
                           f"does not match total GPU count ({self.total_gpus})")
        
        # TP configuration: {gpu_type: max_tp_degree} (cap by instance GPU count)
        if tp_configuration is None:
            self.tp_max_configuration = {
                gpu_type: min(8, gpu_obj.count)
                for gpu_type, gpu_obj in self.gpu_types.items()
            }
        else:
            self.tp_max_configuration = {}
            for gpu_type, gpu_obj in self.gpu_types.items():
                requested = tp_configuration.get(gpu_type, 8)
                capped = min(requested, gpu_obj.count)
                if capped != requested:
                    logger.warning(
                        f"Capping TP for {gpu_type}: requested {requested}, "
                        f"available GPUs {gpu_obj.count} -> {capped}"
                    )
                self.tp_max_configuration[gpu_type] = capped
        
        if skip_solve_init:
            # Lightweight init: only needed for evaluate_manual_placement()
            self.tp_allocations = None
            self.max_segment_size = None
            self.min_segment_size = None
            self.quantized_sizes = None
            self.valid_segments = []
            self.segment_throughputs = {}
            self.valid_connections = []
            self.gpu_pair_throughputs = {}
        else:
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

            if self.throughput_debug_samples > 0:
                self._log_throughput_debug_samples(self.throughput_debug_samples)
                self._log_network_debug_samples(self.throughput_debug_samples)

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
    
    def _load_cloud_instances(self, filename: str) -> Dict[Tuple[str, str], Dict[str, Union[str, float]]]:
        """
        Load cloud instance specs from cloud_instances_specs.csv
        Returns dict: (provider, instance_name) -> specs
        """
        import re
        
        df = pd.read_csv(filename)
        instances = {}
        
        for _, row in df.iterrows():
            provider = row['Cloud Provider']
            instance_name = row['Instance Name']
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
            
            num_gpus = row.get('num_gpus')
            vram_per_gpu = row.get('vram_gb_per_gpu')
            if pd.isna(num_gpus) or pd.isna(vram_per_gpu):
                # Fallback to parsing original columns if new ones are missing
                gpu_count_str = str(row.get('GPU Count', ''))
                gpu_mem_str = str(row.get('GPU Memory per GPU', ''))
                gpu_count_numbers = re.findall(r'\d+', gpu_count_str)
                mem_numbers = re.findall(r'[\d.]+', gpu_mem_str)
                num_gpus = float(gpu_count_numbers[0]) if gpu_count_numbers else None
                vram_per_gpu = float(mem_numbers[0]) if mem_numbers else None
            
            if num_gpus is None or vram_per_gpu is None:
                continue

            instances[(provider, instance_name)] = {
                'instance_name': instance_name,
                'gpu_model': gpu_model,
                'num_gpus': float(num_gpus),
                'vram_gb_per_gpu': float(vram_per_gpu),
                'price_per_hour': price_value
            }
        
        return instances
    
    def _get_cloud_instance_specs(self, instance_name: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Get instance specs by instance name.
        Returns the cheapest instance if provider not specified.
        """
        if not self.cloud_instances:
            return None
        
        matching = []
        for (provider, inst_name), specs in self.cloud_instances.items():
            if inst_name == instance_name:
                if self.cloud_provider is None or provider == self.cloud_provider:
                    matching.append((provider, specs))
        
        if not matching:
            return None
        
        if self.cloud_provider:
            for provider, specs in matching:
                if provider == self.cloud_provider:
                    return specs
            return None
        else:
            # Return cheapest price
            return min(matching, key=lambda x: x[1]['price_per_hour'])[1]

    def _normalize_gpu_model(self, gpu_model: str) -> str:
        """Normalize GPU model names to match ThroughputFunctions.GPU_SPECS keys."""
        model = gpu_model.upper()
        if "A100" in model:
            return "A100"
        if "H100" in model:
            return "H100"
        if "H200" in model:
            return "H200"
        if "B200" in model:
            return "B200"
        if "L40S" in model:
            return "L40S"
        if "L40" in model:
            return "L40"
        if "A10G" in model or "A10" in model:
            return "A10"
        if "V100" in model:
            return "V100"
        if "T4" in model:
            return "T4"
        if "L4" in model:
            return "L4"
        if "RTX4090" in model:
            return "RTX4090"
        if "L20" in model:
            return "L20"
        if "A40" in model:
            return "A40"
        return gpu_model

    def _load_gpu_pool(self, filename: str) -> Dict[str, GPUType]:
        """Load GPU pool configuration (instance_name,count)"""
        df = pd.read_csv(filename)
        gpu_types = {}
        global_id = 0
        
        for _, row in df.iterrows():
            if 'instance_name' not in row or pd.isna(row['instance_name']):
                raise ValueError("gpu_pool.csv must include 'instance_name' column with valid values.")
            instance_name = row['instance_name']
            instance_count = int(row['count'])
            specs = self._get_cloud_instance_specs(instance_name)
            if specs is None:
                raise ValueError(f"Missing cloud specs for instance '{instance_name}' in cloud_instances_specs.csv.")

            num_gpus = int(specs['num_gpus'])
            gpu_model = self._normalize_gpu_model(str(specs['gpu_model']))
            memory_gb = float(specs['vram_gb_per_gpu'])
            price_per_hour_instance = float(specs['price_per_hour'])
            
            # Try to get price from config first, then from cloud pricing
            if 'dollar_per_hour' in row and pd.notna(row['dollar_per_hour']):
                cost_per_hour = float(row['dollar_per_hour'])
                price_source = "config"
            else:
                cost_per_hour = price_per_hour_instance
                price_source = f"cloud ({self.cloud_provider or 'cheapest'})"

            for inst_idx in range(instance_count):
                instance_key = f"{instance_name}#{inst_idx}"
                global_ids = list(range(global_id, global_id + num_gpus))

                gpu_types[instance_key] = GPUType(
                    name=instance_key,
                    count=num_gpus,
                    memory_gb=memory_gb,
                    global_ids=global_ids,
                    cost_per_hour=cost_per_hour,
                    gpu_model=gpu_model
                )
            
                logger.info(
                    f"  Instance {instance_key}: {num_gpus}×{gpu_model}, "
                    f"{memory_gb:.0f}GB, ${cost_per_hour:.2f}/instance-hour (from {price_source})"
                )
            
                global_id += num_gpus
        
        return gpu_types
    
    def _load_network_bandwidth(self, filename: str) -> np.ndarray:
        """Load network bandwidth matrix"""
        df = pd.read_csv(filename, index_col=0)
        matrix = df.values
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Network bandwidth matrix must be square, got {matrix.shape}")
        
        return matrix

    def _get_gpu_model_name(self, gpu_type: str) -> str:
        """Resolve GPU model name for a given instance name."""
        return self.gpu_types[gpu_type].gpu_model
    
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
    
    def _load_config(
        self,
        filename: str,
        workload_phase: Optional[str] = None,
        sequence_length: Optional[int] = None,
        output_length: Optional[int] = None,
        min_batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        total_tokens_to_process: Optional[int] = None,
        max_total_runtime_hours: Optional[float] = None,
    ) -> Config:
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
        if workload_phase is None:
            workload_phase = config_dict.get('workload_phase', None)
        workload_phase = str(workload_phase).lower() if workload_phase is not None else None
        if workload_phase is None:
            raise ValueError("Missing workload_phase: provide via argparse or config.csv")
        if workload_phase not in ['prefill', 'decode', 'aggregated']:
            raise ValueError(f"Invalid workload_phase: '{workload_phase}'. Must be 'prefill', 'decode', or 'aggregated'")
        logger.info(f"Workload phase: {workload_phase}")
        
        # Load output_length (NEW: for decode phase)
        if output_length is None:
            output_length = int(config_dict.get('output_length', 0))
        if workload_phase == 'decode' and output_length == 0:
            logger.warning("Decode phase with output_length=0! This may indicate misconfiguration.")
        if workload_phase == 'aggregated' and output_length == 0:
            raise ValueError("Aggregated phase requires output_length > 0 to model prefill→decode throughput")

        if sequence_length is None:
            if 'sequence_length' not in config_dict:
                raise ValueError("Missing sequence_length: provide via argparse or config.csv")
            sequence_length = int(config_dict['sequence_length'])
        if min_batch_size is None:
            if 'min_batch_size' not in config_dict:
                raise ValueError("Missing min_batch_size: provide via argparse or config.csv")
            min_batch_size = int(config_dict['min_batch_size'])
        if max_batch_size is None:
            if 'max_batch_size' not in config_dict:
                raise ValueError("Missing max_batch_size: provide via argparse or config.csv")
            max_batch_size = int(config_dict['max_batch_size'])

        # NEW: Runtime calibration knobs (optional)
        # real_world_efficiency: overall efficiency multiplier [0,1]
        # micro_batch_size: assumed micro-batch size for pipeline bubble model
        real_world_efficiency = float(config_dict.get('real_world_efficiency', 0.55))
        micro_batch_size = int(config_dict.get('micro_batch_size', 8))

        # max_num_batched_tokens: vLLM default is 16384 (0 = fall back to max_model_len)
        if max_num_batched_tokens is None:
            max_num_batched_tokens = int(config_dict.get('max_num_batched_tokens', 16384))

        # NEW: gpu_memory_utilization (fraction of GPU memory for model/KV)
        if gpu_memory_utilization is None:
            gpu_memory_utilization = float(config_dict.get('gpu_memory_utilization', 0.90))

        # NEW: Penalty for cross-instance PP connections (encourages intra-node placement)
        cross_instance_penalty = float(config_dict.get('cross_instance_penalty', 0.01))

        # NEW: KV block size, enforce_eager, cross-node PP latency
        kv_block_size = int(config_dict.get('kv_block_size', 16))
        enforce_eager = str(config_dict.get('enforce_eager', 'false')).lower() in ('true', '1', 'yes')
        enforce_eager_penalty = float(config_dict.get('enforce_eager_penalty', 0.80))
        cross_node_pp_latency_us = float(config_dict.get('cross_node_pp_latency_us', 0.0))
        
        return Config(
            workload_phase=workload_phase,
            sequence_length=sequence_length,
            output_length=output_length,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            model_name=config_dict['model_name'],
            num_decoder_layers=int(config_dict['num_decoder_layers']),
            d_model=int(config_dict['d_model']),
            d_hidden=int(config_dict['d_hidden']),
            vocab_size=int(config_dict['vocab_size']),
            num_attention_heads=int(config_dict['num_attention_heads']),
            num_kv_heads=int(config_dict.get('num_kv_heads', config_dict['num_attention_heads'])),
            max_position_embeddings=int(config_dict.get('max_position_embeddings', 131072)),  # Default to 128K if not specified
            head_dim=int(config_dict['head_dim']) if 'head_dim' in config_dict else int(config_dict['d_model']) // int(config_dict['num_attention_heads']),
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
            total_tokens_to_process=total_tokens_to_process if total_tokens_to_process is not None else int(config_dict['total_tokens_to_process']),
            max_total_runtime_hours=max_total_runtime_hours if max_total_runtime_hours is not None else float(config_dict['max_total_runtime_hours']),
            real_world_efficiency=real_world_efficiency,
            micro_batch_size=micro_batch_size,
            max_input_tokens=max_input_tokens or 0,
            max_output_tokens=max_output_tokens or 0,
            max_num_seqs=max_num_seqs or 256,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            cross_instance_penalty=cross_instance_penalty,
            kv_block_size=kv_block_size,
            enforce_eager=enforce_eager,
            enforce_eager_penalty=enforce_eager_penalty,
            cross_node_pp_latency_us=cross_node_pp_latency_us
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
        bw = self._get_min_intra_tp_bandwidth(gpu_type, gpu_set)
        logger.debug(
            f"  [tp_bw] {gpu_type} TP={tp_degree}: "
            f"gpu_set={list(gpu_set)}, min_intra_bw={bw:.2f} GB/s"
        )
        return bw
    
    def _get_representative_pp_bandwidth(self, gpu_type: str, tp_degree: int,
                                          pp_stages: int, gpus_per_instance: int,
                                          instances: Optional[List[str]] = None) -> float:
        """
        Get representative inter-stage bandwidth for PP communication.

        If all PP stages fit on one instance, returns intra-node bandwidth.
        If stages span multiple instances, returns inter-node bandwidth (bottleneck).

        Args:
            gpu_type: GPU type name (a single instance, e.g. "p4d.24xlarge#0")
            tp_degree: TP degree (GPUs per TP group)
            pp_stages: Number of pipeline stages
            gpus_per_instance: Number of GPUs per instance
            instances: List of instance names in the same family

        Returns:
            Representative inter-stage bandwidth in GB/s
        """
        if pp_stages <= 1:
            return float('inf')

        gpu_obj = self.gpu_types[gpu_type]
        pp_stages_per_instance = gpus_per_instance // tp_degree

        if pp_stages <= pp_stages_per_instance:
            # All stages fit on one instance — use intra-node bandwidth
            # Between first TP group and second TP group on same instance
            from_gpu = gpu_obj.global_ids[0]
            to_gpu = gpu_obj.global_ids[min(tp_degree, len(gpu_obj.global_ids) - 1)]
            bw = self.network_bandwidth[from_gpu, to_gpu]
            logger.debug(
                f"  [pp_bw] {gpu_type} TP={tp_degree} PP={pp_stages}: "
                f"intra-node gpu_{from_gpu}->gpu_{to_gpu}, bw={bw:.2f} GB/s"
            )
            return bw
        else:
            # Stages span multiple instances — bottleneck is inter-node link
            # Look at bandwidth between first GPU of this instance and first GPU of next instance
            if instances and len(instances) >= 2:
                from_gpu = self.gpu_types[instances[0]].global_ids[0]
                to_gpu = self.gpu_types[instances[1]].global_ids[0]
                bw = self.network_bandwidth[from_gpu, to_gpu]
                logger.debug(
                    f"  [pp_bw] {gpu_type} TP={tp_degree} PP={pp_stages}: "
                    f"inter-node {instances[0]}->gpu_{from_gpu} to {instances[1]}->gpu_{to_gpu}, "
                    f"bw={bw:.2f} GB/s"
                )
                return bw
            else:
                logger.debug(
                    f"  [pp_bw] {gpu_type} TP={tp_degree} PP={pp_stages}: "
                    f"fallback bw=1.0 GB/s (no instances provided)"
                )
                return 1.0  # Conservative fallback

    def _compute_pipeline_efficiency_1f1b(self, pp_stages: int, inter_stage_bw: float,
                                           prefill_tp: float = 0.0, decode_tp: float = 0.0,
                                           effective_decode_batch: int = 0, prefill_batch: int = 0,
                                           stage_throughput: float = 0.0, batch_for_phase: int = 0) -> float:
        """
        Compute pipeline efficiency using 1F1B wave model with communication overhead.

        For aggregated workloads (prefill_tp > 0 and decode_tp > 0):
            Models communication overhead per iteration + startup bubble.
        For non-aggregated workloads:
            Models steady-state compute_time / (compute_time + comm_time).

        Returns 1.0 if pp_stages <= 1.
        """
        if pp_stages <= 1:
            return 1.0

        inter_stage_bw = max(1e-6, inter_stage_bw)
        _d = self.config.d_model
        _bpe = self.config.bytes_per_element

        if prefill_tp > 0 and decode_tp > 0:
            # Aggregated: explicit wave model with comm overhead + startup bubble
            _N = effective_decode_batch
            _P = prefill_batch
            _S = self.config.sequence_length
            _O = max(1, self.config.output_length)

            # Inter-stage activation transfer time per iteration
            latency_s = self.config.cross_node_pp_latency_us * 1e-6
            pf_comm = (_P * _S * _d * _bpe) / (inter_stage_bw * 1e9) + latency_s
            dc_comm = (_N * 1 * _d * _bpe) / (inter_stage_bw * 1e9) + latency_s

            # Per-iteration compute times (from roofline)
            T_pf = (_P * _S) / prefill_tp
            T_dc = _N / decode_tp

            # Step times with communication
            T_pf_c = T_pf + pf_comm
            T_dc_c = T_dc + dc_comm

            K_pf = math.ceil(_N / _P)
            K_dc = _O

            # Wave time without PP overhead (baseline)
            wave_time_no_pp = K_pf * T_pf + K_dc * T_dc
            # Wave time with comm overhead + startup bubble (PP-1 prefill steps)
            wave_time_pp = (K_pf * T_pf_c + K_dc * T_dc_c
                            + (pp_stages - 1) * T_pf_c)

            pipeline_efficiency = wave_time_no_pp / wave_time_pp if wave_time_pp > 0 else 1.0

            logger.info(
                f"    PP efficiency (1F1B): inter_stage_bw={inter_stage_bw:.2f} GB/s, "
                f"pf_comm={pf_comm*1000:.2f}ms, dc_comm={dc_comm*1000:.2f}ms, "
                f"bubble=({pp_stages}-1)*{T_pf_c*1000:.2f}ms, "
                f"pipeline_eff={pipeline_efficiency:.3f}"
            )
        else:
            # Non-aggregated: comm overhead only (no wave structure for bubble)
            if self.config.workload_phase == 'prefill':
                tokens_per_step = max(1, self.config.sequence_length)
            else:
                tokens_per_step = 1
            batch_for_comm = batch_for_phase
            comm_bytes = batch_for_comm * tokens_per_step * _d * _bpe
            comm_time = comm_bytes / (inter_stage_bw * 1e9)
            compute_time = (batch_for_comm * tokens_per_step) / stage_throughput if stage_throughput > 0 else 0
            pipeline_efficiency = compute_time / (compute_time + comm_time) if (compute_time + comm_time) > 0 else 1.0

            logger.debug(
                f"    PP efficiency (non-agg): inter_stage_bw={inter_stage_bw:.2f} GB/s, "
                f"batch={batch_for_comm}, tokens/step={tokens_per_step}, "
                f"comm_bytes={comm_bytes:.0f}B, comm_time={comm_time*1000:.3f}ms, "
                f"compute_time={compute_time*1000:.3f}ms, pipeline_eff={pipeline_efficiency:.3f}"
            )

        return pipeline_efficiency

    def _get_inter_stage_bw_for_placement(self, segments_sorted) -> float:
        """
        Get bottleneck (minimum) inter-stage bandwidth for a heterogeneous placement.

        Args:
            segments_sorted: List of segment tuples sorted by start_layer.
                Each segment is (gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, batch_size).

        Returns:
            Minimum inter-stage bandwidth in GB/s, or float('inf') if <= 1 segment.
        """
        if len(segments_sorted) <= 1:
            return float('inf')

        min_bw = float('inf')
        for i in range(len(segments_sorted) - 1):
            seg1 = segments_sorted[i]
            seg2 = segments_sorted[i + 1]
            gpu_type1, gpu_set1 = seg1[0], seg1[1]
            gpu_type2, gpu_set2 = seg2[0], seg2[1]
            master_gpu1 = self._get_global_gpu_id(gpu_type1, min(gpu_set1))
            master_gpu2 = self._get_global_gpu_id(gpu_type2, min(gpu_set2))
            bw = self.network_bandwidth[master_gpu1, master_gpu2]
            min_bw = min(min_bw, bw)

        return min_bw

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
            # CRITICAL: both num_attention_heads and num_kv_heads must be divisible by tp_degree (vLLM requirement)
            all_possible_tp = [d for d in [1, 2, 4, 8, 16] if d <= max_tp and d <= gpu_obj.count]
            valid_tp_degrees = [d for d in all_possible_tp
                            if self.config.num_attention_heads % d == 0
                            and self.config.num_kv_heads % d == 0]

            # Log filtered TP degrees for debugging
            filtered_out = [d for d in all_possible_tp if d not in valid_tp_degrees]
            if filtered_out:
                logger.warning(
                    f"GPU {gpu_type}: TP degrees {filtered_out} filtered out - "
                    f"num_attention_heads ({self.config.num_attention_heads}) or "
                    f"num_kv_heads ({self.config.num_kv_heads}) not divisible"
                )

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

            # Consistency check: partition_id should map to a single TP degree per GPU type
            partition_id_to_tp = {}
            for _, tp_degree, partition_id in allocations:
                if partition_id in partition_id_to_tp and partition_id_to_tp[partition_id] != tp_degree:
                    raise ValueError(
                        f"Partition ID reuse detected for GPU {gpu_type}: "
                        f"partition_id={partition_id} maps to TP {partition_id_to_tp[partition_id]} "
                        f"and TP {tp_degree}"
                    )
                partition_id_to_tp[partition_id] = tp_degree

        return tp_allocations
    
    def _compute_max_segment_sizes(self) -> Dict[Tuple[str, int], int]:
        """Compute maximum segment size for each (GPU type, TP degree) pair"""
        max_sizes = {}

        for gpu_type, gpu_obj in self.gpu_types.items():
            tp_degree = self.tp_max_configuration[gpu_type]

            # With TP, weights are sharded
            weight_per_gpu_per_layer = self.config.layer_weight_memory_gb / tp_degree
            kv_cache_per_layer = self._calculate_kv_cache_per_layer(tp_degree)

            # Activation memory with TP (excludes KV cache; KV scales with layers)
            activation_memory = self._calculate_activation_memory(tp_degree)

            # Binary search for max layers (with context for logging)
            context = {'gpu_type': gpu_type, 'tp_degree': tp_degree, 'batch_size': 'max'}
            max_layers, is_tight, headroom_pct = self._binary_search_max_layers(
                gpu_obj.memory_gb, weight_per_gpu_per_layer + kv_cache_per_layer,
                activation_memory, context=context
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
            kv_cache_per_layer = self._calculate_kv_cache_per_layer(tp_degree)
            activation_memory = self._calculate_activation_memory(tp_degree)

            # Total memory = (layers * weight_per_layer) + activation_memory
            # We want: total >= min_memory_utilization * gpu_memory
            # Solving for layers:
            min_layers = max(1, math.ceil(
                (self.config.min_memory_utilization * gpu_memory - activation_memory) /
                (weight_per_layer + kv_cache_per_layer)
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
        Generate quantized segment sizes for performance.
        Combines exact divisors of total_layers with powers-of-two sizes.
        """
        total_layers = self.config.num_decoder_layers
        
        sizes = set()

        # # Include all exact divisors (good for even splits like 8+8+8+8 for 32, 10+10+10 for 30)
        # for d in range(1, int(math.sqrt(total_layers)) + 1):
        #     if total_layers % d == 0:
        #         sizes.add(d)
        #         sizes.add(total_layers // d)

        # # Also include powers of two for non-divisible totals
        # power = 4
        # while power <= total_layers:
        #     sizes.add(power)
        #     power *= 2
        
        size = total_layers
        while True:
            sizes.add(size)
            size = size // 2
            if size < 8:
                break

        sizes = sorted(sizes)
        
        logger.info(f"Quantized segment sizes (divisors + powers of two): {sizes}")
        logger.info(f"  Using {len(sizes)} sizes for performance")
        return sizes
    
    def _calculate_kv_cache_per_layer(self, tp_degree: int = 1, batch_size: int = None) -> float:
        """Calculate KV cache memory per layer per GPU (GB)."""
        batch = batch_size if batch_size is not None else self.config.min_batch_size
        bytes_per_elem = self.config.bytes_per_element
        hidden = self.config.d_model
        d_head = self.config.head_dim
        kv_dim = self.config.num_kv_heads * d_head

        if self.config.workload_phase == 'prefill':
            kv_cache_len = self.config.sequence_length
        else:
            # Peak KV cache during decode/aggregated includes generated tokens
            # For aggregated: use decode peak since that's the maximum KV cache needed
            kv_cache_len = self.config.sequence_length + max(self.config.output_length, 0)

        kv_cache_bytes = (2 * batch * kv_cache_len * (kv_dim / tp_degree) *
                          bytes_per_elem)
        return kv_cache_bytes / (1024**3)

    def _calculate_activation_memory(self, tp_degree: int = 1, batch_size: int = None) -> float:
        """
        Calculate runtime memory overhead per GPU beyond weights and KV cache.

        Models actual vLLM/inference framework memory consumers:
        1. Peak activations (intermediate tensors, reused across layers)
        2. CUDA graph workspace (pre-allocated for decode phase)
        3. Sampler workspace (logits buffer, top-k/p workspace)
        4. Allocator fragmentation (PyTorch caching allocator overhead on TOTAL memory)

        Returns: Memory overhead in GB
        """
        batch = batch_size if batch_size is not None else self.config.max_batch_size
        hidden = self.config.d_model
        d_hidden = self.config.d_hidden
        bytes_per_elem = self.config.bytes_per_element
        phase = self.config.workload_phase
        d_head = self.config.head_dim
        kv_dim = self.config.num_kv_heads * d_head
        q_o_dim = self.config.num_attention_heads * d_head

        # 1. PEAK ACTIVATIONS (per-layer, reused across layers)
        if phase == 'prefill' or phase == 'aggregated':
            seq_len = self.config.sequence_length
            tokens = seq_len
        else:  # decode
            tokens = 1

        # Sharded intermediate tensors during computation
        qkv_memory = (batch * tokens * (
            (q_o_dim / tp_degree) +               # Q
            2 * (kv_dim / tp_degree)              # K, V (GQA/MQA aware)
        ) * bytes_per_elem) / (1024**3)
        mlp_intermediate = (batch * tokens * (d_hidden / tp_degree) *
                           bytes_per_elem) / (1024**3)
        sharded_computation = qkv_memory + mlp_intermediate

        # Full activation tensor after all-reduce (NOT sharded)
        full_activation = (batch * tokens * hidden * bytes_per_elem) / (1024**3)

        peak_activation = max(sharded_computation, full_activation)

        # 2. CUDA GRAPH WORKSPACE (critical for decode!)
        # vLLM captures CUDA graphs for decode to avoid kernel launch overhead
        # Each captured graph pre-allocates workspace memory
        if phase in ['decode', 'aggregated']:
            # vLLM captures graphs for batch sizes: 1, 2, 4, 8, ... up to max
            num_captured_graphs = min(6, int(math.log2(max(1, batch))) + 2)
            decode_workspace_per_graph = (batch * 1 * (hidden + d_hidden / tp_degree) *
                                          bytes_per_elem) / (1024**3)
            cuda_graph_memory = num_captured_graphs * decode_workspace_per_graph
            # Add fixed graph metadata overhead (~500 MB)
            cuda_graph_memory += 0.5
        else:
            cuda_graph_memory = 0.0

        # 3. SAMPLER WORKSPACE
        # Logits buffer: [batch, vocab_size] in float32 for numerical stability
        logits_memory = (batch * self.config.vocab_size * 4) / (1024**3)  # float32
        # Top-k/p workspace and temperature scaling buffers
        sampler_memory = logits_memory * 1.5

        # 4. ALLOCATOR FRAGMENTATION
        # PyTorch's caching allocator causes memory fragmentation
        # This applies to ALL runtime overhead, not just activations
        # Empirically: 12-18% depending on memory pressure
        base_overhead = peak_activation + cuda_graph_memory + sampler_memory
        allocator_fragmentation = base_overhead * 0.15

        # Total runtime overhead
        total_overhead = base_overhead + allocator_fragmentation

        logger.debug(
            f"  [activation_mem] TP={tp_degree} batch={batch}: "
            f"peak_act={peak_activation*1024:.1f}MB (sharded={sharded_computation*1024:.1f}MB, "
            f"full={full_activation*1024:.1f}MB), "
            f"cuda_graph={cuda_graph_memory*1024:.1f}MB, "
            f"sampler={sampler_memory*1024:.1f}MB, "
            f"frag={allocator_fragmentation*1024:.1f}MB, "
            f"total={total_overhead:.4f}GB"
        )

        return total_overhead

    def _compute_kv_pool_gb(self, gpu_memory_gb: float, layers_per_stage: int,
                            tp_degree: int) -> float:
        """
        Compute available KV cache memory pool in GB.

        KV pool = GPU memory - (weight memory for layers on this stage) - fixed overhead.
        Fixed overhead uses batch_size=1 (minimal activations).

        Args:
            gpu_memory_gb: Total GPU memory in GB
            layers_per_stage: Number of layers on this pipeline stage
            tp_degree: Tensor parallelism degree

        Returns:
            Available KV pool memory in GB (>= 0)
        """
        usable_memory_gb = gpu_memory_gb * self.config.gpu_memory_utilization
        weight_per_layer = self.config.layer_weight_memory_gb / tp_degree
        weights_gb = layers_per_stage * weight_per_layer
        fixed_overhead = self._calculate_activation_memory(tp_degree, batch_size=1)
        kv_pool = usable_memory_gb - weights_gb - fixed_overhead

        logger.debug(
            f"  [kv_pool] GPU={gpu_memory_gb:.1f}GB, usable={usable_memory_gb:.1f}GB "
            f"(util={self.config.gpu_memory_utilization:.0%}), "
            f"weights={weights_gb:.3f}GB ({layers_per_stage}L × {weight_per_layer:.3f}GB/L, TP={tp_degree}), "
            f"overhead={fixed_overhead:.3f}GB, "
            f"kv_pool={max(0.0, kv_pool):.3f}GB"
        )

        return max(0.0, kv_pool)

    def _kv_bytes_per_token(self, layers_per_stage: int, tp_degree: int) -> float:
        """
        KV cache bytes per token per sequence for the layers on one pipeline stage.

        Formula: 2 * layers_per_stage * (kv_dim / tp) * bytes_per_element
        (factor of 2 for K and V tensors)
        """
        d_head = self.config.head_dim
        kv_dim = self.config.num_kv_heads * d_head
        return 2 * layers_per_stage * (kv_dim / tp_degree) * self.config.bytes_per_element

    def _derive_max_concurrent_sequences(self, gpu_memory_gb: float,
                                          layers_per_stage: int,
                                          tp_degree: int) -> int:
        """
        Derive maximum decode concurrency from available KV cache pool.

        Each concurrent sequence needs KV cache for (avg_input + avg_output) tokens.

        Args:
            gpu_memory_gb: Total GPU memory in GB
            layers_per_stage: Number of layers on this pipeline stage
            tp_degree: Tensor parallelism degree

        Returns:
            Maximum number of concurrent sequences (>= 1)
        """
        kv_pool_gb = self._compute_kv_pool_gb(gpu_memory_gb, layers_per_stage, tp_degree)
        kv_pool_bytes = kv_pool_gb * (1024 ** 3)

        avg_seq_tokens = self.config.sequence_length + max(self.config.output_length, 0)
        if avg_seq_tokens <= 0:
            return 1

        # vLLM allocates KV in blocks of kv_block_size tokens — round up for fragmentation
        kv_block_size = self.config.kv_block_size
        paged_tokens = math.ceil(avg_seq_tokens / kv_block_size) * kv_block_size
        kv_per_seq = self._kv_bytes_per_token(layers_per_stage, tp_degree) * paged_tokens
        if kv_per_seq <= 0:
            return 1

        max_concurrent = max(1, int(kv_pool_bytes / kv_per_seq))
        kv_bpt = self._kv_bytes_per_token(layers_per_stage, tp_degree)

        logger.debug(
            f"  [max_concurrent] kv_pool={kv_pool_gb:.3f}GB, "
            f"kv_bytes/token={kv_bpt:.0f}B, avg_seq_tokens={avg_seq_tokens}, "
            f"kv/seq={kv_per_seq/1024:.1f}KB, max_concurrent={max_concurrent}"
        )

        return max_concurrent

    def _check_max_context_feasibility(self, gpu_memory_gb: float,
                                        layers_per_stage: int,
                                        tp_degree: int) -> bool:
        """
        Check whether 1 sequence at maximum context length fits in KV pool.

        Maximum context = max_input_tokens + max_output_tokens.
        Falls back to avg values if max values are not set.

        Args:
            gpu_memory_gb: Total GPU memory in GB
            layers_per_stage: Number of layers on this pipeline stage
            tp_degree: Tensor parallelism degree

        Returns:
            True if feasible, False otherwise
        """
        max_input = self.config.max_input_tokens or self.config.sequence_length
        max_output = self.config.max_output_tokens or max(self.config.output_length, 0)
        max_context = max_input + max_output
        if max_context <= 0:
            return True

        kv_pool_gb = self._compute_kv_pool_gb(gpu_memory_gb, layers_per_stage, tp_degree)
        kv_pool_bytes = kv_pool_gb * (1024 ** 3)

        kv_for_1_seq = self._kv_bytes_per_token(layers_per_stage, tp_degree) * max_context
        feasible = kv_pool_bytes >= kv_for_1_seq

        logger.debug(
            f"  [context_feasibility] max_input={max_input}, max_output={max_output}, "
            f"max_context={max_context}, kv_pool={kv_pool_gb:.3f}GB, "
            f"kv_for_1_seq={kv_for_1_seq/1024**2:.1f}MB, feasible={feasible}"
        )

        return feasible

    def _compute_max_model_len(self, gpu_memory_gb: float,
                                layers_per_stage: int,
                                tp_degree: int) -> int:
        """
        Compute maximum context length for 1 sequence (for vLLM --max-model-len).

        max_model_len = kv_pool_bytes / kv_bytes_per_token_for_1_seq

        Args:
            gpu_memory_gb: Total GPU memory in GB
            layers_per_stage: Number of layers on this pipeline stage
            tp_degree: Tensor parallelism degree

        Returns:
            Maximum context length (clamped to [1024, max_position_embeddings])
        """
        kv_pool_gb = self._compute_kv_pool_gb(gpu_memory_gb, layers_per_stage, tp_degree)
        kv_pool_bytes = kv_pool_gb * (1024 ** 3)

        kv_bytes_per_token = self._kv_bytes_per_token(layers_per_stage, tp_degree)
        if kv_bytes_per_token <= 0:
            return 1024

        max_len = int(kv_pool_bytes / kv_bytes_per_token)
        raw_max_len = max_len
        max_len = max(1024, min(max_len, self.config.max_position_embeddings))

        logger.debug(
            f"  [max_model_len] kv_pool={kv_pool_gb:.3f}GB, "
            f"kv_bytes/token={kv_bytes_per_token:.0f}B, "
            f"raw_max_len={raw_max_len}, clamped={max_len} "
            f"(bounds=[1024, {self.config.max_position_embeddings}])"
        )

        return max_len

    def _binary_search_max_layers(self, gpu_memory: float, memory_per_layer: float,
                                  activation_memory: float,
                                  context: dict = None) -> tuple:
        """
        Binary search to find maximum layers that fit in GPU memory.

        Args:
            gpu_memory: Total GPU memory in GB
            memory_per_layer: Memory per layer (weights + KV cache) in GB
            activation_memory: Runtime overhead (activations + CUDA graphs + sampler + fragmentation) in GB
            context: Optional dict with 'gpu_type', 'tp_degree', 'batch_size' for logging

        Returns:
            Tuple of (max_feasible_layers, is_tight_memory, headroom_pct)
        """
        left, right = 1, self.config.num_decoder_layers
        max_feasible = 0  # Start at 0 - might not fit ANY layers

        while left <= right:
            mid = (left + right) // 2
            total_memory = mid * memory_per_layer + activation_memory

            if total_memory <= gpu_memory:
                max_feasible = mid
                left = mid + 1
            else:
                right = mid - 1

        # Calculate headroom
        is_tight_memory = False
        headroom_pct = 100.0

        if max_feasible > 0:
            used_memory = max_feasible * memory_per_layer + activation_memory
            headroom = gpu_memory - used_memory
            headroom_pct = headroom / gpu_memory * 100

            # Tight memory threshold (configurable, default 10%)
            tight_memory_threshold = getattr(self.config, 'tight_memory_threshold_pct', 10.0)
            is_tight_memory = headroom_pct < tight_memory_threshold

            if is_tight_memory:
                # Build context string for detailed logging
                ctx_str = ""
                if context:
                    gpu_type = context.get('gpu_type', 'unknown')
                    tp_degree = context.get('tp_degree', '?')
                    batch_size = context.get('batch_size', '?')
                    ctx_str = f" [{gpu_type} TP={tp_degree} BS={batch_size} layers={max_feasible}]"

                logger.warning(
                    f"TIGHT MEMORY{ctx_str}: {used_memory:.1f}/{gpu_memory:.1f} GB used "
                    f"({headroom:.1f} GB / {headroom_pct:.1f}% headroom). "
                    f"Risk of OOM from unmodeled overhead."
                )

        return max_feasible, is_tight_memory, headroom_pct
    
    def _compute_max_segment_size_for_tp(self, gpu_type: str, tp_degree: int, batch_size: int,
                                         filter_tight_memory: bool = None) -> int:
        """
        Compute max segment size for a specific (gpu_type, tp_degree, batch_size).

        Args:
            gpu_type: GPU instance type
            tp_degree: Tensor parallelism degree
            batch_size: Batch size
            filter_tight_memory: If True, returns 0 for tight memory configs (filtered out).
                                 If None, uses config.filter_tight_memory (default True).

        Returns:
            Maximum number of layers, or 0 if filtered out due to tight memory
        """
        gpu_obj = self.gpu_types[gpu_type]
        weight_per_gpu_per_layer = self.config.layer_weight_memory_gb / tp_degree
        kv_cache_per_layer = self._calculate_kv_cache_per_layer(tp_degree, batch_size)
        activation_memory = self._calculate_activation_memory(tp_degree, batch_size)

        context = {
            'gpu_type': gpu_type,
            'tp_degree': tp_degree,
            'batch_size': batch_size
        }

        max_feasible, is_tight_memory, headroom_pct = self._binary_search_max_layers(
            gpu_obj.memory_gb, weight_per_gpu_per_layer + kv_cache_per_layer,
            activation_memory, context=context
        )

        # Determine whether to filter tight memory configs
        if filter_tight_memory is None:
            filter_tight_memory = getattr(self.config, 'filter_tight_memory', True)

        if is_tight_memory and filter_tight_memory:
            logger.info(f"  FILTERED: {gpu_type} TP={tp_degree} BS={batch_size} "
                       f"(headroom={headroom_pct:.1f}% < threshold)")
            return 0

        return max_feasible

    def _compute_min_segment_size_for_tp(self, gpu_type: str, tp_degree: int, batch_size: int) -> int:
        """Compute min segment size for a specific (gpu_type, tp_degree, batch_size)"""
        gpu_obj = self.gpu_types[gpu_type]
        weight_per_layer = self.config.layer_weight_memory_gb / tp_degree
        kv_cache_per_layer = self._calculate_kv_cache_per_layer(tp_degree, batch_size)
        activation_memory = self._calculate_activation_memory(tp_degree, batch_size)
        
        min_layers = max(1, math.ceil(
            (self.config.min_memory_utilization * gpu_obj.memory_gb - activation_memory) /
            (weight_per_layer + kv_cache_per_layer)
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

        # Handle edge case: min_batch_size > max_batch_size (invalid config)
        if not batch_sizes:
            logger.warning(f"Invalid batch size range: min={self.config.min_batch_size} > "
                          f"max={self.config.max_batch_size}. Using min_batch_size.")
            batch_sizes = [self.config.min_batch_size]

        # Ensure max_batch_size is included even if not exact power of 2
        if batch_sizes[-1] != self.config.max_batch_size:
            batch_sizes.append(self.config.max_batch_size)
        logger.info(f"Batch size options: {batch_sizes}")
        return batch_sizes

    def _describe_eval_failure(self, placement: Dict, exc: Exception) -> str:
        """Build a descriptive, root-cause explanation for eval infeasibility."""
        stages = placement.get("stages", [])
        batch_size = placement.get("batch_size", self.config.min_batch_size)
        min_pipeline_segment_size = math.ceil(
            self.config.num_decoder_layers / max(1, self.config.max_pipeline_stages)
        )
        quantized_sizes = self.quantized_sizes or []

        reasons = []
        for idx, stage in enumerate(stages):
            gpu_type = stage.get("gpu_type")
            tp_degree = int(stage.get("tp_degree", 1))
            start_layer = int(stage.get("start_layer", 0))
            end_layer = int(stage.get("end_layer", 0))
            if start_layer <= 0 or end_layer < start_layer or gpu_type not in self.gpu_types:
                continue
            segment_size = end_layer - start_layer + 1

            max_seg = self._compute_max_segment_size_for_tp(gpu_type, tp_degree, batch_size=1)
            min_seg = self._compute_min_segment_size_for_tp(gpu_type, tp_degree, batch_size=1)

            gpu_mem = self.gpu_types[gpu_type].memory_gb
            weight_per_layer = self.config.layer_weight_memory_gb / tp_degree
            kv_cache_per_layer = self._calculate_kv_cache_per_layer(tp_degree, batch_size)
            activation_memory = self._calculate_activation_memory(tp_degree, batch_size)
            min_util_target = self.config.min_memory_utilization * gpu_mem
            per_layer_total = weight_per_layer + kv_cache_per_layer

            min_layers_formula = max(1, math.ceil(
                (min_util_target - activation_memory) / per_layer_total
            )) if per_layer_total > 0 else 1

            stage_header = (
                f"Stage {idx + 1} ({gpu_type}, TP={tp_degree}, layers {start_layer}-{end_layer}, "
                f"size={segment_size})"
            )

            stage_reasons = []
            if segment_size > max_seg:
                total_mem = activation_memory + segment_size * per_layer_total
                max_mem = activation_memory + max_seg * per_layer_total
                stage_reasons.append(
                    f"memory overflow: required={total_mem:.2f}GB "
                    f"(activation={activation_memory:.2f}GB + "
                    f"{segment_size}×per_layer={per_layer_total:.2f}GB) > "
                    f"GPU_mem={gpu_mem:.1f}GB. "
                    f"Max feasible layers={max_seg} "
                    f"(max_required={max_mem:.2f}GB). "
                    f"per_layer breakdown: weights={weight_per_layer:.2f}GB + "
                    f"kv_cache={kv_cache_per_layer:.2f}GB"
                )
            if segment_size < min_seg:
                stage_reasons.append(
                    f"underutilized GPU: segment uses {segment_size} layers, but the minimum to "
                    f"reach {self.config.min_memory_utilization:.0%} of GPU memory is {min_seg} layers. "
                    f"Target memory={min_util_target:.2f}GB (= {self.config.min_memory_utilization:.0%} of "
                    f"{gpu_mem:.1f}GB), activation={activation_memory:.2f}GB, per_layer={per_layer_total:.2f}GB "
                    f"⇒ required_layers=ceil(({min_util_target:.2f} - {activation_memory:.2f}) / "
                    f"{per_layer_total:.2f}) = {min_layers_formula}"
                )
            if segment_size < self.config.min_layers_per_stage:
                stage_reasons.append(
                    f"below min_layers_per_stage={self.config.min_layers_per_stage}"
                )
            if segment_size < min_pipeline_segment_size:
                stage_reasons.append(
                    f"below min_pipeline_segment_size={min_pipeline_segment_size} "
                    f"(num_layers={self.config.num_decoder_layers}, max_pp={self.config.max_pipeline_stages})"
                )
            if quantized_sizes and segment_size not in quantized_sizes:
                stage_reasons.append(
                    f"not in allowed segment sizes {quantized_sizes}"
                )

            if stage_reasons:
                reasons.append(f"{stage_header}: " + "; ".join(stage_reasons))

        if not reasons:
            reasons.append(str(exc))

        return (
            f"INFEASIBLE placement. "
            f"Context: model_layers={self.config.num_decoder_layers}, "
            f"phase={self.config.workload_phase}, seq_len={self.config.sequence_length}, "
            f"output_len={self.config.output_length}, batch={batch_size}, "
            f"pp_stages={len(stages)}. "
            f"Reasons: " + " | ".join(reasons)
        )

    def _compute_network_throughput_between_segments(self, seg1: Tuple, seg2: Tuple) -> float:
        """Compute network throughput between two segments (tokens/sec)."""
        gpu_type1 = seg1[0]
        gpu_type2 = seg2[0]
        gpu_set1 = seg1[1]
        gpu_set2 = seg2[1]
        tp_degree1 = seg1[2]
        tp_degree2 = seg2[2]
        batch_size = seg1[6]

        if self.config.workload_phase in ['prefill', 'aggregated']:
            tensor_size_gb = (batch_size * self.config.sequence_length *
                              self.config.d_model * self.config.bytes_per_element) / (1024**3)
            tokens_per_batch = batch_size * self.config.sequence_length
        else:
            tensor_size_gb = (batch_size * 1 *
                              self.config.d_model * self.config.bytes_per_element) / (1024**3)
            tokens_per_batch = batch_size * 1

        if tp_degree1 > 1:
            min_intra_bw_source = float('inf')
            for id1 in gpu_set1:
                global_id1 = self._get_global_gpu_id(gpu_type1, id1)
                for id2 in gpu_set1:
                    if id1 != id2:
                        global_id2 = self._get_global_gpu_id(gpu_type1, id2)
                        min_intra_bw_source = min(min_intra_bw_source, self.network_bandwidth[global_id1, global_id2])
            all_reduce_bw = min_intra_bw_source * (tp_degree1 - 1) / tp_degree1
        else:
            all_reduce_bw = float('inf')

        master_global_id1 = self._get_global_gpu_id(gpu_type1, min(gpu_set1))
        master_global_id2 = self._get_global_gpu_id(gpu_type2, min(gpu_set2))
        inter_stage_bw = self.network_bandwidth[master_global_id1, master_global_id2]

        if tp_degree2 > 1:
            min_intra_bw_dest = float('inf')
            for id1 in gpu_set2:
                global_id1 = self._get_global_gpu_id(gpu_type2, id1)
                for id2 in gpu_set2:
                    if id1 != id2:
                        global_id2 = self._get_global_gpu_id(gpu_type2, id2)
                        min_intra_bw_dest = min(min_intra_bw_dest, self.network_bandwidth[global_id1, global_id2])
            all_scatter_bw = min_intra_bw_dest * (tp_degree2 - 1) / tp_degree2
        else:
            all_scatter_bw = float('inf')

        effective_bandwidth_gbps = min(all_reduce_bw, inter_stage_bw, all_scatter_bw)
        if effective_bandwidth_gbps > 0:
            transfer_time_sec = tensor_size_gb / effective_bandwidth_gbps
            return tokens_per_batch / transfer_time_sec

        return float('inf')

    def evaluate_manual_placement(self, placement: Dict) -> Dict:
        """
        Evaluate throughput/cost metrics for a user-provided placement.

        Expected format:
        {
          "batch_size": 32,
          "stages": [
            {"gpu_type": "p4d.24xlarge#0", "tp_degree": 8, "start_layer": 1, "end_layer": 16},
            {"gpu_type": "p4d.24xlarge#1", "tp_degree": 8, "start_layer": 17, "end_layer": 32}
          ],
          "sequence_length": 8192,           # optional override
          "output_length": 2048,             # optional override
          "workload_phase": "aggregated"     # optional override
        }
        """
        if "sequence_length" in placement:
            self.config.sequence_length = int(placement["sequence_length"])
        if "output_length" in placement:
            self.config.output_length = int(placement["output_length"])
        if "workload_phase" in placement:
            self.config.workload_phase = str(placement["workload_phase"]).lower()

        # batch_size from placement is treated as an optional cap (user override)
        # Actual decode batch is derived from KV-cache capacity per stage
        user_batch_size = placement.get("batch_size", None)
        if user_batch_size is not None:
            user_batch_size = int(user_batch_size)

        stages = placement.get("stages", [])
        if not stages:
            raise ValueError("Placement must include non-empty 'stages' list.")
        if len(stages) > self.config.max_pipeline_stages:
            raise ValueError(
                f"PP stages {len(stages)} exceed max_pipeline_stages {self.config.max_pipeline_stages}"
            )

        used_local_ids: Dict[str, set] = {gpu_type: set() for gpu_type in self.gpu_types}
        assignments = []
        segments = []
        tp_configuration: Dict[str, int] = {}
        min_pipeline_segment_size = math.ceil(
            self.config.num_decoder_layers / max(1, self.config.max_pipeline_stages)
        )

        for idx, stage in enumerate(stages):
            gpu_type = stage.get("gpu_type")
            if gpu_type not in self.gpu_types:
                raise ValueError(f"Unknown gpu_type '{gpu_type}' in placement.")

            start_layer = int(stage.get("start_layer", 0))
            if "end_layer" in stage:
                end_layer = int(stage["end_layer"])
            elif "segment_size" in stage:
                end_layer = start_layer + int(stage["segment_size"]) - 1
            else:
                raise ValueError("Each stage must include 'end_layer' or 'segment_size'.")

            if start_layer <= 0 or end_layer < start_layer:
                raise ValueError(f"Invalid layer range: start={start_layer}, end={end_layer}")

            gpu_ids = stage.get("gpu_ids")
            if gpu_ids is not None:
                gpu_set = frozenset(int(i) for i in gpu_ids)
                tp_degree = int(stage.get("tp_degree", len(gpu_set)))
                if len(gpu_set) != tp_degree:
                    raise ValueError(f"gpu_ids length does not match tp_degree for {gpu_type}")
                if used_local_ids[gpu_type].intersection(gpu_set):
                    raise ValueError(f"GPU IDs overlap between stages for {gpu_type}: {sorted(gpu_set)}")
            else:
                tp_degree = int(stage.get("tp_degree", 0))
                if tp_degree <= 0:
                    raise ValueError(f"Missing tp_degree for stage {idx} on {gpu_type}")
                if tp_degree > self.gpu_types[gpu_type].count:
                    raise ValueError(
                        f"TP degree {tp_degree} exceeds available GPUs ({self.gpu_types[gpu_type].count}) "
                        f"for {gpu_type}"
                    )
                available = [i for i in range(self.gpu_types[gpu_type].count)
                             if i not in used_local_ids[gpu_type]]
                if len(available) < tp_degree:
                    raise ValueError(f"Not enough available GPUs for {gpu_type}: need {tp_degree}, have {len(available)}")
                gpu_set = frozenset(available[:tp_degree])

            used_local_ids[gpu_type].update(gpu_set)
            segment_size = end_layer - start_layer + 1
            # Use batch_size=1 for weight feasibility (KV capacity checked separately)
            max_seg_size = self._compute_max_segment_size_for_tp(gpu_type, tp_degree, batch_size=1)
            if segment_size > max_seg_size:
                raise ValueError(
                    f"Segment {start_layer}-{end_layer} (size={segment_size}) exceeds "
                    f"max feasible size {max_seg_size} for {gpu_type} TP={tp_degree}"
                )
            min_seg_size = self._compute_min_segment_size_for_tp(gpu_type, tp_degree, batch_size=1)
            if segment_size < min_seg_size:
                raise ValueError(
                    f"Segment {start_layer}-{end_layer} (size={segment_size}) below "
                    f"min feasible size {min_seg_size} for {gpu_type} TP={tp_degree}"
                )
            if segment_size < self.config.min_layers_per_stage:
                raise ValueError(
                    f"Segment size {segment_size} below min_layers_per_stage {self.config.min_layers_per_stage}"
                )
            if segment_size < min_pipeline_segment_size:
                raise ValueError(
                    f"Segment size {segment_size} below min pipeline segment size {min_pipeline_segment_size}"
                )
            if self.quantized_sizes and segment_size not in self.quantized_sizes:
                raise ValueError(
                    f"Segment size {segment_size} not in quantized sizes {self.quantized_sizes}"
                )
            partition_id = idx

            nvlink_bw = self._get_min_intra_tp_bandwidth(gpu_type, gpu_set)

            # --- Derive decode concurrency from KV pool (consistent with solve_homogeneous) ---
            _gpu_mem = self.gpu_types[gpu_type].memory_gb
            max_concurrent = self._derive_max_concurrent_sequences(_gpu_mem, segment_size, tp_degree)
            effective_decode_batch = min(self.config.max_num_seqs, max_concurrent)
            if user_batch_size is not None:
                effective_decode_batch = min(effective_decode_batch, user_batch_size)
            effective_decode_batch = max(1, effective_decode_batch)

            # Check enforce_eager from placement or config
            stage_enforce_eager = placement.get("enforce_eager", self.config.enforce_eager)
            if isinstance(stage_enforce_eager, str):
                stage_enforce_eager = stage_enforce_eager.lower() in ('true', '1', 'yes')

            logger.info(
                f"  Stage {idx}: {gpu_type} TP={tp_degree} layers={start_layer}-{end_layer} "
                f"max_concurrent={max_concurrent}, eff_decode_batch={effective_decode_batch}"
            )

            segment = (gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, effective_decode_batch)
            segments.append(segment)

            # Derive prefill_batch for aggregated mode (consistent with homogeneous solver)
            _max_model_len = self._compute_max_model_len(_gpu_mem, segment_size, tp_degree)
            _mnbt = self.config.max_num_batched_tokens or _max_model_len
            _S = self.config.sequence_length
            _prefill_batch = max(1, min(_mnbt // _S if _S > 0 else 1, effective_decode_batch))

            throughput = ThroughputFunctions.gpu_throughput_with_tp(
                self._get_gpu_model_name(gpu_type), self.config.sequence_length,
                effective_decode_batch, segment_size, self.config.d_model,
                self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                debug=False, phase=self.config.workload_phase,
                output_length=self.config.output_length,
                num_attention_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                prefill_batch=_prefill_batch,
                vocab_size=self.config.vocab_size,
                total_num_layers=self.config.num_decoder_layers
            )
            # Apply enforce_eager penalty to pure decode phase throughput
            if stage_enforce_eager and self.config.workload_phase == 'decode':
                throughput *= self.config.enforce_eager_penalty

            # Compute per-stage prefill/decode throughputs for 1F1B pipeline model
            if self.config.workload_phase == 'aggregated':
                stage_prefill_tp = ThroughputFunctions.gpu_throughput_with_tp(
                    self._get_gpu_model_name(gpu_type), self.config.sequence_length,
                    _prefill_batch, segment_size, self.config.d_model,
                    self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                    debug=False, phase='prefill',
                    num_attention_heads=self.config.num_attention_heads,
                    num_kv_heads=self.config.num_kv_heads,
                    head_dim=self.config.head_dim,
                    vocab_size=self.config.vocab_size,
                    total_num_layers=self.config.num_decoder_layers
                )
                stage_decode_tp = ThroughputFunctions.gpu_throughput_with_tp(
                    self._get_gpu_model_name(gpu_type), self.config.sequence_length,
                    effective_decode_batch, segment_size, self.config.d_model,
                    self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                    debug=False, phase='decode',
                    output_length=self.config.output_length,
                    num_attention_heads=self.config.num_attention_heads,
                    num_kv_heads=self.config.num_kv_heads,
                    head_dim=self.config.head_dim,
                    vocab_size=self.config.vocab_size,
                    total_num_layers=self.config.num_decoder_layers
                )
                # Apply enforce_eager penalty to decode (decode is memory-bound, affected by CUDA graph overhead)
                if stage_enforce_eager:
                    stage_decode_tp *= self.config.enforce_eager_penalty
                    # Recompute aggregated throughput with penalty-adjusted decode
                    throughput = ThroughputFunctions.calculate_aggregated_throughput(
                        prefill_throughput=stage_prefill_tp,
                        decode_throughput=stage_decode_tp,
                        seq_len=self.config.sequence_length,
                        output_len=self.config.output_length,
                        max_num_seqs=effective_decode_batch,
                        prefill_batch=_prefill_batch,
                    )
            else:
                stage_prefill_tp = 0.0
                stage_decode_tp = 0.0

            tp_configuration[gpu_type] = max(tp_configuration.get(gpu_type, 0), tp_degree)
            assignments.append({
                'gpu_type': gpu_type,
                'partition_id': partition_id,
                'gpu_ids': sorted(list(gpu_set)),
                'global_gpu_ids': sorted([self._get_global_gpu_id(gpu_type, i) for i in gpu_set]),
                'tp_degree': tp_degree,
                'start_layer': start_layer,
                'end_layer': end_layer,
                'segment_size': segment_size,
                'throughput': throughput,
                'prefill_tp': stage_prefill_tp,
                'decode_tp': stage_decode_tp,
                'prefill_batch': _prefill_batch,
                'effective_decode_batch': effective_decode_batch,
            })

        assignments.sort(key=lambda x: x['start_layer'])
        segments_sorted = sorted(segments, key=lambda s: s[4])

        expected_start = 1
        last_end = 0
        for seg in segments_sorted:
            start_layer = seg[4]
            end_layer = seg[4] + seg[5] - 1
            if start_layer != expected_start:
                raise ValueError(
                    f"Non-contiguous placement: expected start {expected_start}, got {start_layer}"
                )
            if start_layer <= last_end:
                raise ValueError(
                    f"Overlapping segments detected: start {start_layer} <= previous end {last_end}"
                )
            last_end = end_layer
            expected_start = end_layer + 1
        if last_end != self.config.num_decoder_layers:
            raise ValueError(
                f"Placement does not cover all layers: ends at {last_end}, "
                f"model has {self.config.num_decoder_layers}"
            )

        connections = []
        network_throughputs = []
        for i in range(len(segments_sorted) - 1):
            seg1 = segments_sorted[i]
            seg2 = segments_sorted[i + 1]
            conn_throughput = self._compute_network_throughput_between_segments(seg1, seg2)
            network_throughputs.append(conn_throughput)
            connections.append({
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
                'throughput': conn_throughput
            })

        stage_throughputs = [a['throughput'] for a in assignments]
        raw_throughput = min(stage_throughputs) if stage_throughputs else 0.0

        num_stages = len(assignments)
        inter_stage_bw = self._get_inter_stage_bw_for_placement(segments_sorted)

        # Pipeline-wide effective decode batch = bottleneck across stages
        effective_batches = [a['effective_decode_batch'] for a in assignments]
        pipeline_effective_decode_batch = min(effective_batches) if effective_batches else 1

        if num_stages <= 1:
            pipeline_efficiency = 1.0
        elif self.config.workload_phase == 'aggregated':
            prefill_tps = [a['prefill_tp'] for a in assignments]
            decode_tps = [a['decode_tp'] for a in assignments]
            prefill_batches = [a['prefill_batch'] for a in assignments]
            pipeline_efficiency = self._compute_pipeline_efficiency_1f1b(
                num_stages, inter_stage_bw,
                prefill_tp=min(prefill_tps),
                decode_tp=min(decode_tps),
                effective_decode_batch=pipeline_effective_decode_batch,
                prefill_batch=min(prefill_batches)
            )
        else:
            batch_for_phase = 1 if self.config.workload_phase == 'prefill' else pipeline_effective_decode_batch
            pipeline_efficiency = self._compute_pipeline_efficiency_1f1b(
                num_stages, inter_stage_bw,
                stage_throughput=raw_throughput,
                batch_for_phase=batch_for_phase
            )

        # Apply generation factor from the bottleneck GPU
        _milp_gen_factor = 1.0
        if assignments:
            _milp_gpu_models = [self.gpu_types[a['gpu_type']].gpu_model for a in assignments
                                if a['gpu_type'] in self.gpu_types]
            if _milp_gpu_models:
                _milp_gen_factor = min(
                    ThroughputFunctions.GPU_SPECS.get(m, {}).get('generation_factor', 1.0)
                    for m in _milp_gpu_models
                )
        throughput_per_sec = raw_throughput * pipeline_efficiency * self.config.real_world_efficiency * _milp_gen_factor

        used_instances = sorted({a['gpu_type'] for a in assignments})
        cost_per_hour = sum(self.gpu_types[gpu_type].cost_per_hour for gpu_type in used_instances)

        if throughput_per_sec > 0:
            cost_per_token = cost_per_hour / (throughput_per_sec * 3600)
            total_runtime_hours = self.config.total_tokens_to_process / (throughput_per_sec * 3600)
        else:
            cost_per_token = float('inf')
            total_runtime_hours = float('inf')

        self.solution = {
            'objective_value': 0.0,
            'batch_size': pipeline_effective_decode_batch,
            'effective_decode_batch': pipeline_effective_decode_batch,
            'throughput_tokens_per_sec': throughput_per_sec,
            'raw_throughput_tokens_per_sec': raw_throughput,
            'pipeline_efficiency': pipeline_efficiency,
            'cost_per_hour': cost_per_hour,
            'cost_per_token': cost_per_token,
            'total_runtime_hours': total_runtime_hours,
            'meets_cost_threshold': cost_per_token <= self.config.max_cost_per_token,
            'tp_configuration': tp_configuration,
            'gpu_assignments': assignments,
            'network_connections': connections,
            'solve_status': None,
            'num_pipeline_stages': num_stages
        }

        return self.solution
    
    def _generate_valid_segments(self) -> List[Tuple]:
        """Generate segments with variable TP degree and batch size per segment"""
        valid_segments = []
        batch_size_options = self._get_batch_size_options()
        logger.info(f"Batch size options: {batch_size_options}")
        min_pipeline_segment_size = math.ceil(
            self.config.num_decoder_layers / max(1, self.config.max_pipeline_stages)
        )
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
                            if segment_size < max(self.config.min_layers_per_stage, min_pipeline_segment_size):
                                continue
                            if start_layer + segment_size - 1 <= self.config.num_decoder_layers:
                                # NEW FORMAT: (gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, batch_size)
                                segment = (gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, batch_size)
                                valid_segments.append(segment)
                                
                                # Pre-compute throughput for this segment (OPTIMIZATION)
                                # This avoids recomputing during constraint creation
                                # Get actual network bandwidth from topology (not hardcoded!)
                                nvlink_bw = self._get_min_intra_tp_bandwidth(gpu_type, gpu_set)

                                # Derive prefill_batch for aggregated mode
                                _gpu_mem = self.gpu_types[gpu_type].memory_gb
                                _max_model_len = self._compute_max_model_len(_gpu_mem, segment_size, tp_degree)
                                _mnbt = self.config.max_num_batched_tokens or _max_model_len
                                _S = self.config.sequence_length
                                _prefill_batch = max(1, min(_mnbt // _S if _S > 0 else 1, batch_size))

                                throughput = ThroughputFunctions.gpu_throughput_with_tp(
                                    self._get_gpu_model_name(gpu_type), self.config.sequence_length,
                                    batch_size, segment_size, self.config.d_model,
                                    self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                                    debug=False, phase=self.config.workload_phase,
                                    output_length=self.config.output_length,
                                    num_attention_heads=self.config.num_attention_heads,
                                    num_kv_heads=self.config.num_kv_heads,
                                    head_dim=self.config.head_dim,
                                    prefill_batch=_prefill_batch,
                                    vocab_size=self.config.vocab_size,
                                    total_num_layers=self.config.num_decoder_layers
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

        # Track verification results
        verification_passed = 0
        verification_failed = 0
        verification_missing = 0

        for seg in sample_segments:
            gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, batch_size = seg
            precomputed = throughput_dict.get(seg, None)
            nvlink_bw = self._get_min_intra_tp_bandwidth(gpu_type, gpu_set)
            _gpu_mem = self.gpu_types[gpu_type].memory_gb
            _max_model_len = self._compute_max_model_len(_gpu_mem, segment_size, tp_degree)
            _mnbt = self.config.max_num_batched_tokens or _max_model_len
            _S = self.config.sequence_length
            _prefill_batch = max(1, min(_mnbt // _S if _S > 0 else 1, batch_size))
            recalculated = ThroughputFunctions.gpu_throughput_with_tp(
                self._get_gpu_model_name(gpu_type), self.config.sequence_length,
                batch_size, segment_size, self.config.d_model,
                self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                debug=False, phase=self.config.workload_phase,
                output_length=self.config.output_length,
                num_attention_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                prefill_batch=_prefill_batch,
                vocab_size=self.config.vocab_size,
                total_num_layers=self.config.num_decoder_layers
            )
            if precomputed is None:
                logger.error(f"  ✗ Missing pre-computed value for segment: {seg}")
                verification_missing += 1
            elif abs(precomputed - recalculated) > 0.01:
                logger.error(f"  ✗ Mismatch for {gpu_type} TP={tp_degree} layers={segment_size}: "
                            f"pre-computed={precomputed:.2f}, recalculated={recalculated:.2f}")
                verification_failed += 1
            else:
                logger.debug(f"  ✓ Verified segment throughput: {precomputed:.2f} tokens/sec")
                verification_passed += 1

        # Report verification results accurately
        if verification_failed > 0 or verification_missing > 0:
            logger.error(f"  ✗ Verification FAILED: {verification_passed}/{verification_sample} passed, "
                        f"{verification_failed} mismatches, {verification_missing} missing")
        else:
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
            tp_degree2 = seg2[2]
            batch_size1 = seg1[6]
            batch_size2 = seg2[6]
            
            # PHASE-AWARE: Tensor size depends on workload phase
            # NOTE: batch_size1 should equal batch_size2 due to global constraint
            if self.config.workload_phase == 'prefill':
                # Prefill: Transfer activations for all tokens in the sequence
                tensor_size_gb = (batch_size1 * self.config.sequence_length *
                                self.config.d_model * self.config.bytes_per_element) / (1024**3)
            elif self.config.workload_phase == 'aggregated':
                # Aggregated: Use PREFILL tensor size for inter-stage communication
                # In PP, the full prefill tensor must transfer before any decode can happen
                # This is the bottleneck for network throughput, not the averaged size
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
                elif self.config.workload_phase == 'aggregated':
                    # Aggregated: Use PREFILL tokens per batch (consistent with tensor size)
                    # Network throughput is limited by the prefill phase transfer
                    tokens_per_batch = batch_size1 * self.config.sequence_length
                else:  # decode
                    tokens_per_batch = batch_size1 * 1  # 1 token per forward pass
                throughput = tokens_per_batch / transfer_time_sec
            elif tensor_size_gb == 0:
                # No data to transfer (e.g., edge case with 0 batch size) - effectively infinite
                throughput = float('inf')
            else:
                # Zero bandwidth but non-zero tensor - this indicates missing/invalid data
                logger.warning(f"Zero bandwidth between segments {gpu_type1}->>{gpu_type2} "
                              f"(all_reduce={all_reduce_bw:.2f}, inter={inter_stage_bw:.2f}, "
                              f"all_scatter={all_scatter_bw:.2f} GB/s). Setting low throughput.")
                # Use very low throughput to discourage this connection (but not crash)
                throughput = 1.0  # 1 token/sec - effectively infeasible
            
            gpu_pair_throughputs[(seg1, seg2)] = throughput
        
        return gpu_pair_throughputs

    def _log_throughput_debug_samples(self, sample_count: int) -> None:
        """Log detailed throughput components for a few representative segments."""
        if sample_count <= 0:
            return

        logger.info("=" * 80)
        logger.info("THROUGHPUT DEBUG SAMPLES (roofline + TP + comm)")
        logger.info("=" * 80)

        min_batch = min(self._get_batch_size_options())
        samples = []
        seen_models = set()
        remaining = []
        for gpu_type in sorted(self.gpu_types.keys()):
            max_tp = self.tp_max_configuration[gpu_type]
            candidates = [
                seg for seg in self.valid_segments
                if seg[0] == gpu_type and seg[2] == max_tp and seg[6] == min_batch
            ]
            if not candidates:
                continue
            gpu_model = self._get_gpu_model_name(gpu_type)
            if gpu_model not in seen_models:
                samples.append(candidates[0])
                seen_models.add(gpu_model)
            else:
                remaining.append(candidates[0])
            if len(samples) >= sample_count:
                break
        if len(samples) < sample_count:
            samples.extend(remaining[:max(0, sample_count - len(samples))])

        for seg in samples:
            gpu_type, gpu_set, tp_degree, _, _, segment_size, batch_size = seg
            nvlink_bw = self._get_min_intra_tp_bandwidth(gpu_type, gpu_set)
            gpu_model = self._get_gpu_model_name(gpu_type)
            logger.info(
                f"Sample segment: {gpu_type} (model={gpu_model}) "
                f"TP={tp_degree}, batch={batch_size}, layers={segment_size}, "
                f"nvlink_bw={nvlink_bw:.2f} GB/s"
            )
            _gpu_mem = self.gpu_types[gpu_type].memory_gb
            _max_model_len = self._compute_max_model_len(_gpu_mem, segment_size, tp_degree)
            _mnbt = self.config.max_num_batched_tokens or _max_model_len
            _S = self.config.sequence_length
            _prefill_batch = max(1, min(_mnbt // _S if _S > 0 else 1, batch_size))
            ThroughputFunctions.gpu_throughput_with_tp(
                gpu_model,
                self.config.sequence_length,
                batch_size,
                segment_size,
                self.config.d_model,
                self.config.bytes_per_element,
                tp_degree,
                d_hidden=self.config.d_hidden,
                nvlink_bw_gbps=nvlink_bw,
                debug=True,
                phase=self.config.workload_phase,
                output_length=self.config.output_length,
                num_attention_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                prefill_batch=_prefill_batch,
                vocab_size=self.config.vocab_size,
                total_num_layers=self.config.num_decoder_layers
            )

    def _log_network_debug_samples(self, sample_count: int) -> None:
        """Log detailed network throughput components for a few connections."""
        if sample_count <= 0:
            return
        if not self.valid_connections:
            logger.info("No valid connections available for network debug samples.")
            return

        logger.info("=" * 80)
        logger.info("NETWORK DEBUG SAMPLES (all-reduce -> inter-stage -> all-scatter)")
        logger.info("=" * 80)

        samples = self.valid_connections[:sample_count]
        for seg1, seg2 in samples:
            gpu_type1, gpu_set1, tp_degree1 = seg1[0], seg1[1], seg1[2]
            gpu_type2, gpu_set2, tp_degree2 = seg2[0], seg2[1], seg2[2]
            batch_size1 = seg1[6]

            if self.config.workload_phase == 'prefill':
                tensor_size_gb = (batch_size1 * self.config.sequence_length *
                                  self.config.d_model * self.config.bytes_per_element) / (1024**3)
                tokens_per_batch = batch_size1 * self.config.sequence_length
            else:
                tensor_size_gb = (batch_size1 * 1 *
                                  self.config.d_model * self.config.bytes_per_element) / (1024**3)
                tokens_per_batch = batch_size1

            # Source all-reduce
            if tp_degree1 > 1:
                min_intra_bw_source = float('inf')
                for id1 in gpu_set1:
                    g1 = self._get_global_gpu_id(gpu_type1, id1)
                    for id2 in gpu_set1:
                        if id1 != id2:
                            g2 = self._get_global_gpu_id(gpu_type1, id2)
                            min_intra_bw_source = min(min_intra_bw_source, self.network_bandwidth[g1, g2])
                all_reduce_bw = min_intra_bw_source * (tp_degree1 - 1) / tp_degree1
            else:
                all_reduce_bw = float('inf')

            # Inter-stage transfer
            master_global_id1 = self._get_global_gpu_id(gpu_type1, min(gpu_set1))
            master_global_id2 = self._get_global_gpu_id(gpu_type2, min(gpu_set2))
            inter_stage_bw = self.network_bandwidth[master_global_id1, master_global_id2]

            # Dest all-scatter
            if tp_degree2 > 1:
                min_intra_bw_dest = float('inf')
                for id1 in gpu_set2:
                    g1 = self._get_global_gpu_id(gpu_type2, id1)
                    for id2 in gpu_set2:
                        if id1 != id2:
                            g2 = self._get_global_gpu_id(gpu_type2, id2)
                            min_intra_bw_dest = min(min_intra_bw_dest, self.network_bandwidth[g1, g2])
                all_scatter_bw = min_intra_bw_dest * (tp_degree2 - 1) / tp_degree2
            else:
                all_scatter_bw = float('inf')

            effective_bw = min(all_reduce_bw, inter_stage_bw, all_scatter_bw)
            if effective_bw > 0:
                transfer_time = tensor_size_gb / effective_bw
                throughput = tokens_per_batch / transfer_time
            else:
                transfer_time = float('inf')
                throughput = float('inf')

            logger.info(
                f"{gpu_type1}(TP={tp_degree1}) -> {gpu_type2}(TP={tp_degree2}), "
                f"tensor={tensor_size_gb:.4f}GB, "
                f"all-reduce={all_reduce_bw:.2f}GB/s, inter={inter_stage_bw:.2f}GB/s, "
                f"all-scatter={all_scatter_bw:.2f}GB/s, "
                f"effective={effective_bw:.2f}GB/s, "
                f"throughput={throughput:,.0f} tokens/s"
            )
    
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
        """Build the Gurobi optimization model.

        NOTE: The MILP path still uses batch_size as a decision variable via b[bs]
        binary variables. Since skip_gurobi=True is always set in production, this
        path is unused. When MILP is re-enabled, it needs the same refactor as
        solve_homogeneous(): remove batch_size loop, derive max_concurrent_sequences
        from KV pool, add context feasibility checking.
        """
        build_start = time.time()
        logger.info("Building optimization model with TP and practical constraints...")
        
        model_init_start = time.time()
        if self.env is None:
            self.model = gp.Model("llm_placement_with_tp_constrained")
        else:
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
        
        # TP partition usage: z[gpu_type, tp_degree, partition_id]
        partition_keys = [(seg[0], seg[2], seg[3]) for seg in self.valid_segments]
        self.z = self.model.addVars(
            set(partition_keys),
            vtype=GRB.BINARY,
            name="partition_usage"
        )

        # Instance usage: y[gpu_type] (instance_key)
        self.y = self.model.addVars(
            list(self.gpu_types.keys()),
            vtype=GRB.BINARY,
            name="instance_usage"
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
                    # If an instance is used, all GPUs in that instance must be used exactly once.
                    self.model.addConstr(
                        gp.quicksum(self.x[seg] for seg in segments_using_gpu) == self.y[gpu_type],
                        name=f"gpu_exclusivity_{gpu_type}_{local_gpu_id}"
                    )
                    
        existing_partition_keys = set((seg[0], seg[2], seg[3]) for seg in self.valid_segments)
        # 3. Partition usage indicators
        for gpu_type, allocations in self.tp_allocations.items():
            for gpu_set, tp_degree, partition_id in allocations:
                if (gpu_type, tp_degree, partition_id) in existing_partition_keys:
                    self.model.addConstr(
                        self.z[gpu_type, tp_degree, partition_id] ==
                        gp.quicksum(self.x[seg] for seg in self.valid_segments 
                                if seg[0] == gpu_type and seg[2] == tp_degree and seg[3] == partition_id),
                        name=f"partition_usage_{gpu_type}_tp{tp_degree}_{partition_id}"
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
        # Cost definition: sum(instance_usage × instance_cost_per_hour)
        cost_expr = gp.quicksum(
            self.y[gpu_type] * self.gpu_types[gpu_type].cost_per_hour
            for gpu_type in self.gpu_types.keys()
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
                if (gpu_type, tp_degree, partition_id) in existing_partition_keys:
                    # Use pre-computed throughput (OPTIMIZATION)
                    # This avoids recomputing throughput for every segment in every model build
                    throughput_expr = gp.quicksum(
                        self.segment_throughputs[seg] * self.x[seg]
                        for seg in self.valid_segments 
                        if seg[0] == gpu_type and seg[2] == tp_degree and seg[3] == partition_id  # seg[3]=partition_id
                    )
                    self.model.addConstr(
                        self.tau[gpu_type, tp_degree, partition_id] == throughput_expr,
                        name=f"partition_throughput_{gpu_type}_tp{tp_degree}_{partition_id}"
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
        existing_partition_keys = set((seg[0], seg[2], seg[3]) for seg in self.valid_segments)
        
        # STEP 1: Find the maximum possible throughput across ALL GPU types
        max_throughput_global = 0
        throughput_by_type = {}
        
        min_pipeline_segment_size = math.ceil(
            self.config.num_decoder_layers / max(1, self.config.max_pipeline_stages)
        )
        for gpu_type, allocations in self.tp_allocations.items():
            tp_degree = self.tp_max_configuration[gpu_type]
            max_size = self.max_segment_size[(gpu_type, tp_degree)]
            
            if max_size >= min_pipeline_segment_size:
                # Use max_batch_size for upper bound calculation
                nvlink_bw = self._get_representative_tp_bandwidth(gpu_type, tp_degree)
                _gpu_mem = self.gpu_types[gpu_type].memory_gb
                _max_model_len = self._compute_max_model_len(_gpu_mem, max_size, tp_degree)
                _mnbt = self.config.max_num_batched_tokens or _max_model_len
                _S = self.config.sequence_length
                _prefill_batch = max(1, min(_mnbt // _S if _S > 0 else 1, self.config.max_batch_size))
                max_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                    self._get_gpu_model_name(gpu_type), self.config.sequence_length,
                    self.config.max_batch_size, max_size, self.config.d_model,
                    self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                    debug=False, phase=self.config.workload_phase,
                    output_length=self.config.output_length,
                    num_attention_heads=self.config.num_attention_heads,
                    num_kv_heads=self.config.num_kv_heads,
                    head_dim=self.config.head_dim,
                    prefill_batch=_prefill_batch,
                    vocab_size=self.config.vocab_size,
                    total_num_layers=self.config.num_decoder_layers
                )
                throughput_by_type[gpu_type] = max_throughput
                max_throughput_global = max(max_throughput_global, max_throughput)
        
        # STEP 2: Use the GLOBAL maximum for ALL partitions (with safety factor)
        M_global = max_throughput_global * 3
        
        for gpu_type, allocations in self.tp_allocations.items():
            for gpu_set, tp_degree_alloc, partition_id in allocations:
                if (gpu_type, tp_degree_alloc, partition_id) in existing_partition_keys:
                    M_partition[(gpu_type, tp_degree_alloc, partition_id)] = M_global
        
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
        existing_partition_keys = set((seg[0], seg[2], seg[3]) for seg in self.valid_segments)
        
        # Group partitions by GPU type AND TP degree (not just GPU type)
        for gpu_type, allocations in self.tp_allocations.items():
            # Group by TP degree
            from collections import defaultdict
            by_tp = defaultdict(list)
            for gpu_set, tp_degree, partition_id in allocations:
                if (gpu_type, tp_degree, partition_id) in existing_partition_keys:
                    by_tp[tp_degree].append(partition_id)
            
            # Apply symmetry breaking ONLY within same TP degree
            for tp_degree, partition_ids in by_tp.items():
                sorted_ids = sorted(partition_ids)
                if len(sorted_ids) > 1:
                    for i in range(len(sorted_ids) - 1):
                        self.model.addConstr(
                            self.z[gpu_type, tp_degree, sorted_ids[i]] >= self.z[gpu_type, tp_degree, sorted_ids[i+1]],
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

        Objective = w*(t/T_norm) - (1-w)*(cost/C_norm) - penalty*cross_instance_edges
        where w = cost_throughput_weight

        Special cases:
        - w=0: Pure throughput maximization (backward compatible)
        - w=1: Pure cost minimization
        - 0<w<1: Weighted trade-off

        NEW: Cross-instance penalty encourages intra-node PP placement
        """
        w = self.config.cost_throughput_weight

        # NEW: Compute cross-instance penalty term
        # Penalize edges that cross instance boundaries (seg1[0] != seg2[0])
        cross_instance_edges = [
            conn for conn in self.valid_connections
            if conn[0][0] != conn[1][0]  # Different instance (gpu_type includes instance ID)
        ]
        cross_instance_penalty_term = gp.quicksum(
            self.e[conn] for conn in cross_instance_edges
        ) * self.config.cross_instance_penalty

        num_cross_instance = len(cross_instance_edges)
        num_same_instance = len(self.valid_connections) - num_cross_instance
        logger.info(f"Cross-instance penalty: {self.config.cross_instance_penalty} per edge")
        logger.info(f"  - Same-instance connections: {num_same_instance}")
        logger.info(f"  - Cross-instance connections: {num_cross_instance}")

        if w == 0.0:
            # Pure throughput maximization (original objective)
            obj_expr = self.t - cross_instance_penalty_term
            logger.info("Objective: Maximize throughput (cost ignored)")

        elif w == 1.0:
            # Pure cost minimization (minimize = maximize negative)
            obj_expr = -self.cost - cross_instance_penalty_term
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

            obj_expr = t_norm - lambda_param * c_norm - cross_instance_penalty_term

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
        
        logger.info('='*80)
        logger.info(f"ITERATIVE $/M TOKENS OPTIMIZATION")
        logger.info(f"{'='*80}")
        target_cost_per_million_token = target_cost_per_token * 1_000_000
        logger.info(f"Target $/M tokens: ${target_cost_per_million_token:.6f}")
        
        # Step 1: Estimate minimum throughput (single GPU, single layer, min batch)
        min_tp_estimates = []
        for gpu_type in self.gpu_types:
            # Use min_batch_size for lower bound estimation
            t_est = ThroughputFunctions.gpu_throughput(
                self._get_gpu_model_name(gpu_type), self.config.sequence_length, self.config.min_batch_size,
                1, self.config.d_model, self.config.bytes_per_element, self.config.d_hidden,
                num_attention_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_kv_heads
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
            logger.info(f"Iteration {iteration + 1}/{max_iterations}:")
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
            logger.info(f"Best $/M tokens found: ${best_cpm:.6f}")
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
                config_cost = gpu_info.cost_per_hour
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
        logger.info(f"Configuration Coverage Analysis:")
        covered = []
        not_covered = []
        for gpu_type, gpu_info in self.gpu_types.items():
            for tp_degree in [1, 2, 4, 8]:
                if tp_degree > gpu_info.count:
                    continue
                config_cost = gpu_info.cost_per_hour
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
        logger.info(f'='*80)
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
                logger.info("Phase 1: Iterative search to find ballpark...")
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
                    logger.info("Phase 2: Coarse enumeration around ballpark...")
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
            logger.info(f"[{i+1}/{len(budget_points)}] Budget: ${budget:.2f}/hour")
            
            # Log which TP configurations are feasible at this budget
            feasible_configs = []
            for gpu_type, gpu_info in self.gpu_types.items():
                for tp_degree in [1, 2, 4, 8]:
                    if tp_degree > gpu_info.count:
                        continue
                    config_cost = gpu_info.cost_per_hour
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
        logger.info(f"Total enumeration time: {total_enumeration_time:.1f}s (model build: {model_build_time:.1f}s, solving: {total_enumeration_time - model_build_time:.1f}s)")
        
        # Log coverage summary
        logger.info(f'='*80)
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
                config_cost = gpu_info.cost_per_hour
                config_name = f"{gpu_type} TP={tp_degree}"
                all_configs.append((config_name, config_cost))
                if config_cost <= max_budget_tested:
                    explored.append((config_name, config_cost))
                else:
                    not_explored.append((config_name, config_cost))
        
        logger.info(f"✓ Explored: {len(explored)}/{len(all_configs)} configurations")
        for name, cost in explored:
            logger.info(f"  {name:<20} ${cost:.2f}/h")
        
        if not_explored:
            logger.warning(f"✗ NOT explored: {len(not_explored)} configurations (budget too low)")
            for name, cost in not_explored:
                logger.warning(f"  {name:<20} ${cost:.2f}/h (need >= ${cost:.2f}/h budget)")
        
        if best_solution:
            self.solution = best_solution
            self.all_enumeration_results = all_results  # Store for CSV export
            best_cpm = best_cpt * 1_000_000
            logger.info(f'='*80)
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
                logger.info(f"MEETS TARGET: ${best_cpm:.6f} <= ${target_per_million:.6f}")
                logger.info(f"   {improvement:.1f}% better than target!")
            else:
                shortfall = (best_cpt - target) / target * 100
                logger.info(f"MISSES TARGET: ${best_cpm:.6f} > ${target_per_million:.6f}")
                logger.info(f"   {shortfall:.1f}% worse than target (infeasible to meet)")
            
            # Log Pareto frontier
            logger.info(f"Pareto Frontier (all solutions):")
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

    def set_warm_start_from_homogeneous(self, homogeneous_solution: dict) -> bool:
        """
        Set MIP warm start values from a homogeneous solution.

        This provides Gurobi with a known feasible solution to start from,
        which can significantly accelerate the branch-and-bound search and
        help prune inferior branches early.

        Args:
            homogeneous_solution: Solution dict from solve_homogeneous() containing
                                  gpu_assignments, batch_size, cost_per_token, etc.

        Returns:
            bool: True if warm start was successfully set, False otherwise
        """
        if not hasattr(self, 'model') or self.model is None:
            logger.error("Cannot set warm start: model not built yet")
            return False

        if not homogeneous_solution:
            logger.warning("Empty homogeneous solution, skipping warm start")
            return False

        logger.info("="*80)
        logger.info("SETTING MIP WARM START FROM HOMOGENEOUS SOLUTION")
        logger.info("="*80)

        gpu_assignments = homogeneous_solution.get('gpu_assignments', [])
        batch_size = homogeneous_solution.get('batch_size')
        cost_per_token = homogeneous_solution.get('cost_per_token', float('inf'))
        cost_per_hour = homogeneous_solution.get('cost_per_hour', 0)
        throughput = homogeneous_solution.get('throughput_tokens_per_sec', 0)

        if not gpu_assignments or batch_size is None:
            logger.warning("Homogeneous solution missing gpu_assignments or batch_size")
            return False

        logger.info(f"Homogeneous solution: {len(gpu_assignments)} stages, "
                   f"batch_size={batch_size}, $/M tokens=${cost_per_token*1e6:.6f}")

        # Initialize all variables to 0
        for seg in self.x:
            self.x[seg].Start = 0.0
        for key in self.z:
            self.z[key].Start = 0.0
        for gpu_type in self.y:
            self.y[gpu_type].Start = 0.0
        for conn in self.e:
            self.e[conn].Start = 0.0
        for bs in self._get_batch_size_options():
            self.b[bs].Start = 0.0

        # Set batch size
        if batch_size in self.b:
            self.b[batch_size].Start = 1.0
            logger.info(f"  Set batch_size={batch_size}")

        # Match homogeneous assignments to MILP segments
        # Segment tuple: (gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, batch_size)
        matched_segments = []
        used_instances = set()
        used_partitions = set()

        for assignment in gpu_assignments:
            gpu_type = assignment['gpu_type']
            partition_id = assignment.get('partition_id', 0)
            tp_degree = assignment['tp_degree']
            start_layer = assignment['start_layer']
            segment_size = assignment['segment_size']
            gpu_ids = tuple(sorted(assignment.get('gpu_ids', [])))

            # Find matching segment in valid_segments
            found = False
            for seg in self.valid_segments:
                seg_gpu_type, seg_gpu_set, seg_tp, seg_part, seg_start, seg_size, seg_bs = seg

                # Match all components
                if (seg_gpu_type == gpu_type and
                    seg_tp == tp_degree and
                    seg_part == partition_id and
                    seg_start == start_layer and
                    seg_size == segment_size and
                    seg_bs == batch_size and
                    tuple(sorted(seg_gpu_set)) == gpu_ids):

                    # Set this segment as active
                    self.x[seg].Start = 1.0
                    matched_segments.append(seg)
                    used_instances.add(gpu_type)
                    used_partitions.add((gpu_type, tp_degree, partition_id))
                    found = True
                    logger.info(f"  Matched segment: {gpu_type} partition={partition_id} "
                               f"layers={start_layer}-{start_layer+segment_size-1} tp={tp_degree}")
                    break

            if not found:
                logger.warning(f"  Could not match assignment: {gpu_type} partition={partition_id} "
                             f"layers={start_layer}-{start_layer+segment_size-1} tp={tp_degree}")

        # Set partition usage (z variables)
        for gpu_type, tp_degree, partition_id in used_partitions:
            if (gpu_type, tp_degree, partition_id) in self.z:
                self.z[gpu_type, tp_degree, partition_id].Start = 1.0

        # Set instance usage (y variables)
        for gpu_type in used_instances:
            if gpu_type in self.y:
                self.y[gpu_type].Start = 1.0

        # Set network connections (e variables)
        # Find connections between consecutive segments
        for i in range(len(matched_segments) - 1):
            seg1 = matched_segments[i]
            seg2 = matched_segments[i + 1]
            if (seg1, seg2) in self.e:
                self.e[seg1, seg2].Start = 1.0
                logger.info(f"  Set connection: stage {i} -> stage {i+1}")

        # Set throughput and cost start values
        if throughput > 0:
            self.t.Start = throughput
        if cost_per_hour > 0:
            self.cost.Start = cost_per_hour

        # Set cutoff to prune branches worse than homogeneous solution
        # The objective is cost_per_token (minimization), so set upper bound
        # We add a small margin to ensure the homogeneous solution is accepted
        if cost_per_token < float('inf') and cost_per_token > 0:
            cutoff_margin = 1.001  # Allow 0.1% margin
            cutoff_value = cost_per_token * cutoff_margin
            self.model.setParam('Cutoff', cutoff_value)
            logger.info(f"  Set Cutoff={cutoff_value:.10f} (homogeneous={cost_per_token:.10f})")

        logger.info(f"Warm start set: {len(matched_segments)}/{len(gpu_assignments)} segments matched")
        logger.info("="*80)

        return len(matched_segments) == len(gpu_assignments)

    def _extract_solution(self):
        """Extract solution from solved model"""
        # Compute metrics
        cost_per_hour = self.cost.x

        # Count number of pipeline stages in solution
        num_stages = sum(1 for key in self.z.keys() if self.z[key].x > 0.5)

        # Raw throughput = min of active stage throughputs (bottleneck stage)
        active_stage_tps = [self.tau[key].x for key in self.z if self.z[key].x > 0.5]
        raw_throughput = min(active_stage_tps) if active_stage_tps else 0.0

        # Get optimal batch size from solution
        optimal_batch_size = None
        for bs in self._get_batch_size_options():
            if self.b[bs].x > 0.5:
                optimal_batch_size = bs
                break

        # Guard against missing batch size (should not happen in valid solution)
        if optimal_batch_size is None:
            logger.error("No valid batch size found in solution - using min_batch_size as fallback")
            optimal_batch_size = self.config.min_batch_size

        # Pipeline efficiency using 1F1B wave model (consistent with homogeneous solver)
        active_segments = []
        if num_stages <= 1:
            pipeline_efficiency = 1.0
        else:
            # Collect active segments sorted by start_layer
            active_segments = sorted(
                [seg for seg in self.valid_segments if self.x[seg].x > 0.5],
                key=lambda s: s[4]  # sort by start_layer
            )
            inter_stage_bw = self._get_inter_stage_bw_for_placement(active_segments)

            if self.config.workload_phase == 'aggregated':
                # Compute per-stage prefill/decode throughputs
                prefill_tps = []
                decode_tps = []
                prefill_batches = []
                for seg in active_segments:
                    gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size, seg_batch = seg
                    nvlink_bw = self._get_min_intra_tp_bandwidth(gpu_type, gpu_set)
                    gpu_model = self._get_gpu_model_name(gpu_type)

                    _gpu_mem = self.gpu_types[gpu_type].memory_gb
                    _max_model_len = self._compute_max_model_len(_gpu_mem, segment_size, tp_degree)
                    _mnbt = self.config.max_num_batched_tokens or _max_model_len
                    _S = self.config.sequence_length
                    _prefill_batch = max(1, min(_mnbt // _S if _S > 0 else 1, optimal_batch_size))

                    pf_tp = ThroughputFunctions.gpu_throughput_with_tp(
                        gpu_model, self.config.sequence_length,
                        _prefill_batch, segment_size, self.config.d_model,
                        self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                        debug=False, phase='prefill',
                        num_attention_heads=self.config.num_attention_heads,
                        num_kv_heads=self.config.num_kv_heads,
                        head_dim=self.config.head_dim,
                        vocab_size=self.config.vocab_size,
                        total_num_layers=self.config.num_decoder_layers
                    )
                    dc_tp = ThroughputFunctions.gpu_throughput_with_tp(
                        gpu_model, self.config.sequence_length,
                        optimal_batch_size, segment_size, self.config.d_model,
                        self.config.bytes_per_element, tp_degree, self.config.d_hidden, nvlink_bw,
                        debug=False, phase='decode',
                        output_length=self.config.output_length,
                        num_attention_heads=self.config.num_attention_heads,
                        num_kv_heads=self.config.num_kv_heads,
                        head_dim=self.config.head_dim,
                        vocab_size=self.config.vocab_size,
                        total_num_layers=self.config.num_decoder_layers
                    )
                    prefill_tps.append(pf_tp)
                    decode_tps.append(dc_tp)
                    prefill_batches.append(_prefill_batch)

                pipeline_efficiency = self._compute_pipeline_efficiency_1f1b(
                    num_stages, inter_stage_bw,
                    prefill_tp=min(prefill_tps),
                    decode_tp=min(decode_tps),
                    effective_decode_batch=optimal_batch_size,
                    prefill_batch=min(prefill_batches)
                )
            else:
                batch_for_phase = 1 if self.config.workload_phase == 'prefill' else optimal_batch_size
                pipeline_efficiency = self._compute_pipeline_efficiency_1f1b(
                    num_stages, inter_stage_bw,
                    stage_throughput=raw_throughput,
                    batch_for_phase=batch_for_phase
                )

        # Apply real-world efficiency factor with GPU generation penalty
        real_world_efficiency = self.config.real_world_efficiency
        # Get generation factor from active segments' GPU model
        _diag_gen_factor = 1.0
        if active_segments:
            _seg_gpu_types = [seg[0].split('#')[0] for seg in active_segments]
            for _sgt in _seg_gpu_types:
                _sgt_model = self.gpu_types[_sgt].gpu_model if _sgt in self.gpu_types else ''
                _gf = ThroughputFunctions.GPU_SPECS.get(_sgt_model, {}).get('generation_factor', 1.0)
                _diag_gen_factor = min(_diag_gen_factor, _gf)
        real_world_efficiency *= _diag_gen_factor

        throughput_per_sec = raw_throughput * pipeline_efficiency * real_world_efficiency

        logger.info(f"Throughput Corrections:")
        logger.info(f"  Batch size: {optimal_batch_size}")
        logger.info(f"  Pipeline stages: {num_stages}")
        logger.info(f"  Pipeline efficiency (1F1B): {pipeline_efficiency:.1%}")
        logger.info(f"  Real-world efficiency: {real_world_efficiency:.1%} (gen_factor={_diag_gen_factor:.2f})")
        logger.info(f"  Combined efficiency: {pipeline_efficiency * real_world_efficiency:.1%}")
        logger.info(f"  Raw throughput: {raw_throughput:.2f} tokens/sec")
        logger.info(f"  After pipeline efficiency: {raw_throughput * pipeline_efficiency:.2f} tokens/sec")
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
        logger.info("="*80)
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
        
        logger.info(f"Core Metrics:")
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
                    'throughput': self.tau[gpu_type, tp_degree, partition_id].x
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
        logger.info("="*80)
        logger.info("BATCH SIZE ANALYSIS")
        logger.info("="*80)
        
        batch_size_options = self._get_batch_size_options()
        optimal_batch = self.solution.get('batch_size')
        
        logger.info(f"Batch Size Selection:")
        logger.info(f"  Optimal: {optimal_batch}")
        logger.info(f"  Search range: [{self.config.min_batch_size}, {self.config.max_batch_size}]")
        logger.info(f"  Options considered: {batch_size_options}")
        
        # Show batch efficiency factor
        if optimal_batch:
            efficiency = ThroughputFunctions.batch_efficiency_factor(optimal_batch)
            logger.info(f"Batch Efficiency Factor: {efficiency:.1%}")
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
            logger.info(f"Batch Size Impact (estimated for current GPU configuration):")
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
        logger.info("="*80)
        logger.info("ROOFLINE MODEL ANALYSIS")
        logger.info("="*80)
        
        optimal_batch = self.solution.get('batch_size', self.config.min_batch_size)
        
        logger.info(f"Performance Bottleneck Analysis (Roofline Model):")
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
                phase=self.config.workload_phase,
                num_attention_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_kv_heads,
                vocab_size=self.config.vocab_size,
                total_num_layers=self.config.num_decoder_layers
            )

            # Get ridge point
            ridge_point = ThroughputFunctions.get_ridge_point(self._get_gpu_model_name(gpu_type))
            
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
        logger.info("-" * 80)
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
                phase=self.config.workload_phase,
                num_attention_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_kv_heads,
                vocab_size=self.config.vocab_size,
                total_num_layers=self.config.num_decoder_layers
            )
            ridge_point = ThroughputFunctions.get_ridge_point(self._get_gpu_model_name(gpu_type))
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
            logger.info(f"Most segments are memory-bound.")
            logger.info(f"- Consider GPUs with higher memory bandwidth for better performance.")
            logger.info(f"- Increasing TP degree can help (reduces memory per GPU).")
        elif compute_bound_count > memory_bound_count:
            logger.info(f"Most segments are compute-bound.")
            logger.info(f"- Consider GPUs with higher TFLOPS for better performance.")
            logger.info(f"- Memory bandwidth is not the bottleneck here.")
    
    def _log_solution_analysis(self):
        """Comprehensive analysis of why the solution looks the way it does"""
        logger.info("="*80)
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
                phase=self.config.workload_phase,
                num_attention_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_kv_heads,
                vocab_size=self.config.vocab_size,
                total_num_layers=self.config.num_decoder_layers
            )
            ridge_point = ThroughputFunctions.get_ridge_point(self._get_gpu_model_name(gpu_type))
            regime = ThroughputFunctions.determine_regime(ai, ridge_point)
            
            if regime == "COMPUTE_BOUND":
                compute_bound_count += 1
            else:
                memory_bound_count += 1
        
        is_memory_bound = memory_bound_count > compute_bound_count
        
        # 1. Compute instance efficiency ratios (both FLOP/$ and Bandwidth/$)
        # Note: specs are per-GPU, cost is per-instance, so we multiply by GPU count
        gpu_efficiency = {}
        for gpu_type, gpu_obj in self.gpu_types.items():
            specs = ThroughputFunctions.GPU_SPECS.get(self._get_gpu_model_name(gpu_type))
            if specs:
                # Per-GPU specs
                effective_flops_per_gpu = specs['tflops'] * specs['compute_efficiency']
                effective_bw_per_gpu = specs['mem_bw'] * specs['memory_efficiency']
                # Total instance capacity (all GPUs in instance)
                total_flops = effective_flops_per_gpu * gpu_obj.count
                total_bw = effective_bw_per_gpu * gpu_obj.count
                # Efficiency = total capacity / instance cost
                compute_efficiency = total_flops / gpu_obj.cost_per_hour
                memory_efficiency = total_bw / gpu_obj.cost_per_hour
                gpu_efficiency[gpu_type] = {
                    'flops': effective_flops_per_gpu,
                    'bandwidth': effective_bw_per_gpu,
                    'total_flops': total_flops,
                    'total_bw': total_bw,
                    'cost': gpu_obj.cost_per_hour,
                    'compute_ratio': compute_efficiency,
                    'memory_ratio': memory_efficiency
                }
        
        # Sort by the relevant metric
        if is_memory_bound:
            sorted_efficiency = sorted(gpu_efficiency.items(), key=lambda x: x[1]['memory_ratio'], reverse=True)
            metric_name = "Memory Bandwidth/$ ratio"
            logger.info(f"WARNING WORKLOAD IS MEMORY-BOUND -> Memory Bandwidth/$ is the key metric!")
        else:
            sorted_efficiency = sorted(gpu_efficiency.items(), key=lambda x: x[1]['compute_ratio'], reverse=True)
            metric_name = "Compute (TFLOP/$) ratio"
            logger.info(f"OK WORKLOAD IS COMPUTE-BOUND -> TFLOP/$ is the key metric!")
        
        logger.info(f"GPU Efficiency Ranking ({metric_name}):")
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
                    'total_cost': self.gpu_types[gpu_type].cost_per_hour,  # Instance cost (paid once per instance)
                    'total_layers': 0,
                    'tp_degrees': []
                }
            gpu_usage[gpu_type]['segments'] += 1
            gpu_usage[gpu_type]['total_gpus'] += assignment['tp_degree']
            # Note: total_cost is set once per instance, not accumulated per segment
            gpu_usage[gpu_type]['total_layers'] += assignment['segment_size']
            gpu_usage[gpu_type]['tp_degrees'].append(assignment['tp_degree'])
        
        logger.info("Actual GPU Usage in Solution:")
        logger.info("-" * 80)
        total_cost = 0
        for gpu_type in sorted(gpu_usage.keys()):
            data = gpu_usage[gpu_type]
            efficiency_rank = next(i for i, (gt, _) in enumerate(sorted_efficiency, 1) if gt == gpu_type)
            logger.info(f"  {gpu_type:<10} {data['segments']} segments, {data['total_gpus']} GPUs, "
                       f"{data['total_layers']} layers, ${data['total_cost']:.2f}/h "
                       f"(Efficiency rank: #{efficiency_rank})")
            total_cost += data['total_cost']
        
        logger.info(f"  TOTAL COST: ${total_cost:.2f}/hour")
        
        # 3. Analyze unused GPUs
        unused_gpus = set(self.gpu_types.keys()) - set(gpu_usage.keys())
        if unused_gpus:
            logger.info("Unused GPUs (and why they might not be chosen):")
            logger.info("-" * 80)
            min_pipeline_segment_size = math.ceil(
                self.config.num_decoder_layers / max(1, self.config.max_pipeline_stages)
            )
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
                if max_seg < min_pipeline_segment_size:
                    logger.info(
                        f"             Excluded by min pipeline segment size: "
                        f"{min_pipeline_segment_size} layers"
                    )
                    continue
                
                # Hypothetical: what if we used this GPU?
                hypothetical_tp = min(4, max_tp)
                hyp_max_seg = self._compute_max_segment_size_for_tp(gpu_type, hypothetical_tp, optimal_batch)
                if hyp_max_seg >= 5:  # Can fit at least 5 layers
                    # Use max_batch_size for optimistic throughput estimate
                    nvlink_bw = self._get_representative_tp_bandwidth(gpu_type, hypothetical_tp)
                    _gpu_mem = self.gpu_types[gpu_type].memory_gb
                    _max_model_len = self._compute_max_model_len(_gpu_mem, 5, hypothetical_tp)
                    _mnbt = self.config.max_num_batched_tokens or _max_model_len
                    _S = self.config.sequence_length
                    _prefill_batch = max(1, min(_mnbt // _S if _S > 0 else 1, self.config.max_batch_size))
                    hyp_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                        self._get_gpu_model_name(gpu_type), self.config.sequence_length, self.config.max_batch_size,
                        5, self.config.d_model, self.config.bytes_per_element, hypothetical_tp, self.config.d_hidden, nvlink_bw,
                        debug=False, phase=self.config.workload_phase,
                        output_length=self.config.output_length,
                        num_attention_heads=self.config.num_attention_heads,
                        num_kv_heads=self.config.num_kv_heads,
                        head_dim=self.config.head_dim,
                        prefill_batch=_prefill_batch,
                        vocab_size=self.config.vocab_size,
                        total_num_layers=self.config.num_decoder_layers
                    )
                    hyp_cost = gpu_obj.cost_per_hour  # Instance cost (same regardless of TP)
                    logger.info(
                        f"             Hypothetical (5 layers, TP={hypothetical_tp}): "
                        f"{hyp_throughput:.0f} tokens/s, ${hyp_cost:.2f}/h"
                    )
        
        # 4. Segment-level contribution analysis
        logger.info("="*80)
        logger.info("SEGMENT-LEVEL ANALYSIS")
        logger.info("="*80)
        
        # Find bottleneck
        min_throughput = min(a['throughput'] for a in self.solution['gpu_assignments'])
        bottleneck_segments = [a for a in self.solution['gpu_assignments'] if abs(a['throughput'] - min_throughput) < 0.01]
        
        logger.info(f"Bottleneck Throughput: {min_throughput:.2f} tokens/sec")
        logger.info(f"Bottleneck Segments:")
        for seg in bottleneck_segments:
            seg_cost = self.gpu_types[seg['gpu_type']].cost_per_hour  # Instance cost (same regardless of TP)
            logger.info(f"  {seg['gpu_type']} TP={seg['tp_degree']}, layers {seg['start_layer']}-{seg['end_layer']}, "
                       f"${seg_cost:.2f}/h, {seg['throughput']:.0f} tokens/s")
        
        # 5. Alternative scenario: Check single-segment solution with best efficiency GPU
        best_gpu_type = sorted_efficiency[0][0]
        logger.info("="*80)
        logger.info(f"ALTERNATIVE SCENARIO: Single segment with {best_gpu_type} (best efficiency)")
        logger.info("="*80)
        
        best_gpu = self.gpu_types[best_gpu_type]
        max_tp = self.tp_max_configuration[best_gpu_type]
        optimal_batch = self.solution.get('batch_size', self.config.max_batch_size)
        max_seg_size = self._compute_max_segment_size_for_tp(best_gpu_type, max_tp, optimal_batch)
        
        logger.info(f"{best_gpu_type} specs:")
        logger.info(f"  Available: {best_gpu.count} GPUs")
        logger.info(f"  Cost: ${best_gpu.cost_per_hour:.2f}/hour (instance)")
        logger.info(f"  Max segment size (TP={max_tp}): {max_seg_size} layers")
        logger.info(f"  Model needs: {self.config.num_decoder_layers} layers")
        
        if max_seg_size >= self.config.num_decoder_layers:
            # Can fit entire model in one segment!
            # Use max_batch_size for optimal throughput estimate
            nvlink_bw = self._get_representative_tp_bandwidth(best_gpu_type, max_tp)
            _gpu_mem = self.gpu_types[best_gpu_type].memory_gb
            _max_model_len = self._compute_max_model_len(_gpu_mem, self.config.num_decoder_layers, max_tp)
            _mnbt = self.config.max_num_batched_tokens or _max_model_len
            _S = self.config.sequence_length
            _prefill_batch = max(1, min(_mnbt // _S if _S > 0 else 1, self.config.max_batch_size))
            alt_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                self._get_gpu_model_name(best_gpu_type), self.config.sequence_length, self.config.max_batch_size,
                self.config.num_decoder_layers, self.config.d_model,
                self.config.bytes_per_element, max_tp, self.config.d_hidden, nvlink_bw,
                debug=False, phase=self.config.workload_phase,
                output_length=self.config.output_length,
                num_attention_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                prefill_batch=_prefill_batch,
                vocab_size=self.config.vocab_size,
                total_num_layers=self.config.num_decoder_layers
            )
            alt_cost = best_gpu.cost_per_hour  # Instance cost (same regardless of TP)
            alt_cost_per_token = alt_cost / (alt_throughput * 3600)
            alt_cost_per_million = alt_cost_per_token * 1_000_000

            logger.info(f"CAN FIT Single segment with TP={max_tp}:")
            logger.info(f"  Throughput: {alt_throughput:.0f} tokens/sec")
            logger.info(f"  Cost: ${alt_cost:.2f}/hour")
            logger.info(f"  $/M tokens: ${alt_cost_per_million:.6f}")
            
            # Compare with actual solution
            actual_cpt = self.solution['cost_per_token']
            actual_cpm = actual_cpt * 1_000_000
            logger.info(f"Comparison with current multi-segment solution:")
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
            logger.info(f"CANNOT FIT: Model needs {self.config.num_decoder_layers} layers but max is {max_seg_size}")
        
        # 6. Objective function check
        logger.info("="*80)
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
                    _gpu_mem = self.gpu_types[best_gpu_type].memory_gb
                    _max_model_len = self._compute_max_model_len(_gpu_mem, self.config.num_decoder_layers, max_tp)
                    _mnbt = self.config.max_num_batched_tokens or _max_model_len
                    _S = self.config.sequence_length
                    _prefill_batch = max(1, min(_mnbt // _S if _S > 0 else 1, self.config.max_batch_size))
                    t_alt = ThroughputFunctions.gpu_throughput_with_tp(
                        self._get_gpu_model_name(best_gpu_type), self.config.sequence_length, self.config.max_batch_size,
                        self.config.num_decoder_layers, self.config.d_model,
                        self.config.bytes_per_element, max_tp, self.config.d_hidden, nvlink_bw,
                        debug=False, phase=self.config.workload_phase,
                        output_length=self.config.output_length,
                        num_attention_heads=self.config.num_attention_heads,
                        num_kv_heads=self.config.num_kv_heads,
                        head_dim=self.config.head_dim,
                        prefill_batch=_prefill_batch,
                        vocab_size=self.config.vocab_size,
                        total_num_layers=self.config.num_decoder_layers
                    )
                    c_alt = best_gpu.cost_per_hour  # Instance cost (same regardless of TP)
                    obj_alt = (t_alt / self.config.throughput_normalization) - \
                             (w/(1-w)) * (c_alt / self.config.cost_normalization)
                    
                    logger.info(f"Alternative ({best_gpu_type} single segment) objective: {obj_alt:.6f}")
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
        
        logger.info("="*100)
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
            segment_cost = self.gpu_types[gpu_type].cost_per_hour
            
            logger.info(f"{gpu_type:<12} {tp_degree:<4} "
                  f"{gpu_ids_str:<20} {layers_str:<15} "
                  f"{assignment['segment_size']:<6} {assignment['throughput']:<12.2f} "
                  f"${segment_cost:<9.2f}")
        
        if self.solution['network_connections']:
            logger.info("NETWORK CONNECTIONS:")
            logger.info("-" * 80)
            for i, conn in enumerate(self.solution['network_connections']):
                from_seg = conn['from_segment']
                to_seg = conn['to_segment']
                logger.info(f"Connection {i+1}: {from_seg['gpu_type']} partition {from_seg['partition_id']} "
                      f"(layers {from_seg['start_layer']}-{from_seg['end_layer']}) -> "
                      f"{to_seg['gpu_type']} partition {to_seg['partition_id']} "
                      f"(layers {to_seg['start_layer']}-{to_seg['end_layer']}) "
                      f"[Throughput: {conn['throughput']:.2f}]")
        
        logger.info("="*100)
    
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
                'output_length': self.config.output_length,
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
            json.dump(_json_safe(output_data), f, indent=2)
        
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
        
        def format_placement(gpu_assignments):
            """
            Format placement as: {PP_1:{GPU_type:TP_degree}, PP_2:{GPU_type:TP_degree}, ...}
            Example: {PP_1:{A100:2}, PP_2:{L40:4}, PP_3:{L40:4}}
            """
            if not gpu_assignments:
                return '{}'
            
            placement_parts = []
            for i, assignment in enumerate(gpu_assignments, 1):
                gpu_type = assignment['gpu_type']
                tp_degree = assignment['tp_degree']
                placement_parts.append(f"PP_{i}:{{{gpu_type}:{tp_degree}}}")
            
            return '{' + ', '.join(placement_parts) + '}'

        def format_layer_mapping(gpu_assignments):
            """Format layer ranges per PP stage: {PP_1:0-15, PP_2:16-31}"""
            if not gpu_assignments:
                return '{}'
            parts = []
            for i, assignment in enumerate(gpu_assignments, 1):
                parts.append(f"PP_{i}:{assignment['start_layer']}-{assignment['end_layer']}")
            return '{' + ', '.join(parts) + '}'

        def format_tp_per_stage(gpu_assignments):
            """Format TP degree per PP stage: {PP_1:2, PP_2:4}"""
            if not gpu_assignments:
                return '{}'
            parts = []
            for i, assignment in enumerate(gpu_assignments, 1):
                parts.append(f"PP_{i}:{assignment['tp_degree']}")
            return '{' + ', '.join(parts) + '}'

        def format_stage_mem_gb(gpu_assignments):
            """Format per-stage GPU memory: {PP_1:48, PP_2:80}"""
            if not gpu_assignments:
                return '{}'
            parts = []
            for i, assignment in enumerate(gpu_assignments, 1):
                gpu_type = assignment['gpu_type']
                mem_gb = self.gpu_types[gpu_type].memory_gb
                parts.append(f"PP_{i}:{mem_gb}")
            return '{' + ', '.join(parts) + '}'

        def get_device_types(gpu_assignments):
            if not gpu_assignments:
                return ''
            unique_types = sorted({a['gpu_type'] for a in gpu_assignments})
            return ','.join(unique_types)

        def get_phase_throughputs(total_throughput):
            if self.config.workload_phase == 'prefill':
                return total_throughput, 0.0
            if self.config.workload_phase == 'decode':
                return 0.0, total_throughput
            if self.config.workload_phase == 'aggregated':
                # Report prefill and decode components proportionally
                prefill_tokens = self.config.sequence_length
                decode_tokens = self.config.output_length
                total_tokens = prefill_tokens + decode_tokens
                if total_tokens > 0:
                    # Proportional allocation based on token count
                    prefill_tps = total_throughput * prefill_tokens / total_tokens
                    decode_tps = total_throughput * decode_tokens / total_tokens
                    return prefill_tps, decode_tps
            return 0.0, 0.0
        
        # Check if we have enumeration results
        has_enumeration = hasattr(self, 'all_enumeration_results') and self.all_enumeration_results
        
        if include_all_results and has_enumeration:
            # Save all enumeration results
            rows = []
            for result in sorted(self.all_enumeration_results, key=lambda x: x['cost_per_token']):
                sol = result['solution']
                
                # Extract aggregated GPU info
                num_gpus = ''
                total_layers = ''
                placement = ''
                layer_mapping = ''
                tp_per_stage = ''
                stage_mem_gb = ''
                device_types = ''
                
                if sol['gpu_assignments']:
                    num_gpus = sum(len(a['gpu_ids']) for a in sol['gpu_assignments'])
                    total_layers = sum(a['segment_size'] for a in sol['gpu_assignments'])
                    placement = format_placement(sol['gpu_assignments'])
                    layer_mapping = format_layer_mapping(sol['gpu_assignments'])
                    tp_per_stage = format_tp_per_stage(sol['gpu_assignments'])
                    stage_mem_gb = format_stage_mem_gb(sol['gpu_assignments'])
                    device_types = get_device_types(sol['gpu_assignments'])
                
                cost_per_million = result['cost_per_token'] * 1_000_000
                is_best = (result['cost_per_token'] == min(r['cost_per_token'] for r in self.all_enumeration_results))
                
                # Calculate total runtime and cost based on workload
                total_runtime_hours = self.config.total_tokens_to_process / result['throughput'] / 3600
                total_cost = result['cost'] * total_runtime_hours
                
                input_tps, output_tps = get_phase_throughputs(result['throughput'])
                
                rows.append({
                    'model_name': self.config.model_name,
                    'max_input_length': self.config.sequence_length,
                    'max_output_length': self.config.output_length,
                    'batch_size': sol.get('batch_size', ''),
                    'budget_tested': f"{result['budget']:.2f}",
                    'total_tokens_per_sec': f"{result['throughput']:.2f}",
                    'input_tokens_per_sec': f"{input_tps:.2f}",
                    'output_tokens_per_sec': f"{output_tps:.2f}",
                    'cost_per_hour': f"{result['cost']:.2f}",
                    'dollar_per_million_token': f"{cost_per_million:.6f}",
                    'total_runtime_hours': f"{total_runtime_hours:.2f}",
                    'total_cost': f"{total_cost:.2f}",
                    'pipeline_stages': sol['num_pipeline_stages'],
                    'device_type': device_types,
                    'pp': sol['num_pipeline_stages'],
                    'mem_per_gpu_gb': stage_mem_gb,
                    'placement': placement,
                    'layer_mapping': layer_mapping,
                    'tp_per_stage': tp_per_stage,
                    'num_gpus': num_gpus,
                    'total_layers': total_layers,
                    'is_best': 'YES' if is_best else 'NO',
                    'status': 'SUCCESS'
                })
            
            # Write CSV with all results
            with open(output_file, 'w', newline='') as f:
                fieldnames = [
                    'model_name', 'max_input_length', 'max_output_length', 'batch_size', 'budget_tested',
                    'total_tokens_per_sec', 'input_tokens_per_sec', 'output_tokens_per_sec',
                    'cost_per_hour', 'dollar_per_million_token', 'total_runtime_hours', 'total_cost',
                    'pipeline_stages', 'device_type', 'pp', 'mem_per_gpu_gb',
                    'placement', 'layer_mapping', 'tp_per_stage',
                    'num_gpus', 'total_layers', 'is_best', 'status'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Solution CSV saved with {len(rows)} enumeration results to {output_file}")
        else:
            # Save only the best solution (original behavior)
            num_gpus = ''
            total_layers = ''
            placement = ''
            layer_mapping = ''
            tp_per_stage = ''
            stage_mem_gb = ''
            device_types = ''
            
            if self.solution['gpu_assignments']:
                num_gpus = sum(len(a['gpu_ids']) for a in self.solution['gpu_assignments'])
                total_layers = sum(a['segment_size'] for a in self.solution['gpu_assignments'])
                placement = format_placement(self.solution['gpu_assignments'])
                layer_mapping = format_layer_mapping(self.solution['gpu_assignments'])
                tp_per_stage = format_tp_per_stage(self.solution['gpu_assignments'])
                stage_mem_gb = format_stage_mem_gb(self.solution['gpu_assignments'])
                device_types = get_device_types(self.solution['gpu_assignments'])
            
            cost_per_million = self.solution['cost_per_token'] * 1_000_000
            
            # Calculate total runtime and cost based on workload
            total_runtime_hours = self.config.total_tokens_to_process / self.solution['throughput_tokens_per_sec'] / 3600
            total_cost = self.solution['cost_per_hour'] * total_runtime_hours
            
            input_tps, output_tps = get_phase_throughputs(self.solution['throughput_tokens_per_sec'])
            
            row = {
                'model_name': self.config.model_name,
                'max_input_length': self.config.sequence_length,
                'max_output_length': self.config.output_length,
                'batch_size': self.solution.get('batch_size', ''),
                'total_tokens_per_sec': f"{self.solution['throughput_tokens_per_sec']:.2f}",
                'input_tokens_per_sec': f"{input_tps:.2f}",
                'output_tokens_per_sec': f"{output_tps:.2f}",
                'cost_per_hour': f"{self.solution['cost_per_hour']:.2f}",
                'dollar_per_million_token': f"{cost_per_million:.6f}",
                'total_runtime_hours': f"{total_runtime_hours:.2f}",
                'total_cost': f"{total_cost:.2f}",
                'pipeline_stages': self.solution['num_pipeline_stages'],
                'device_type': device_types,
                'pp': self.solution['num_pipeline_stages'],
                'mem_per_gpu_gb': stage_mem_gb,
                'placement': placement,
                'layer_mapping': layer_mapping,
                'tp_per_stage': tp_per_stage,
                'num_gpus': num_gpus,
                'total_layers': total_layers,
                'status': 'SUCCESS'
            }
            
            # Write CSV (with header)
            with open(output_file, 'w', newline='') as f:
                fieldnames = [
                    'model_name', 'max_input_length', 'max_output_length', 'batch_size',
                    'total_tokens_per_sec', 'input_tokens_per_sec', 'output_tokens_per_sec',
                    'cost_per_hour', 'dollar_per_million_token', 'total_runtime_hours', 'total_cost',
                    'pipeline_stages', 'device_type', 'pp', 'mem_per_gpu_gb',
                'placement', 'layer_mapping', 'tp_per_stage',
                    'num_gpus', 'total_layers', 'status'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            
            logger.info(f"Solution CSV saved to {output_file}")

    @staticmethod
    def _get_instance_family(instance_name: str) -> str:
        """Extract instance family from instance name (e.g., 'g6e.48xlarge#0' -> 'g6e.48xlarge')"""
        if '#' in instance_name:
            return instance_name.split('#')[0]
        return instance_name

    def solve_homogeneous(self) -> bool:
        """
        Solve for optimal homogeneous placement using fast enumeration.

        Homogeneous placement constraints:
        1. All PP stages use the same instance family (e.g., all g6e.48xlarge)
        2. All PP stages have the same number of layers
        3. All PP stages use the same TP degree

        This method bypasses MILP and directly enumerates all valid configurations,
        which is much faster than the full heterogeneous solver.

        Returns:
            bool: True if a valid solution was found, False otherwise
        """
        # Attach log capture handler so all solver output is available to callers
        _capture = _LogCapture()
        _capture.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(_capture)

        logger.info("=" * 80)
        logger.info("HOMOGENEOUS PLACEMENT SOLVER (wave model)")
        logger.info("=" * 80)

        # --- Solver inputs ---
        logger.info(f"Model: {self.config.model_name} ({self.config.num_decoder_layers}L, "
                    f"d={self.config.d_model}, h={self.config.num_attention_heads}, "
                    f"kv_heads={self.config.num_kv_heads}, head_dim={self.config.head_dim})")
        logger.info(f"Workload: seq_len={self.config.sequence_length}, output_len={self.config.output_length}, "
                    f"phase={self.config.workload_phase}")
        logger.info(f"Max token limits: max_input={self.config.max_input_tokens}, "
                    f"max_output={self.config.max_output_tokens}")
        logger.info(f"vLLM params: max_num_seqs={self.config.max_num_seqs}, "
                    f"max_num_batched_tokens={self.config.max_num_batched_tokens} "
                    f"({'default' if self.config.max_num_batched_tokens == 16384 else 'explicit'}), "
                    f"gpu_memory_utilization={self.config.gpu_memory_utilization:.2f}")
        logger.info(f"Efficiency: real_world={self.config.real_world_efficiency:.0%}, "
                    f"weight_mem/layer={self.config.layer_weight_memory_gb:.3f} GB, "
                    f"bytes_per_elem={self.config.bytes_per_element}")

        # SLO constraint
        slo_active = self.config.max_total_runtime_hours < 999999.0
        if slo_active:
            min_throughput_for_slo = self.config.total_tokens_to_process / (
                self.config.max_total_runtime_hours * 3600
            )
            logger.info(f"SLO: {self.config.max_total_runtime_hours:.2f} hours, "
                        f"tokens_per_replica={self.config.total_tokens_to_process}, "
                        f"min_throughput={min_throughput_for_slo:.0f} tok/s")
        else:
            min_throughput_for_slo = 0.0
            logger.info("SLO: none (optimizing $/token only)")

        total_layers = self.config.num_decoder_layers
        max_pp_stages = self.config.max_pipeline_stages

        # Group instances by family
        instance_families: Dict[str, List[str]] = {}
        for instance_name in self.gpu_types.keys():
            family = self._get_instance_family(instance_name)
            if family not in instance_families:
                instance_families[family] = []
            instance_families[family].append(instance_name)

        logger.info(f"Instance families: {list(instance_families.keys())}")
        logger.info(f"Total layers: {total_layers}, Max PP stages: {max_pp_stages}")

        best_solution = None
        best_cost_per_token = float('inf')
        all_valid_configs = []

        # Enumerate all valid homogeneous configurations
        for family, instances in instance_families.items():
            # Get instance properties (all instances in family have same specs)
            sample_instance = instances[0]
            gpu_obj = self.gpu_types[sample_instance]
            gpus_per_instance = gpu_obj.count
            instance_cost = gpu_obj.cost_per_hour
            gpu_model = gpu_obj.gpu_model
            gpu_memory_gb = gpu_obj.memory_gb
            num_instances_available = len(instances)

            usable_gb = gpu_memory_gb * self.config.gpu_memory_utilization
            logger.info(f"--- {family}: {gpus_per_instance}x {gpu_model} {gpu_memory_gb}GB "
                        f"(usable {usable_gb:.1f}GB @ {self.config.gpu_memory_utilization:.0%}), "
                        f"${instance_cost:.2f}/hr, avail={num_instances_available} ---")

            # Valid TP degrees for this instance type
            valid_tp_degrees = [d for d in [1, 2, 4, 8]
                               if d <= gpus_per_instance and gpus_per_instance % d == 0]

            # Fixed-size instances (only offered as 8-GPU by cloud provider):
            # packing multiple PP stages per node is necessary.
            # Variable-size families (g6e, g5, etc.): 1 PP stage per node,
            # TP stays intra-node, PP crosses node boundaries.
            FIXED_SIZE_PREFIXES = ('p3.', 'p3dn.', 'p4d.', 'p4de.', 'p5.', 'p5e.')
            is_fixed_size = any(family.startswith(p) for p in FIXED_SIZE_PREFIXES)

            for tp_degree in valid_tp_degrees:
                # Variable-size instances: skip if instance has more GPUs than TP needs
                # (pick the right-sized instance instead of wasting GPUs)
                if not is_fixed_size and gpus_per_instance != tp_degree:
                    continue

                # Calculate number of TP partitions per instance
                partitions_per_instance = gpus_per_instance // tp_degree

                for pp_stages in range(1, max_pp_stages + 1):
                    # Check if layers divide evenly
                    if total_layers % pp_stages != 0:
                        continue
                    layers_per_stage = total_layers // pp_stages

                    # Calculate how many instances we need
                    if is_fixed_size:
                        # Pack multiple PP stages per node (only option)
                        instances_needed = math.ceil(pp_stages / partitions_per_instance)
                    else:
                        # 1 PP stage per node; gpus_per_instance == tp_degree
                        instances_needed = pp_stages

                    if instances_needed > num_instances_available:
                        continue  # Not enough instances of this family

                    # --- Weight feasibility: can weights + minimal overhead fit? ---
                    max_seg_for_tp = self._compute_max_segment_size_for_tp(
                        sample_instance, tp_degree, batch_size=1
                    )
                    if layers_per_stage > max_seg_for_tp:
                        continue  # Weights don't fit

                    # --- Context feasibility: can 1 sequence at max length fit? ---
                    if not self._check_max_context_feasibility(
                        gpu_memory_gb, layers_per_stage, tp_degree
                    ):
                        max_input = self.config.max_input_tokens or self.config.sequence_length
                        max_output = self.config.max_output_tokens or max(self.config.output_length, 0)
                        logger.info(
                            f"  FILTERED: {family} TP={tp_degree} PP={pp_stages} "
                            f"(cannot fit 1 seq at max context {max_input + max_output})"
                        )
                        continue

                    # --- Derive decode concurrency from KV pool ---
                    max_concurrent = self._derive_max_concurrent_sequences(
                        gpu_memory_gb, layers_per_stage, tp_degree
                    )

                    # --- Compute max_model_len for this config ---
                    max_model_len = self._compute_max_model_len(
                        gpu_memory_gb, layers_per_stage, tp_degree
                    )

                    # Get NVLink bandwidth for this instance type
                    nvlink_bw = self._get_representative_tp_bandwidth(sample_instance, tp_degree)

                    # --- Phase-aware throughput ---
                    # Effective decode batch = min(max_num_seqs, KV-pool capacity)
                    # max_num_seqs is the vLLM scheduler limit (default 256)
                    # max_concurrent is the KV-pool capacity (could be much larger)
                    effective_decode_batch = min(self.config.max_num_seqs, max_concurrent)

                    logger.debug(
                        f"  [{family} TP={tp_degree} PP={pp_stages}] "
                        f"layers/stage={layers_per_stage}, instances={instances_needed}, "
                        f"max_concurrent={max_concurrent}, max_model_len={max_model_len}, "
                        f"eff_decode_batch={effective_decode_batch}, tp_bw={nvlink_bw:.1f}GB/s"
                    )

                    if self.config.workload_phase == 'aggregated':
                        # Compute prefill_batch: how many sequences vLLM prefills per iteration
                        S = self.config.sequence_length
                        mnbt = self.config.max_num_batched_tokens or max_model_len
                        prefill_batch = max(1, min(mnbt // S if S > 0 else 1, effective_decode_batch))

                        # Prefill: batch_size=prefill_batch (vLLM batches multiple prefills)
                        prefill_tp = ThroughputFunctions.gpu_throughput_with_tp(
                            gpu_model, self.config.sequence_length, prefill_batch,
                            layers_per_stage, self.config.d_model,
                            self.config.bytes_per_element, tp_degree,
                            self.config.d_hidden, nvlink_bw,
                            debug=False, phase='prefill',
                            num_attention_heads=self.config.num_attention_heads,
                            num_kv_heads=self.config.num_kv_heads,
                            head_dim=self.config.head_dim,
                            vocab_size=self.config.vocab_size,
                            total_num_layers=self.config.num_decoder_layers
                        )
                        # Decode: batch_size=effective_decode_batch (actual engine concurrency)
                        decode_tp = ThroughputFunctions.gpu_throughput_with_tp(
                            gpu_model, self.config.sequence_length, effective_decode_batch,
                            layers_per_stage, self.config.d_model,
                            self.config.bytes_per_element, tp_degree,
                            self.config.d_hidden, nvlink_bw,
                            debug=False, phase='decode',
                            output_length=self.config.output_length,
                            num_attention_heads=self.config.num_attention_heads,
                            num_kv_heads=self.config.num_kv_heads,
                            head_dim=self.config.head_dim,
                            vocab_size=self.config.vocab_size,
                            total_num_layers=self.config.num_decoder_layers
                        )
                        # Apply enforce_eager penalty to decode (decode is memory-bound, affected by CUDA graph overhead)
                        if self.config.enforce_eager:
                            decode_tp *= self.config.enforce_eager_penalty
                        # Combine with wave model (replaces Little's Law + interference_factor)
                        stage_throughput = ThroughputFunctions.calculate_aggregated_throughput(
                            prefill_throughput=prefill_tp,
                            decode_throughput=decode_tp,
                            seq_len=self.config.sequence_length,
                            output_len=self.config.output_length,
                            max_num_seqs=effective_decode_batch,
                            prefill_batch=prefill_batch,
                        )

                        # Log throughput breakdown for valid configs
                        if stage_throughput > 0:
                            O = self.config.output_length
                            N = effective_decode_batch
                            P = prefill_batch
                            prefill_iters = math.ceil(N / P)
                            t_prefill = prefill_iters * (P * S / prefill_tp) if prefill_tp > 0 else float('inf')
                            t_decode = O * N / decode_tp if decode_tp > 0 else float('inf')
                            logger.info(
                                f"  {family} TP={tp_degree} PP={pp_stages}: "
                                f"eff_batch={N} (min(max_num_seqs={self.config.max_num_seqs}, kv_pool={max_concurrent})), "
                                f"prefill_batch={P} (mnbt={mnbt}), prefill_iters={prefill_iters}, "
                                f"prefill_tp={prefill_tp:.0f} tok/s, decode_tp={decode_tp:.0f} tok/s (system), "
                                f"t_prefill={t_prefill:.4f}s, t_decode={t_decode:.4f}s, "
                                f"stage_tp={stage_throughput:.0f} tok/s"
                            )
                    else:
                        # Pure prefill or decode phase
                        batch_for_phase = 1 if self.config.workload_phase == 'prefill' else effective_decode_batch
                        stage_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                            gpu_model, self.config.sequence_length, batch_for_phase,
                            layers_per_stage, self.config.d_model,
                            self.config.bytes_per_element, tp_degree,
                            self.config.d_hidden, nvlink_bw,
                            debug=False, phase=self.config.workload_phase,
                            num_attention_heads=self.config.num_attention_heads,
                            num_kv_heads=self.config.num_kv_heads,
                            head_dim=self.config.head_dim,
                            vocab_size=self.config.vocab_size,
                            total_num_layers=self.config.num_decoder_layers
                        )
                        # Apply enforce_eager penalty to decode phase
                        if self.config.enforce_eager and self.config.workload_phase == 'decode':
                            stage_throughput *= self.config.enforce_eager_penalty

                    if stage_throughput <= 0:
                        continue

                    # --- Pipeline efficiency: comm overhead + bubble ---
                    if pp_stages > 1:
                        inter_stage_bw = self._get_representative_pp_bandwidth(
                            sample_instance, tp_degree, pp_stages, gpus_per_instance,
                            instances=instances
                        )

                        if self.config.workload_phase == 'aggregated' and prefill_tp > 0 and decode_tp > 0:
                            pipeline_efficiency = self._compute_pipeline_efficiency_1f1b(
                                pp_stages, inter_stage_bw,
                                prefill_tp=prefill_tp, decode_tp=decode_tp,
                                effective_decode_batch=effective_decode_batch,
                                prefill_batch=prefill_batch
                            )
                        else:
                            pipeline_efficiency = self._compute_pipeline_efficiency_1f1b(
                                pp_stages, inter_stage_bw,
                                stage_throughput=stage_throughput,
                                batch_for_phase=batch_for_phase
                            )
                    else:
                        pipeline_efficiency = 1.0

                    # Apply real-world efficiency factor with GPU generation penalty
                    real_world_efficiency = self.config.real_world_efficiency
                    _specs = ThroughputFunctions.GPU_SPECS.get(gpu_model, {})
                    generation_factor = _specs.get('generation_factor', 1.0)
                    real_world_efficiency *= generation_factor

                    effective_throughput = stage_throughput * pipeline_efficiency * real_world_efficiency

                    # Calculate cost
                    total_cost_per_hour = instances_needed * instance_cost

                    # Calculate $/token
                    cost_per_token = total_cost_per_hour / (effective_throughput * 3600)
                    cost_per_million = cost_per_token * 1_000_000

                    # --- SLO feasibility: reject configs that can't finish in time ---
                    if self.config.max_total_runtime_hours < 999999.0:
                        runtime_hours = self.config.total_tokens_to_process / (effective_throughput * 3600)
                        if runtime_hours > self.config.max_total_runtime_hours:
                            logger.info(
                                f"  FILTERED: {family} TP={tp_degree} PP={pp_stages} "
                                f"(runtime {runtime_hours:.2f}h > SLO {self.config.max_total_runtime_hours:.2f}h)"
                            )
                            continue

                    logger.debug(
                        f"  [{family} TP={tp_degree} PP={pp_stages}] "
                        f"stage_tp={stage_throughput:.0f} tok/s, "
                        f"pp_eff={pipeline_efficiency:.4f}, rw_eff={real_world_efficiency:.2f} (gen={generation_factor:.2f}), "
                        f"eff_tp={effective_throughput:.0f} tok/s, "
                        f"${total_cost_per_hour:.2f}/hr, ${cost_per_million:.4f}/Mtok"
                    )

                    # SLO feasibility check: can this config finish within the deadline?
                    estimated_runtime_hours = None
                    meets_slo = True
                    if slo_active and effective_throughput > 0:
                        estimated_runtime_hours = self.config.total_tokens_to_process / (
                            effective_throughput * 3600
                        )
                        meets_slo = estimated_runtime_hours <= self.config.max_total_runtime_hours

                    config_info = {
                        'family': family,
                        'instances_used': instances[:instances_needed],
                        'num_instances': instances_needed,
                        'tp_degree': tp_degree,
                        'pp_stages': pp_stages,
                        'layers_per_stage': layers_per_stage,
                        'max_concurrent_sequences': max_concurrent,
                        'max_model_len': max_model_len,
                        'stage_throughput': stage_throughput,
                        'pipeline_efficiency': pipeline_efficiency,
                        'effective_throughput': effective_throughput,
                        'cost_per_hour': total_cost_per_hour,
                        'cost_per_token': cost_per_token,
                        'cost_per_million': cost_per_million,
                        'gpu_model': gpu_model,
                        'gpus_per_instance': gpus_per_instance,
                        'gpu_memory_gb': gpu_memory_gb,
                        'nvlink_bw': nvlink_bw,
                        'estimated_runtime_hours': estimated_runtime_hours,
                        'meets_slo': meets_slo,
                    }

                    all_valid_configs.append(config_info)

                    if meets_slo and cost_per_token < best_cost_per_token:
                        best_cost_per_token = cost_per_token
                        best_solution = config_info

        slo_feasible = [c for c in all_valid_configs if c['meets_slo']]
        slo_rejected = len(all_valid_configs) - len(slo_feasible)
        logger.info(f"\nEnumerated {len(all_valid_configs)} valid configurations")
        if slo_active:
            logger.info(f"SLO filter: {len(slo_feasible)} meet deadline, {slo_rejected} too slow")

        # If SLO is active but nothing meets it, fall back to fastest config
        if not best_solution and slo_active and all_valid_configs:
            fastest = max(all_valid_configs, key=lambda c: c['effective_throughput'])
            best_solution = fastest
            best_cost_per_token = fastest['cost_per_token']
            est = fastest.get('estimated_runtime_hours')
            logger.warning(
                f"No config meets SLO ({self.config.max_total_runtime_hours:.2f}h). "
                f"Falling back to fastest: {fastest['family']} TP={fastest['tp_degree']} "
                f"PP={fastest['pp_stages']} ({fastest['effective_throughput']:.0f} tok/s, "
                f"est {est:.2f}h)"
            )

        if not best_solution:
            logger.error("No valid homogeneous placement found!")
            self.solve_log = "\n".join(_capture.lines)
            logger.removeHandler(_capture)
            return False

        # Sort all configs by cost_per_token for reporting
        all_valid_configs.sort(key=lambda x: x['cost_per_token'])

        # Log top 10 configurations
        logger.info("="*80)
        logger.info("TOP 10 HOMOGENEOUS CONFIGURATIONS (by $/M tokens)")
        logger.info("="*80)
        slo_hdr = " {'ETA(h)':<8}" if slo_active else ""
        logger.info(f"{'Rank':<5} {'Family':<20} {'PP':<4} {'TP':<4} {'Layers/Stage':<13} "
                   f"{'KVPool':<8} {'EffBatch':<9} "
                   f"{'Throughput':<12} {'$/hour':<10} {'$/M tokens':<12}"
                   + (" {'ETA(h)':<8} {'SLO':<4}" if slo_active else ""))
        logger.info("-"*115)

        for i, cfg in enumerate(all_valid_configs[:10], 1):
            eff_batch = min(self.config.max_num_seqs, cfg['max_concurrent_sequences'])
            slo_suffix = ""
            if slo_active:
                eta = cfg.get('estimated_runtime_hours')
                eta_str = f"{eta:.2f}" if eta is not None else "N/A"
                slo_mark = "OK" if cfg['meets_slo'] else "MISS"
                slo_suffix = f" {eta_str:<8} {slo_mark:<4}"
            logger.info(f"{i:<5} {cfg['family']:<20} {cfg['pp_stages']:<4} {cfg['tp_degree']:<4} "
                       f"{cfg['layers_per_stage']:<13} {cfg['max_concurrent_sequences']:<8} "
                       f"{eff_batch:<9} "
                       f"{cfg['effective_throughput']:<12.0f} ${cfg['cost_per_hour']:<9.2f} "
                       f"${cfg['cost_per_million']:<11.6f}"
                       + slo_suffix)

        # Build solution in the standard format
        best = best_solution

        # Create GPU assignments
        gpu_assignments = []
        start_layer = 1
        instances_to_use = best['instances_used']
        stages_assigned = 0

        for inst_idx, instance_name in enumerate(instances_to_use):
            gpu_obj = self.gpu_types[instance_name]
            partitions_per_instance = gpu_obj.count // best['tp_degree']

            for partition_idx in range(partitions_per_instance):
                if stages_assigned >= best['pp_stages']:
                    break

                # GPU IDs for this partition
                start_gpu = partition_idx * best['tp_degree']
                gpu_ids = list(range(start_gpu, start_gpu + best['tp_degree']))
                global_gpu_ids = [self._get_global_gpu_id(instance_name, gid) for gid in gpu_ids]

                end_layer = start_layer + best['layers_per_stage'] - 1

                assignment = {
                    'gpu_type': instance_name,
                    'partition_id': partition_idx,
                    'gpu_ids': gpu_ids,
                    'global_gpu_ids': global_gpu_ids,
                    'tp_degree': best['tp_degree'],
                    'start_layer': start_layer,
                    'end_layer': end_layer,
                    'segment_size': best['layers_per_stage'],
                    'throughput': best['stage_throughput']
                }
                gpu_assignments.append(assignment)

                start_layer = end_layer + 1
                stages_assigned += 1

            if stages_assigned >= best['pp_stages']:
                break

        # Create network connections
        network_connections = []
        for i in range(len(gpu_assignments) - 1):
            from_seg = gpu_assignments[i]
            to_seg = gpu_assignments[i + 1]

            # Calculate network throughput between stages
            from_gpu = from_seg['global_gpu_ids'][0]
            to_gpu = to_seg['global_gpu_ids'][0]

            # Phase-aware inter-stage tensor size
            # Prefill: prefill_batch × seq_len × d_model × bytes
            # Decode: effective_decode_batch × 1 × d_model × bytes
            # Use the larger as bottleneck
            S = self.config.sequence_length
            eff_batch = min(self.config.max_num_seqs, best['max_concurrent_sequences'])
            best_prefill_batch = max(1, min(
                (self.config.max_num_batched_tokens or best['max_model_len']) // S if S > 0 else 1,
                eff_batch
            ))
            prefill_tensor_gb = (best_prefill_batch * S * self.config.d_model *
                                self.config.bytes_per_element) / (1024**3)
            decode_tensor_gb = (eff_batch * 1 * self.config.d_model *
                               self.config.bytes_per_element) / (1024**3)
            tensor_size_gb = max(prefill_tensor_gb, decode_tensor_gb)
            bw = self.network_bandwidth[from_gpu, to_gpu]
            net_throughput = (bw / tensor_size_gb) * eff_batch if tensor_size_gb > 0 else float('inf')

            connection = {
                'from_segment': {
                    'gpu_type': from_seg['gpu_type'],
                    'partition_id': from_seg['partition_id'],
                    'start_layer': from_seg['start_layer'],
                    'end_layer': from_seg['end_layer']
                },
                'to_segment': {
                    'gpu_type': to_seg['gpu_type'],
                    'partition_id': to_seg['partition_id'],
                    'start_layer': to_seg['start_layer'],
                    'end_layer': to_seg['end_layer']
                },
                'throughput': net_throughput
            }
            network_connections.append(connection)

        # Calculate total runtime
        total_runtime_hours = self.config.total_tokens_to_process / best['effective_throughput'] / 3600

        # Build the solution object
        self.solution = {
            'objective_value': best['stage_throughput'],
            'max_concurrent_sequences': best['max_concurrent_sequences'],
            'max_model_len': best['max_model_len'],
            'throughput_tokens_per_sec': best['effective_throughput'],
            'cost_per_hour': best['cost_per_hour'],
            'cost_per_token': best['cost_per_token'],
            'total_runtime_hours': total_runtime_hours,
            'estimated_runtime_hours': best.get('estimated_runtime_hours'),
            'meets_slo': best.get('meets_slo', True),
            'meets_cost_threshold': best['cost_per_token'] <= self.config.max_cost_per_token,
            'tp_configuration': {inst: best['tp_degree'] for inst in best['instances_used']},
            'gpu_assignments': gpu_assignments,
            'network_connections': network_connections,
            'solve_status': 2,  # Optimal
            'num_pipeline_stages': best['pp_stages'],
            'homogeneous_config': {
                'family': best['family'],
                'tp_degree': best['tp_degree'],
                'pp_stages': best['pp_stages'],
                'layers_per_stage': best['layers_per_stage'],
                'instances_used': best['num_instances'],
                'pipeline_efficiency': best['pipeline_efficiency'],
            }
        }

        # Store all enumeration results for CSV export
        # Build proper gpu_assignments with full instance names
        def build_gpu_assignments_for_config(cfg):
            assignments = []
            instances_to_use = cfg['instances_used']
            start_layer = 1
            stages_assigned = 0
            gpus_per_inst = cfg['gpus_per_instance']
            partitions_per_inst = gpus_per_inst // cfg['tp_degree']

            for inst_name in instances_to_use:
                for part_idx in range(partitions_per_inst):
                    if stages_assigned >= cfg['pp_stages']:
                        break
                    end_layer = start_layer + cfg['layers_per_stage'] - 1
                    assignments.append({
                        'gpu_type': inst_name,
                        'tp_degree': cfg['tp_degree'],
                        'start_layer': start_layer,
                        'end_layer': end_layer,
                        'segment_size': cfg['layers_per_stage'],
                        'gpu_ids': list(range(part_idx * cfg['tp_degree'],
                                              (part_idx + 1) * cfg['tp_degree'])),
                    })
                    start_layer = end_layer + 1
                    stages_assigned += 1
                if stages_assigned >= cfg['pp_stages']:
                    break
            return assignments

        self.all_enumeration_results = [
            {
                'budget': cfg['cost_per_hour'],
                'throughput': cfg['effective_throughput'],
                'cost': cfg['cost_per_hour'],
                'cost_per_token': cfg['cost_per_token'],
                'solution': {
                    'max_concurrent_sequences': cfg['max_concurrent_sequences'],
                    'num_pipeline_stages': cfg['pp_stages'],
                    'gpu_assignments': build_gpu_assignments_for_config(cfg)
                }
            }
            for cfg in all_valid_configs
        ]

        logger.info("=" * 80)
        logger.info("OPTIMAL HOMOGENEOUS SOLUTION")
        logger.info("=" * 80)
        eff_batch = min(self.config.max_num_seqs, best['max_concurrent_sequences'])
        S = self.config.sequence_length
        mnbt = self.config.max_num_batched_tokens or best['max_model_len']
        opt_prefill_batch = max(1, min(mnbt // S if S > 0 else 1, eff_batch))
        opt_prefill_iters = math.ceil(eff_batch / opt_prefill_batch)

        logger.info(f"Instance: {best['family']} x {best['num_instances']}")
        logger.info(f"GPU: {best['gpu_model']}, TP={best['tp_degree']}, PP={best['pp_stages']}, "
                    f"layers/stage={best['layers_per_stage']}")
        logger.info(f"Memory: KV pool={best['max_concurrent_sequences']} seqs, "
                    f"max_model_len={best['max_model_len']}")
        logger.info(f"Batching: eff_decode_batch={eff_batch} "
                    f"(min(max_num_seqs={self.config.max_num_seqs}, kv_pool={best['max_concurrent_sequences']}))")
        logger.info(f"Wave model: prefill_batch={opt_prefill_batch} "
                    f"(mnbt={mnbt} // seq_len={S}), "
                    f"prefill_iters={opt_prefill_iters} (ceil({eff_batch}/{opt_prefill_batch}))")
        _best_gen = ThroughputFunctions.GPU_SPECS.get(best.get('gpu_model', ''), {}).get('generation_factor', 1.0)
        logger.info(f"Throughput: stage={best['stage_throughput']:.0f} tok/s, "
                    f"pipeline_eff={best['pipeline_efficiency']:.0%}, "
                    f"real_world_eff={self.config.real_world_efficiency:.0%}×gen={_best_gen:.2f}, "
                    f"effective={best['effective_throughput']:.0f} tok/s")
        logger.info(f"Cost: ${best['cost_per_hour']:.2f}/hr, "
                    f"${best['cost_per_million']:.4f}/M tokens")

        # ===== DETAILED DIAGNOSTIC BREAKDOWN (for verification) =====
        logger.info("")
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC BREAKDOWN (verify against vLLM)")
        logger.info("=" * 80)
        _tp = best['tp_degree']
        _lps = best['layers_per_stage']
        _gpu_mem = best['gpu_memory_gb']
        _nvlink = best['nvlink_bw']
        _gpu = best['gpu_model']
        _specs = ThroughputFunctions.GPU_SPECS.get(_gpu, ThroughputFunctions.GPU_SPECS['A100'])

        # --- GPU specs ---
        logger.info(f"[GPU] {_gpu}: peak={_specs['tflops']} TFLOPS, "
                    f"mem_bw={_specs['mem_bw']} GB/s, "
                    f"compute_eff={_specs['compute_efficiency']:.0%}, "
                    f"memory_eff={_specs['memory_efficiency']:.0%}, "
                    f"nvlink_bw={_nvlink:.0f} GB/s")
        _ridge = _specs['tflops'] * 1e12 / (_specs['mem_bw'] * 1e9)
        logger.info(f"[GPU] Ridge point: {_ridge:.1f} FLOPs/byte")

        # --- Memory breakdown ---
        _usable = _gpu_mem * self.config.gpu_memory_utilization
        _wperlayer = self.config.layer_weight_memory_gb / _tp
        _weights_gb = _lps * _wperlayer
        _act_overhead = self._calculate_activation_memory(_tp, batch_size=1)
        _kv_pool_gb = self._compute_kv_pool_gb(_gpu_mem, _lps, _tp)
        logger.info(f"\n[Memory] GPU={_gpu_mem}GB, usable={_usable:.1f}GB "
                    f"(gpu_memory_utilization={self.config.gpu_memory_utilization})")
        logger.info(f"[Memory] Weights: {_lps} layers × {_wperlayer:.3f} GB/layer = {_weights_gb:.3f} GB "
                    f"(layer_weight_memory_gb={self.config.layer_weight_memory_gb}, TP={_tp})")
        logger.info(f"[Memory] Activation overhead (batch=1): {_act_overhead:.4f} GB")

        # Activation breakdown
        _batch1 = 1
        _tokens1 = self.config.sequence_length
        _d_head = self.config.head_dim
        _kv_dim = self.config.num_kv_heads * _d_head
        _q_o_dim = self.config.num_attention_heads * _d_head
        _qkv_mem = (_batch1 * _tokens1 * ((_q_o_dim / _tp) + 2 * (_kv_dim / _tp))
                    * self.config.bytes_per_element) / (1024**3)
        _mlp_mem = (_batch1 * _tokens1 * (self.config.d_hidden / _tp)
                    * self.config.bytes_per_element) / (1024**3)
        _sharded = _qkv_mem + _mlp_mem
        _full_act = (_batch1 * _tokens1 * self.config.d_model * self.config.bytes_per_element) / (1024**3)
        _peak_act = max(_sharded, _full_act)
        _num_graphs = min(6, int(math.log2(max(1, _batch1))) + 2)
        _graph_ws = (_batch1 * 1 * (self.config.d_model + self.config.d_hidden / _tp)
                     * self.config.bytes_per_element) / (1024**3)
        _cuda_graph = _num_graphs * _graph_ws + 0.5
        _logits = (_batch1 * self.config.vocab_size * 4) / (1024**3)
        _sampler = _logits * 1.5
        logger.info(f"[Memory]   peak_activation={_peak_act*1024:.1f} MB "
                    f"(sharded={_sharded*1024:.1f} MB, full={_full_act*1024:.1f} MB)")
        logger.info(f"[Memory]   cuda_graph_mem={_cuda_graph*1024:.1f} MB "
                    f"({_num_graphs} graphs + 512 MB fixed)")
        logger.info(f"[Memory]   sampler={_sampler*1024:.1f} MB "
                    f"(logits={_logits*1024:.1f} MB × 1.5)")
        logger.info(f"[Memory]   allocator_frag=15% of {(_peak_act+_cuda_graph+_sampler)*1024:.1f} MB")
        logger.info(f"[Memory] KV pool: {_usable:.1f} - {_weights_gb:.3f} - {_act_overhead:.4f} "
                    f"= {_kv_pool_gb:.4f} GB")

        # KV per token
        _kv_bpt = self._kv_bytes_per_token(_lps, _tp)
        logger.info(f"[Memory] KV bytes/token: 2 × {_lps} layers × {_kv_dim}/{_tp} × "
                    f"{self.config.bytes_per_element}B = {_kv_bpt:.0f} bytes/token")

        # max_model_len
        _mml = self._compute_max_model_len(_gpu_mem, _lps, _tp)
        _mml_raw = int((_kv_pool_gb * 1024**3) / _kv_bpt) if _kv_bpt > 0 else 0
        logger.info(f"[Memory] max_model_len: raw={_mml_raw}, "
                    f"clamped=[1024, {self.config.max_position_embeddings}] → {_mml}")

        # Concurrent sequences
        _avg_seq = self.config.sequence_length + max(self.config.output_length, 0)
        _kv_per_seq = _kv_bpt * _avg_seq
        _max_conc = max(1, int((_kv_pool_gb * 1024**3) / _kv_per_seq)) if _kv_per_seq > 0 else 1
        logger.info(f"[Memory] max_concurrent: kv_pool_bytes / (kv_bpt × avg_seq_tokens) "
                    f"= {_kv_pool_gb*1024**3:.0f} / ({_kv_bpt:.0f} × {_avg_seq}) = {_max_conc}")
        logger.info(f"[Memory]   eff_decode_batch = min(max_num_seqs={self.config.max_num_seqs}, "
                    f"kv_pool={_max_conc}) = {eff_batch}")

        # --- Roofline breakdown for PREFILL ---
        logger.info(f"\n[Roofline-Prefill] batch={opt_prefill_batch}, "
                    f"seq_len={S}, layers={_lps}, TP={_tp}")

        # Recompute FLOPs for prefill
        _pb = opt_prefill_batch
        _layer_frac = _lps / self.config.num_decoder_layers
        _pf_q = 2 * _pb * S * self.config.d_model * (_q_o_dim / _tp)
        _pf_k = 2 * _pb * S * self.config.d_model * (_kv_dim / _tp)
        _pf_v = 2 * _pb * S * self.config.d_model * (_kv_dim / _tp)
        _pf_o = 2 * _pb * S * _q_o_dim * (self.config.d_model / _tp)
        _pf_attn_proj = _pf_q + _pf_k + _pf_v + _pf_o
        _pf_attn_score = 4 * _pb * S * S * (_q_o_dim / _tp)
        _pf_ffn = 2 * _pb * S * (
            self.config.d_model * (self.config.d_hidden / _tp) +
            self.config.d_model * (self.config.d_hidden / _tp) +
            (self.config.d_hidden / _tp) * self.config.d_model
        )
        _pf_rmsnorm_flops = 2 * 5 * _pb * S * self.config.d_model
        _pf_flops_layer = _pf_attn_proj + _pf_attn_score + _pf_ffn + _pf_rmsnorm_flops
        _pf_total_flops = _lps * _pf_flops_layer
        # LM head (amortized)
        _pf_lmhead_flops = 2 * _pb * S * self.config.d_model * self.config.vocab_size * _layer_frac
        _pf_total_flops += _pf_lmhead_flops

        logger.info(f"[Roofline-Prefill] FLOPs/layer: attn_proj={_pf_attn_proj:.3e}, "
                    f"attn_score(O(n²))={_pf_attn_score:.3e}, ffn={_pf_ffn:.3e}, "
                    f"rmsnorm={_pf_rmsnorm_flops:.3e}")
        logger.info(f"[Roofline-Prefill] Total FLOPs: {_pf_flops_layer:.3e}/layer × {_lps} "
                    f"+ lm_head={_pf_lmhead_flops:.3e} = {_pf_total_flops:.3e}")

        # Memory access for prefill
        _pf_wt_bytes = _lps * (
            (self.config.d_model * (_q_o_dim / _tp) + _q_o_dim * (self.config.d_model / _tp)) +
            (2 * self.config.d_model * (_kv_dim / _tp)) +
            (3 * self.config.d_model * (self.config.d_hidden / _tp))
        ) * self.config.bytes_per_element
        _pf_act_bytes = _lps * _pb * S * self.config.d_model * self.config.bytes_per_element
        _pf_kv_bytes = _lps * 2 * _pb * S * (_kv_dim / _tp) * self.config.bytes_per_element
        # Attention score intermediate tensors (QK^T write + softmax read + V-multiply read)
        _pf_attn_score_bytes = _lps * _pb * (self.config.num_attention_heads / _tp) * S * S * self.config.bytes_per_element * 3
        # RMSNorm bytes: 2 norms × (read input + write output + read norm weight)
        _pf_rmsnorm_bytes = _lps * 2 * (2 * _pb * S * self.config.d_model + self.config.d_model) * self.config.bytes_per_element
        # Residual connections: 2 per layer × (read + write) × d_model per token
        _pf_residual_bytes = _lps * 2 * 2 * _pb * S * self.config.d_model * self.config.bytes_per_element
        # LM head bytes (amortized)
        _pf_lmhead_wt_bytes = self.config.d_model * self.config.vocab_size * self.config.bytes_per_element / _tp * _layer_frac
        _pf_lmhead_act_bytes = _pb * S * self.config.vocab_size * self.config.bytes_per_element / _tp * _layer_frac
        _pf_lmhead_bytes = _pf_lmhead_wt_bytes + _pf_lmhead_act_bytes
        _pf_total_bytes = (_pf_wt_bytes + _pf_act_bytes + _pf_kv_bytes +
                           _pf_attn_score_bytes + _pf_rmsnorm_bytes + _pf_residual_bytes + _pf_lmhead_bytes)

        logger.info(f"[Roofline-Prefill] Memory: weights={_pf_wt_bytes/1e9:.3f}GB, "
                    f"act={_pf_act_bytes/1e9:.3f}GB, kv_write={_pf_kv_bytes/1e9:.3f}GB, "
                    f"attn_score={_pf_attn_score_bytes/1e9:.3f}GB, "
                    f"rmsnorm={_pf_rmsnorm_bytes/1e9:.3f}GB, residual={_pf_residual_bytes/1e9:.3f}GB, "
                    f"lm_head={_pf_lmhead_bytes/1e9:.3f}GB, total={_pf_total_bytes/1e9:.3f}GB")

        _pf_ai = _pf_total_flops / _pf_total_bytes if _pf_total_bytes > 0 else 0
        _pf_regime = "COMPUTE" if _pf_ai > _ridge else "MEMORY"
        logger.info(f"[Roofline-Prefill] AI={_pf_ai:.1f} FLOPs/byte → {_pf_regime}-BOUND "
                    f"(ridge={_ridge:.1f})")

        if _pf_regime == "COMPUTE":
            _pf_base_time = _pf_total_flops / (_specs['tflops'] * 1e12 * _specs['compute_efficiency'])
        else:
            _pf_base_time = _pf_total_bytes / (_specs['mem_bw'] * 1e9 * _specs['memory_efficiency'])

        _pf_comm_size = _pb * S * self.config.d_model * self.config.bytes_per_element
        # Per-op latency and effective BW: PCIe degrades with TP degree
        _allreduce_latency = 5e-6 if _nvlink >= 100 else 200e-6
        if _nvlink >= 100:
            _eff_comm_bw = _nvlink
        else:
            _pcie_scale = {1: 1.0, 2: 0.75, 4: 0.375, 8: 0.125}.get(_tp, max(0.10, 0.5 / _tp))
            _eff_comm_bw = _nvlink * _pcie_scale
        _pf_comm_per_layer = 2 * (2 * _pf_comm_size / (_eff_comm_bw * 1e9) * (_tp - 1) / _tp + _allreduce_latency) if _tp > 1 else 0
        _pf_total_comm = _lps * _pf_comm_per_layer
        # Simplified TP efficiency: residual overhead only (per-op latency in comm_time)
        if _tp == 1:
            _pf_tp_eff = 1.0
        else:
            _residual_oh = {2: 0.01, 4: 0.02, 8: 0.03, 16: 0.05}.get(_tp, 0.06)
            _pf_tp_eff = max(0.50, 1.0 - _residual_oh)
        _pf_time = _pf_base_time / _pf_tp_eff
        _pf_total_time = _pf_time + _pf_total_comm
        _pf_batch_eff = min(1.0, 0.70 + 0.30 * min(1.0, S / 512))
        _pf_throughput = (_pb * S / _pf_total_time) * _pf_batch_eff

        logger.info(f"[Roofline-Prefill] base_time={_pf_base_time*1000:.2f}ms, "
                    f"tp_eff={_pf_tp_eff:.4f} "
                    f"(residual_oh={_residual_oh if _tp > 1 else 0:.3f}), "
                    f"adjusted={_pf_time*1000:.2f}ms")
        logger.info(f"[Roofline-Prefill] comm/layer={_pf_comm_per_layer*1000:.4f}ms "
                    f"(act_size={_pf_comm_size/1e6:.1f}MB, "
                    f"nvlink={_nvlink:.0f}GB/s, latency={_allreduce_latency*1e6:.0f}μs/op), "
                    f"total_comm={_pf_total_comm*1000:.2f}ms")
        logger.info(f"[Roofline-Prefill] total_time={_pf_total_time*1000:.2f}ms, "
                    f"batch_eff={_pf_batch_eff:.2f}, "
                    f"throughput={_pf_throughput:.0f} tok/s")

        # --- Roofline breakdown for DECODE ---
        _db = eff_batch
        _O = self.config.output_length
        _avg_kv = S + (_O - 1) / 2.0 if _O > 0 else float(S)
        logger.info(f"\n[Roofline-Decode] batch={_db}, "
                    f"avg_kv_cache_len={_avg_kv:.0f} (seq={S}+({_O}-1)/2), "
                    f"layers={_lps}, TP={_tp}")

        _dc_q = 2 * _db * 1 * self.config.d_model * (_q_o_dim / _tp)
        _dc_k = 2 * _db * 1 * self.config.d_model * (_kv_dim / _tp)
        _dc_v = 2 * _db * 1 * self.config.d_model * (_kv_dim / _tp)
        _dc_o = 2 * _db * 1 * _q_o_dim * (self.config.d_model / _tp)
        _dc_attn_proj = _dc_q + _dc_k + _dc_v + _dc_o
        _dc_attn_score = 4 * _db * 1 * _avg_kv * (_q_o_dim / _tp)
        _dc_ffn = 2 * _db * 1 * (
            self.config.d_model * (self.config.d_hidden / _tp) +
            self.config.d_model * (self.config.d_hidden / _tp) +
            (self.config.d_hidden / _tp) * self.config.d_model
        )
        _dc_rmsnorm_flops = 2 * 5 * _db * 1 * self.config.d_model
        _dc_flops_layer = _dc_attn_proj + _dc_attn_score + _dc_ffn + _dc_rmsnorm_flops
        _dc_total_flops = _lps * _dc_flops_layer
        # LM head (amortized)
        _dc_lmhead_flops = 2 * _db * 1 * self.config.d_model * self.config.vocab_size * _layer_frac
        _dc_total_flops += _dc_lmhead_flops

        logger.info(f"[Roofline-Decode] FLOPs/layer: attn_proj={_dc_attn_proj:.3e}, "
                    f"attn_score(O(n))={_dc_attn_score:.3e}, ffn={_dc_ffn:.3e}, "
                    f"rmsnorm={_dc_rmsnorm_flops:.3e}")
        logger.info(f"[Roofline-Decode] Total FLOPs: {_dc_flops_layer:.3e}/layer × {_lps} "
                    f"+ lm_head={_dc_lmhead_flops:.3e} = {_dc_total_flops:.3e}")

        _dc_wt_bytes = _lps * (
            (self.config.d_model * (_q_o_dim / _tp) + _q_o_dim * (self.config.d_model / _tp)) +
            (2 * self.config.d_model * (_kv_dim / _tp)) +
            (3 * self.config.d_model * (self.config.d_hidden / _tp))
        ) * self.config.bytes_per_element
        _dc_act_bytes = _lps * _db * 1 * self.config.d_model * self.config.bytes_per_element
        _dc_kv_bytes = _lps * 2 * _db * _avg_kv * (_kv_dim / _tp) * self.config.bytes_per_element
        # PagedAttention scatter/gather overhead
        _dc_kv_bytes *= 1.15
        # Attention score intermediate tensors (1 query × kv_cache_len)
        _dc_attn_score_bytes = _lps * _db * (self.config.num_attention_heads / _tp) * _avg_kv * self.config.bytes_per_element * 3
        # RMSNorm bytes
        _dc_rmsnorm_bytes = _lps * 2 * (2 * _db * 1 * self.config.d_model + self.config.d_model) * self.config.bytes_per_element
        # Residual connections
        _dc_residual_bytes = _lps * 2 * 2 * _db * 1 * self.config.d_model * self.config.bytes_per_element
        # LM head bytes (amortized)
        _dc_lmhead_wt_bytes = self.config.d_model * self.config.vocab_size * self.config.bytes_per_element / _tp * _layer_frac
        _dc_lmhead_act_bytes = _db * 1 * self.config.vocab_size * self.config.bytes_per_element / _tp * _layer_frac
        _dc_lmhead_bytes = _dc_lmhead_wt_bytes + _dc_lmhead_act_bytes
        _dc_total_bytes = (_dc_wt_bytes + _dc_act_bytes + _dc_kv_bytes +
                           _dc_attn_score_bytes + _dc_rmsnorm_bytes + _dc_residual_bytes + _dc_lmhead_bytes)

        logger.info(f"[Roofline-Decode] Memory: weights={_dc_wt_bytes/1e9:.3f}GB, "
                    f"act={_dc_act_bytes/1e6:.2f}MB, kv_read={_dc_kv_bytes/1e9:.3f}GB, "
                    f"attn_score={_dc_attn_score_bytes/1e6:.2f}MB, "
                    f"rmsnorm={_dc_rmsnorm_bytes/1e6:.2f}MB, residual={_dc_residual_bytes/1e6:.2f}MB, "
                    f"lm_head={_dc_lmhead_bytes/1e6:.2f}MB, total={_dc_total_bytes/1e9:.3f}GB")

        _dc_ai = _dc_total_flops / _dc_total_bytes if _dc_total_bytes > 0 else 0
        _dc_regime = "COMPUTE" if _dc_ai > _ridge else "MEMORY"
        logger.info(f"[Roofline-Decode] AI={_dc_ai:.1f} FLOPs/byte → {_dc_regime}-BOUND")

        if _dc_regime == "COMPUTE":
            _dc_base_time = _dc_total_flops / (_specs['tflops'] * 1e12 * _specs['compute_efficiency'])
        else:
            _dc_base_time = _dc_total_bytes / (_specs['mem_bw'] * 1e9 * _specs['memory_efficiency'])

        _dc_comm_size = _db * 1 * self.config.d_model * self.config.bytes_per_element
        _dc_comm_per_layer = 2 * (2 * _dc_comm_size / (_eff_comm_bw * 1e9) * (_tp - 1) / _tp + _allreduce_latency) if _tp > 1 else 0
        _dc_total_comm = _lps * _dc_comm_per_layer
        _dc_tp_eff = _pf_tp_eff
        _dc_time = _dc_base_time / _dc_tp_eff
        _dc_total_time = _dc_time + _dc_total_comm
        # Decode batch efficiency: memory-bound, high utilization even at small batch
        if _db >= 16:
            _dc_batch_eff = 1.0
        elif _db >= 4:
            _dc_batch_eff = 0.95
        elif _db >= 2:
            _dc_batch_eff = 0.90
        else:
            _dc_batch_eff = 0.85
        _dc_throughput = (_db / _dc_total_time) * _dc_batch_eff

        logger.info(f"[Roofline-Decode] base_time={_dc_base_time*1000:.3f}ms, "
                    f"comm={_dc_total_comm*1000:.4f}ms, "
                    f"total_time={_dc_total_time*1000:.3f}ms")
        logger.info(f"[Roofline-Decode] batch_eff={_dc_batch_eff:.2f}, "
                    f"throughput={_dc_throughput:.0f} tok/s (per forward pass)")

        # --- Wave model summary ---
        logger.info(f"\n[Wave] prefill_tp={_pf_throughput:.0f} tok/s, "
                    f"decode_tp={_dc_throughput:.0f} tok/s")
        _w_N = eff_batch
        _w_P = opt_prefill_batch
        _w_piters = math.ceil(_w_N / _w_P)
        _w_t_pf = _w_piters * (_w_P * S / _pf_throughput) if _pf_throughput > 0 else float('inf')
        _w_t_dc = _O * _w_N / _dc_throughput if _dc_throughput > 0 else float('inf')
        _w_time = _w_t_pf + _w_t_dc
        _w_stage_tp = _w_N * (S + _O) / _w_time if _w_time > 0 else 0
        logger.info(f"[Wave] t_prefill={_w_piters}×({_w_P}×{S}/{_pf_throughput:.0f}) = {_w_t_pf:.4f}s")
        logger.info(f"[Wave] t_decode={_O}×({_w_N}/{_dc_throughput:.0f}) = {_w_t_dc:.4f}s")
        logger.info(f"[Wave] stage_tp={_w_N}×{S+_O}/{_w_time:.4f} = {_w_stage_tp:.0f} tok/s")

        # --- Final throughput chain ---
        logger.info(f"\n[Final] stage_tp={best['stage_throughput']:.0f} "
                    f"× pipeline_eff={best['pipeline_efficiency']:.2f} "
                    f"× real_world_eff={self.config.real_world_efficiency:.2f}×gen={_best_gen:.2f} "
                    f"= {best['effective_throughput']:.0f} tok/s")

        # --- What to compare against vLLM ---
        logger.info(f"\n[Verify] Compare these with vLLM startup logs:")
        logger.info(f"[Verify]   GPU memory used for weights: ~{_weights_gb:.1f} GB "
                    f"(vLLM: 'Model weights: X.XX GiB')")
        logger.info(f"[Verify]   KV cache pool: ~{_kv_pool_gb:.2f} GB "
                    f"(vLLM: 'KV cache memory pool: X.XX GiB')")
        logger.info(f"[Verify]   max_model_len: {_mml} "
                    f"(vLLM: '--max-model-len' or 'Maximum context length: X')")
        logger.info(f"[Verify]   max_num_seqs: {self.config.max_num_seqs} "
                    f"(vLLM: '--max-num-seqs')")
        logger.info(f"[Verify]   Predicted total time for 1 wave ({_w_N} seqs × {S+_O} tok): "
                    f"{_w_time:.2f}s")
        logger.info(f"[Verify]   Predicted effective throughput: "
                    f"{best['effective_throughput']:.0f} tok/s "
                    f"(measure: total_tokens / wall_time)")

        # Store captured log for callers (adapter/server)
        self.solve_log = "\n".join(_capture.lines)
        logger.removeHandler(_capture)

        return True


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
            tier1 = GPU_PERFORMANCE_TIERS.get(gpu_types[gpu1].gpu_model, 2)
            
            for gpu2, tp2 in config.items():
                tier2 = GPU_PERFORMANCE_TIERS.get(gpu_types[gpu2].gpu_model, 2)
                
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
                                     num_layers: int) -> List[Dict[str, int]]:  # noqa: ARG001
    """
    Filter extreme TP-PP combinations based on heuristics.
    - If max TP >= 8, estimated PP should be <= 8
    - If estimated PP >= 16, max TP should be <= 4

    Note: num_layers is kept for potential future use in PP depth estimation.
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
        GPU_PERFORMANCE_TIERS.get(gpu_types[gpu].gpu_model, 2) * tp_degree 
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
        logger.info("="*80)
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
        logger.info("="*80)
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
    parser.add_argument('--output-dir', required=False, help='Output directory')
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
    parser.add_argument('--sequence-length', type=int, required=True,
                       help='Sequence length for prefill (or KV cache length for decode)')
    parser.add_argument('--workload-phase', type=str, required=True,
                       choices=['prefill', 'decode', 'aggregated'],
                       help='Workload phase to model: prefill, decode, or aggregated (combined prefill→decode)')
    parser.add_argument('--output-length', type=int, default=0,
                       help='Output length for decode phase (default: 0)')
    parser.add_argument('--min-batch-size', type=int, required=True,
                       help='Minimum batch size to consider')
    parser.add_argument('--max-batch-size', type=int, required=True,
                       help='Maximum batch size to consider')
    
    # Network bandwidth generation arguments
    parser.add_argument('--generate-network', type=float, nargs=2, metavar=('INTRA_BW', 'INTER_BW'),
                       help='Generate network bandwidth matrix instead of reading from CSV. '
                            'Args: intra_bandwidth (GB/s within same GPU type) inter_bandwidth (GB/s between different GPU types)')

    # Debugging arguments
    parser.add_argument('--throughput-debug-samples', type=int, default=0,
                       help='Log detailed throughput/network debug for N sample segments/connections')
    
    # Cloud pricing arguments
    parser.add_argument('--cloud-provider', type=str, 
                        default='AWS',
                       choices=['AWS', 'GCP', 'Azure', 'Lambda', 'CoreWeave', 'Nebius'],
                       help='Cloud provider for pricing (uses cheapest if not specified). '
                            'Prices loaded from cloud_instances_specs.csv')
    parser.add_argument('--evaluate-placement', type=str,
                       help='Path to placement JSON file to evaluate (skips optimization)')
    parser.add_argument('--eval-output-root', type=str,
                       help='Root directory for eval outputs (results under {root}/{config}/{result_dir})')
    parser.add_argument('--evaluate-throughput', action='store_true',
                       help='Evaluate throughput/cost for a uniform PP/TP placement (skips optimization)')
    parser.add_argument('--eval-instance-family', type=str,
                       help='Instance family/name from gpu_pool.csv (e.g., p4d.24xlarge)')
    parser.add_argument('--eval-tp-degree', type=int,
                       help='TP degree to use for all PP stages in eval mode')
    parser.add_argument('--eval-pp-stages', type=int,
                       help='Number of PP stages in eval mode')

    # Homogeneous placement mode
    parser.add_argument('--homogeneous', type=int, default=0, choices=[0, 1],
                       help='Restrict to homogeneous placement: same instance family, '
                            'same layer count per stage, and same TP degree across all PP stages. '
                            'This bypasses MILP and uses fast enumeration.')

    # Warm start heterogeneous MILP with homogeneous solution
    parser.add_argument('--warm-start-homogeneous', type=int, default=0, choices=[0, 1],
                       help='Run homogeneous solver first and use its solution as MIP warm start '
                            'for heterogeneous MILP. This provides a known good starting point and '
                            'sets a cutoff bound to prune inferior branches. Only used when --homogeneous=0.')

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
            'cloud_provider': args.cloud_provider,
            'throughput_debug_samples': args.throughput_debug_samples,
            'sequence_length': args.sequence_length,
            'workload_phase': args.workload_phase,
            'output_length': args.output_length,
            'min_batch_size': args.min_batch_size,
            'max_batch_size': args.max_batch_size
        }

        if not (args.evaluate_throughput or args.evaluate_placement) and not args.output_dir:
            raise ValueError("Solver mode requires --output-dir")

        # Create output directory if it doesn't exist
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        if args.evaluate_throughput:
            if not args.output_dir and not args.eval_output_root:
                raise ValueError("Eval mode requires --output-dir or --eval-output-root")
            if not args.eval_instance_family:
                raise ValueError("--eval-instance-family is required for --evaluate-throughput")
            if not args.eval_tp_degree or args.eval_tp_degree <= 0:
                raise ValueError("--eval-tp-degree must be a positive integer")
            if not args.eval_pp_stages or args.eval_pp_stages <= 0:
                raise ValueError("--eval-pp-stages must be a positive integer")

            if args.output_dir:
                output_dir = args.output_dir
            else:
                result_dir_name = (
                    f"eval_family-{args.eval_instance_family}-pp{args.eval_pp_stages}-"
                    f"tp{args.eval_tp_degree}-in{args.sequence_length}-out{args.output_length}-"
                    f"bs{args.min_batch_size}_{args.max_batch_size}-"
                    f"{time.strftime('%Y%m%d_%H%M%S')}"
                )
                output_dir = _resolve_eval_output_dir(args, args.config_dir, result_dir_name)
            os.makedirs(output_dir, exist_ok=True)
            config_src = os.path.join(args.config_dir, "config.csv")
            if os.path.exists(config_src):
                shutil.copy2(config_src, os.path.join(output_dir, "config.csv"))

            solver = LLMPlacementSolverWithTP(
                args.config_dir,
                tp_configuration=None,
                **solver_kwargs,
                skip_gurobi=True
            )

            family = args.eval_instance_family
            matching_instances = [
                instance_key for instance_key in solver.gpu_types.keys()
                if instance_key.split("#", 1)[0] == family
            ]
            if not matching_instances:
                raise ValueError(f"No instances found for family '{family}' in gpu_pool.csv")

            pp_stages = args.eval_pp_stages
            if pp_stages > len(matching_instances):
                raise ValueError(
                    f"Not enough instances for PP={pp_stages} using family '{family}': "
                    f"available {len(matching_instances)}"
                )

            matching_instances = sorted(matching_instances)[:pp_stages]
            per_instance_gpu_count = solver.gpu_types[matching_instances[0]].count
            if args.eval_tp_degree > per_instance_gpu_count:
                raise ValueError(
                    f"TP degree {args.eval_tp_degree} exceeds GPUs per instance "
                    f"({per_instance_gpu_count}) for family '{family}'"
                )

            total_layers = solver.config.num_decoder_layers
            base = total_layers // pp_stages
            remainder = total_layers % pp_stages

            stages = []
            start_layer = 1
            for idx in range(pp_stages):
                segment_size = base + (1 if idx < remainder else 0)
                end_layer = start_layer + segment_size - 1
                stages.append({
                    "gpu_type": matching_instances[idx],
                    "tp_degree": args.eval_tp_degree,
                    "gpu_ids": list(range(args.eval_tp_degree)),
                    "start_layer": start_layer,
                    "end_layer": end_layer
                })
                start_layer = end_layer + 1

            placement = {
                "batch_size": args.min_batch_size,
                "sequence_length": args.sequence_length,
                "output_length": args.output_length,
                "workload_phase": args.workload_phase,
                "stages": stages
            }

            try:
                solver.evaluate_manual_placement(placement)
                solver.print_solution()

                output_file = os.path.join(output_dir, 'placement_metrics.json')
                solver.save_solution(output_file)

                csv_file = os.path.join(output_dir, 'placement_metrics.csv')
                solver.save_solution_csv(csv_file)
            except ValueError as exc:
                detailed_error = solver._describe_eval_failure(placement, exc)
                failure_payload = {
                    "config": {
                        "model_name": solver.config.model_name,
                        "num_decoder_layers": solver.config.num_decoder_layers,
                        "sequence_length": solver.config.sequence_length,
                        "output_length": solver.config.output_length,
                        "min_batch_size": solver.config.min_batch_size,
                        "max_batch_size": solver.config.max_batch_size,
                        "d_model": solver.config.d_model,
                        "d_hidden": solver.config.d_hidden,
                        "max_pipeline_stages": solver.config.max_pipeline_stages,
                        "min_memory_utilization": solver.config.min_memory_utilization
                    },
                    "placement": placement,
                    "solution": {
                        "status": "INFEASIBLE",
                        "error": detailed_error
                    }
                }
                with open(os.path.join(output_dir, "placement_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(_json_safe(failure_payload), f, indent=2)
                logger.error(f"Eval placement infeasible: {detailed_error}")
            return 0

        if args.evaluate_placement:
            if not args.output_dir and not args.eval_output_root:
                raise ValueError("Eval mode requires --output-dir or --eval-output-root")
            if args.output_dir:
                output_dir = args.output_dir
            else:
                result_dir_name = f"eval_placement-{time.strftime('%Y%m%d_%H%M%S')}"
                output_dir = _resolve_eval_output_dir(args, args.config_dir, result_dir_name)
            os.makedirs(output_dir, exist_ok=True)
            config_src = os.path.join(args.config_dir, "config.csv")
            if os.path.exists(config_src):
                shutil.copy2(config_src, os.path.join(output_dir, "config.csv"))

            solver = LLMPlacementSolverWithTP(
                args.config_dir,
                tp_configuration=None,
                **solver_kwargs,
                skip_gurobi=True
            )
            with open(args.evaluate_placement, "r", encoding="utf-8") as f:
                placement = json.load(f)
            try:
                solver.evaluate_manual_placement(placement)
                solver.print_solution()

                output_file = os.path.join(output_dir, 'placement_metrics.json')
                solver.save_solution(output_file)

                csv_file = os.path.join(output_dir, 'placement_metrics.csv')
                solver.save_solution_csv(csv_file)
            except ValueError as exc:
                detailed_error = solver._describe_eval_failure(placement, exc)
                failure_payload = {
                    "config": {
                        "model_name": solver.config.model_name,
                        "num_decoder_layers": solver.config.num_decoder_layers,
                        "sequence_length": solver.config.sequence_length,
                        "output_length": solver.config.output_length,
                        "min_batch_size": solver.config.min_batch_size,
                        "max_batch_size": solver.config.max_batch_size,
                        "d_model": solver.config.d_model,
                        "d_hidden": solver.config.d_hidden,
                        "max_pipeline_stages": solver.config.max_pipeline_stages,
                        "min_memory_utilization": solver.config.min_memory_utilization
                    },
                    "placement": placement,
                    "solution": {
                        "status": "INFEASIBLE",
                        "error": detailed_error
                    }
                }
                with open(os.path.join(output_dir, "placement_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(_json_safe(failure_payload), f, indent=2)
                logger.error(f"Eval placement infeasible: {detailed_error}")
            return 0

        if args.homogeneous == 1:
            # Homogeneous placement mode - bypasses MILP, uses fast enumeration
            logger.info("Using HOMOGENEOUS mode (fast enumeration, no MILP)")

            # Create solver with skip_gurobi=True since we don't need MILP
            solver = LLMPlacementSolverWithTP(
                args.config_dir,
                tp_configuration=None,
                **solver_kwargs,
                skip_gurobi=True
            )

            success = solver.solve_homogeneous()

            if success:
                solver.print_solution()
                output_file = os.path.join(args.output_dir, 'solution_homogeneous.json')
                solver.save_solution(output_file)

                # Also save CSV summary
                csv_file = os.path.join(args.output_dir, 'solution_summary.csv')
                solver.save_solution_csv(csv_file)
            else:
                logger.error("Failed to find valid homogeneous placement")
                return 1

            end_time = time.time()
            logger.info(f"Total time: {end_time - start_time:.0f} seconds")
            return 0

        if args.search_all_tp:
            # Search all TP configurations
            best_solution, best_tp_config = solve_all_tp_configurations(
                args.config_dir, **solver_kwargs
            )
            
            if best_solution:
                # Print summary
                logger.info("="*100)
                logger.info("BEST SOLUTION SUMMARY")
                logger.info("="*100)
                logger.info(f"Best TP Configuration: {best_tp_config}")
                logger.info(f"Throughput: {best_solution['objective_value']:.2f} tokens/sec")
                logger.info(f"Pipeline Stages: {best_solution['num_pipeline_stages']}")
                logger.info("="*100)
                
                # Save best solution
                output_file = os.path.join(args.output_dir, 'solution_best_tp.json')
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
            # Single TP configuration (heterogeneous MILP)
            tp_config = None
            if args.tp_config:
                tp_config = json.loads(args.tp_config)

            # Optionally run homogeneous solver first for warm start
            homogeneous_solution = None
            if args.warm_start_homogeneous == 1:
                logger.info("="*80)
                logger.info("PHASE 1: Running homogeneous solver for MIP warm start")
                logger.info("="*80)

                # Create a separate solver for homogeneous (with skip_gurobi=True)
                homogeneous_solver = LLMPlacementSolverWithTP(
                    args.config_dir,
                    tp_configuration=None,
                    **solver_kwargs,
                    skip_gurobi=True
                )

                if homogeneous_solver.solve_homogeneous():
                    homogeneous_solution = homogeneous_solver.solution
                    logger.info(f"Homogeneous baseline: ${homogeneous_solution['cost_per_token']*1e6:.6f}/M tokens, "
                               f"{homogeneous_solution['throughput_tokens_per_sec']:.0f} tokens/sec")

                    # Save homogeneous solution for reference
                    homo_output = os.path.join(args.output_dir, 'solution_homogeneous_warmstart.json')
                    homogeneous_solver.save_solution(homo_output)
                    logger.info(f"Homogeneous baseline saved to {homo_output}")
                else:
                    logger.warning("Homogeneous solver failed, proceeding without warm start")

                logger.info("="*80)
                logger.info("PHASE 2: Running heterogeneous MILP with warm start")
                logger.info("="*80)

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

            # Set warm start from homogeneous solution if available
            if homogeneous_solution is not None:
                warm_start_success = solver.set_warm_start_from_homogeneous(homogeneous_solution)
                if warm_start_success:
                    logger.info("MIP warm start successfully set from homogeneous solution")
                else:
                    logger.warning("Could not fully set MIP warm start (some segments not matched)")

            # Choose optimization method
            if args.method == 'weighted':
                logger.info("Using WEIGHTED method (fast, approximate)")
                success = solver.solve()
            elif args.method == 'enumeration':
                logger.info("Using ENUMERATION method (slow, guaranteed optimal)")
                success = solver.solve_optimal_cost_per_token()

            if success:
                solver.print_solution()
                output_file = os.path.join(args.output_dir, 'solution_with_tp.json')
                solver.save_solution(output_file)

                # Also save CSV summary
                csv_file = os.path.join(args.output_dir, 'solution_summary.csv')
                solver.save_solution_csv(csv_file)

                # Log comparison with homogeneous baseline if available
                if homogeneous_solution is not None:
                    homo_cost = homogeneous_solution['cost_per_token'] * 1e6
                    hetero_cost = solver.solution['cost_per_token'] * 1e6
                    improvement = (homo_cost - hetero_cost) / homo_cost * 100 if homo_cost > 0 else 0

                    logger.info("="*80)
                    logger.info("COMPARISON: Homogeneous vs Heterogeneous")
                    logger.info("="*80)
                    logger.info(f"Homogeneous: ${homo_cost:.6f}/M tokens, "
                               f"{homogeneous_solution['throughput_tokens_per_sec']:.0f} tokens/sec")
                    logger.info(f"Heterogeneous: ${hetero_cost:.6f}/M tokens, "
                               f"{solver.solution['throughput_tokens_per_sec']:.0f} tokens/sec")
                    logger.info(f"Improvement: {improvement:.2f}% cost reduction")
                    if improvement < 0:
                        logger.warning("Heterogeneous solver found WORSE solution than homogeneous baseline!")
                    logger.info("="*80)
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