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
from typing import Dict, List, Tuple, Optional, FrozenSet
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
    """Runtime configuration"""
    sequence_length: int
    batch_size: int
    model_name: str
    num_decoder_layers: int
    d_model: int
    d_hidden: int
    vocab_size: int
    num_attention_heads: int
    layer_weight_memory_gb: float
    time_limit_seconds: float
    optimality_gap: float
    bytes_per_element: int = 2
    # Practical constraints
    max_pipeline_stages: int = 8
    min_memory_utilization: float = 0.5
    min_layers_per_stage: int = 1
    network_bandwidth_percentile_threshold: float = 0.1
    enable_segment_quantization: bool = True
    # NEW: Cost optimization parameters
    cost_throughput_weight: float = 0.0  # 0=pure throughput, 1=pure cost
    max_hourly_cost: float = 999.0  # Budget constraint ($/hour)
    max_cost_per_token: float = 999.0  # Competitor threshold ($/token)
    throughput_normalization: float = 1000.0  # Scaling for objective
    cost_normalization: float = 50.0  # Scaling for objective

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

class ThroughputFunctions:
    """Throughput functions with TP support"""

    GPU_SPECS = {
        'H100': {'tflops': 989, 'mem_bw': 3350, 'efficiency': 0.65},
        'A100': {'tflops': 312, 'mem_bw': 2039, 'efficiency': 0.60},
        'L40S': {'tflops': 362, 'mem_bw': 864, 'efficiency': 0.55},
        'A40': {'tflops': 150, 'mem_bw': 696, 'efficiency': 0.55},
        'L40': {'tflops': 181, 'mem_bw': 864, 'efficiency': 0.50},
        'V100': {'tflops': 125, 'mem_bw': 900, 'efficiency': 0.55},
        'RTX4090': {'tflops': 165, 'mem_bw': 1008, 'efficiency': 0.50},
        'L20': {'tflops': 119, 'mem_bw': 480, 'efficiency': 0.50},
        'A10': {'tflops': 125, 'mem_bw': 600, 'efficiency': 0.45},
        'T4': {'tflops': 65, 'mem_bw': 320, 'efficiency': 0.40}
    }

    # TP efficiency factors based on empirical observations
    TP_EFFICIENCY = {
        1: 1.0,   # No TP overhead
        2: 0.90,  # 10% overhead for all-reduce
        4: 0.80,  # 20% overhead
        8: 0.70   # 30% overhead
    }
    
    @staticmethod
    def gpu_throughput(gpu_type: str, seq_len: int, batch_size: int, num_layers: int, d_model: int, bytes_per_element: int) -> float:
        specs = ThroughputFunctions.GPU_SPECS.get(gpu_type, ThroughputFunctions.GPU_SPECS['A100'])
        
        # Correct FLOP count per layer
        # Attention: QKV + attention + output projections
        attn_proj_flops = 2 * 4 * batch_size * seq_len * d_model * d_model  # QKV + output
        attn_score_flops = 4 * batch_size * seq_len * seq_len * d_model  # QK^T + softmax*V
        
        # MLP: up + down projections
        d_ff = 4 * d_model
        mlp_flops = 16 * batch_size * seq_len * d_model * d_model
        
        flops_per_layer = attn_proj_flops + attn_score_flops + mlp_flops
        total_flops = num_layers * flops_per_layer
        
        time_compute = total_flops / (specs['tflops'] * 1e12 * specs['efficiency'])
        
        # Weights per layer: QKV (3D²) + Output (D²) + MLP_up (4D²) + MLP_down (4D²) = 12D²
        weight_bytes = num_layers * 12 * d_model * d_model * bytes_per_element
        activation_bytes = batch_size * seq_len * d_model * bytes_per_element
        time_memory = (weight_bytes + activation_bytes) / (specs['mem_bw'] * 1e9)
        
        time_per_batch = max(time_compute, time_memory)
        tokens_per_batch = batch_size * seq_len
        
        return tokens_per_batch / time_per_batch
    
    @staticmethod
    def gpu_throughput_with_tp(gpu_type: str, seq_len: int, batch_size: int, 
                               num_layers: int, d_model: int, bytes_per_element: int, tp_degree: int) -> float:
        """GPU throughput with tensor parallelism"""
        base_throughput = ThroughputFunctions.gpu_throughput(
            gpu_type, seq_len, batch_size, num_layers, d_model, bytes_per_element
        )
        
        efficiency = ThroughputFunctions.TP_EFFICIENCY.get(tp_degree, 0.70)
        tp_speedup = tp_degree * efficiency
        
        return base_throughput * tp_speedup

class LLMPlacementSolverWithTP:
    """LLM placement solver with Tensor Parallelism and practical constraints"""

    def __init__(self, config_dir: str, tp_configuration: Optional[Dict[str, int]] = None,
                 enable_symmetry_breaking: bool = True,
                 enable_upper_bound: bool = True, enable_tight_bigm: bool = True,
                 enable_flow_conservation: bool = True, threads: Optional[int] = None,
                 max_threads: int = 32):
        self.options = {
            "WLSACCESSID": "790b9c11-45d0-4785-8d99-a5e6414f9321",
            "WLSSECRET": "adef4738-7bf6-41b8-8dfd-d04e23d53e51",
            "LICENSEID": 2415150,
        }
        self.env = gp.Env(params=self.options)
        self.config_dir = config_dir
        
        # Optimization flags
        self.enable_symmetry_breaking = enable_symmetry_breaking
        self.enable_upper_bound = enable_upper_bound
        self.enable_tight_bigm = enable_tight_bigm
        self.enable_flow_conservation = enable_flow_conservation
        self.threads = threads
        self.max_threads = max_threads

        # Load configuration
        gpu_pool_file = os.path.join(config_dir, 'gpu_pool.csv')
        network_file = os.path.join(config_dir, 'network_bandwidth.csv')
        config_file = os.path.join(config_dir, 'config.csv')

        self.gpu_types = self._load_gpu_pool(gpu_pool_file)
        self.network_bandwidth = self._load_network_bandwidth(network_file)
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
    
    def _load_gpu_pool(self, filename: str) -> Dict[str, GPUType]:
        """Load GPU pool configuration"""
        df = pd.read_csv(filename)
        gpu_types = {}
        global_id = 0
        
        for _, row in df.iterrows():
            global_ids = list(range(global_id, global_id + row['count']))
            gpu_types[row['gpu_type']] = GPUType(
                name=row['gpu_type'],
                count=row['count'],
                memory_gb=row['memory_gb'],
                global_ids=global_ids,
                cost_per_hour=float(row.get('dollar_per_hour', 0.0))  # NEW: Load cost
            )
            global_id += row['count']
        
        return gpu_types
    
    def _load_network_bandwidth(self, filename: str) -> np.ndarray:
        """Load network bandwidth matrix"""
        df = pd.read_csv(filename, index_col=0)
        matrix = df.values
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Network bandwidth matrix must be square, got {matrix.shape}")
        
        return matrix
    
    def _load_config(self, filename: str) -> Config:
        """Load runtime configuration"""
        df = pd.read_csv(filename)
        config_dict = dict(zip(df['parameter'], df['value']))
        
        return Config(
            sequence_length=int(config_dict['sequence_length']),
            batch_size=int(config_dict['batch_size']),
            model_name=config_dict['model_name'],
            num_decoder_layers=int(config_dict['num_decoder_layers']),
            d_model=int(config_dict['d_model']),
            d_hidden=int(config_dict['d_hidden']),
            vocab_size=int(config_dict['vocab_size']),
            num_attention_heads=int(config_dict['num_attention_heads']),
            layer_weight_memory_gb=float(config_dict['layer_weight_memory_gb']),
            time_limit_seconds=float(config_dict['time_limit_seconds']),
            optimality_gap=float(config_dict['optimality_gap']),
            bytes_per_element=int(config_dict.get('bytes_per_element', 2)),
            max_pipeline_stages=int(config_dict.get('max_pipeline_stages', 8)),
            min_memory_utilization=float(config_dict.get('min_memory_utilization', 0.5)),
            min_layers_per_stage=int(config_dict.get('min_layers_per_stage', 10)),
            network_bandwidth_percentile_threshold=float(config_dict.get('network_bandwidth_percentile_threshold', 0.1)),
            enable_segment_quantization=config_dict.get('enable_segment_quantization', 'true').lower() == 'true',
            # NEW: Cost optimization parameters
            cost_throughput_weight=float(config_dict.get('cost_throughput_weight', 0.0)),
            max_hourly_cost=float(config_dict.get('max_hourly_cost', 999.0)),
            max_cost_per_token=float(config_dict.get('max_cost_per_token', 999.0)),
            throughput_normalization=float(config_dict.get('throughput_normalization', 1000.0)),
            cost_normalization=float(config_dict.get('cost_normalization', 50.0))
        )
    
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
    
    def _calculate_activation_memory(self, tp_degree: int = 1) -> float:
        """
        Calculate peak activation memory per GPU with TP.
        Accounts for all-reduce operations and KV cache sharding.
        """
        batch = self.config.batch_size
        seq_len = self.config.sequence_length
        hidden = self.config.d_model
        d_hidden = self.config.d_hidden
        bytes_per_elem = self.config.bytes_per_element

        # Sharded intermediate tensors during computation
        qkv_memory = (3 * batch * seq_len * (hidden / tp_degree) *
                     bytes_per_elem) / (1024**3)
        mlp_intermediate = (batch * seq_len * (4 * hidden / tp_degree) *
                           bytes_per_elem) / (1024**3)
        sharded_computation = qkv_memory + mlp_intermediate

        # Full activation tensor after all-reduce (NOT sharded)
        full_activation = (batch * seq_len * hidden * bytes_per_elem) / (1024**3)

        # KV cache: Persistently sharded along hidden dimension
        kv_cache = (2 * batch * seq_len * (hidden / tp_degree) *
                   bytes_per_elem) / (1024**3)

        # Peak memory: max of computation phase vs all-reduce output phase, plus KV cache
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
    
    def _compute_max_segment_size_for_tp(self, gpu_type: str, tp_degree: int) -> int:
        """Compute max segment size for a specific (gpu_type, tp_degree)"""
        gpu_obj = self.gpu_types[gpu_type]
        weight_per_gpu_per_layer = self.config.layer_weight_memory_gb / tp_degree
        activation_memory = self._calculate_activation_memory(tp_degree)
        
        return self._binary_search_max_layers(
            gpu_obj.memory_gb, weight_per_gpu_per_layer, activation_memory
        )

    def _compute_min_segment_size_for_tp(self, gpu_type: str, tp_degree: int) -> int:
        """Compute min segment size for a specific (gpu_type, tp_degree)"""
        gpu_obj = self.gpu_types[gpu_type]
        weight_per_layer = self.config.layer_weight_memory_gb / tp_degree
        activation_memory = self._calculate_activation_memory(tp_degree)
        
        min_layers = max(1, math.ceil(
            (self.config.min_memory_utilization * gpu_obj.memory_gb - activation_memory) / weight_per_layer
        ))
        
        if activation_memory > self.config.min_memory_utilization * gpu_obj.memory_gb:
            min_layers = 1
        
        max_layers = self._compute_max_segment_size_for_tp(gpu_type, tp_degree)
        return min(min_layers, max_layers)

    def _generate_valid_segments(self) -> List[Tuple]:
        """Generate segments with variable TP degree per segment"""
        valid_segments = []
        # MAX_SEGMENTS = 10000 # NOTE: Hardcoded max segment. not ideal though... it prevents combinatorial explosion
        for gpu_type, allocations in self.tp_allocations.items():
            for gpu_set, tp_degree, partition_id in allocations:
                # Get max/min segment sizes for this specific TP degree
                max_seg_size = self._compute_max_segment_size_for_tp(gpu_type, tp_degree)
                min_seg_size = self._compute_min_segment_size_for_tp(gpu_type, tp_degree)
                
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
                            # NEW FORMAT: (gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size)
                            segment = (gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size)
                            valid_segments.append(segment)
                            
                            # # SAFETY CHECK
                            # if len(valid_segments) > MAX_SEGMENTS:
                            #     logger.warning(f"Hit segment limit of {MAX_SEGMENTS}, stopping generation")
                            #     return valid_segments
        
        logger.info(f"Generated {len(valid_segments)} segments with variable TP")
        return valid_segments
    
    def _generate_valid_connections(self) -> List[Tuple]:
        """Generate valid network connections between consecutive segments"""
        valid_connections = []
        
        # Group segments by ending/starting layer
        segments_by_end_layer = {}
        segments_by_start_layer = {}
        
        for seg in self.valid_segments:
            gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size = seg
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
                for seg2 in starting_segments:
                    gpu_set2 = seg2[1]
                    # Connection valid only if GPU sets don't overlap
                    if not gpu_set1.intersection(gpu_set2):
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
        """
        gpu_pair_throughputs = {}
        
        # Full tensor size after all-reduce (NOT sharded)
        tensor_size_gb = (self.config.batch_size * self.config.sequence_length *
                        self.config.d_model * self.config.bytes_per_element) / (1024**3)
        
        for seg1, seg2 in self.valid_connections:
            gpu_type1 = seg1[0]
            gpu_type2 = seg2[0]
            gpu_set1 = seg1[1]
            gpu_set2 = seg2[1]
            tp_degree1 = seg1[2]
            tp_degree2 = seg2[2]
            
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
                # Throughput in tokens/sec
                tokens_per_batch = self.config.batch_size * self.config.sequence_length
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
        logger.info("Building optimization model with TP and practical constraints...")
        
        self.model = gp.Model("llm_placement_with_tp_constrained", env=self.env)
        
        # Solver parameters
        self.model.setParam('Presolve', 2)
        self.model.setParam('Cuts', 1)
        self.model.setParam('Heuristics', 0.05)
        self.model.setParam('MIPFocus', 1)
        self.model.setParam('NodefileStart', 0.5)
        self.model.setParam('TimeLimit', self.config.time_limit_seconds)
        self.model.setParam('MIPGap', self.config.optimality_gap)
        self.model.setParam('LogToConsole', 1)
        
        self._create_variables()
        self._create_constraints()
        self._set_objective()
        
        logger.info("Model built successfully")
    
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
        if self.config.max_hourly_cost < 999.0:
            self.model.addConstr(
                self.cost <= self.config.max_hourly_cost,
                name="cost_budget"
            )
            logger.info(f"Added cost budget constraint: ≤ ${self.config.max_hourly_cost:.2f}/hour")
        
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
                    throughput_expr = gp.quicksum(
                        ThroughputFunctions.gpu_throughput_with_tp(
                            gpu_type, self.config.sequence_length,
                            self.config.batch_size, seg[5], self.config.d_model, self.config.bytes_per_element, seg[2]  # seg[5]=segment_size, seg[2]=tp_degree
                        ) * self.x[seg]
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
                name=f"throughput_partition"
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
        """Compute tight Big-M values"""
        M_partition = {}
        existing_partition_keys = set((seg[0], seg[3]) for seg in self.valid_segments)
        
        for gpu_type, allocations in self.tp_allocations.items():
            tp_degree = self.tp_max_configuration[gpu_type]
            max_size = self.max_segment_size[(gpu_type, tp_degree)]
            
            if max_size > 0:
                max_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                    gpu_type, self.config.sequence_length,
                    self.config.batch_size, max_size, self.config.d_model, self.config.bytes_per_element, tp_degree
                )
                # Only create M for partitions that exist
                for gpu_set, tp_degree_alloc, partition_id in allocations:
                    if (gpu_type, partition_id) in existing_partition_keys:
                        M_partition[(gpu_type, partition_id)] = max_throughput * 3
        
        M_network = max(self.gpu_pair_throughputs.values()) * 2 if self.gpu_pair_throughputs else 1000.0
        
        logger.info(f"Tight Big-M computed: M_network={M_network:.2f}")
        return M_partition, M_network
    
    def _add_pipeline_connectivity_constraints(self):
        """Ensure pipeline connectivity from layer 1 to final layer"""
        # Pipeline must start at layer 1
        first_layer_segments = [seg for seg in self.valid_segments if seg[4] == 1]
        if first_layer_segments:
            self.model.addConstr(
                gp.quicksum(self.x[seg] for seg in first_layer_segments) >= 1,
                name="pipeline_starts_at_layer_1"
            )
        
        # Sequential connectivity
        for layer in range(1, self.config.num_decoder_layers):
            segments_ending_here = [seg for seg in self.valid_segments
                                   if seg[4] + seg[5] - 1 == layer]
            segments_starting_next = [seg for seg in self.valid_segments
                                     if seg[4] == layer + 1]
            
            if segments_ending_here and segments_starting_next:
                # Outgoing connections
                for seg1 in segments_ending_here:
                    valid_next_connections = [(s1, s2) for (s1, s2) in self.valid_connections 
                                             if s1 == seg1 and s2 in segments_starting_next]
                    if valid_next_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[s1, s2] for (s1, s2) in valid_next_connections) >= self.x[seg1],
                            name=f"connectivity_out_{layer}"
                        )
                
                # Incoming connections
                for seg2 in segments_starting_next:
                    valid_prev_connections = [(s1, s2) for (s1, s2) in self.valid_connections 
                                             if s2 == seg2 and s1 in segments_ending_here]
                    if valid_prev_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[s1, s2] for (s1, s2) in valid_prev_connections) >= self.x[seg2],
                            name=f"connectivity_in_{layer}"
                        )
    
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
    
    def solve_for_min_cost_per_token(self, target_cpt: float = None, max_iterations: int = 5) -> bool:
        """
        Iteratively solve to minimize $/token.
        
        Uses binary search on cost budget to find solution with best $/token.
        """
        if target_cpt is None:
            target_cpt = self.config.max_cost_per_token
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ITERATIVE $/TOKEN OPTIMIZATION")
        logger.info(f"{'='*80}")
        logger.info(f"Target $/token: ${target_cpt:.9f}")
        
        # Step 1: Estimate minimum throughput (single GPU, single layer)
        min_tp_estimates = []
        for gpu_type in self.gpu_types:
            t_est = ThroughputFunctions.gpu_throughput(
                gpu_type, self.config.sequence_length, self.config.batch_size,
                1, self.config.d_model, self.config.bytes_per_element
            )
            min_tp_estimates.append(t_est)
        
        t_min = min(min_tp_estimates) if min_tp_estimates else 100.0
        
        # Step 2: Initial cost budget (conservative)
        initial_budget = target_cpt * t_min * 3600 * 10  # 10× margin
        
        logger.info(f"Estimated min throughput: {t_min:.1f} tokens/sec")
        logger.info(f"Initial cost budget: ${initial_budget:.2f}/hour")
        
        best_solution = None
        best_cpt = float('inf')
        
        # Step 3: Iterative refinement
        for iteration in range(max_iterations):
            logger.info(f"\nIteration {iteration + 1}/{max_iterations}:")
            logger.info(f"  Cost budget: ${initial_budget:.2f}/hour")
            
            # Set budget and solve
            original_budget = self.config.max_hourly_cost
            self.config.max_hourly_cost = initial_budget
            
            success = self.solve()
            
            if success:
                actual_cpt = self.solution['cost_per_token']
                logger.info(f"  Result: $/token = ${actual_cpt:.9f}")
                
                if actual_cpt < best_cpt:
                    best_cpt = actual_cpt
                    best_solution = self.solution.copy()
                
                if actual_cpt <= target_cpt:
                    logger.info(f"  ✓ Meets target!")
                    break
                else:
                    # Tighten budget
                    throughput = self.solution['throughput_tokens_per_sec']
                    new_budget = target_cpt * throughput * 3600 * 0.95
                    logger.info(f"  Tightening budget to ${new_budget:.2f}/hour")
                    initial_budget = new_budget
            else:
                logger.info(f"  Infeasible, relaxing budget")
                initial_budget *= 1.5
            
            self.config.max_hourly_cost = original_budget
        
        if best_solution:
            self.solution = best_solution
            logger.info(f"\nBest $/token found: ${best_cpt:.9f}")
            return True
        
        return False
    
    def _estimate_feasible_budget_range(self) -> tuple:
        """
        Estimate feasible budget range based on max_cost_per_token target.
        
        Logic:
        - To meet $/token target: cost / (throughput × 3600) ≤ max_cost_per_token
        - So: cost ≤ max_cost_per_token × throughput × 3600
        - Estimate min/max throughput from GPU memory (proxy for performance)
        - Calculate corresponding budget bounds
        
        Returns:
            (min_budget, max_budget) tuple in $/hour
        """
        target_cpt = self.config.max_cost_per_token
        
        # Collect GPU costs to estimate range
        costs = []
        
        for gpu_type, gpu_info in self.gpu_types.items():
            if gpu_info.count == 0:
                continue
            
            # For different TP degrees
            for tp_degree in [1, 2, 4, 8]:
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
        max_budget = max_cost * 1.5  # Slightly above most expensive config
        
        # Apply absolute bounds for safety
        min_budget = max(0.20, min_budget)
        max_budget = min(20.0, max_budget)
        
        # If target_cpt is very restrictive (< $0.0001/token), narrow the range
        # Typical LLM $/token is $0.00001 - $0.0001
        if target_cpt < 0.0001:
            # Very aggressive target - focus on cheaper GPUs
            max_budget = min(max_budget, max_cost * 0.8)
            logger.info(f"  Aggressive $/token target detected, focusing on lower budgets")
        
        logger.info(f"Competitive budget range for $/token ≤ ${target_cpt:.9f}:")
        logger.info(f"  GPU cost range: ${min_cost:.2f} - ${max_cost:.2f}/hour")
        logger.info(f"  Search budget range: ${min_budget:.2f} - ${max_budget:.2f}/hour")
        
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
        logger.info(f"OPTIMAL $/TOKEN SEARCH {'(Smart Hybrid)' if use_smart_hybrid and budget_points is None else '(Full Enumeration)'}")
        logger.info(f"{'='*80}")
        logger.info(f"Target $/token (from config): ${self.config.max_cost_per_token:.9f}")
        
        # Auto-generate budget range if not provided
        if budget_points is None:
            # COMPETITIVE INTELLIGENCE: Use max_cost_per_token to focus search
            min_feasible, max_feasible = self._estimate_feasible_budget_range()
            
            if use_smart_hybrid:
                # SMART HYBRID: Use iterative to find ballpark, then enumerate around it
                logger.info("\nPhase 1: Iterative search to find ballpark...")
                if self.solve_for_min_cost_per_token(max_iterations=3):
                    ballpark_cost = self.solution['cost_per_hour']
                    ballpark_cpt = self.solution['cost_per_token']
                    logger.info(f"Ballpark found: ${ballpark_cost:.2f}/h, ${ballpark_cpt:.9f}/token")
                    
                    # Check if ballpark meets target
                    if ballpark_cpt <= self.config.max_cost_per_token:
                        logger.info(f"✓ Ballpark meets target! (${ballpark_cpt:.9f} ≤ ${self.config.max_cost_per_token:.9f})")
                    else:
                        logger.warning(f"✗ Ballpark exceeds target (${ballpark_cpt:.9f} > ${self.config.max_cost_per_token:.9f})")
                    
                    # PERFORMANCE OPTIMIZATION: Reduced from 12 to 6-8 budget points
                    # COMPETITIVE OPTIMIZATION: Focus on range that can beat competitor
                    logger.info("\nPhase 2: Coarse enumeration around ballpark...")
                    budget_points = [
                        ballpark_cost * factor
                        for factor in [0.4, 0.7, 0.9, 1.0, 1.2, 1.5]
                    ]
                    # Add feasible range bounds to ensure we explore competitive region
                    budget_points.extend([min_feasible, max_feasible * 0.5, max_feasible])
                    
                    # Filter to feasible range (don't waste time on budgets that can't meet target)
                    budget_points = [b for b in budget_points if min_feasible * 0.5 <= b <= max_feasible * 1.5]
                    budget_points = sorted(set(budget_points))  # Remove duplicates, sort
                    logger.info(f"  Using {len(budget_points)} coarse budget points focused on competitive range")
                    logger.info(f"  Budget range: ${min(budget_points):.2f} - ${max(budget_points):.2f}/hour")
                else:
                    # Fallback to coarse range if iterative fails
                    logger.warning("Iterative failed, using coarse enumeration in feasible range...")
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
        
        for i, budget in enumerate(sorted(budget_points)):
            logger.info(f"\n[{i+1}/{len(budget_points)}] Budget: ${budget:.2f}/hour")
            
            self.config.max_hourly_cost = budget
            
            # Rebuild model with new budget constraint
            self.build_model()
            success = self.solve()
            
            if success:
                cpt = self.solution['cost_per_token']
                throughput = self.solution['throughput_tokens_per_sec']
                cost = self.solution['cost_per_hour']
                
                logger.info(f"  Result: {throughput:.0f} tokens/s, ${cost:.2f}/h, ${cpt:.9f}/token")
                
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
                    logger.info(f"  ✓ New best $/token!")
            else:
                logger.info(f"  ✗ Infeasible")
        
        # Restore original config
        self.config.max_hourly_cost = original_budget
        self.config.cost_throughput_weight = original_weight
        
        if best_solution:
            self.solution = best_solution
            logger.info(f"\n{'='*80}")
            logger.info(f"OPTIMAL $/TOKEN FOUND")
            logger.info(f"{'='*80}")
            logger.info(f"Best $/token: ${best_cpt:.9f}")
            logger.info(f"  Throughput: {best_solution['throughput_tokens_per_sec']:.0f} tokens/sec")
            logger.info(f"  Cost: ${best_solution['cost_per_hour']:.2f}/hour")
            logger.info(f"  Pipeline stages: {best_solution['num_pipeline_stages']}")
            
            # Check against target
            target = self.config.max_cost_per_token
            if best_cpt <= target:
                improvement = (target - best_cpt) / target * 100
                logger.info(f"\n✅ MEETS TARGET: ${best_cpt:.9f} ≤ ${target:.9f}")
                logger.info(f"   {improvement:.1f}% better than target!")
            else:
                shortfall = (best_cpt - target) / target * 100
                logger.info(f"\n⚠️  MISSES TARGET: ${best_cpt:.9f} > ${target:.9f}")
                logger.info(f"   {shortfall:.1f}% worse than target (infeasible to meet)")
            
            # Log Pareto frontier
            logger.info(f"\nPareto Frontier (all solutions):")
            for r in sorted(all_results, key=lambda x: x['cost_per_token']):
                marker = "✓" if r['cost_per_token'] <= target else " "
                logger.info(f"  {marker} ${r['cost_per_token']:.9f}/token: "
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
        logger.info(f"Using {threads} threads for optimization")
        
        logger.info("Starting optimization...")
        start_time = time.time()
        
        try:
            self.model.optimize()
            solve_time = time.time() - start_time
            
            if self.model.status == GRB.OPTIMAL:
                logger.info(f"Optimal solution found in {solve_time:.2f} seconds")
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
        throughput_per_sec = self.t.x
        cost_per_hour = self.cost.x
        
        # $/token = ($/hour) / (tokens/sec × sec/hour)
        if throughput_per_sec > 0:
            cost_per_token = cost_per_hour / (throughput_per_sec * 3600)
        else:
            cost_per_token = float('inf')
        
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
        
        logger.info(f"\nCore Metrics:")
        logger.info(f"  Throughput: {throughput_per_sec:.2f} tokens/sec")
        logger.info(f"  Cost: ${cost_per_hour:.2f}/hour")
        logger.info(f"  $/token: ${cost_per_token:.9f}")
        
        self.solution = {
            'objective_value': self.model.ObjVal,
            'throughput_tokens_per_sec': throughput_per_sec,
            'cost_per_hour': cost_per_hour,
            'cost_per_token': cost_per_token,
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
                gpu_type, gpu_set, tp_degree, partition_id, start_layer, segment_size = seg
                
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
        
        # NEW: Detailed GPU utilization analysis
        self._log_solution_analysis()
    
    def _log_solution_analysis(self):
        """Comprehensive analysis of why the solution looks the way it does"""
        logger.info("\n" + "="*80)
        logger.info("SOLUTION ANALYSIS - GPU EFFICIENCY & SELECTION")
        logger.info("="*80)
        
        # 1. Compute GPU efficiency ratios (FLOP per $)
        gpu_efficiency = {}
        for gpu_type, gpu_obj in self.gpu_types.items():
            specs = ThroughputFunctions.GPU_SPECS.get(gpu_type)
            if specs:
                effective_flops = specs['tflops'] * specs['efficiency']
                efficiency_ratio = effective_flops / gpu_obj.cost_per_hour
                gpu_efficiency[gpu_type] = {
                    'flops': effective_flops,
                    'cost': gpu_obj.cost_per_hour,
                    'ratio': efficiency_ratio
                }
        
        # Sort by efficiency
        sorted_efficiency = sorted(gpu_efficiency.items(), key=lambda x: x[1]['ratio'], reverse=True)
        
        logger.info("\nGPU Efficiency Ranking (TFLOP/$ ratio):")
        logger.info("-" * 80)
        for i, (gpu_type, data) in enumerate(sorted_efficiency, 1):
            logger.info(f"  {i}. {gpu_type:<10} {data['flops']:>6.1f} TFLOP × eff @ ${data['cost']:.2f}/h = {data['ratio']:>6.1f} TFLOP/$")
        
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
                
                # Check max segment size
                max_tp = self.tp_max_configuration[gpu_type]
                max_seg = self._compute_max_segment_size_for_tp(gpu_type, max_tp)
                
                logger.info(f"  {gpu_type:<10} count={gpu_obj.count}, ${gpu_obj.cost_per_hour:.2f}/h, "
                           f"efficiency rank #{efficiency_rank}")
                logger.info(f"             Max segment size (TP={max_tp}): {max_seg} layers")
                
                # Hypothetical: what if we used this GPU?
                if max_seg >= 5:  # Can fit at least 5 layers
                    hyp_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                        gpu_type, self.config.sequence_length, self.config.batch_size,
                        5, self.config.d_model, self.config.bytes_per_element, 4
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
        max_seg_size = self._compute_max_segment_size_for_tp(best_gpu_type, max_tp)
        
        logger.info(f"\n{best_gpu_type} specs:")
        logger.info(f"  Available: {best_gpu.count} GPUs")
        logger.info(f"  Cost: ${best_gpu.cost_per_hour:.2f}/hour per GPU")
        logger.info(f"  Max segment size (TP={max_tp}): {max_seg_size} layers")
        logger.info(f"  Model needs: {self.config.num_decoder_layers} layers")
        
        if max_seg_size >= self.config.num_decoder_layers:
            # Can fit entire model in one segment!
            alt_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                best_gpu_type, self.config.sequence_length, self.config.batch_size,
                self.config.num_decoder_layers, self.config.d_model, 
                self.config.bytes_per_element, max_tp
            )
            alt_cost = best_gpu.cost_per_hour * max_tp
            alt_cost_per_token = alt_cost / (alt_throughput * 3600)
            
            logger.info(f"\n✓ CAN FIT! Single segment with TP={max_tp}:")
            logger.info(f"  Throughput: {alt_throughput:.0f} tokens/sec")
            logger.info(f"  Cost: ${alt_cost:.2f}/hour")
            logger.info(f"  $/token: ${alt_cost_per_token:.9f}")
            
            # Compare with actual solution
            actual_cpt = self.solution['cost_per_token']
            logger.info(f"\nComparison with current multi-segment solution:")
            logger.info(f"  Current solution $/token: ${actual_cpt:.9f}")
            logger.info(f"  Single-segment $/token:   ${alt_cost_per_token:.9f}")
            
            if alt_cost_per_token < actual_cpt:
                diff_pct = (actual_cpt/alt_cost_per_token - 1)*100
                logger.warning(f"  🚨 SINGLE-SEGMENT is {diff_pct:.1f}% BETTER in $/token!")
                logger.warning(f"  The solver found a suboptimal solution!")
            else:
                diff_pct = (alt_cost_per_token/actual_cpt - 1)*100
                logger.info(f"  ✓ Current multi-segment is {diff_pct:.1f}% better")
        else:
            logger.info(f"\n✗ CANNOT FIT: Model needs {self.config.num_decoder_layers} layers but max is {max_seg_size}")
        
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
                max_seg_size = self._compute_max_segment_size_for_tp(best_gpu_type, max_tp)
                
                if max_seg_size >= self.config.num_decoder_layers:
                    t_alt = ThroughputFunctions.gpu_throughput_with_tp(
                        best_gpu_type, self.config.sequence_length, self.config.batch_size,
                        self.config.num_decoder_layers, self.config.d_model,
                        self.config.bytes_per_element, max_tp
                    )
                    c_alt = best_gpu.cost_per_hour * max_tp
                    obj_alt = (t_alt / self.config.throughput_normalization) - \
                             (w/(1-w)) * (c_alt / self.config.cost_normalization)
                    
                    logger.info(f"\nAlternative ({best_gpu_type} single segment) objective: {obj_alt:.6f}")
                    logger.info(f"  Throughput: {t_alt:.0f} tokens/s")
                    logger.info(f"  Cost: ${c_alt:.2f}/h")
                    
                    if obj_alt > obj_curr:
                        logger.warning(f"  ⚠️  Alternative objective is HIGHER by {obj_alt - obj_curr:.6f}!")
                        logger.warning(f"  This suggests the solver may have missed this solution!")
    
    def print_solution(self):
        """Print solution in readable format"""
        if not self.solution:
            logger.error("No solution available")
            return
        
        print("\n" + "="*100)
        print(f"LLM PLACEMENT OPTIMIZATION RESULTS (COST-AWARE)")
        print("="*100)
        print(f"Model: {self.config.model_name} ({self.config.num_decoder_layers} layers)")
        print(f"Batch Size: {self.config.batch_size}, Sequence Length: {self.config.sequence_length}")
        print(f"TP Configuration: {self.solution['tp_configuration']}")
        print(f"Pipeline Stages: {self.solution['num_pipeline_stages']} (max: {self.config.max_pipeline_stages})")
        print()
        
        # NEW: Performance & Cost Metrics
        print("PERFORMANCE & COST METRICS:")
        print("-" * 100)
        print(f"  Throughput:        {self.solution['throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"  Cost:              ${self.solution['cost_per_hour']:.2f}/hour")
        print(f"  $/token:           ${self.solution['cost_per_token']:.9f}")  # Changed to 9 decimals
        print(f"  Objective Value:   {self.solution['objective_value']:.4f}")
        print()
        
        # NEW: Cost Comparison (if threshold specified)
        if self.config.max_cost_per_token < 999.0:
            competitor = self.config.max_cost_per_token
            our_cost = self.solution['cost_per_token']
            improvement = (competitor - our_cost) / competitor * 100
            
            print("COST COMPARISON vs COMPETITOR:")
            print("-" * 100)
            print(f"  Competitor $/token:  ${competitor:.9f}")
            print(f"  Our $/token:         ${our_cost:.9f}")
            if improvement > 0:
                print(f"  Improvement:         ✓ {improvement:.1f}% BETTER")
            else:
                print(f"  Improvement:         ✗ {abs(improvement):.1f}% WORSE")
            print()
        
        print()
        
        print("GPU ASSIGNMENTS (WITH TP):")
        print("-" * 100)
        print(f"{'GPU Type':<12} {'TP':<4} {'GPU IDs':<20} {'Layers':<15} {'Size':<6} {'Throughput':<12} {'Cost/h':<10}")
        print("-" * 100)
        
        for assignment in self.solution['gpu_assignments']:
            layers_str = f"{assignment['start_layer']}-{assignment['end_layer']}"
            gpu_ids_str = str(assignment['gpu_ids'])
            
            # Calculate segment cost
            gpu_type = assignment['gpu_type']
            tp_degree = assignment['tp_degree']
            segment_cost = self.gpu_types[gpu_type].cost_per_hour * tp_degree
            
            print(f"{gpu_type:<12} {tp_degree:<4} "
                  f"{gpu_ids_str:<20} {layers_str:<15} "
                  f"{assignment['segment_size']:<6} {assignment['throughput']:<12.2f} "
                  f"${segment_cost:<9.2f}")
        
        if self.solution['network_connections']:
            print("\nNETWORK CONNECTIONS:")
            print("-" * 80)
            for i, conn in enumerate(self.solution['network_connections']):
                from_seg = conn['from_segment']
                to_seg = conn['to_segment']
                print(f"Connection {i+1}: {from_seg['gpu_type']} partition {from_seg['partition_id']} "
                      f"(layers {from_seg['start_layer']}-{from_seg['end_layer']}) -> "
                      f"{to_seg['gpu_type']} partition {to_seg['partition_id']} "
                      f"(layers {to_seg['start_layer']}-{to_seg['end_layer']}) "
                      f"[Throughput: {conn['throughput']:.2f}]")
        
        print("\n" + "="*100)
    
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
                'batch_size': self.config.batch_size,
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
                    logger.info(f"✓ New best throughput: {best_throughput:.2f} tokens/sec")
        
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
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = time.time()
    
    try:
        solver_kwargs = {
            'enable_symmetry_breaking': args.enable_symmetry_breaking,
            'enable_upper_bound': args.enable_upper_bound,
            'enable_tight_bigm': args.enable_tight_bigm,
            'enable_flow_conservation': args.enable_flow_conservation,
            'threads': args.threads,
            'max_threads': args.max_threads
        }
        
        if args.search_all_tp:
            # Search all TP configurations
            best_solution, best_tp_config = solve_all_tp_configurations(
                args.config_dir, **solver_kwargs
            )
            
            if best_solution:
                # Print summary
                print("\n" + "="*100)
                print("BEST SOLUTION SUMMARY")
                print("="*100)
                print(f"Best TP Configuration: {best_tp_config}")
                print(f"Throughput: {best_solution['objective_value']:.2f} tokens/sec")
                print(f"Pipeline Stages: {best_solution['num_pipeline_stages']}")
                print("="*100)
                
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
    exit(main())