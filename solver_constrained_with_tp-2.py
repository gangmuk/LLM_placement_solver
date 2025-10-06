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
                global_ids=global_ids
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
            enable_segment_quantization=config_dict.get('enable_segment_quantization', 'true').lower() == 'true'
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
            valid_tp_degrees = [d for d in [1, 2, 4, 8, 16]
                            if d <= max_tp and d <= gpu_obj.count and gpu_obj.count % d == 0]

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
        Generate quantized segment sizes: [1, 2, 4, 8, 16, 32, ...]
        Powers of 2 for easier scheduling and load balancing.
        """
        sizes = []
        power = 0
        while 2**power <= self.config.num_decoder_layers:
            sizes.append(2**power)
            power += 1
        
        # Also include total layer count if not a power of 2
        if self.config.num_decoder_layers not in sizes:
            sizes.append(self.config.num_decoder_layers)
        additions = 5
        while additions < self.config.num_decoder_layers:
            sizes.append(additions)
            additions += 5
        sizes.sort()
        logger.info(f"Quantized segment sizes: {sizes}")
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
        existing_partition_keys = set((seg[0], seg[3]) for seg in self.valid_segments)  # ADD THIS
        for gpu_type, partitions in self.tp_allocations.items():
            existing_ids = sorted([pid for (gt, pid) in existing_partition_keys if gt == gpu_type])
            if len(existing_ids) > 1:
                for i in range(len(existing_ids) - 1):
                    self.model.addConstr(
                        self.z[gpu_type, existing_ids[i]] >= self.z[gpu_type, existing_ids[i+1]],
                        name=f"symmetry_break_{gpu_type}_{i}"
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
        """Set optimization objective"""
        self.model.setObjective(self.t, GRB.MAXIMIZE)
    
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
        self.solution = {
            'objective_value': self.t.x,
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
    
    def print_solution(self):
        """Print solution in readable format"""
        if not self.solution:
            logger.error("No solution available")
            return
        
        print("\n" + "="*100)
        print(f"LLM PLACEMENT OPTIMIZATION RESULTS (WITH TP + PRACTICAL CONSTRAINTS)")
        print("="*100)
        print(f"Model: {self.config.model_name} ({self.config.num_decoder_layers} layers)")
        print(f"Batch Size: {self.config.batch_size}, Sequence Length: {self.config.sequence_length}")
        print(f"TP Configuration: {self.solution['tp_configuration']}")
        print(f"Pipeline Stages: {self.solution['num_pipeline_stages']} (max: {self.config.max_pipeline_stages})")
        print(f"Optimal End-to-End Throughput: {self.solution['objective_value']:.2f} tokens/sec")
        print()
        
        print("GPU ASSIGNMENTS (WITH TP):")
        print("-" * 100)
        print(f"{'GPU Type':<12} {'TP':<4} {'GPU IDs':<20} {'Layers':<15} {'Size':<6} {'Throughput':<12}")
        print("-" * 100)
        
        for assignment in self.solution['gpu_assignments']:
            layers_str = f"{assignment['start_layer']}-{assignment['end_layer']}"
            gpu_ids_str = str(assignment['gpu_ids'])
            
            print(f"{assignment['gpu_type']:<12} {assignment['tp_degree']:<4} "
                  f"{gpu_ids_str:<20} {layers_str:<15} "
                  f"{assignment['segment_size']:<6} {assignment['throughput']:<12.2f}")
        
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
            
            solver.build_model()
            
            if solver.solve():
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