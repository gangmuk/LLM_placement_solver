#!/usr/bin/env python3
"""
LLM Model Parallelism Placement Solver - WITH TENSOR PARALLELISM
Uses original formulation extended with TP support.

Key Features:
- Tensor Parallelism (TP) within same GPU type
- Pipeline Parallelism (PP) across segments
- Each GPU type independently selects its TP degree
- Non-overlapping TP partition generation
- Multi-configuration optimization to find best TP assignment

Memory Model:
- Weight memory: Sharded by TP degree (layer_weight / tp_degree)
- Activation memory: Accounts for all-reduce operations
  * Sharded during computation (QKV, MLP intermediate)
  * Full tensor after all-reduce (before next layer)
  * Peak memory = max(sharded_computation, full_activation) + kv_cache
- KV cache: Persistently sharded by TP degree

Network Model:
- Communication pattern: All-reduce → Master send → All-scatter
- Step 1: All-reduce within source TP group (ring algorithm)
- Step 2: Master-to-master transfer between pipeline stages
- Step 3: All-scatter within destination TP group
- Bottleneck: min(all_reduce_bw, inter_stage_bw, all_scatter_bw)
- Matches typical LLM frameworks (Megatron-LM, DeepSpeed, vLLM)
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

class ThroughputFunctions:
    """Throughput functions with TP support"""

    GPU_THROUGHPUT_COEFFS = {
        'A100': {'seq_len': -0.01, 'batch_size': 5.0, 'num_layers': 2.0, 'constant': 150.0},
        'V100': {'seq_len': -0.008, 'batch_size': 3.0, 'num_layers': 1.5, 'constant': 100.0},
        'H100': {'seq_len': -0.012, 'batch_size': 6.0, 'num_layers': 2.5, 'constant': 200.0},
        'RTX4090': {'seq_len': -0.006, 'batch_size': 2.5, 'num_layers': 1.0, 'constant': 80.0},
        'L20': {'seq_len': -0.009, 'batch_size': 4.0, 'num_layers': 1.8, 'constant': 120.0},
        'L40': {'seq_len': -0.009, 'batch_size': 4.0, 'num_layers': 1.8, 'constant': 120.0},
        'L40S': {'seq_len': -0.009, 'batch_size': 4.5, 'num_layers': 1.9, 'constant': 130.0},
        'A10': {'seq_len': -0.007, 'batch_size': 2.0, 'num_layers': 1.2, 'constant': 70.0},
        'A40': {'seq_len': -0.009, 'batch_size': 4.5, 'num_layers': 1.9, 'constant': 140.0},
        'T4': {'seq_len': -0.005, 'batch_size': 1.5, 'num_layers': 0.8, 'constant': 50.0}
    }

    # TP efficiency factors based on empirical observations
    TP_EFFICIENCY = {
        1: 1.0,   # No TP overhead
        2: 0.90,  # 10% overhead for all-reduce
        4: 0.80,  # 20% overhead
        8: 0.70   # 30% overhead
    }
    
    @staticmethod
    def gpu_throughput(gpu_type: str, seq_len: int, batch_size: int, num_layers: int) -> float:
        """Base GPU throughput function (tokens/sec)"""
        coeffs = ThroughputFunctions.GPU_THROUGHPUT_COEFFS.get(
            gpu_type, 
            ThroughputFunctions.GPU_THROUGHPUT_COEFFS['A100']  # Default fallback
        )
        throughput = (coeffs['seq_len'] * seq_len + 
                     coeffs['batch_size'] * batch_size + 
                     coeffs['num_layers'] * num_layers + 
                     coeffs['constant'])
        return max(1.0, throughput)
    
    @staticmethod
    def gpu_throughput_with_tp(gpu_type: str, seq_len: int, batch_size: int, 
                               num_layers: int, tp_degree: int) -> float:
        """
        GPU throughput with tensor parallelism.
        
        TP provides near-linear scaling but with communication overhead:
        - TP=1: baseline (no overhead)
        - TP=2: ~1.8x speedup (90% efficiency)
        - TP=4: ~3.2x speedup (80% efficiency)
        - TP=8: ~5.6x speedup (70% efficiency)
        """
        base_throughput = ThroughputFunctions.gpu_throughput(
            gpu_type, seq_len, batch_size, num_layers
        )
        
        efficiency = ThroughputFunctions.TP_EFFICIENCY.get(tp_degree, 0.70)
        tp_speedup = tp_degree * efficiency
        
        return base_throughput * tp_speedup
    
    @staticmethod
    def network_throughput(bandwidth_gbps: float, seq_len: int, batch_size: int, hidden_dim: int) -> float:
        """Network throughput function (transfers/sec)"""
        tensor_size_gb = (batch_size * seq_len * hidden_dim * 2) / (1024**3)  # FP16
        if tensor_size_gb > 0:
            return bandwidth_gbps / tensor_size_gb
        return 1000.0

class LLMPlacementSolverWithTP:
    """LLM placement solver with Tensor Parallelism support"""

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
        
        # TP configuration: {gpu_type: tp_degree}
        self.tp_configuration = tp_configuration or {gpu_type: 1 for gpu_type in self.gpu_types}
        
        # Generate TP partitions based on configuration
        self.tp_partitions = self._generate_tp_partitions()
        
        # Generate valid segments and connections with TP
        self.max_segment_size = self._compute_max_segment_sizes()
        self.valid_segments = self._generate_valid_segments()
        self.valid_connections = self._generate_valid_connections()
        self.gpu_pair_throughputs = self._precompute_network_throughputs()
        
        # Validate problem size
        self._validate_problem_size()
        
        logger.info(f"Initialized solver with TP: {len(self.gpu_types)} GPU types, {self.total_gpus} total GPUs")
        logger.info(f"TP Configuration: {self.tp_configuration}")
        logger.info(f"Model: {self.config.num_decoder_layers} layers, batch_size={self.config.batch_size}")
        logger.info(f"Problem size: {len(self.valid_segments)} segments, {len(self.valid_connections)} connections")
    
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
        
        bytes_per_element = int(config_dict.get('bytes_per_element', 2))
        
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
            bytes_per_element=bytes_per_element
        )
    
    def _generate_tp_partitions(self) -> Dict[str, List[FrozenSet[int]]]:
        """
        Generate non-overlapping TP partitions for each GPU type.
        Returns: {gpu_type: [partition_0, partition_1, ...]}
        """
        tp_partitions = {}
        
        for gpu_type, gpu_obj in self.gpu_types.items():
            tp_degree = self.tp_configuration[gpu_type]
            
            if gpu_obj.count % tp_degree != 0:
                raise ValueError(f"GPU type {gpu_type} has {gpu_obj.count} GPUs, "
                               f"which is not divisible by TP degree {tp_degree}")
            
            num_partitions = gpu_obj.count // tp_degree
            partitions = []
            
            for partition_id in range(num_partitions):
                start_id = partition_id * tp_degree
                gpu_set = frozenset(range(start_id, start_id + tp_degree))
                partitions.append(gpu_set)
            
            tp_partitions[gpu_type] = partitions
            
            logger.info(f"GPU {gpu_type}: TP={tp_degree}, {num_partitions} partitions of {tp_degree} GPUs each")
        
        return tp_partitions
    
    def _compute_max_segment_sizes(self) -> Dict[Tuple[str, int], int]:
        """
        Compute maximum segment size for each (GPU type, TP degree) pair.
        With TP, model weights are sharded across GPUs.
        """
        max_sizes = {}

        precision_name = "FP16" if self.config.bytes_per_element == 2 else "FP32" if self.config.bytes_per_element == 4 else f"{self.config.bytes_per_element}-byte"
        logger.info(f"Memory analysis for batch_size={self.config.batch_size}, seq_len={self.config.sequence_length} ({precision_name}):")

        for gpu_type, gpu_obj in self.gpu_types.items():
            tp_degree = self.tp_configuration[gpu_type]

            # With TP, weights are sharded
            weight_per_gpu_per_layer = self.config.layer_weight_memory_gb / tp_degree

            # Activation memory with TP (accounts for all-reduce and KV cache sharding)
            activation_memory = self._calculate_activation_memory(tp_degree)

            # Binary search for max layers
            max_layers = self._binary_search_max_layers(
                gpu_obj.memory_gb, weight_per_gpu_per_layer, activation_memory
            )

            max_sizes[(gpu_type, tp_degree)] = max_layers
            total_memory_used = max_layers * weight_per_gpu_per_layer + activation_memory
            memory_efficiency = (total_memory_used / gpu_obj.memory_gb) * 100

            logger.info(f"GPU {gpu_type} with TP={tp_degree}: max {max_layers} layers "
                       f"(memory: {gpu_obj.memory_gb}GB, usage: {total_memory_used:.2f}GB, "
                       f"efficiency: {memory_efficiency:.1f}%)")

        return max_sizes
    
    def _calculate_activation_memory(self, tp_degree: int = 1) -> float:
        """
        Calculate peak activation memory per GPU with TP.

        Key insight: With TP, there are two all-reduce operations per layer:
        1. After attention output projection (row-parallel)
        2. After MLP second linear (row-parallel)

        After all-reduce, each GPU temporarily holds the FULL activation tensor
        before the next layer, even though intermediate computation is sharded.

        Args:
            tp_degree: Tensor parallelism degree (1 = no TP)

        Returns:
            Peak activation memory in GB per GPU
        """
        batch = self.config.batch_size
        seq_len = self.config.sequence_length
        hidden = self.config.d_model
        d_hidden = self.config.d_hidden
        bytes_per_elem = self.config.bytes_per_element

        # Sharded intermediate tensors during computation
        # QKV projection outputs (column-parallel): sharded along hidden dim
        qkv_memory = (3 * batch * seq_len * (hidden / tp_degree) *
                     bytes_per_elem) / (1024**3)

        # MLP intermediate (column-parallel): sharded along hidden dim
        # Typically 4x hidden size for MLP
        mlp_intermediate = (batch * seq_len * (4 * hidden / tp_degree) *
                           bytes_per_elem) / (1024**3)

        # Total sharded computation memory
        sharded_computation = qkv_memory + mlp_intermediate

        # Full activation tensor after all-reduce (NOT sharded)
        # Each GPU holds the complete tensor after row-parallel all-reduce
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
    
    def _generate_valid_segments(self) -> List[Tuple[str, int, int, int, int]]:
        """
        Generate segments with TP partitions.
        Segment format: (gpu_type, partition_id, start_layer, segment_size, tp_degree)
        """
        valid_segments = []
        
        for gpu_type, partitions in self.tp_partitions.items():
            tp_degree = self.tp_configuration[gpu_type]
            max_seg_size = self.max_segment_size[(gpu_type, tp_degree)]
            
            if max_seg_size == 0:
                logger.warning(f"GPU type {gpu_type} with TP={tp_degree} cannot hold any layers!")
                continue
            
            # Minimum segment size for memory efficiency
            min_seg_size = max(1, max_seg_size // 3)
            
            for partition_id, gpu_set in enumerate(partitions):
                # Step size for large problems
                step_size = max(1, self.config.num_decoder_layers // 20) if self.config.num_decoder_layers > 40 else 1
                
                for start_layer in range(1, self.config.num_decoder_layers + 1, step_size):
                    for segment_size in range(min_seg_size, min(max_seg_size + 1,
                                                   self.config.num_decoder_layers - start_layer + 2)):
                        if start_layer + segment_size - 1 <= self.config.num_decoder_layers:
                            # Store GPU set for later use
                            segment = (gpu_type, partition_id, start_layer, segment_size, tp_degree, gpu_set)
                            valid_segments.append(segment)
        
        logger.info(f"Generated {len(valid_segments)} segments with TP partitions")
        return valid_segments
    
    def _generate_valid_connections(self) -> List[Tuple]:
        """Generate valid network connections between consecutive segments"""
        valid_connections = []
        
        # Group segments by ending/starting layer
        segments_by_end_layer = {}
        segments_by_start_layer = {}
        
        for seg in self.valid_segments:
            gpu_type, partition_id, start_layer, segment_size, tp_degree, gpu_set = seg
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
                gpu_set1 = seg1[5]
                for seg2 in starting_segments:
                    gpu_set2 = seg2[5]
                    # Connection valid only if GPU sets don't overlap
                    if not gpu_set1.intersection(gpu_set2):
                        valid_connections.append((seg1, seg2))
        
        logger.info(f"Generated {len(valid_connections)} valid connections")
        return valid_connections
    
    def _get_global_gpu_id(self, gpu_type: str, local_gpu_id: int) -> int:
        """Convert (gpu_type, local_gpu_id) to global GPU ID"""
        return self.gpu_types[gpu_type].global_ids[local_gpu_id]
    
    def _precompute_network_throughputs(self) -> Dict:
        """
        Pre-compute network throughput for all GPU pairs using master all-reduce/scatter pattern.

        Communication pattern between TP groups:
        1. All-reduce within source TP group → master GPU has full tensor
        2. Master-to-master transfer between PP stages
        3. All-scatter within destination TP group

        This matches typical LLM serving frameworks (Megatron-LM, DeepSpeed, vLLM).
        """
        gpu_pair_throughputs = {}

        # Full tensor size (NOT sharded - each GPU holds full activation after all-reduce)
        tensor_size_gb = (self.config.batch_size * self.config.sequence_length *
                         self.config.d_model * self.config.bytes_per_element) / (1024**3)

        precision_name = "FP16" if self.config.bytes_per_element == 2 else "FP32"
        logger.info(f"Tensor size per inter-stage transfer: {tensor_size_gb:.3f} GB ({precision_name})")
        logger.info(f"Network model: All-reduce → Master send → All-scatter")

        # Assume intra-node bandwidth (NVLink for NVIDIA GPUs)
        # This is a reasonable default; could be made configurable
        nvlink_bandwidth_gbps = 600.0  # A100 NVLink bandwidth

        for seg1 in self.valid_segments:
            gpu_set1 = seg1[5]
            gpu_type1 = seg1[0]
            tp_degree1 = seg1[4]

            for seg2 in self.valid_segments:
                if seg1[2] + seg1[3] != seg2[2]:  # Not consecutive layers
                    continue

                gpu_set2 = seg2[5]
                gpu_type2 = seg2[0]
                tp_degree2 = seg2[4]

                if gpu_set1.intersection(gpu_set2):  # Overlapping GPUs - invalid
                    continue

                # Step 1: All-reduce bandwidth within source TP group
                # Ring all-reduce transfers (tp_degree - 1) / tp_degree of the data
                all_reduce_efficiency = (tp_degree1 - 1) / tp_degree1 if tp_degree1 > 1 else 1.0
                all_reduce_bw = all_reduce_efficiency * nvlink_bandwidth_gbps

                # Step 2: Master-to-master inter-stage bandwidth
                # Choose master GPU (e.g., lowest ID in partition)
                master_local_id1 = min(gpu_set1)
                master_local_id2 = min(gpu_set2)
                master_global_id1 = self._get_global_gpu_id(gpu_type1, master_local_id1)
                master_global_id2 = self._get_global_gpu_id(gpu_type2, master_local_id2)
                inter_stage_bw = self.network_bandwidth[master_global_id1, master_global_id2]

                # Step 3: All-scatter bandwidth within destination TP group
                all_scatter_efficiency = (tp_degree2 - 1) / tp_degree2 if tp_degree2 > 1 else 1.0
                all_scatter_bw = all_scatter_efficiency * nvlink_bandwidth_gbps

                # Bottleneck: minimum of all three steps
                effective_bandwidth = min(all_reduce_bw, inter_stage_bw, all_scatter_bw)

                # Network throughput (transfers per second)
                throughput = effective_bandwidth / tensor_size_gb if tensor_size_gb > 0 else 1000.0
                gpu_pair_throughputs[(seg1, seg2)] = throughput

        logger.info(f"Pre-computed {len(gpu_pair_throughputs)} network throughputs")
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
        logger.info("Building optimization model with TP...")
        
        self.model = gp.Model("llm_placement_with_tp", env=self.env)
        
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
        partition_keys = [(seg[0], seg[1]) for seg in self.valid_segments]
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
                           if seg[2] <= layer <= seg[2] + seg[3] - 1) == 1,
                name=f"layer_coverage_{layer}"
            )
        
        # 2. TP partition capacity: each partition processes at most one segment
        for gpu_type, partitions in self.tp_partitions.items():
            for partition_id in range(len(partitions)):
                self.model.addConstr(
                    gp.quicksum(self.x[seg] for seg in self.valid_segments 
                               if seg[0] == gpu_type and seg[1] == partition_id) <= 1,
                    name=f"partition_capacity_{gpu_type}_{partition_id}"
                )
        
        # 3. Partition usage indicators
        for gpu_type, partitions in self.tp_partitions.items():
            for partition_id in range(len(partitions)):
                self.model.addConstr(
                    self.z[gpu_type, partition_id] == 
                    gp.quicksum(self.x[seg] for seg in self.valid_segments 
                               if seg[0] == gpu_type and seg[1] == partition_id),
                    name=f"partition_usage_{gpu_type}_{partition_id}"
                )
        
        # 4. Network connection constraints
        for (seg1, seg2) in self.valid_connections:
            self.model.addConstr(self.e[seg1, seg2] <= self.x[seg1], name=f"conn_seg1")
            self.model.addConstr(self.e[seg1, seg2] <= self.x[seg2], name=f"conn_seg2")
            self.model.addConstr(
                self.e[seg1, seg2] >= self.x[seg1] + self.x[seg2] - 1,
                name=f"conn_both"
            )
        
        # 5. Partition throughput definition (with TP)
        for gpu_type, partitions in self.tp_partitions.items():
            tp_degree = self.tp_configuration[gpu_type]
            
            for partition_id in range(len(partitions)):
                throughput_expr = gp.quicksum(
                    ThroughputFunctions.gpu_throughput_with_tp(
                        gpu_type, self.config.sequence_length,
                        self.config.batch_size, seg[3], tp_degree
                    ) * self.x[seg]
                    for seg in self.valid_segments 
                    if seg[0] == gpu_type and seg[1] == partition_id
                )
                self.model.addConstr(
                    self.tau[gpu_type, partition_id] == throughput_expr,
                    name=f"partition_throughput_{gpu_type}_{partition_id}"
                )
        
        # 6. Network throughput definition
        for (seg1, seg2) in self.valid_connections:
            net_throughput = self.gpu_pair_throughputs.get((seg1, seg2), 100.0)
            self.model.addConstr(
                self.rho[seg1, seg2] == net_throughput * self.e[seg1, seg2],
                name=f"network_throughput"
            )
        
        # 7. End-to-end throughput constraints
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
        
        # 8. Pipeline connectivity constraints
        self._add_pipeline_connectivity_constraints()
        
        # Optional optimizations
        if self.enable_symmetry_breaking:
            self._add_symmetry_breaking_constraints()
        
        if self.enable_flow_conservation:
            self._add_flow_conservation_constraints()
    
    def _compute_tight_bigM(self) -> Tuple[Dict, float]:
        """Compute tight Big-M values"""
        # Partition Big-M: maximum possible throughput per partition
        M_partition = {}
        for gpu_type, partitions in self.tp_partitions.items():
            tp_degree = self.tp_configuration[gpu_type]
            max_size = self.max_segment_size[(gpu_type, tp_degree)]
            if max_size > 0:
                max_throughput = ThroughputFunctions.gpu_throughput_with_tp(
                    gpu_type, self.config.sequence_length,
                    self.config.batch_size, max_size, tp_degree
                )
                for partition_id in range(len(partitions)):
                    M_partition[(gpu_type, partition_id)] = max_throughput * 3
        
        # Network Big-M: maximum network throughput
        M_network = max(self.gpu_pair_throughputs.values()) * 2 if self.gpu_pair_throughputs else 1000.0
        
        logger.info(f"Tight Big-M computed: M_network={M_network:.2f}")
        return M_partition, M_network
    
    def _add_pipeline_connectivity_constraints(self):
        """Ensure pipeline connectivity from layer 1 to final layer"""
        # Pipeline must start at layer 1
        first_layer_segments = [seg for seg in self.valid_segments if seg[2] == 1]
        if first_layer_segments:
            self.model.addConstr(
                gp.quicksum(self.x[seg] for seg in first_layer_segments) >= 1,
                name="pipeline_starts_at_layer_1"
            )
        
        # Sequential connectivity
        for layer in range(1, self.config.num_decoder_layers):
            segments_ending_here = [seg for seg in self.valid_segments
                                   if seg[2] + seg[3] - 1 == layer]
            segments_starting_next = [seg for seg in self.valid_segments
                                     if seg[2] == layer + 1]
            
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
        
        for gpu_type, partitions in self.tp_partitions.items():
            if len(partitions) > 1:
                for i in range(len(partitions) - 1):
                    self.model.addConstr(
                        self.z[gpu_type, i] >= self.z[gpu_type, i+1],
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
                              if seg[2] + seg[3] - 1 == layer]
            segments_starting = [seg for seg in self.valid_segments
                               if seg[2] == layer + 1]
            
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
            'tp_configuration': self.tp_configuration,
            'gpu_assignments': [],
            'network_connections': [],
            'solve_status': self.model.status
        }
        
        # Extract GPU assignments
        for seg in self.valid_segments:
            if self.x[seg].x > 0.5:
                gpu_type, partition_id, start_layer, segment_size, tp_degree, gpu_set = seg
                
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
                        'partition_id': seg1[1],
                        'start_layer': seg1[2],
                        'end_layer': seg1[2] + seg1[3] - 1
                    },
                    'to_segment': {
                        'gpu_type': seg2[0],
                        'partition_id': seg2[1],
                        'start_layer': seg2[2],
                        'end_layer': seg2[2] + seg2[3] - 1
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
        print(f"LLM PLACEMENT OPTIMIZATION RESULTS (WITH TENSOR PARALLELISM)")
        print("="*100)
        print(f"Model: {self.config.model_name} ({self.config.num_decoder_layers} layers)")
        print(f"Batch Size: {self.config.batch_size}, Sequence Length: {self.config.sequence_length}")
        print(f"TP Configuration: {self.solution['tp_configuration']}")
        print(f"Optimal End-to-End Throughput: {self.solution['objective_value']:.2f} tokens/sec")
        print()
        
        print("GPU ASSIGNMENTS (WITH TP):")
        print("-" * 100)
        print(f"{'GPU Type':<12} {'TP Deg':<8} {'GPU IDs':<20} {'Layers':<15} {'Size':<6} {'Throughput':<12}")
        print("-" * 100)
        
        for assignment in self.solution['gpu_assignments']:
            layers_str = f"{assignment['start_layer']}-{assignment['end_layer']}"
            gpu_ids_str = str(assignment['gpu_ids'])
            
            print(f"{assignment['gpu_type']:<12} {assignment['tp_degree']:<8} "
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
                'd_hidden': self.config.d_hidden
            },
            'solution': self.solution
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Solution saved to {output_file}")


def enumerate_tp_configurations(gpu_types: Dict[str, GPUType]) -> List[Dict[str, int]]:
    """Enumerate all valid TP configurations across GPU types"""
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
    # print(f"Enumerated {len(configs)} TP configurations")
    # print(f"TP configurations: {configs}")
    # exit()
    return configs


def solve_all_tp_configurations(config_dir: str, **kwargs) -> Dict:
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
    
    # Enumerate all TP configurations
    tp_configs = enumerate_tp_configurations(gpu_types)
    logger.info(f"Evaluating {len(tp_configs)} TP configurations...")
    
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
    
    return best_solution, best_tp_config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LLM Placement Optimizer with Tensor Parallelism')
    parser.add_argument('--config-dir', required=True, help='Configuration directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--tp-config', type=str, help='TP configuration as JSON (e.g., \'{"A100": 4, "V100": 1}\')')
    parser.add_argument('--search-all-tp', action='store_true', 
                       help='Search all valid TP configurations (ignores --tp-config)')
    
    # Optimization flags
    parser.add_argument('--enable-symmetry-breaking', action='store_true', default=True)
    parser.add_argument('--disable-symmetry-breaking', dest='enable_symmetry_breaking', action='store_false')
    parser.add_argument('--enable-upper-bound', action='store_true', default=True)
    parser.add_argument('--disable-upper-bound', dest='enable_upper_bound', action='store_false')
    parser.add_argument('--enable-tight-bigm', action='store_true', default=True)
    parser.add_argument('--disable-tight-bigm', dest='enable_tight_bigm', action='store_false')
    parser.add_argument('--enable-flow-conservation', action='store_true', default=True)
    parser.add_argument('--disable-flow-conservation', dest='enable_flow_conservation', action='store_false')
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