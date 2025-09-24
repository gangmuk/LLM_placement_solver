#!/usr/bin/env python3
"""
LLM Model Parallelism Placement Solver - FLOW-BASED REFORMULATION
Uses layer assignment variables instead of segment variables for better scalability.
Variables: x[gpu, layer] instead of x[gpu, start_layer, segment_size]
"""

import os
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GPUType:
    """GPU type specification"""
    name: str
    count: int
    memory_gb: float
    global_ids: List[int]  # Global GPU IDs for this type

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

class ThroughputFunctions:
    """Throughput functions with configurable coefficients"""

    # FIXED: Positive coefficients to ensure feasibility
    GPU_THROUGHPUT_COEFFS = {
        'A100': {'seq_len': -0.01, 'batch_size': 5.0, 'num_layers': 2.0, 'constant': 150.0},
        'V100': {'seq_len': -0.008, 'batch_size': 3.0, 'num_layers': 1.5, 'constant': 100.0},
        'H100': {'seq_len': -0.012, 'batch_size': 6.0, 'num_layers': 2.5, 'constant': 200.0},
        'RTX4090': {'seq_len': -0.006, 'batch_size': 2.5, 'num_layers': 1.0, 'constant': 80.0},
        # Additional GPU types
        'L20': {'seq_len': -0.009, 'batch_size': 4.0, 'num_layers': 1.8, 'constant': 120.0},
        'A10': {'seq_len': -0.007, 'batch_size': 2.0, 'num_layers': 1.2, 'constant': 70.0},
        'A40': {'seq_len': -0.009, 'batch_size': 4.5, 'num_layers': 1.9, 'constant': 140.0},
        'T4': {'seq_len': -0.005, 'batch_size': 1.5, 'num_layers': 0.8, 'constant': 50.0}
    }

    NETWORK_COEFFS = {
        'bandwidth': 1.2, 'seq_len': -0.001, 'batch_size': -0.1, 'hidden_dim': -0.00001, 'constant': 50.0
    }
    
    @staticmethod
    def gpu_throughput(gpu_type: str, seq_len: int, batch_size: int, num_layers: int) -> float:
        """Linear GPU throughput function (tokens/sec) - FIXED"""
        coeffs = ThroughputFunctions.GPU_THROUGHPUT_COEFFS[gpu_type]
        throughput = (coeffs['seq_len'] * seq_len + 
                     coeffs['batch_size'] * batch_size + 
                     coeffs['num_layers'] * num_layers + 
                     coeffs['constant'])
        return max(1.0, throughput)  # Ensure positive throughput
    
    @staticmethod
    def network_throughput(bandwidth_gbps: float, seq_len: int, batch_size: int, hidden_dim: int) -> float:
        """Linear network throughput function (transfers/sec) - FIXED"""
        coeffs = ThroughputFunctions.NETWORK_COEFFS
        throughput = (coeffs['bandwidth'] * bandwidth_gbps + 
                     coeffs['seq_len'] * seq_len + 
                     coeffs['batch_size'] * batch_size + 
                     coeffs['hidden_dim'] * hidden_dim + 
                     coeffs['constant'])
        return max(1.0, throughput)  # Ensure positive throughput
    
    @staticmethod
    def memory_usage(seq_len: int, batch_size: int, num_layers: int, layer_weight_gb: float,
                    d_model: int, d_hidden: int) -> float:
        """Memory usage in GB - FIXED"""
        # Model weights
        weight_memory = num_layers * layer_weight_gb

        # FIXED: Correct intermediate tensor memory calculation
        # Attention matrix: batch_size × seq_len × seq_len × d_model (for QK^T)
        attention_memory = batch_size * seq_len * seq_len * d_model * 4 / (1024**3)
        
        # K,V cache: 2 × batch_size × seq_len × d_model
        kv_cache_memory = 2 * batch_size * seq_len * d_model * 4 / (1024**3)
        
        # Hidden states: batch_size × seq_len × d_hidden
        hidden_memory = batch_size * seq_len * d_hidden * 4 / (1024**3)
        
        # Intermediate memory per layer (not total - pipeline processing)
        intermediate_memory_per_layer = (attention_memory + kv_cache_memory + hidden_memory) / 1024  # More reasonable
        
        total_intermediate = intermediate_memory_per_layer * min(num_layers, 2)  # At most 2 layers worth of intermediates
        
        return weight_memory + total_intermediate

class LLMPlacementFlowSolver:
    """Flow-based solver using layer assignment variables for better scalability"""

    def __init__(self, config_dir: str):
        self.options = {
            "WLSACCESSID": "790b9c11-45d0-4785-8d99-a5e6414f9321",
            "WLSSECRET": "adef4738-7bf6-41b8-8dfd-d04e23d53e51",
            "LICENSEID": 2415150,
        }
        self.env = gp.Env(params=self.options)
        self.config_dir = config_dir

        # FIXED: Correct file names
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
        
        # Validate network matrix matches GPU count
        if self.network_bandwidth.shape[0] != self.total_gpus:
            raise ValueError(f"Network bandwidth matrix size ({self.network_bandwidth.shape[0]}) "
                           f"does not match total GPU count ({self.total_gpus})")
        
        self.max_segment_size = self._compute_max_segment_sizes()
        self.gpu_pair_throughputs = self._precompute_network_throughputs()
        
        # Validate flow-based problem size
        self._validate_flow_problem_size()
        
        logger.info(f"Initialized flow-based solver: {len(self.gpu_types)} GPU types, {self.total_gpus} total GPUs")
        logger.info(f"Model: {self.config.num_decoder_layers} layers, batch_size={self.config.batch_size}")
    
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
        """Load network bandwidth matrix with validation"""
        df = pd.read_csv(filename, index_col=0)
        matrix = df.values
        
        # Validate matrix is square
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
            optimality_gap=float(config_dict['optimality_gap'])
        )
    
    def _compute_max_segment_sizes(self) -> Dict[str, int]:
        """Compute maximum segment size for each GPU type based on precise memory constraints"""
        max_sizes = {}
        
        # Pre-calculate memory components that don't depend on layer count
        base_memory_per_layer = self.config.layer_weight_memory_gb
        
        # Activation memory (depends on batch_size, seq_len, model dimensions)
        # This is per-batch, not per-layer
        activation_memory = self._calculate_activation_memory()
        
        logger.info(f"Memory analysis for batch_size={self.config.batch_size}, seq_len={self.config.sequence_length}:")
        logger.info(f"  - Weight memory per layer: {base_memory_per_layer:.2f} GB")
        logger.info(f"  - Activation memory (constant): {activation_memory:.2f} GB")
        
        for gpu_type_name, gpu_type in self.gpu_types.items():
            # Binary search for maximum layers that fit in memory
            max_layers = self._binary_search_max_layers(
                gpu_type, base_memory_per_layer, activation_memory
            )
            
            max_sizes[gpu_type_name] = max_layers
            total_memory_used = max_layers * base_memory_per_layer + activation_memory
            memory_efficiency = (total_memory_used / gpu_type.memory_gb) * 100
            
            logger.info(f"GPU {gpu_type_name}: max {max_layers} layers "
                       f"(memory: {gpu_type.memory_gb}GB, usage: {total_memory_used:.2f}GB, "
                       f"efficiency: {memory_efficiency:.1f}%)")
        
        return max_sizes
    
    def _calculate_activation_memory(self) -> float:
        """Calculate activation memory that doesn't scale with number of layers (FIXED)"""
        # FIXED: Attention matrices per layer are much smaller - we don't store full seq_len×seq_len
        # Only store attention outputs: batch_size × seq_len × d_model
        attention_memory = (self.config.batch_size * self.config.sequence_length * 
                          self.config.d_model * 4) / (1024**3)
        
        # K,V cache per layer: 2 × batch_size × seq_len × d_model  
        kv_cache_memory = (2 * self.config.batch_size * self.config.sequence_length * 
                          self.config.d_model * 4) / (1024**3)
        
        # Hidden states: batch_size × seq_len × d_hidden (intermediate computation)
        hidden_memory = (self.config.batch_size * self.config.sequence_length * 
                        self.config.d_hidden * 4) / (1024**3)
        
        # FIXED: Total activation memory is much more reasonable
        total_activation = attention_memory + kv_cache_memory + hidden_memory
        
        # Framework overhead (typically 10-20% of total memory)
        framework_overhead = 0.15  # 15% overhead
        total_with_overhead = total_activation * (1 + framework_overhead)
        
        return total_with_overhead
    
    def _binary_search_max_layers(self, gpu_type: GPUType, memory_per_layer: float, 
                                 activation_memory: float) -> int:
        """Binary search to find maximum layers that fit in GPU memory"""
        left, right = 1, self.config.num_decoder_layers
        max_feasible = 1
        
        while left <= right:
            mid = (left + right) // 2
            total_memory = mid * memory_per_layer + activation_memory
            
            if total_memory <= gpu_type.memory_gb:
                max_feasible = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return max_feasible
    
    def _generate_valid_segments(self) -> List[Tuple[str, int, int, int]]:
        """Generate all valid (gpu_type, gpu_id, start_layer, segment_size) combinations"""
        valid_segments = []
        
        for gpu_type_name, gpu_type in self.gpu_types.items():
            max_seg_size = self.max_segment_size[gpu_type_name]
            if max_seg_size == 0:
                logger.warning(f"GPU type {gpu_type_name} cannot hold any layers!")
                continue
                
            for gpu_id in range(gpu_type.count):
                for start_layer in range(1, self.config.num_decoder_layers + 1):
                    for segment_size in range(1, min(max_seg_size + 1,
                                                   self.config.num_decoder_layers - start_layer + 2)):
                        if start_layer + segment_size - 1 <= self.config.num_decoder_layers:
                            valid_segments.append((gpu_type_name, gpu_id, start_layer, segment_size))
        
        logger.info(f"Generated {len(valid_segments)} valid segments")
        return valid_segments
    
    def _generate_valid_connections(self) -> List[Tuple[Tuple[str, int, int, int], Tuple[str, int, int, int]]]:
        """Generate valid network connections efficiently using layer-based grouping"""
        valid_connections = []
        
        # Group segments by ending layer for efficient lookup
        segments_by_end_layer = {}
        for seg in self.valid_segments:
            gpu_type, gpu_id, start_layer, segment_size = seg
            end_layer = start_layer + segment_size - 1
            if end_layer not in segments_by_end_layer:
                segments_by_end_layer[end_layer] = []
            segments_by_end_layer[end_layer].append(seg)
        
        # Group segments by starting layer for efficient lookup
        segments_by_start_layer = {}
        for seg in self.valid_segments:
            gpu_type, gpu_id, start_layer, segment_size = seg
            if start_layer not in segments_by_start_layer:
                segments_by_start_layer[start_layer] = []
            segments_by_start_layer[start_layer].append(seg)
        
        # Generate connections only between consecutive layers - O(layers × segments_per_layer)
        for layer in range(1, self.config.num_decoder_layers):
            ending_segments = segments_by_end_layer.get(layer, [])
            starting_segments = segments_by_start_layer.get(layer + 1, [])
            
            for seg1 in ending_segments:
                gpu_type1, gpu_id1 = seg1[0], seg1[1]
                for seg2 in starting_segments:
                    gpu_type2, gpu_id2 = seg2[0], seg2[1]
                    # Connection valid only between different GPUs
                    if gpu_type1 != gpu_type2 or gpu_id1 != gpu_id2:
                        valid_connections.append((seg1, seg2))
        
        logger.info(f"Generated {len(valid_connections)} valid connections efficiently")
        return valid_connections
    
    def _get_global_gpu_id(self, gpu_type: str, gpu_id: int) -> int:
        """Convert (gpu_type, gpu_id) to global GPU ID"""
        return self.gpu_types[gpu_type].global_ids[gpu_id]
    
    def _precompute_network_throughputs(self) -> Dict[Tuple[Tuple[str, int], Tuple[str, int]], float]:
        """Pre-compute network throughput for all GPU pairs"""
        gpu_pair_throughputs = {}
        
        # Constant tensor size for all layer-to-layer transfers
        tensor_size_gb = (self.config.batch_size * self.config.sequence_length * 
                         self.config.d_model * 4) / (1024**3)
        
        logger.info(f"Tensor size per transfer: {tensor_size_gb:.3f} GB")
        
        for gpu_type1, gpu_obj1 in self.gpu_types.items():
            for gpu_id1 in range(gpu_obj1.count):
                global_id1 = self._get_global_gpu_id(gpu_type1, gpu_id1)
                
                for gpu_type2, gpu_obj2 in self.gpu_types.items():
                    for gpu_id2 in range(gpu_obj2.count):
                        if gpu_type1 == gpu_type2 and gpu_id1 == gpu_id2:
                            continue  # Skip same GPU
                            
                        global_id2 = self._get_global_gpu_id(gpu_type2, gpu_id2)
                        bandwidth_gbps = self.network_bandwidth[global_id1, global_id2]
                        
                        # Improved throughput: bandwidth / tensor_size (transfers per second)
                        throughput = bandwidth_gbps / tensor_size_gb if tensor_size_gb > 0 else 1000.0
                        gpu_pair_throughputs[((gpu_type1, gpu_id1), (gpu_type2, gpu_id2))] = throughput
        
        logger.info(f"Pre-computed {len(gpu_pair_throughputs)} GPU pair throughputs")
        return gpu_pair_throughputs
    
    def _validate_problem_size(self):
        """Validate that problem size is manageable and suggest optimizations"""
        num_segments = len(self.valid_segments)
        num_connections = len(self.valid_connections)
        total_binary_vars = num_segments + num_connections
        
        logger.info(f"Problem size validation:")
        logger.info(f"  - Segments: {num_segments}")
        logger.info(f"  - Connections: {num_connections}")
        logger.info(f"  - Binary variables: ~{total_binary_vars}")
        
        # Analyze memory utilization efficiency
        self._analyze_memory_efficiency()
        
        # Practical solving thresholds based on experience
        if total_binary_vars > 100000:
            logger.warning(f"Large problem ({total_binary_vars} binary variables). "
                         f"Consider using segment size constraints or hierarchical solving.")
            
            # Suggest specific optimizations
            self._suggest_problem_reduction()
            
        elif total_binary_vars > 50000:
            logger.info(f"Medium-large problem ({total_binary_vars} binary variables). "
                       f"Solving may take 5-15 minutes.")
        
        elif total_binary_vars > 10000:
            logger.info(f"Medium problem ({total_binary_vars} binary variables). "
                       f"Should solve in 1-5 minutes.")
        
        else:
            logger.info(f"Small problem ({total_binary_vars} binary variables). "
                       f"Should solve quickly.")
    
    def _analyze_memory_efficiency(self):
        """Analyze how efficiently we're using GPU memory"""
        logger.info("Memory efficiency analysis:")
        
        total_model_memory = self.config.num_decoder_layers * self.config.layer_weight_memory_gb
        total_cluster_memory = sum(gpu_type.memory_gb * gpu_type.count 
                                 for gpu_type in self.gpu_types.values())
        
        memory_utilization = (total_model_memory / total_cluster_memory) * 100
        
        logger.info(f"  - Total model memory: {total_model_memory:.1f} GB")
        logger.info(f"  - Total cluster memory: {total_cluster_memory:.1f} GB")
        logger.info(f"  - Memory utilization: {memory_utilization:.1f}%")
        
        if memory_utilization < 30:
            logger.info("  → Cluster is over-provisioned. Consider reducing GPU count.")
        elif memory_utilization > 80:
            logger.warning("  → Cluster may be under-provisioned. Limited placement options.")
    
    def _suggest_problem_reduction(self):
        """Suggest specific ways to reduce problem complexity"""
        logger.info("Problem reduction suggestions:")
        
        # Analyze segment size distribution
        segment_sizes = {}
        for gpu_type, max_size in self.max_segment_size.items():
            segment_sizes[gpu_type] = max_size
        
        # Suggest minimum segment sizes
        avg_max_size = sum(segment_sizes.values()) / len(segment_sizes)
        suggested_min_size = max(1, int(avg_max_size * 0.5))
        
        logger.info(f"  1. Set minimum segment size to {suggested_min_size} layers")
        logger.info(f"  2. Use only high-memory GPUs for this workload")
        logger.info(f"  3. Consider model sharding across fewer, larger segments")
    
    def _validate_flow_problem_size(self):
        """Validate flow-based problem size"""
        num_layer_vars = self.total_gpus * self.config.num_decoder_layers
        num_flow_vars = self.total_gpus * (self.total_gpus - 1) * (self.config.num_decoder_layers - 1)
        total_binary_vars = num_layer_vars + num_flow_vars + self.total_gpus
        
        logger.info(f"Flow-based problem size:")
        logger.info(f"  - Layer assignment vars: {num_layer_vars}")
        logger.info(f"  - Network flow vars: {num_flow_vars}")
        logger.info(f"  - Total binary variables: {total_binary_vars}")
        
        if total_binary_vars > 100000:
            logger.warning(f"Large flow-based problem ({total_binary_vars} variables). May take time to solve.")
        else:
            logger.info(f"Flow-based problem should be tractable.")
        
        # Compare to segment-based approach
        estimated_segments = sum(self.max_segment_size[gpu_type] * gpu_obj.count * self.config.num_decoder_layers 
                               for gpu_type, gpu_obj in self.gpu_types.items()) // 4  # Rough estimate
        logger.info(f"Estimated segment-based variables would be: ~{estimated_segments * 20} (much larger)")
    
    def build_model(self):
        """Build the Gurobi optimization model"""
        logger.info("Building optimization model...")
        
        # Create model
        self.model = gp.Model("llm_placement", env=self.env)
        
        # Optimized solver parameters for large problems
        self.model.setParam('Presolve', 2)              # Aggressive presolving
        self.model.setParam('Cuts', 1)                  # Moderate cut generation  
        self.model.setParam('Heuristics', 0.05)         # Limited heuristics time
        self.model.setParam('MIPFocus', 1)              # Focus on feasible solutions
        self.model.setParam('NodefileStart', 0.5)       # Use disk for large problems
        self.model.setParam('Threads', 4)               # Limit threads to avoid memory issues
        self.model.setParam('TimeLimit', self.config.time_limit_seconds)
        self.model.setParam('MIPGap', self.config.optimality_gap)
        self.model.setParam('LogToConsole', 1)
        
        # Decision variables
        self._create_variables()
        self._create_constraints()
        self._set_objective()
        
        logger.info("Model built successfully")
    
    def _create_variables(self):
        """Create decision variables using flow-based formulation"""
        # FLOW-BASED: Layer assignment variables x[gpu_type, gpu_id, layer]
        self.x = self.model.addVars(
            [(gpu_type, gpu_id, layer) 
             for gpu_type, gpu_type_obj in self.gpu_types.items()
             for gpu_id in range(gpu_type_obj.count)
             for layer in range(1, self.config.num_decoder_layers + 1)],
            vtype=GRB.BINARY,
            name="layer_assignment"
        )
        
        # GPU usage indicators: z[gpu_type, gpu_id]
        self.z = self.model.addVars(
            [(gpu_type, gpu_id) for gpu_type, gpu_type_obj in self.gpu_types.items() 
             for gpu_id in range(gpu_type_obj.count)],
            vtype=GRB.BINARY,
            name="gpu_usage"
        )
        
        # FLOW-BASED: Network flow variables for consecutive layers
        self.flow = self.model.addVars(
            [(gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer)
             for layer in range(1, self.config.num_decoder_layers)  # Flow between layer and layer+1
             for gpu_type1, gpu_obj1 in self.gpu_types.items()
             for gpu_id1 in range(gpu_obj1.count)
             for gpu_type2, gpu_obj2 in self.gpu_types.items()
             for gpu_id2 in range(gpu_obj2.count)
             if not (gpu_type1 == gpu_type2 and gpu_id1 == gpu_id2)],
            vtype=GRB.BINARY,
            name="network_flow"
        )
        
        # GPU throughput variables
        self.tau = self.model.addVars(
            [(gpu_type, gpu_id) for gpu_type, gpu_type_obj in self.gpu_types.items() 
             for gpu_id in range(gpu_type_obj.count)],
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="gpu_throughput"
        )
        
        # Network throughput variables (per flow)
        self.rho = self.model.addVars(
            [(gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer)
             for layer in range(1, self.config.num_decoder_layers)
             for gpu_type1, gpu_obj1 in self.gpu_types.items()
             for gpu_id1 in range(gpu_obj1.count)
             for gpu_type2, gpu_obj2 in self.gpu_types.items()
             for gpu_id2 in range(gpu_obj2.count)
             if not (gpu_type1 == gpu_type2 and gpu_id1 == gpu_id2)],
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="network_throughput"
        )
        
        # End-to-end throughput
        self.t = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="end_to_end_throughput")
        
        logger.info(f"Flow-based variables created:")
        logger.info(f"  - Layer assignments: {len(self.x)}")
        logger.info(f"  - Network flows: {len(self.flow)}")
        logger.info(f"  - Total binary variables: {len(self.x) + len(self.flow) + len(self.z)}")
    
    def _create_constraints(self):
        """Create flow-based constraints"""
        
        # 1. Layer coverage: each layer assigned to exactly one GPU
        for layer in range(1, self.config.num_decoder_layers + 1):
            self.model.addConstr(
                gp.quicksum(self.x[gpu_type, gpu_id, layer]
                           for gpu_type, gpu_type_obj in self.gpu_types.items()
                           for gpu_id in range(gpu_type_obj.count)) == 1,
                name=f"layer_coverage_{layer}"
            )
        
        # 2. Memory constraints: total layers per GPU must not exceed memory capacity
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            max_layers = self.max_segment_size[gpu_type]
            for gpu_id in range(gpu_type_obj.count):
                self.model.addConstr(
                    gp.quicksum(self.x[gpu_type, gpu_id, layer]
                               for layer in range(1, self.config.num_decoder_layers + 1)) <= max_layers,
                    name=f"memory_capacity_{gpu_type}_{gpu_id}"
                )
        
        # 3. GPU usage indicators
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                self.model.addConstr(
                    self.z[gpu_type, gpu_id] >= (1.0 / self.config.num_decoder_layers) *
                    gp.quicksum(self.x[gpu_type, gpu_id, layer]
                               for layer in range(1, self.config.num_decoder_layers + 1)),
                    name=f"gpu_usage_{gpu_type}_{gpu_id}"
                )
        
        # 4. Consecutive layer constraints: layers on same GPU must be consecutive
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                for layer in range(2, self.config.num_decoder_layers - 1):  # FIXED: -1 to avoid out of bounds
                    # If layer L and L+2 are on same GPU, then L+1 must also be on same GPU
                    self.model.addConstr(
                        self.x[gpu_type, gpu_id, layer] + self.x[gpu_type, gpu_id, layer + 2] - 1 
                        <= self.x[gpu_type, gpu_id, layer + 1],
                        name=f"consecutive_{gpu_type}_{gpu_id}_{layer}"
                    )
        
        # 5. Network flow constraints: flow exists if layers are on different GPUs
        for layer in range(1, self.config.num_decoder_layers):
            for gpu_type1, gpu_obj1 in self.gpu_types.items():
                for gpu_id1 in range(gpu_obj1.count):
                    for gpu_type2, gpu_obj2 in self.gpu_types.items():
                        for gpu_id2 in range(gpu_obj2.count):
                            if gpu_type1 == gpu_type2 and gpu_id1 == gpu_id2:
                                continue
                            
                            # Flow exists if layer L is on gpu1 and layer L+1 is on gpu2
                            self.model.addConstr(
                                self.flow[gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer] >=
                                self.x[gpu_type1, gpu_id1, layer] + self.x[gpu_type2, gpu_id2, layer + 1] - 1,
                                name=f"flow_def_{gpu_type1}_{gpu_id1}_{gpu_type2}_{gpu_id2}_{layer}"
                            )
                            
                            # Flow cannot exist without both layers
                            self.model.addConstr(
                                self.flow[gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer] <=
                                self.x[gpu_type1, gpu_id1, layer],
                                name=f"flow_src_{gpu_type1}_{gpu_id1}_{gpu_type2}_{gpu_id2}_{layer}"
                            )
                            
                            self.model.addConstr(
                                self.flow[gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer] <=
                                self.x[gpu_type2, gpu_id2, layer + 1],
                                name=f"flow_dst_{gpu_type1}_{gpu_id1}_{gpu_type2}_{gpu_id2}_{layer}"
                            )
        
        # 6. GPU throughput definition (flow-based)
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                # Count layers assigned to this GPU
                layers_assigned = gp.quicksum(
                    self.x[gpu_type, gpu_id, layer]
                    for layer in range(1, self.config.num_decoder_layers + 1)
                )
                
                # Throughput based on number of layers assigned
                # Use average throughput per layer for this GPU type
                avg_throughput_per_layer = ThroughputFunctions.gpu_throughput(
                    gpu_type, self.config.sequence_length, 
                    self.config.batch_size, 1  # Per layer
                )
                
                self.model.addConstr(
                    self.tau[gpu_type, gpu_id] == avg_throughput_per_layer * layers_assigned,
                    name=f"gpu_throughput_def_{gpu_type}_{gpu_id}"
                )
        
        # 7. Network throughput definition (flow-based)
        for layer in range(1, self.config.num_decoder_layers):
            for gpu_type1, gpu_obj1 in self.gpu_types.items():
                for gpu_id1 in range(gpu_obj1.count):
                    for gpu_type2, gpu_obj2 in self.gpu_types.items():
                        for gpu_id2 in range(gpu_obj2.count):
                            if gpu_type1 == gpu_type2 and gpu_id1 == gpu_id2:
                                continue
                            
                            # Use precomputed throughput for this GPU pair
                            net_throughput = self.gpu_pair_throughputs.get(
                                ((gpu_type1, gpu_id1), (gpu_type2, gpu_id2)), 100.0
                            )
                            
                            self.model.addConstr(
                                self.rho[gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer] == 
                                net_throughput * self.flow[gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer],
                                name=f"network_throughput_def_{gpu_type1}_{gpu_id1}_{gpu_type2}_{gpu_id2}_{layer}"
                            )
        
        # 8. End-to-end throughput constraints (flow-based)
        # Compute Big-M intelligently based on max possible throughputs
        max_gpu_throughput = max(
            ThroughputFunctions.gpu_throughput(gpu_type, self.config.sequence_length,
                                             self.config.batch_size, max_size)
            for gpu_type, max_size in self.max_segment_size.items()
            if max_size > 0
        )
        max_net_throughput = max(self.gpu_pair_throughputs.values()) if self.gpu_pair_throughputs else 1000.0
        M = max(max_gpu_throughput, max_net_throughput) * 1.1  # 10% buffer
        logger.info(f"Using computed Big-M value: {M:.2f}")

        # GPU throughput constraints - only for used GPUs
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                self.model.addConstr(
                    self.t <= self.tau[gpu_type, gpu_id] + M * (1 - self.z[gpu_type, gpu_id]),
                    name=f"throughput_gpu_{gpu_type}_{gpu_id}"
                )

        # Network throughput constraints - only for active flows
        for layer in range(1, self.config.num_decoder_layers):
            for gpu_type1, gpu_obj1 in self.gpu_types.items():
                for gpu_id1 in range(gpu_obj1.count):
                    for gpu_type2, gpu_obj2 in self.gpu_types.items():
                        for gpu_id2 in range(gpu_obj2.count):
                            if gpu_type1 == gpu_type2 and gpu_id1 == gpu_id2:
                                continue
                            
                            self.model.addConstr(
                                self.t <= self.rho[gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer] + 
                                M * (1 - self.flow[gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer]),
                                name=f"throughput_network_{gpu_type1}_{gpu_id1}_{gpu_type2}_{gpu_id2}_{layer}"
                            )

        # 9. Pipeline connectivity constraints (flow-based)
        # Ensure pipeline starts at layer 1
        self.model.addConstr(
            gp.quicksum(self.x[gpu_type, gpu_id, 1]
                       for gpu_type, gpu_type_obj in self.gpu_types.items()
                       for gpu_id in range(gpu_type_obj.count)) >= 1,
            name="pipeline_starts_at_layer_1"
        )
    
    def _set_objective(self):
        """Set optimization objective"""
        self.model.setObjective(self.t, GRB.MAXIMIZE)
    
    def solve(self) -> bool:
        """Solve the optimization problem"""
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
                    logger.error("Time limit reached with no feasible solution found")
                    return False
            else:
                logger.error(f"No solution found. Status: {self.model.status}")
                if self.model.status == GRB.INFEASIBLE:
                    logger.error("Model is infeasible - check memory constraints and segment generation")
                return False
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return False
    
    def _extract_solution(self):
        """Extract solution from solved flow-based model"""
        self.solution = {
            'objective_value': self.t.x,
            'gpu_assignments': [],
            'network_connections': [],
            'solve_status': self.model.status
        }
        
        # Extract layer assignments and group into segments
        layer_assignments = {}
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                layers = []
                for layer in range(1, self.config.num_decoder_layers + 1):
                    if self.x[gpu_type, gpu_id, layer].x > 0.5:
                        layers.append(layer)
                
                if layers:
                    # Group consecutive layers into segments
                    segments = self._group_consecutive_layers(layers)
                    global_gpu_id = self._get_global_gpu_id(gpu_type, gpu_id)
                    
                    for start_layer, end_layer in segments:
                        assignment = {
                            'gpu_type': gpu_type,
                            'gpu_id': gpu_id,
                            'global_gpu_id': global_gpu_id,
                            'start_layer': start_layer,
                            'end_layer': end_layer,
                            'segment_size': end_layer - start_layer + 1,
                            'throughput': self.tau[gpu_type, gpu_id].x
                        }
                        self.solution['gpu_assignments'].append(assignment)
        
        # Extract network flows
        for layer in range(1, self.config.num_decoder_layers):
            for gpu_type1, gpu_obj1 in self.gpu_types.items():
                for gpu_id1 in range(gpu_obj1.count):
                    for gpu_type2, gpu_obj2 in self.gpu_types.items():
                        for gpu_id2 in range(gpu_obj2.count):
                            if gpu_type1 == gpu_type2 and gpu_id1 == gpu_id2:
                                continue
                            
                            if self.flow[gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer].x > 0.5:
                                connection = {
                                    'from_gpu': (gpu_type1, gpu_id1),
                                    'to_gpu': (gpu_type2, gpu_id2),
                                    'layer_boundary': layer,
                                    'throughput': self.rho[gpu_type1, gpu_id1, gpu_type2, gpu_id2, layer].x
                                }
                                self.solution['network_connections'].append(connection)
        
        # Sort assignments by start layer
        self.solution['gpu_assignments'].sort(key=lambda x: x['start_layer'])
    
    def _group_consecutive_layers(self, layers):
        """Group consecutive layers into segments"""
        if not layers:
            return []
        
        layers.sort()
        segments = []
        start = layers[0]
        end = layers[0]
        
        for i in range(1, len(layers)):
            if layers[i] == end + 1:
                end = layers[i]
            else:
                segments.append((start, end))
                start = layers[i]
                end = layers[i]
        
        segments.append((start, end))
        return segments
    
    def print_solution(self):
        """Print the solution in a readable format"""
        if not self.solution:
            logger.error("No solution available")
            return
        
        print("\n" + "="*80)
        print(f"LLM PLACEMENT OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Model: {self.config.model_name} ({self.config.num_decoder_layers} layers)")
        print(f"Batch Size: {self.config.batch_size}, Sequence Length: {self.config.sequence_length}")
        print(f"Optimal End-to-End Throughput: {self.solution['objective_value']:.2f} tokens/sec")
        print()
        
        print("GPU ASSIGNMENTS:")
        print("-" * 80)
        print(f"{'GPU Type':<10} {'GPU ID':<8} {'Global ID':<10} {'Layers':<15} {'Size':<6} {'Throughput':<12}")
        print("-" * 80)
        
        for assignment in self.solution['gpu_assignments']:
            layers_str = f"{assignment['start_layer']}-{assignment['end_layer']}"
            print(f"{assignment['gpu_type']:<10} {assignment['gpu_id']:<8} "
                  f"{assignment['global_gpu_id']:<10} {layers_str:<15} "
                  f"{assignment['segment_size']:<6} {assignment['throughput']:<12.2f}")
        
        if self.solution['network_connections']:
            print("\nNETWORK CONNECTIONS:")
            print("-" * 60)
            for i, conn in enumerate(self.solution['network_connections']):
                seg1, seg2 = conn['from_segment'], conn['to_segment']
                print(f"Connection {i+1}: GPU({seg1[0]},{seg1[1]}) -> GPU({seg2[0]},{seg2[1]}) "
                      f"[Throughput: {conn['throughput']:.2f}]")
        
        print("\n" + "="*80)
    
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


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='LLM Model Parallelism Placement Optimizer - Flow-Based')
    parser.add_argument('--config-dir', required=True, help='Configuration directory containing all CSV files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize flow-based solver
        solver = LLMPlacementFlowSolver(args.config_dir)
        
        # Build and solve model
        solver.build_model()
        
        if solver.solve():
            solver.print_solution()

            # Save solution to config directory
            output_file = os.path.join(args.config_dir, 'solution.json')
            solver.save_solution(output_file)
        else:
            logger.error("Failed to find optimal solution")
            return 1
            
    except Exception as e:
        logger.error(f"Solver failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())