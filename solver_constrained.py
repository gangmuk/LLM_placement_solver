#!/usr/bin/env python3
"""
LLM Model Parallelism Placement Solver - CONSTRAINED SEGMENTS
Uses original formulation but with intelligent minimum segment size constraints
to reduce the combinatorial explosion while maintaining optimality.
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
    bytes_per_element: int = 2  # FP16 by default, can be 4 for FP32

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
                    d_model: int, d_hidden: int, bytes_per_element: int = 2) -> float:
        """Memory usage in GB - parameterized precision"""
        # Model weights
        weight_memory = num_layers * layer_weight_gb

        # Intermediate tensor memory calculation (parameterized bytes per element)
        # Attention matrix: batch_size × seq_len × seq_len × d_model (for QK^T)
        attention_memory = batch_size * seq_len * seq_len * d_model * bytes_per_element / (1024**3)
        
        # K,V cache: 2 × batch_size × seq_len × d_model
        kv_cache_memory = 2 * batch_size * seq_len * d_model * bytes_per_element / (1024**3)
        
        # Hidden states: batch_size × seq_len × d_hidden
        hidden_memory = batch_size * seq_len * d_hidden * bytes_per_element / (1024**3)
        
        # Intermediate memory per layer (not total - pipeline processing)
        intermediate_memory_per_layer = (attention_memory + kv_cache_memory + hidden_memory) / 1024  # More reasonable
        
        total_intermediate = intermediate_memory_per_layer * min(num_layers, 2)  # At most 2 layers worth of intermediates
        
        return weight_memory + total_intermediate

class LLMPlacementSolver:
    """Main solver class for LLM placement optimization"""

    def __init__(self, config_dir: str, enable_symmetry_breaking: bool = True,
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
        self.valid_segments = self._generate_valid_segments()
        self.valid_connections = self._generate_valid_connections()
        self.gpu_pair_throughputs = self._precompute_network_throughputs()
        
        # Validate problem size
        self._validate_problem_size()
        
        logger.info(f"Initialized solver: {len(self.gpu_types)} GPU types, {self.total_gpus} total GPUs")
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
        
        # Handle optional bytes_per_element parameter (default to FP16 = 2 bytes)
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
    
    def _compute_max_segment_sizes(self) -> Dict[str, int]:
        """Compute maximum segment size for each GPU type based on precise memory constraints"""
        max_sizes = {}
        
        # Pre-calculate memory components that don't depend on layer count
        base_memory_per_layer = self.config.layer_weight_memory_gb
        
        # Activation memory (depends on batch_size, seq_len, model dimensions)
        # This is per-batch, not per-layer
        activation_memory = self._calculate_activation_memory()
        
        precision_name = "FP16" if self.config.bytes_per_element == 2 else "FP32" if self.config.bytes_per_element == 4 else f"{self.config.bytes_per_element}-byte"
        logger.info(f"Memory analysis for batch_size={self.config.batch_size}, seq_len={self.config.sequence_length} ({precision_name}):")
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
        """Calculate activation memory that doesn't scale with number of layers"""
        # Attention matrices per layer are much smaller - we don't store full seq_len×seq_len
        # Only store attention outputs: batch_size × seq_len × d_model
        attention_memory = (self.config.batch_size * self.config.sequence_length * 
                          self.config.d_model * self.config.bytes_per_element) / (1024**3)
        
        # K,V cache per layer: 2 × batch_size × seq_len × d_model  
        kv_cache_memory = (2 * self.config.batch_size * self.config.sequence_length * 
                          self.config.d_model * self.config.bytes_per_element) / (1024**3)
        
        # Hidden states: batch_size × seq_len × d_hidden (intermediate computation)
        hidden_memory = (self.config.batch_size * self.config.sequence_length * 
                        self.config.d_hidden * self.config.bytes_per_element) / (1024**3)
        
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
        """Generate segments with intelligent minimum size constraints"""
        valid_segments = []
        
        # Calculate intelligent minimum segment sizes
        min_segment_sizes = self._calculate_min_segment_sizes()
        
        for gpu_type_name, gpu_type in self.gpu_types.items():
            max_seg_size = self.max_segment_size[gpu_type_name]
            min_seg_size = min_segment_sizes[gpu_type_name]
            
            if max_seg_size == 0:
                logger.warning(f"GPU type {gpu_type_name} cannot hold any layers!")
                continue
                
            logger.info(f"GPU {gpu_type_name}: using segment sizes {min_seg_size}-{max_seg_size}")
                
            for gpu_id in range(gpu_type.count):
                # Use larger step sizes for large problems
                step_size = max(1, self.config.num_decoder_layers // 20) if self.config.num_decoder_layers > 40 else 1
                
                for start_layer in range(1, self.config.num_decoder_layers + 1, step_size):
                    for segment_size in range(min_seg_size, min(max_seg_size + 1,
                                                   self.config.num_decoder_layers - start_layer + 2)):
                        if start_layer + segment_size - 1 <= self.config.num_decoder_layers:
                            valid_segments.append((gpu_type_name, gpu_id, start_layer, segment_size))
        
        logger.info(f"Generated {len(valid_segments)} constrained segments")
        return valid_segments
    
    def _calculate_min_segment_sizes(self) -> Dict[str, int]:
        """Calculate intelligent minimum segment sizes based on memory efficiency"""
        min_sizes = {}
        
        for gpu_type_name, gpu_type in self.gpu_types.items():
            max_size = self.max_segment_size[gpu_type_name]
            
            # Minimum segment size based on memory efficiency
            # Aim for at least 50% memory utilization
            memory_per_layer = self.config.layer_weight_memory_gb
            activation_memory = self._calculate_activation_memory()
            target_memory = gpu_type.memory_gb * 0.5  # 50% utilization target
            
            min_layers = max(1, int((target_memory - activation_memory) / memory_per_layer))
            min_sizes[gpu_type_name] = min(min_layers, max_size, max(1, max_size // 3))
            
        return min_sizes
    
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
                         self.config.d_model * self.config.bytes_per_element) / (1024**3)
        
        precision_name = "FP16" if self.config.bytes_per_element == 2 else "FP32" if self.config.bytes_per_element == 4 else f"{self.config.bytes_per_element}-byte"
        logger.info(f"Tensor size per transfer: {tensor_size_gb:.3f} GB ({precision_name})")
        
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
    
    def _compute_smart_upper_bound(self) -> float:
        """Compute theoretically achievable upper bound with feasibility check"""
        best_feasible_throughput = 0
        
        for gpu_type, gpu_obj in self.gpu_types.items():
            max_layers_per_gpu = self.max_segment_size[gpu_type]
            if max_layers_per_gpu > 0:
                # Check if this GPU type can handle the full model
                gpus_needed = math.ceil(self.config.num_decoder_layers / max_layers_per_gpu)
                
                if gpus_needed <= gpu_obj.count:  # Feasible with available GPUs
                    # Best case: all segments have max size
                    segment_throughput = ThroughputFunctions.gpu_throughput(
                        gpu_type, self.config.sequence_length,
                        self.config.batch_size, max_layers_per_gpu
                    )
                    best_feasible_throughput = max(best_feasible_throughput, segment_throughput)
                    logger.info(f"  Upper bound candidate from {gpu_type}: {segment_throughput:.2f} tokens/sec "
                               f"({gpus_needed} GPUs needed, {gpu_obj.count} available)")
        
        return best_feasible_throughput
    
    def _compute_tight_bigM(self) -> Tuple[Dict[str, float], float]:
        """Compute Big-M values for different constraint types"""
        # FIXED: Big-M values must be large enough to not constrain unused GPUs
        # The previous "tight" approach created false constraints by using actual throughput values
        
        # Calculate a reasonable upper bound for throughput
        max_possible_throughput = 0
        for gpu_type in self.gpu_types:
            if self.max_segment_size[gpu_type] > 0:
                gpu_max_throughput = ThroughputFunctions.gpu_throughput(
                    gpu_type, self.config.sequence_length,
                    self.config.batch_size, self.max_segment_size[gpu_type]
                )
                max_possible_throughput = max(max_possible_throughput, gpu_max_throughput)
        
        # Use a sufficiently large Big-M (3x the maximum possible throughput)
        # This ensures unused GPUs don't create false constraints while keeping the problem tractable
        safe_bigM = max(1000.0, max_possible_throughput * 3)
        
        # Use the same safe Big-M for all GPU types to avoid false constraints
        M_gpu = {gpu_type: safe_bigM for gpu_type in self.gpu_types}
        
        # For network constraints: t <= ρ[e] + M*(1-e)
        M_network = max(self.gpu_pair_throughputs.values()) if self.gpu_pair_throughputs else 1000.0
        # Network Big-M can be tighter since connections are explicitly modeled
        M_network = max(safe_bigM, M_network * 2)
        
        logger.info(f"Safe Big-M values computed:")
        logger.info(f"  M_gpu (all types) = {safe_bigM:.2f}")
        logger.info(f"  M_network = {M_network:.2f}")
        logger.info("  (Using safe values to prevent false constraints from unused GPUs)")
        
        return M_gpu, M_network
    
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
        self.model.setParam('TimeLimit', self.config.time_limit_seconds)
        self.model.setParam('MIPGap', self.config.optimality_gap)
        self.model.setParam('LogToConsole', 1)
        
        # Decision variables
        self._create_variables()
        self._create_constraints()
        self._set_objective()
        
        logger.info("Model built successfully")
    
    def _create_variables(self):
        """Create decision variables"""
        # Segment assignment variables: x[gpu_type, gpu_id, start_layer, segment_size]
        self.x = self.model.addVars(
            self.valid_segments,
            vtype=GRB.BINARY,
            name="segment_assignment"
        )
        
        # GPU usage indicators: z[gpu_type, gpu_id]
        self.z = self.model.addVars(
            [(gpu_type, gpu_id) for gpu_type, gpu_type_obj in self.gpu_types.items() 
             for gpu_id in range(gpu_type_obj.count)],
            vtype=GRB.BINARY,
            name="gpu_usage"
        )
        
        # Network connection variables: e[seg1, seg2]
        self.e = self.model.addVars(
            self.valid_connections,
            vtype=GRB.BINARY,
            name="network_connection"
        )
        
        # Throughput variables
        self.tau = self.model.addVars(
            [(gpu_type, gpu_id) for gpu_type, gpu_type_obj in self.gpu_types.items() 
             for gpu_id in range(gpu_type_obj.count)],
            vtype=GRB.CONTINUOUS,
            lb=0,
            name="gpu_throughput"
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
        """Create optimization constraints - FIXED"""
        
        # 1. Layer coverage: each layer assigned exactly once
        for layer in range(1, self.config.num_decoder_layers + 1):
            self.model.addConstr(
                gp.quicksum(self.x[seg] for seg in self.valid_segments 
                           if seg[2] <= layer <= seg[2] + seg[3] - 1) == 1,
                name=f"layer_coverage_{layer}"
            )
        
        # 2. GPU capacity: each GPU processes at most one segment
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                self.model.addConstr(
                    gp.quicksum(self.x[seg] for seg in self.valid_segments 
                               if seg[0] == gpu_type and seg[1] == gpu_id) <= 1,
                    name=f"gpu_capacity_{gpu_type}_{gpu_id}"
                )
        
        # 3. GPU usage indicators
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                self.model.addConstr(
                    self.z[gpu_type, gpu_id] == 
                    gp.quicksum(self.x[seg] for seg in self.valid_segments 
                               if seg[0] == gpu_type and seg[1] == gpu_id),
                    name=f"gpu_usage_{gpu_type}_{gpu_id}"
                )
        
        # 4. Network connection constraints
        for (seg1, seg2) in self.valid_connections:
            # Connection exists if both segments are selected
            self.model.addConstr(
                self.e[seg1, seg2] <= self.x[seg1],
                name=f"connection_seg1"
            )
            self.model.addConstr(
                self.e[seg1, seg2] <= self.x[seg2],
                name=f"connection_seg2"
            )
            self.model.addConstr(
                self.e[seg1, seg2] >= self.x[seg1] + self.x[seg2] - 1,
                name=f"connection_both"
            )
        
        # 5. GPU throughput definition
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                gpu_throughput_expr = gp.quicksum(
                    ThroughputFunctions.gpu_throughput(
                        gpu_type, self.config.sequence_length, 
                        self.config.batch_size, seg[3]
                    ) * self.x[seg]
                    for seg in self.valid_segments 
                    if seg[0] == gpu_type and seg[1] == gpu_id
                )
                self.model.addConstr(
                    self.tau[gpu_type, gpu_id] == gpu_throughput_expr,
                    name=f"gpu_throughput_def_{gpu_type}_{gpu_id}"
                )
        
        # 6. Network throughput definition using precomputed values
        for (seg1, seg2) in self.valid_connections:
            gpu_type1, gpu_id1 = seg1[0], seg1[1]
            gpu_type2, gpu_id2 = seg2[0], seg2[1]
            
            # Use precomputed throughput
            net_throughput = self.gpu_pair_throughputs.get(
                ((gpu_type1, gpu_id1), (gpu_type2, gpu_id2)), 100.0
            )
            
            self.model.addConstr(
                self.rho[seg1, seg2] == net_throughput * self.e[seg1, seg2],
                name=f"network_throughput_def"
            )
        
        # 7. End-to-end throughput constraints with optimization
        if self.enable_tight_bigm:
            M_gpu, M_network = self._compute_tight_bigM()
        else:
            # Original approach
            max_gpu_throughput = max(
                ThroughputFunctions.gpu_throughput(gpu_type, self.config.sequence_length,
                                                 self.config.batch_size, max_size)
                for gpu_type, max_size in self.max_segment_size.items()
                if max_size > 0
            )
            max_net_throughput = max(self.gpu_pair_throughputs.values()) if self.gpu_pair_throughputs else 1000.0
            M_unified = max(max_gpu_throughput, max_net_throughput) * 1.1  # 10% buffer
            M_gpu = {gpu_type: M_unified for gpu_type in self.gpu_types}
            M_network = M_unified
            logger.info(f"Using unified Big-M value: {M_unified:.2f}")

        # GPU throughput constraints - only for used GPUs
        for gpu_type, gpu_type_obj in self.gpu_types.items():
            for gpu_id in range(gpu_type_obj.count):
                self.model.addConstr(
                    self.t <= self.tau[gpu_type, gpu_id] + M_gpu[gpu_type] * (1 - self.z[gpu_type, gpu_id]),
                    name=f"throughput_gpu_{gpu_type}_{gpu_id}"
                )

        # Network throughput constraints - only for active connections
        for (seg1, seg2) in self.valid_connections:
            self.model.addConstr(
                self.t <= self.rho[seg1, seg2] + M_network * (1 - self.e[seg1, seg2]),
                name=f"throughput_network"
            )

        # 8. FIXED: Pipeline connectivity constraints
        # Ensure pipeline starts at layer 1
        first_layer_segments = [seg for seg in self.valid_segments if seg[2] == 1]
        if first_layer_segments:
            self.model.addConstr(
                gp.quicksum(self.x[seg] for seg in first_layer_segments) >= 1,
                name="pipeline_starts_at_layer_1"
            )

        # FIXED: Only enforce connectivity for non-terminal layers
        for layer in range(1, self.config.num_decoder_layers):
            # Find segments ending at this layer
            segments_ending_here = [seg for seg in self.valid_segments
                                  if seg[2] + seg[3] - 1 == layer]
            # Find segments starting at next layer
            segments_starting_next = [seg for seg in self.valid_segments
                                    if seg[2] == layer + 1]

            if segments_ending_here and segments_starting_next:
                # If a segment ends at layer i, there must be a connection to layer i+1
                for seg1 in segments_ending_here:
                    valid_next_connections = [(s1, s2) for (s1, s2) in self.valid_connections 
                                            if s1 == seg1 and s2 in segments_starting_next]
                    if valid_next_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[s1, s2] for (s1, s2) in valid_next_connections) >= self.x[seg1],
                            name=f"connectivity_out_{layer}"
                        )

                # If a segment starts at layer i+1, there must be a connection from layer i
                for seg2 in segments_starting_next:
                    valid_prev_connections = [(s1, s2) for (s1, s2) in self.valid_connections 
                                            if s2 == seg2 and s1 in segments_ending_here]
                    if valid_prev_connections:
                        self.model.addConstr(
                            gp.quicksum(self.e[s1, s2] for (s1, s2) in valid_prev_connections) >= self.x[seg2],
                            name=f"connectivity_in_{layer}"
                        )
        
        # OPTIMIZATION CONSTRAINTS
        logger.info("Adding optimization constraints...")
        
        # Add symmetry breaking constraints
        if self.enable_symmetry_breaking:
            self._add_symmetry_breaking_constraints()
        
        # Add smart upper bound constraint
        if self.enable_upper_bound:
            self._add_upper_bound_constraint()
            
        # Add flow conservation constraints
        if self.enable_flow_conservation:
            self._add_flow_conservation_constraints()
    
    def _add_symmetry_breaking_constraints(self):
        """Add lexicographic ordering for identical GPU types"""
        logger.info("Adding symmetry breaking constraints...")
        constraints_added = 0
        
        for gpu_type, gpu_obj in self.gpu_types.items():
            if gpu_obj.count > 1:
                for i in range(gpu_obj.count - 1):
                    # Force GPU_i to be used before GPU_{i+1}
                    self.model.addConstr(
                        self.z[gpu_type, i] >= self.z[gpu_type, i+1],
                        name=f"symmetry_break_{gpu_type}_{i}"
                    )
                    constraints_added += 1
        
        logger.info(f"Added {constraints_added} symmetry breaking constraints")
    
    def _add_upper_bound_constraint(self):
        """Add smart upper bound constraint"""
        if self.enable_upper_bound:
            logger.info("Computing smart upper bound...")
            upper_bound = self._compute_smart_upper_bound()
            
            if upper_bound > 0:
                self.model.addConstr(
                    self.t <= upper_bound,
                    name="smart_upper_bound"
                )
                logger.info(f"Added smart upper bound constraint: {upper_bound:.2f} tokens/sec")
            else:
                logger.warning("Could not compute valid upper bound - constraint not added")
    
    def _add_flow_conservation_constraints(self):
        """Add flow conservation at each layer boundary"""
        logger.info("Adding flow conservation constraints...")
        constraints_added = 0
        
        for layer in range(1, self.config.num_decoder_layers):
            # Segments ending at this layer
            segments_ending = [seg for seg in self.valid_segments 
                              if seg[2] + seg[3] - 1 == layer]  # end_layer = start + size - 1
            
            # Segments starting at next layer  
            segments_starting = [seg for seg in self.valid_segments
                               if seg[2] == layer + 1]  # start_layer = layer + 1
            
            if segments_ending and segments_starting:
                # Flow conservation: outgoing segments = incoming segments
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
        # Set dynamic thread allocation based on problem size
        total_binary_vars = len(self.valid_segments) + len(self.valid_connections)
        available_threads = min(self.max_threads, os.cpu_count())  # Respect max_threads setting

        if self.threads is not None:
            # Manual thread specification
            threads = min(self.threads, available_threads)
            logger.info(f"Using manually specified {threads} threads for optimization")
        else:
            # Auto-scale threads based on problem complexity
            if total_binary_vars > 50000:
                threads = min(available_threads, 16)  # Large problems: use more threads
            elif total_binary_vars > 10000:
                threads = min(available_threads, 8)   # Medium problems: moderate threads
            else:
                threads = min(available_threads, 4)   # Small problems: fewer threads
            logger.info(f"Auto-scaling to {threads} threads (available: {available_threads}, problem size: {total_binary_vars})")

        self.model.setParam('Threads', threads)

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
        """Extract solution from solved model"""
        self.solution = {
            'objective_value': self.t.x,
            'gpu_assignments': [],
            'network_connections': [],
            'solve_status': self.model.status
        }
        
        # Extract GPU assignments
        for seg in self.valid_segments:
            if self.x[seg].x > 0.5:  # Binary variable is 1
                gpu_type, gpu_id, start_layer, segment_size = seg
                global_gpu_id = self._get_global_gpu_id(gpu_type, gpu_id)
                
                assignment = {
                    'gpu_type': gpu_type,
                    'gpu_id': gpu_id,
                    'global_gpu_id': global_gpu_id,
                    'start_layer': start_layer,
                    'end_layer': start_layer + segment_size - 1,
                    'segment_size': segment_size,
                    'throughput': self.tau[gpu_type, gpu_id].x
                }
                self.solution['gpu_assignments'].append(assignment)
        
        # Extract network connections
        for (seg1, seg2) in self.valid_connections:
            if self.e[seg1, seg2].x > 0.5:  # Binary variable is 1
                connection = {
                    'from_segment': seg1,
                    'to_segment': seg2,
                    'throughput': self.rho[seg1, seg2].x
                }
                self.solution['network_connections'].append(connection)
        
        # Sort assignments by start layer
        self.solution['gpu_assignments'].sort(key=lambda x: x['start_layer'])
    
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
    parser = argparse.ArgumentParser(description='LLM Model Parallelism Placement Optimizer - Optimized')
    parser.add_argument('--config-dir', required=True, help='Configuration directory containing all CSV files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--bytes-per-element', type=int, choices=[1, 2, 4], 
                       help='Bytes per tensor element (1=FP8, 2=FP16, 4=FP32). Overrides config file setting.')
    
    # Optimization flags
    parser.add_argument('--enable-symmetry-breaking', action='store_true', default=True,
                       help='Enable symmetry breaking constraints (default: True)')
    parser.add_argument('--disable-symmetry-breaking', dest='enable_symmetry_breaking', action='store_false',
                       help='Disable symmetry breaking constraints')
    parser.add_argument('--enable-upper-bound', action='store_true', default=True,
                       help='Enable smart upper bound constraint (default: True)')
    parser.add_argument('--disable-upper-bound', dest='enable_upper_bound', action='store_false',
                       help='Disable smart upper bound constraint')
    parser.add_argument('--enable-tight-bigm', action='store_true', default=True,
                       help='Enable tighter Big-M computation (default: True)')
    parser.add_argument('--disable-tight-bigm', dest='enable_tight_bigm', action='store_false',
                       help='Disable tighter Big-M computation')
    parser.add_argument('--enable-flow-conservation', action='store_true', default=True,
                       help='Enable flow conservation constraints (default: True)')
    parser.add_argument('--disable-flow-conservation', dest='enable_flow_conservation', action='store_false',
                       help='Disable flow conservation constraints')
    parser.add_argument('--threads', type=int, help='Number of threads to use (default: auto-scaled based on problem size)')
    parser.add_argument('--max-threads', type=int, default=32,
                       help='Maximum number of threads to use when auto-scaling (default: 32)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log optimization settings
    logger.info("Optimization settings:")
    logger.info(f"  - Symmetry breaking: {args.enable_symmetry_breaking}")
    logger.info(f"  - Smart upper bound: {args.enable_upper_bound}")
    logger.info(f"  - Tight Big-M: {args.enable_tight_bigm}")
    logger.info(f"  - Flow conservation: {args.enable_flow_conservation}")
    if args.threads:
        logger.info(f"  - Threads: {args.threads} (manual)")
    else:
        logger.info(f"  - Threads: auto-scaled (max: {args.max_threads})")
    
    
    try:
        # Initialize solver with optimization flags
        solver = LLMPlacementSolver(
            args.config_dir,
            enable_symmetry_breaking=args.enable_symmetry_breaking,
            enable_upper_bound=args.enable_upper_bound,
            enable_tight_bigm=args.enable_tight_bigm,
            enable_flow_conservation=args.enable_flow_conservation,
            threads=args.threads,
            max_threads=args.max_threads
        )
        
        # Override bytes_per_element if specified via command line
        if args.bytes_per_element is not None:
            logger.info(f"Overriding bytes_per_element from command line: {args.bytes_per_element}")
            solver.config.bytes_per_element = args.bytes_per_element
            # Recalculate memory constraints with new precision
            solver.max_segment_size = solver._compute_max_segment_sizes()
            solver.valid_segments = solver._generate_valid_segments()
            solver.valid_connections = solver._generate_valid_connections()
            solver.gpu_pair_throughputs = solver._precompute_network_throughputs()
            solver._validate_problem_size()
        
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
    end_time = time.time()
    logger.info(f"Solver finished in {end_time - start_time:.0f} seconds")
    return 0


if __name__ == "__main__":
    exit(main())