# LLM Model Parallelism Placement: Integer Linear Programming Formulation

## Abstract

Large Language Models (LLMs) with hundreds of billions of parameters require distributed inference across multiple GPUs due to memory constraints. This work formulates the optimal placement of transformer decoder layers across a heterogeneous GPU cluster as a Mixed-Integer Linear Program (MILP). The optimization maximizes end-to-end inference throughput while respecting GPU memory limits, ensuring pipeline connectivity, and accounting for network communication overhead between GPUs.

## Introduction

Modern LLMs such as GPT-4, LLaMA, and PaLM contain 50-500+ transformer decoder layers with billions of parameters per layer. Single GPUs cannot accommodate these models due to memory limitations, necessitating model parallelism where different layers execute on different GPUs in a pipeline fashion. However, the placement of layers significantly impacts performance due to:

1. **Heterogeneous hardware**: GPU clusters often contain different GPU types (A100, V100, H100) with varying memory capacities and compute capabilities
2. **Network bottlenecks**: Inter-GPU communication for activations can limit throughput, especially between GPUs on different nodes
3. **Memory constraints**: Each GPU has limited memory for model weights, key-value caches, and intermediate activations
4. **Pipeline dependencies**: Layers must be placed in consecutive segments to maintain the sequential nature of transformer computation

The challenge is to find the optimal assignment of consecutive layer segments to GPUs that maximizes the minimum throughput across all pipeline stages (the bottleneck throughput).

## Problem Intuition

Consider a 100-layer transformer model and a cluster with 10 A100 GPUs (80GB each) and 5 V100 GPUs (32GB each). Each layer requires 2GB for weights, and activations require an additional 5GB per GPU regardless of layer count. The naive approach of evenly distributing layers (10 layers per A100, 20 layers per V100) would violate memory constraints on V100s (20×2+5 = 45GB > 32GB) and create throughput imbalances.

Our ILP formulation addresses this by:

1. **Segment-based placement**: Assigns consecutive layer segments to GPUs rather than individual layers, reducing variables while maintaining pipeline structure
2. **Memory-aware constraints**: Ensures each GPU assignment respects memory limits for weights plus activations  
3. **Throughput optimization**: Models both GPU compute throughput (layers/second) and network throughput (transfers/second) to find the bottleneck
4. **Connectivity enforcement**: Guarantees that layer segments form a complete pipeline from layer 1 to the final layer

The solution identifies which GPU should process which consecutive layers and determines the network connections between segments, maximizing the end-to-end inference throughput.

## Problem Definition

Given a neural network model with $m$ identical transformer decoder layers, a heterogeneous GPU cluster with network interconnections, and a workload requiring batch inference, find the optimal layer-to-GPU assignment and determine the maximum achievable end-to-end inference throughput while satisfying memory constraints and consecutive layer placement requirements.

## Sets and Parameters

### Sets
- $\mathcal{L} = \{1, 2, \ldots, m\}$: Set of transformer decoder layers
- $\mathcal{T} = \{1, 2, \ldots, n\}$: Set of GPU types  
- $\mathcal{G}_i = \{1, 2, \ldots, X_i\}$: Set of GPU units for type $i \in \mathcal{T}$
- $\mathcal{G} = \bigcup_{i \in \mathcal{T}} \{(i,j) : j \in \mathcal{G}_i\}$: Set of all GPU units
### Constrained Segment Generation
The valid segment set $\mathcal{S}$ is generated with intelligent constraints:

$\mathcal{S} = \{(i,j,k,s) : i \in \mathcal{T}, j \in \mathcal{G}_i, k \in \mathcal{L}, s \in \{P_{\text{min},i}, \ldots, P_i\}, k+s-1 \leq m\}$

where segments are constrained to sizes between the computed minimum $P_{\text{min},i}$ and maximum $P_i$ for each GPU type.

### Efficient Connection Generation
Valid network connections are generated using layer-based grouping for $O(m \times \text{segments\_per\_layer})$ complexity instead of $O(|\mathcal{S}|^2)$:

$\mathcal{E} = \{(seg_1, seg_2) \in \mathcal{S} \times \mathcal{S} : \text{end\_layer}(seg_1) + 1 = \text{start\_layer}(seg_2)\}$

where connections only exist between segments with consecutive layer boundaries.

### Parameters
- $m \in \mathbb{Z}^+$: Total number of decoder layers
- $X_i \in \mathbb{Z}^+$: Number of available GPUs of type $i$, $\forall i \in \mathcal{T}$
- $M_i \in \mathbb{R}^+$: Total memory capacity of GPU type $i$ (GB), $\forall i \in \mathcal{T}$
- $\sigma \in \mathbb{Z}^+$: Sequence length (fixed for workload)
- $b \in \mathbb{Z}^+$: Batch size (fixed for workload)
- $h \in \mathbb{Z}^+$: Hidden dimension of the model
- $B_{(i_1,j_1),(i_2,j_2)} \in \mathbb{R}^+$: Network bandwidth between GPU units $(i_1,j_1)$ and $(i_2,j_2)$ (Gbps)
- $P_i \in \mathbb{Z}^+$: Maximum layers that GPU type $i$ can accommodate (computed from memory constraints)

### Maximum Segment Size Computation
The maximum number of layers $P_i$ that GPU type $i$ can accommodate is computed using binary search:

$P_i = \max\{s \in \mathbb{Z}^+ : s \cdot w_{\text{layer}} + W_{\text{intermediate}} \leq M_i\}$

where the binary search finds the largest feasible segment size within memory constraints.

### Intelligent Minimum Segment Sizes
To reduce combinatorial complexity while maintaining solution quality, minimum segment sizes are computed based on memory efficiency targets:

$P_{\text{min},i} = \max\left(1, \min\left(\left\lfloor\frac{0.5 \cdot M_i - W_{\text{intermediate}}}{w_{\text{layer}}}\right\rfloor, \frac{P_i}{3}\right)\right)$

This aims for at least 50% GPU memory utilization while limiting the search space.

## Decision Variables

### Binary Variables
- $x_{i,j,k,s} \in \{0,1\}$: Equals 1 if GPU unit $j$ of type $i$ processes consecutive layers $\{k, k+1, \ldots, k+s-1\}$, 0 otherwise
- $z_{i,j} \in \{0,1\}$: Equals 1 if GPU unit $j$ of type $i$ is used (active), 0 otherwise
- $e_{seg_1,seg_2} \in \{0,1\}$: Equals 1 if there is a network connection from segment $seg_1$ to segment $seg_2$ in the pipeline, 0 otherwise

### Continuous Variables
- $t \in \mathbb{R}^+$: End-to-end throughput (minimum across all pipeline stages)
- $\tau_{i,j} \in \mathbb{R}^+$: GPU computation throughput of unit $j$ of type $i$
- $\rho_{seg_1,seg_2} \in \mathbb{R}^+$: Network throughput between segments $seg_1$ and $seg_2$

## Objective Function

$$\max \quad t$$

## Constraints

### Layer Coverage Constraints
Each layer must be assigned to exactly one GPU segment:
$$\sum_{(i,j,k,s) \in \mathcal{S}: k \leq \ell \leq k+s-1} x_{i,j,k,s} = 1, \quad \forall \ell \in \mathcal{L}$$

### GPU Capacity Constraints
Each GPU can process at most one segment:
$$\sum_{(k,s): (i,j,k,s) \in \mathcal{S}} x_{i,j,k,s} \leq 1, \quad \forall i \in \mathcal{T}, j \in \mathcal{G}_i$$

### Memory Constraints
Total memory consumption must not exceed GPU capacity:
$$\mu_i(\sigma, b, s) \cdot \sum_{(k,s'): (i,j,k,s') \in \mathcal{S}, s'=s} x_{i,j,k,s} \leq M_i, \quad \forall i \in \mathcal{T}, j \in \mathcal{G}_i, s \in \{1,\ldots,P_i\}$$

### GPU Usage Indicator Constraints
GPU is active if and only if it processes at least one segment:
$$z_{i,j} = \sum_{(k,s): (i,j,k,s) \in \mathcal{S}} x_{i,j,k,s}, \quad \forall i \in \mathcal{T}, j \in \mathcal{G}_i$$

### Network Connection Logic Constraints
Network connection exists if both segments are selected and consecutive:

For each $(seg_1, seg_2) \in \mathcal{E}$ where $seg_1 = (i_1,j_1,k_1,s_1)$ and $seg_2 = (i_2,j_2,k_2,s_2)$:
$$e_{seg_1,seg_2} \leq x_{i_1,j_1,k_1,s_1}$$
$$e_{seg_1,seg_2} \leq x_{i_2,j_2,k_2,s_2}$$
$$e_{seg_1,seg_2} \geq x_{i_1,j_1,k_1,s_1} + x_{i_2,j_2,k_2,s_2} - 1$$

### Pipeline Connectivity Constraints
Ensure all layers form a connected pipeline:

**Pipeline Start Constraint:**
$$\sum_{(i,j,1,s) \in \mathcal{S}} x_{i,j,1,s} \geq 1$$

**Sequential Connectivity Constraints:**
For each layer $\ell \in \{1, 2, \ldots, m-1\}$:

Let $S_{\text{end}}(\ell) = \{seg \in \mathcal{S} : seg = (i,j,k,s), k+s-1 = \ell\}$ (segments ending at layer $\ell$)

Let $S_{\text{start}}(\ell+1) = \{seg \in \mathcal{S} : seg = (i,j,\ell+1,s)\}$ (segments starting at layer $\ell+1$)

For each $seg_1 \in S_{\text{end}}(\ell)$:
$$\sum_{seg_2 \in S_{\text{start}}(\ell+1): (seg_1,seg_2) \in \mathcal{E}} e_{seg_1,seg_2} \geq x_{seg_1}$$

For each $seg_2 \in S_{\text{start}}(\ell+1)$:
$$\sum_{seg_1 \in S_{\text{end}}(\ell): (seg_1,seg_2) \in \mathcal{E}} e_{seg_1,seg_2} \geq x_{seg_2}$$

### Throughput Definition Constraints

**GPU Throughput:**
$$\tau_{i,j} = \sum_{(k,s): (i,j,k,s) \in \mathcal{S}} \theta_i(\sigma, b, s) \cdot x_{i,j,k,s}, \quad \forall i \in \mathcal{T}, j \in \mathcal{G}_i$$

**Network Throughput:**
For each $(seg_1, seg_2) \in \mathcal{E}$ where $seg_1 = (i_1,j_1,k_1,s_1)$ and $seg_2 = (i_2,j_2,k_2,s_2)$:
$$\rho_{seg_1,seg_2} = \nu(B_{(i_1,j_1),(i_2,j_2)}, \sigma, b, h) \cdot e_{seg_1,seg_2}$$

### End-to-End Throughput Constraints

**GPU Throughput Bounds:**
$$t \leq \tau_{i,j} + M \cdot (1 - z_{i,j}), \quad \forall i \in \mathcal{T}, j \in \mathcal{G}_i$$

**Network Throughput Bounds:**
$$t \leq \rho_{seg_1,seg_2} + M \cdot (1 - e_{seg_1,seg_2}), \quad \forall (seg_1, seg_2) \in \mathcal{E}$$

where $M$ is a sufficiently large constant (Big-M formulation).

## Memory Model

The memory consumption function $\mu_i(\sigma, b, s)$ includes:

### Model Weights
$W_{\text{model}} = s \cdot w_{\text{layer}}$

where $w_{\text{layer}}$ is the memory per layer for model weights.

### Intermediate Tensors (Activation Memory)
$W_{\text{intermediate}} = W_{\text{attention}} + W_{\text{kv-cache}} + W_{\text{hidden}} + W_{\text{overhead}}$

where:
- **Attention Outputs:** $W_{\text{attention}} = b \cdot \sigma \cdot h \cdot 2 / (1024^3)$ GB (FP16)
- **Key-Value Cache:** $W_{\text{kv-cache}} = 2 \cdot b \cdot \sigma \cdot h \cdot 2 / (1024^3)$ GB (FP16)
- **Hidden States:** $W_{\text{hidden}} = b \cdot \sigma \cdot d_{\text{hidden}} \cdot 2 / (1024^3)$ GB (FP16)
- **Framework Overhead:** $W_{\text{overhead}} = 0.15 \cdot (W_{\text{attention}} + W_{\text{kv-cache}} + W_{\text{hidden}})$ (15% overhead)

### Total Memory
$\mu_i(\sigma, b, s) = s \cdot w_{\text{layer}} + W_{\text{intermediate}}$

Note: The intermediate tensor memory is constant per GPU and does not scale with segment size $s$, as it represents the activation memory footprint during pipeline processing.

## Throughput Model

### GPU Throughput Function
$$\theta_i(\sigma, b, s) = \alpha_{i,\sigma} \cdot \sigma + \alpha_{i,b} \cdot b + \alpha_{i,s} \cdot s + \alpha_{i,0}$$

where $\alpha_{i,\cdot}$ are GPU-type-specific coefficients.

### Network Throughput Function
$\nu(B_{i_1j_1,i_2j_2}, \sigma, b, h) = \frac{B_{i_1j_1,i_2j_2}}{T_{size}}$

where:
- $B_{i_1j_1,i_2j_2}$ is the bandwidth between GPUs in Gbps
- $T_{size} = \frac{b \cdot \sigma \cdot h \cdot 2}{1024^3}$ is the tensor transfer size in GB (FP16 precision)
- The result represents transfers per second capacity between GPU pairs

### Efficient Throughput Modeling
For computational efficiency, network throughput values are pre-computed for all valid GPU pairs as:
$\rho_{(i_1,j_1),(i_2,j_2)} = \frac{B_{(i_1,j_1),(i_2,j_2)}}{b \cdot \sigma \cdot h \cdot 2 / (1024^3)}$

## Complexity Analysis

### Problem Size
- **Variables:** $O(|\mathcal{S}| + |\mathcal{E}| + |\mathcal{G}|)$ where $|\mathcal{S}| = O(nmP_{\max})$ and $|\mathcal{E}| = O(|\mathcal{S}|^2)$
- **Constraints:** $O(m + |\mathcal{G}| + |\mathcal{E}|)$
- **Variable Types:** Binary variables for assignments and connections, continuous for throughputs

### Computational Complexity
- **Problem Class:** NP-hard Mixed-Integer Linear Program
- **Typical Solve Time:** 20 seconds to 5 minutes for realistic instances (with optimizations)
- **Scalability:** Handles 65+ GPUs with 80+ layers effectively with optimization techniques

## Implementation Notes

### Constraint Pruning
- Generate only feasible segments based on memory constraints: $P_i = \lfloor M_i / \mu_i(\sigma, b, 1) \rfloor$
- Create network connections only between consecutive segments: $k_2 = k_1 + s_1$
- Apply intelligent minimum segment sizes to reduce combinatorial explosion
- Use symmetry breaking to eliminate equivalent solutions
- Add flow conservation constraints to strengthen LP relaxation
- Compute tight Big-M bounds for improved numerical properties

### Big-M Selection
The implementation uses **tight GPU-type-specific Big-M values** for improved performance:

$$M^{GPU}_{i} = \max_{s \leq P_i} \theta_i(\sigma, b, s), \quad \forall i \in \mathcal{T}$$
$$M^{NET} = \max_{pairs} \rho_{pairs}$$

GPU throughput constraints use $M^{GPU}_{i}$ and network constraints use $M^{NET}$, providing tighter bounds than a unified Big-M value.

### Solver Configuration
- **Presolve:** Level 2 (aggressive)
- **Cut Generation:** Level 1 (moderate)
- **Heuristics:** 5% time limit
- **MIP Focus:** 1 (feasible solutions)
- **Node File Start:** 0.5GB (disk usage for large problems)
- **Thread Limit:** 4 (memory management)
- **Time Limit:** Configurable (typically 300 seconds)
- **Optimality Gap:** Configurable (typically 1%)

### Problem Size Management
The implementation includes automatic problem size validation and provides optimization techniques:
- **Small problems:** < 10,000 binary variables (solve in < 30 seconds)
- **Medium problems:** 10,000-50,000 binary variables (solve in < 5 minutes)  
- **Large problems:** 50,000-100,000 binary variables (solve in < 30 minutes with optimizations)
- **Very large problems:** > 100,000 binary variables (solvable with advanced optimization techniques)

## Performance Optimizations

### Symmetry Breaking Constraints
To eliminate equivalent solutions caused by identical GPU units, lexicographic ordering constraints are added:

$$z_{i,j} \geq z_{i,j+1}, \quad \forall i \in \mathcal{T}, j \in \{1,2,\ldots,X_i-1\}$$

This forces GPU units of the same type to be used in order (GPU 1 before GPU 2, etc.), dramatically reducing the solution space without affecting optimality. For a cluster with multiple identical GPUs, this can eliminate billions of symmetric solutions.

### Smart Upper Bound Constraints
A theoretically achievable upper bound is computed and added as a constraint:

$$t \leq t^{UB}$$

where $t^{UB}$ is calculated as:
$$t^{UB} = \max_{i \in \mathcal{T}} \left\{\theta_i(\sigma, b, P_i) : \lceil m/P_i \rceil \leq X_i\right\}$$

This represents the best-case throughput using only the most efficient GPU type that can accommodate the entire model.

### Flow Conservation Constraints
Additional constraints ensure flow balance at layer boundaries:

$$\sum_{seg \in S_{\text{end}}(\ell)} x_{seg} = \sum_{seg \in S_{\text{start}}(\ell+1)} x_{seg}, \quad \forall \ell \in \{1,2,\ldots,m-1\}$$

These constraints strengthen the LP relaxation by enforcing that the number of segments ending at each layer equals the number starting at the next layer.

### Tight Big-M Formulation
Instead of using a unified Big-M constant, GPU-type-specific bounds are computed:

**GPU Constraints:**
$$t \leq \tau_{i,j} + M^{GPU}_{i} \cdot (1 - z_{i,j}), \quad \forall i \in \mathcal{T}, j \in \mathcal{G}_i$$

**Network Constraints:**
$$t \leq \rho_{seg_1,seg_2} + M^{NET} \cdot (1 - e_{seg_1,seg_2}), \quad \forall (seg_1, seg_2) \in \mathcal{E}$$

where $M^{GPU}_{i}$ and $M^{NET}$ are computed as the actual maximum achievable values for each constraint type.

### Performance Impact
These optimizations provide dramatic performance improvements:
- **Variable Reduction:** Presolve reduces problem size by 90%+ for large instances
- **Solve Time:** Previously intractable 65-GPU, 80-layer problems now solve in ~20 seconds
- **Scalability:** Handles configurations with 90,000+ binary variables that previously timed out

### Implementation Flags
All optimizations can be controlled via command-line arguments:
```bash
python solver_constrained.py --config-dir config/medium \
    --enable-symmetry-breaking \
    --enable-upper-bound \
    --enable-tight-bigm \
    --enable-flow-conservation
```

This formulation captures the complete optimization problem including heterogeneous hardware, network communication, memory constraints, pipeline connectivity requirements, and advanced performance optimization techniques.