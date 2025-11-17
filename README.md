# LLM Placement Solver

**Mathematical optimization for LLM inference deployment across heterogeneous GPU clusters**

Automatically find **optimal layer-to-GPU mappings** with Tensor + Pipeline parallelism. Supports any LLM size (32-80+ layers), heterogeneous GPU clusters, real cloud pricing, and advanced performance modeling with roofline analysis, memory consumption tracking, and network topology awareness.

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```
**Note**: Requires Gurobi license.

### Basic Usage

Change the setup(config) that you want to run in the `run-batch-sweep.sh` script. e.g., `config_dir_list=("config/medium")`

Run `./run-batch-sweep.sh`

The results will be saved in the `config_dir/output-${timestamp}` directory.

For summary of the results, check the `config_dir/output-${timestamp}/batch_sweep_results.csv` file.

Example output log:
```text
RESULTS SUMMARY:
--------------------------------------------------------------------------------------------
Batch Size   Throughput      Cost ($/h)      $/M tokens           Status                   
--------------------------------------------------------------------------------------------
32           2035.99         8.19            1.117735             ✓ SUCCESS (5 tested)   
64           2035.99         8.19            1.117735             ✓ SUCCESS (5 tested)   
128          3775.18         16.39           1.205609             ✓ SUCCESS (5 tested)   
256          3775.18         16.39           1.205609             ✓ SUCCESS (5 tested)   
--------------------------------------------------------------------------------------------

* Best batch size = 32
Cost per M tokens: $1.117735
```

Example batch_sweep_results.csv:
```csv
batch_size,budget_tested,throughput_tokens_per_sec,cost_per_hour,cost_per_million_tokens,total_runtime_hours,total_cost,pipeline_stages,gpu_type,tp_degree,num_gpus,total_layers,is_best,status
32,8.19,2035.99,8.19,1.117735,0.14,1.12,1,A100,2,2,32,YES,SUCCESS
32,9.83,2035.99,8.19,1.117735,0.14,1.12,1,A100,2,2,32,YES,SUCCESS
32,12.29,2035.99,8.19,1.117735,0.14,1.12,1,A100,2,2,32,YES,SUCCESS
32,157.30,6858.55,32.77,1.327215,0.04,1.33,1,A100,8,8,32,NO,SUCCESS
32,314.59,6858.55,32.77,1.327215,0.04,1.33,1,A100,8,8,32,NO,SUCCESS
64,8.19,2035.99,8.19,1.117735,0.14,1.12,1,A100,2,2,32,YES,SUCCESS
64,9.83,2035.99,8.19,1.117735,0.14,1.12,1,A100,2,2,32,YES,SUCCESS
64,12.29,2035.99,8.19,1.117735,0.14,1.12,1,A100,2,2,32,YES,SUCCESS
64,157.30,6858.55,32.77,1.327215,0.04,1.33,1,A100,8,8,32,NO,SUCCESS
64,314.59,6858.55,32.77,1.327215,0.04,1.33,1,A100,8,8,32,NO,SUCCESS
128,16.39,3775.18,16.39,1.205609,0.07,1.21,1,A100,4,4,32,YES,SUCCESS
128,19.66,3775.18,16.39,1.205609,0.07,1.21,1,A100,4,4,32,YES,SUCCESS
128,24.58,3775.18,16.39,1.205609,0.07,1.21,1,A100,4,4,32,YES,SUCCESS
128,157.30,6858.55,32.77,1.327215,0.04,1.33,1,A100,8,8,32,NO,SUCCESS
128,314.59,6858.55,32.77,1.327215,0.04,1.33,1,A100,8,8,32,NO,SUCCESS
256,16.39,3775.18,16.39,1.205609,0.07,1.21,1,A100,4,4,32,YES,SUCCESS
256,19.66,3775.18,16.39,1.205609,0.07,1.21,1,A100,4,4,32,YES,SUCCESS
256,24.58,3775.18,16.39,1.205609,0.07,1.21,1,A100,4,4,32,YES,SUCCESS
256,157.30,6858.55,32.77,1.327215,0.04,1.33,1,A100,8,8,32,NO,SUCCESS
256,314.59,6858.55,32.77,1.327215,0.04,1.33,1,A100,8,8,32,NO,SUCCESS
```


## What It Supports

### Model Architecture
- **Arbitrary number of decoder layers**: Supports any LLM size (Llama3 8B: 32 layers, Llama3 70B: 80 layers, etc.)
- **Standard transformer architecture**: Attention + FFN layers with configurable dimensions
- **Memory consumption modeling**: Accounts for model weights, activations, and KV cache tensors

### GPU Cluster Configuration
- **Heterogeneous GPUs**: Mix of different GPU types (A100, V100, H100, L40S, etc.) in any combination
- **Arbitrary cluster sizes**: From single GPU to hundreds of GPUs
- **Network topology awareness**: Custom bandwidth matrices between GPU pairs
- **GPU performance hierarchy**: Automatically prioritizes better GPUs for higher tensor parallelism

### Cost Optimization
- **Real cloud pricing**: Integrated pricing for AWS, GCP, Azure, Lambda, CoreWeave, Nebius
- **Multi-objective optimization**: Balance throughput vs. cost with configurable weights
- **Cost-per-token minimization**: Guaranteed optimal solutions via mathematical programming
- **Total cost budgeting**: Respect maximum hourly costs and total processing budgets

### Performance Modeling
- **Roofline model**: Accurate throughput prediction based on compute/memory bottlenecks
- **Arithmetic intensity analysis**: Determines if workloads are compute-bound or memory-bound
- **Batch size optimization**: Power-of-2 batch sizes with efficiency factors
- **Pipeline bubble modeling**: Accounts for communication overhead in pipelined execution

### Practical Constraints
- **Memory limits**: Per-GPU memory constraints with tensor parallelism sharding
- **Network bandwidth filtering**: Automatically filters low-bandwidth connections
- **Pipeline depth limits**: Configurable maximum number of pipeline stages
- **Minimum memory utilization**: Ensures efficient GPU usage

## What It Does

**Automatically finds optimal mappings from model layers to GPU instances with parallelism strategies**

### The Problem
You have a large language model (e.g., 40 layers) and a heterogeneous GPU cluster (mix of A100s, V100s, etc.). You need to:
- Split layers across pipeline stages (Pipeline Parallelism)
- Shard weights within each stage (Tensor Parallelism)
- Respect memory constraints and network bandwidth
- Minimize cost while maximizing throughput

### The Solution
The solver outputs an optimized placement like this:

```
LLM Model (40 layers) → GPU Cluster Mapping

Pipeline Stage 1: Layers 1-10 on 2×A100 GPUs (TP=2)
    ├── GPU 0 (A100): Layers 1-10, weights sharded ½
    └── GPU 1 (A100): Layers 1-10, weights sharded ½

Pipeline Stage 2: Layers 11-25 on 4×V100 GPUs (TP=4)
    ├── GPU 2 (V100): Layers 11-25, weights sharded ¼
    ├── GPU 3 (V100): Layers 11-25, weights sharded ¼
    ├── GPU 4 (V100): Layers 11-25, weights sharded ¼
    └── GPU 5 (V100): Layers 11-25, weights sharded ¼

Pipeline Stage 3: Layers 26-40 on 1×H100 GPU (TP=1)
    └── GPU 6 (H100): Layers 26-40, full weights

Result: 2,035 tokens/sec @ $8.19/hour ($1.12 per million tokens)
```

**Key Insights:**
- **Pipeline Parallelism**: Layers distributed across 3 stages for memory efficiency
- **Tensor Parallelism**: Within stages, weights sharded across multiple GPUs
- **GPU Hierarchy**: Better GPUs (H100) handle final layers, older GPUs (V100s) in middle
- **Cost Optimization**: Balances compute efficiency vs. electricity/network costs

## Who This Is For

- **ML Engineers** deploying LLMs in production
- **Cloud Architects** optimizing infrastructure costs
- **Researchers** studying distributed LLM inference
- **DevOps Teams** managing GPU clusters

## Configurations:

### Cost

### config-dir: `config/medium`, `config/large`, `config/hal`
config-dir has three config files

1. `config.csv`: model and solver configuration, e.g., sequence length, model name, number of decoder layers, 
2. `gpu_pool.csv`: the number of GPUs of each type
3. `network_bandwidth.csv`: the network bandwidth between each GPU type

### method: `weighted`, `enumeration`
- weighted: quick approximate solution
- enumeration: more accurate solution but slower

### generate-network: `generate-network [intra_bandwidth] [inter_bandwidth]`
- intra_bandwidth: bandwidth (GB/s) within same GPU type
- inter_bandwidth: bandwidth (GB/s) between different GPU types

If this argument is not given, it will use the network bandwidth configuration in the config-dir/network_bandwidth.csv
