# LLM_placement_solver

NOTE: You need gurobi license to run the solver.

Prerequisite:
```bash
pip install -r requirements.txt
```

## how to run the solver:

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

## Configurations:

#### config-dir: `config/medium`, `config/large`, `config/hal`
config-dir has three config files

1. `config.csv`: model and solver configuration, e.g., sequence length, model name, number of decoder layers, 
2. `gpu_pool.csv`: the number of GPUs of each type
3. `network_bandwidth.csv`: the network bandwidth between each GPU type

#### method: `weighted`, `enumeration`
- weighted: quick approximate solution
- enumeration: more accurate solution but slower

#### generate-network: `generate-network [intra_bandwidth] [inter_bandwidth]`
- intra_bandwidth: bandwidth (GB/s) within same GPU type
- inter_bandwidth: bandwidth (GB/s) between different GPU types

If this argument is not given, it will use the network bandwidth configuration in the config-dir/network_bandwidth.csv
