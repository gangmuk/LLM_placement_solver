# LLM_placement_solver

NOTE: You need gurobi license to run the solver.

Prerequisite:
```bash
pip install -r requirements.txt
```

## how to run the solver:

### Using the existing network bandwidth configuration:

config/medium, dollar per token objective optimization method: weighted
```bash
python solver_constrained_with_tp-2.py --config-dir config/medium --method weighted
```

config/medium, dollar per token objective optimization method: enumeration
```bash
python solver_constrained_with_tp-2.py --config-dir config/medium --method enumeration
```

### Using the generated network bandwidth configuration:

generate-network will force the solver code to generate the network bandwidth matrix instead of reading from the existing network_bandwidth.csv file.

`--generate-network [intra_bandwidth] [inter_bandwidth]`
- intra_bandwidth: bandwidth (GB/s) within same GPU type
- inter_bandwidth: bandwidth (GB/s) between different GPU types

config/medium, dollar per token objective optimization method: weighted, network bandwidth: 600 GB/s within same GPU type, 400 GB/s between different GPU types
```bash
python solver_constrained_with_tp-2.py --config-dir config/medium --method weighted --generate-network 600 400
```

config/medium, dollar per token objective optimization method: enumeration, network bandwidth: 600 GB/s within same GPU type, 400 GB/s between different GPU types
```bash
python solver_constrained_with_tp-2.py --config-dir config/medium --method enumeration --generate-network 600 400
```

for larger model, change the config_dir to config/large


Or, check `run-solver.sh` for comprehensive runs with different configurations.