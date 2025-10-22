#!/bin/bash

# ctrl+c interrupt the script, kill the background processes
trap cleanup SIGINT
trap cleanup SIGTERM
trap cleanup SIGKILL
trap cleanup SIGQUIT

function cleanup() {
    echo "** Cleaning up..."
    kill $(jobs -p)
    exit 1
}

# config_dir_list=("config/medium" "config/large")
# cost_optimization_method_list=("weighted" "enumeration")
# network_config_list=("600 400" "400 200")
# run_on_background=true

config_dir_list=("config/medium")
# config_dir_list=("config/large")
# cost_optimization_method_list=("weighted")
cost_optimization_method_list=("enumeration")
network_config_list=("400 200")
run_on_background=false
cloud_provider="AWS"

timestamp=$(date +%Y%m%d_%H%M%S)
solver="solver_constrained_with_tp-2.py"
for config_dir in "${config_dir_list[@]}"; do
    for cost_optimization_method in "${cost_optimization_method_list[@]}"; do
        for network_config in "${network_config_list[@]}"; do
            intra_bw=$(echo ${network_config} | cut -d' ' -f1)
            inter_bw=$(echo ${network_config} | cut -d' ' -f2)
            output_log_dir="${config_dir}/output-${timestamp}"
            mkdir -p ${output_log_dir}
            output_log_path="${output_log_dir}/method_${cost_optimization_method}-network_${intra_bw}_${inter_bw}.txt"
            echo "** Starting solver, output_log_path: ${output_log_path}"
            start_time=$(date +%s)
            if [ "${network_config}" != "none" ]; then
                if [ "${run_on_background}" = true ]; then
                    python3 ${solver} --config-dir ${config_dir} --method ${cost_optimization_method} --generate-network ${intra_bw} ${inter_bw} --cloud-provider ${cloud_provider} &> ${output_log_path} &
                else
                    python3 ${solver} --config-dir ${config_dir} --method ${cost_optimization_method} --generate-network ${intra_bw} ${inter_bw} --cloud-provider ${cloud_provider} &> ${output_log_path}
                fi
            else
                if [ "${run_on_background}" = true ]; then
                    python3 ${solver} --config-dir ${config_dir} --method ${cost_optimization_method} --cloud-provider ${cloud_provider} &> ${output_log_path} &
                else
                    python3 ${solver} --config-dir ${config_dir} --method ${cost_optimization_method} --cloud-provider ${cloud_provider} &> ${output_log_path}
                fi
            fi
            if [ "${run_on_background}" = true ]; then
                solver_pid=$!
                echo "** Solver PID: ${solver_pid} for config: ${config_dir} method: ${cost_optimization_method} network: ${intra_bw} ${inter_bw}"
            else
                echo "** Solver finished for config: ${config_dir} with method: ${cost_optimization_method} and network: ${intra_bw} ${inter_bw}"
                end_time=$(date +%s)
                runtime=$((end_time - start_time))
                echo "** Solver output_log_path: ${output_log_path}, total runtime: ${runtime}"
            fi
        done
    done
done