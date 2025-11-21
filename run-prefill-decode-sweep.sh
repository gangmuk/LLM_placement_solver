#!/bin/bash

# Prefill-Decode Batch Size Sweep Runner
# Runs solver for BOTH prefill and decode phases

trap cleanup SIGINT SIGTERM SIGKILL SIGQUIT

function cleanup() {
    echo "** Cleaning up..."
    kill $(jobs -p) 2>/dev/null
    exit 1
}

# Configuration
batch_size_list=(32 64 128 256)
cost_optimization_method_list=("enumeration")
network_config_list=("400 200")
cloud_provider="AWS"
solver="solver.py"
# target_config="hal"
target_config="medium"
timestamp=$(date +%Y%m%d_%H%M%S)

echo "========================================================================"
echo "PREFILL-DECODE DISAGGREGATION BATCH SWEEP"
echo "========================================================================"
echo "Running for BOTH prefill and decode phases"
echo "Batch sizes: ${batch_size_list[@]}"
echo "Timestamp: ${timestamp}"
echo ""

# Run PREFILL phase
echo "========================================================================"
echo "PHASE 1: PREFILL (Compute-bound, O(n²) attention)"
echo "========================================================================"
config_dir="config/${target_config}-prefill"
phase_label="PREFILL"

declare -a pids_prefill
declare -a output_logs_prefill
declare -a batch_sizes_prefill

for cost_optimization_method in "${cost_optimization_method_list[@]}"; do
    for network_config in "${network_config_list[@]}"; do
        intra_bw=$(echo ${network_config} | cut -d' ' -f1)
        inter_bw=$(echo ${network_config} | cut -d' ' -f2)
        
        for batch_size in "${batch_size_list[@]}"; do
            output_log_dir="${config_dir}/output-${timestamp}/batch_${batch_size}"
            mkdir -p ${output_log_dir}
            
            output_log_path="${output_log_dir}/method_${cost_optimization_method}-network_${intra_bw}_${inter_bw}.txt"
            
            # Copy config files
            cp ${config_dir}/gpu_pool.csv ${output_log_dir}/ 2>/dev/null || true
            cp ${config_dir}/network_bandwidth.csv ${output_log_dir}/ 2>/dev/null || true
            
            # Modify batch size in config
            original_config="${config_dir}/config.csv"
            batch_config="${output_log_dir}/config.csv"
            
            awk -F',' -v bs="${batch_size}" '
                BEGIN {OFS=","}
                $1 == "min_batch_size" {$2 = bs}
                $1 == "max_batch_size" {$2 = bs}
                {print}
            ' "${original_config}" > "${batch_config}"
            
            echo "** [PREFILL-Batch ${batch_size}] Starting solver..."
            echo "   Config dir: ${output_log_dir}"
            
            python3 ${solver} \
                --config-dir ${output_log_dir} \
                --method ${cost_optimization_method} \
                --generate-network ${intra_bw} ${inter_bw} \
                --cloud-provider ${cloud_provider} \
                &> ${output_log_path} &
            
            solver_pid=$!
            pids_prefill+=($solver_pid)
            output_logs_prefill+=("${output_log_path}")
            batch_sizes_prefill+=($batch_size)
            
            echo "   PID: ${solver_pid}"
            echo ""
        done
    done
done

echo "Waiting for PREFILL phase to complete..."
for i in "${!pids_prefill[@]}"; do
    pid=${pids_prefill[$i]}
    batch_size=${batch_sizes_prefill[$i]}
    echo "  Waiting for PREFILL batch ${batch_size} (PID: ${pid})..."
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ PREFILL batch ${batch_size} completed"
    else
        echo "  ✗ PREFILL batch ${batch_size} failed (exit code ${exit_code})"
    fi
done

echo ""
echo "========================================================================"
echo "PHASE 2: DECODE (Memory-bound, O(n) KV cache access)"
echo "========================================================================"
config_dir="config/${target_config}-decode"

declare -a pids_decode
declare -a output_logs_decode
declare -a batch_sizes_decode

for cost_optimization_method in "${cost_optimization_method_list[@]}"; do
    for network_config in "${network_config_list[@]}"; do
        intra_bw=$(echo ${network_config} | cut -d' ' -f1)
        inter_bw=$(echo ${network_config} | cut -d' ' -f2)
        
        for batch_size in "${batch_size_list[@]}"; do
            output_log_dir="${config_dir}/output-${timestamp}/batch_${batch_size}"
            mkdir -p ${output_log_dir}
            
            output_log_path="${output_log_dir}/method_${cost_optimization_method}-network_${intra_bw}_${inter_bw}.txt"
            
            # Copy config files
            cp ${config_dir}/gpu_pool.csv ${output_log_dir}/ 2>/dev/null || true
            cp ${config_dir}/network_bandwidth.csv ${output_log_dir}/ 2>/dev/null || true
            
            # Modify batch size in config
            original_config="${config_dir}/config.csv"
            batch_config="${output_log_dir}/config.csv"
            
            awk -F',' -v bs="${batch_size}" '
                BEGIN {OFS=","}
                $1 == "min_batch_size" {$2 = bs}
                $1 == "max_batch_size" {$2 = bs}
                {print}
            ' "${original_config}" > "${batch_config}"
            
            echo "** [DECODE-Batch ${batch_size}] Starting solver..."
            echo "   Config dir: ${output_log_dir}"
            
            python3 ${solver} \
                --config-dir ${output_log_dir} \
                --method ${cost_optimization_method} \
                --generate-network ${intra_bw} ${inter_bw} \
                --cloud-provider ${cloud_provider} \
                &> ${output_log_path} &
            
            solver_pid=$!
            pids_decode+=($solver_pid)
            output_logs_decode+=("${output_log_path}")
            batch_sizes_decode+=($batch_size)
            
            echo "   PID: ${solver_pid}"
            echo ""
        done
    done
done

echo "Waiting for DECODE phase to complete..."
for i in "${!pids_decode[@]}"; do
    pid=${pids_decode[$i]}
    batch_size=${batch_sizes_decode[$i]}
    echo "  Waiting for DECODE batch ${batch_size} (PID: ${pid})..."
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ DECODE batch ${batch_size} completed"
    else
        echo "  ✗ DECODE batch ${batch_size} failed (exit code ${exit_code})"
    fi
done

echo ""
echo "========================================================================"
echo "GENERATING SUMMARY REPORTS"
echo "========================================================================"

# Analyze PREFILL results
echo ""
echo "=== PREFILL PHASE RESULTS ==="
config_dir="config/${target_config}-prefill"
results_csv="${config_dir}/output-${timestamp}/batch_sweep_results.csv"

first_batch=true
for batch_size in "${batch_size_list[@]}"; do
    csv_file="${config_dir}/output-${timestamp}/batch_${batch_size}/solution_summary.csv"
    if [ -f "$csv_file" ]; then
        if [ "$first_batch" = true ]; then
            cat "$csv_file" > "$results_csv"
            first_batch=false
        else
            tail -n +2 "$csv_file" >> "$results_csv"
        fi
    fi
done

printf "%-10s %-12s %-12s %-10s %-8s %-8s %-15s %-15s\n" "Batch" "Throughput" "Cost(\$/h)" "\$/M tok" "PP" "TP" "#GPUs" "GPU"
echo "------------------------------------------------------------------------------------------------"
for batch_size in "${batch_size_list[@]}"; do
    csv_file="${config_dir}/output-${timestamp}/batch_${batch_size}/solution_summary.csv"
    if [ -f "$csv_file" ]; then
        if grep -q "is_best" "$csv_file"; then
            data=$(grep ",YES," "$csv_file" | head -1)
            throughput=$(echo "$data" | cut -d',' -f3)
            cost=$(echo "$data" | cut -d',' -f4)
            cost_per_m=$(echo "$data" | cut -d',' -f5)
            stages=$(echo "$data" | cut -d',' -f8)
            gpu_type=$(echo "$data" | cut -d',' -f9)
            tp_degree=$(echo "$data" | cut -d',' -f10)
            num_gpus=$(echo "$data" | cut -d',' -f11)
        else
            data=$(tail -n 1 "$csv_file")
            throughput=$(echo "$data" | cut -d',' -f2)
            cost=$(echo "$data" | cut -d',' -f3)
            cost_per_m=$(echo "$data" | cut -d',' -f4)
            stages=$(echo "$data" | cut -d',' -f7)
            gpu_type=$(echo "$data" | cut -d',' -f8)
            tp_degree=$(echo "$data" | cut -d',' -f9)
            num_gpus=$(echo "$data" | cut -d',' -f10)
        fi
        printf "%-10s %-12s %-12s %-10s %-8s %-8s %-15s %-15s\n" \
            "$batch_size" "${throughput:-N/A}" "${cost:-N/A}" "${cost_per_m:-N/A}" \
            "${stages:-N/A}" "${tp_degree:-N/A}" "${num_gpus:-N/A}" "${gpu_type:-N/A}"
    else
        printf "%-10s %-12s %-12s %-10s %-8s %-8s %-15s %-15s\n" \
            "$batch_size" "FAILED" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A"
    fi
done

# Analyze DECODE results
echo ""
echo "=== DECODE PHASE RESULTS ==="
config_dir="config/${target_config}-decode"
results_csv="${config_dir}/output-${timestamp}/batch_sweep_results.csv"

first_batch=true
for batch_size in "${batch_size_list[@]}"; do
    csv_file="${config_dir}/output-${timestamp}/batch_${batch_size}/solution_summary.csv"
    if [ -f "$csv_file" ]; then
        if [ "$first_batch" = true ]; then
            cat "$csv_file" > "$results_csv"
            first_batch=false
        else
            tail -n +2 "$csv_file" >> "$results_csv"
        fi
    fi
done

printf "%-10s %-12s %-12s %-10s %-8s %-8s %-15s %-15s\n" "Batch" "Throughput" "Cost(\$/h)" "\$/M tok" "Stages" "TP" "#GPUs" "GPU"
echo "------------------------------------------------------------------------------------------------"
for batch_size in "${batch_size_list[@]}"; do
    csv_file="${config_dir}/output-${timestamp}/batch_${batch_size}/solution_summary.csv"
    if [ -f "$csv_file" ]; then
        if grep -q "is_best" "$csv_file"; then
            data=$(grep ",YES," "$csv_file" | head -1)
            throughput=$(echo "$data" | cut -d',' -f3)
            cost=$(echo "$data" | cut -d',' -f4)
            cost_per_m=$(echo "$data" | cut -d',' -f5)
            stages=$(echo "$data" | cut -d',' -f8)
            gpu_type=$(echo "$data" | cut -d',' -f9)
            tp_degree=$(echo "$data" | cut -d',' -f10)
            num_gpus=$(echo "$data" | cut -d',' -f11)
        else
            data=$(tail -n 1 "$csv_file")
            throughput=$(echo "$data" | cut -d',' -f2)
            cost=$(echo "$data" | cut -d',' -f3)
            cost_per_m=$(echo "$data" | cut -d',' -f4)
            stages=$(echo "$data" | cut -d',' -f7)
            gpu_type=$(echo "$data" | cut -d',' -f8)
            tp_degree=$(echo "$data" | cut -d',' -f9)
            num_gpus=$(echo "$data" | cut -d',' -f10)
        fi
        printf "%-10s %-12s %-12s %-10s %-8s %-8s %-15s %-15s\n" \
            "$batch_size" "${throughput:-N/A}" "${cost:-N/A}" "${cost_per_m:-N/A}" \
            "${stages:-N/A}" "${tp_degree:-N/A}" "${num_gpus:-N/A}" "${gpu_type:-N/A}"
    else
        printf "%-10s %-12s %-12s %-10s %-8s %-8s %-15s %-15s\n" \
            "$batch_size" "FAILED" "N/A" "N/A" "N/A" "N/A" "N/A" "N/A"
    fi
done

echo ""
echo "========================================================================"
echo "PREFILL-DECODE SWEEP COMPLETE!"
echo "========================================================================"
echo "PREFILL results: config/${target_config}-prefill/output-${timestamp}/"
echo "DECODE results:  config/${target_config}-decode/output-${timestamp}/"
echo "========================================================================"

