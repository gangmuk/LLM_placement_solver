#!/bin/bash

# Batch Size Sweep Runner
# Runs solver in parallel for different batch sizes and picks the best result

trap cleanup SIGINT SIGTERM SIGKILL SIGQUIT

function cleanup() {
    echo "** Cleaning up..."
    kill $(jobs -p) 2>/dev/null
    exit 1
}

# Configuration
config_dir_list=("config/medium")
# config_dir_list=("config/hal")
batch_size_list=(32 64 128 256)
cost_optimization_method_list=("enumeration")
# network_config_list=("400 200")
network_config_list=("none")
cloud_provider="AWS"
solver="solver_constrained_with_tp-2.py"

timestamp=$(date +%Y%m%d_%H%M%S)

echo "========================================================================"
echo "BATCH SIZE SWEEP - Running ${#batch_size_list[@]} batch sizes in parallel"
echo "========================================================================"
echo "Batch sizes: ${batch_size_list[@]}"
echo "Timestamp: ${timestamp}"
echo ""

# Array to store PIDs and output paths for later comparison
declare -a pids
declare -a output_logs
declare -a batch_sizes_running

for config_dir in "${config_dir_list[@]}"; do
    for cost_optimization_method in "${cost_optimization_method_list[@]}"; do
        for network_config in "${network_config_list[@]}"; do
            intra_bw=$(echo ${network_config} | cut -d' ' -f1)
            inter_bw=$(echo ${network_config} | cut -d' ' -f2)
            
            # Run each batch size in parallel
            for batch_size in "${batch_size_list[@]}"; do
                # Create output directory for this batch size - this will be the config dir
                output_log_dir="${config_dir}/output-${timestamp}/batch_${batch_size}"
                mkdir -p ${output_log_dir}
                
                output_log_path="${output_log_dir}/method_${cost_optimization_method}-network_${intra_bw}_${inter_bw}.txt"
                
                # Copy all config files to output directory
                cp ${config_dir}/gpu_pool.csv ${output_log_dir}/ 2>/dev/null || true
                cp ${config_dir}/network_bandwidth.csv ${output_log_dir}/ 2>/dev/null || true
                
                # Modify config.csv to set the specific batch size
                original_config="${config_dir}/config.csv"
                batch_config="${output_log_dir}/config.csv"
                
                # Use awk to modify both min and max batch size
                awk -F',' -v bs="${batch_size}" '
                    BEGIN {OFS=","}
                    $1 == "min_batch_size" {$2 = bs}
                    $1 == "max_batch_size" {$2 = bs}
                    {print}
                ' "${original_config}" > "${batch_config}"
                
                echo "** [Batch ${batch_size}] Starting solver..."
                echo "   Config dir: ${output_log_dir}"
                echo "   Output log: ${output_log_path}"
                
                start_time=$(date +%s)
                
                # Run solver in background - use output_log_dir as config dir
                if [ "${network_config}" != "none" ]; then
                    python3 ${solver} \
                        --config-dir ${output_log_dir} \
                        --method ${cost_optimization_method} \
                        --generate-network ${intra_bw} ${inter_bw} \
                        --cloud-provider ${cloud_provider} \
                        &> ${output_log_path} &
                else
                    python3 ${solver} \
                        --config-dir ${output_log_dir} \
                        --method ${cost_optimization_method} \
                        --cloud-provider ${cloud_provider} \
                        &> ${output_log_path} &
                fi
                
                solver_pid=$!
                pids+=($solver_pid)
                output_logs+=("${output_log_path}")
                batch_sizes_running+=($batch_size)
                
                echo "   PID: ${solver_pid}"
                echo ""
            done
        done
    done
done

echo "========================================================================"
echo "All ${#pids[@]} solver processes started, waiting for completion..."
echo "PIDs: ${pids[@]}"
echo "========================================================================"
echo ""

# Wait for all processes and track which ones complete
completed=0
total=${#pids[@]}

for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    batch_size=${batch_sizes_running[$i]}
    
    echo "Waiting for batch size ${batch_size} (PID: ${pid})..."
    wait $pid
    exit_code=$?
    completed=$((completed + 1))
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ [${completed}/${total}] Batch ${batch_size} completed successfully"
    else
        echo "✗ [${completed}/${total}] Batch ${batch_size} failed with exit code ${exit_code}"
    fi
    echo ""
done

echo "========================================================================"
echo "All solvers completed! Analyzing results..."
echo "========================================================================"
echo ""

# Merge CSV results from each batch (no cleanup needed - files are already in place)
results_csv="${config_dir}/output-${timestamp}/batch_sweep_results.csv"

echo "RESULTS SUMMARY:"
echo "--------------------------------------------------------------------------------------------"
printf "%-12s %-15s %-15s %-20s %-25s\n" "Batch Size" "Throughput" "Cost (\$/h)" "\$/M tokens" "Status"
echo "--------------------------------------------------------------------------------------------"

best_cost_per_token=999999.0
best_batch_size=""
best_log_path=""

# First pass: collect all CSV files and merge them
first_batch=true
for batch_size in "${batch_size_list[@]}"; do
    csv_file="${config_dir}/output-${timestamp}/batch_${batch_size}/solution_summary.csv"
    
    if [ -f "$csv_file" ]; then
        if [ "$first_batch" = true ]; then
            # Copy header from first file
            cat "$csv_file" > "$results_csv"
            first_batch=false
        else
            # Append data rows only (skip header)
            tail -n +2 "$csv_file" >> "$results_csv"
        fi
    fi
done

# Second pass: display results and find best
for i in "${!output_logs[@]}"; do
    log_path="${output_logs[$i]}"
    batch_size="${batch_sizes_running[$i]}"
    csv_file="${config_dir}/output-${timestamp}/batch_${batch_size}/solution_summary.csv"
    
    if [ -f "$csv_file" ]; then
        # Count how many solutions were explored (rows minus header)
        num_solutions=$(($(wc -l < "$csv_file") - 1))
        
        # Read BEST solution (marked with is_best=YES or last row if no is_best column)
        if grep -q "is_best" "$csv_file"; then
            # Find row with is_best=YES
            data=$(grep ",YES," "$csv_file" | head -1)
            # Extract fields (with budget_tested column present)
            throughput=$(echo "$data" | cut -d',' -f3)
            cost=$(echo "$data" | cut -d',' -f4)
            cost_per_m=$(echo "$data" | cut -d',' -f5)
        else
            # Old format without is_best column
            data=$(tail -n 1 "$csv_file")
            throughput=$(echo "$data" | cut -d',' -f2)
            cost=$(echo "$data" | cut -d',' -f3)
            cost_per_m=$(echo "$data" | cut -d',' -f4)
        fi
        
        status="✓ SUCCESS (${num_solutions} tested)"
        
        # Compare to find best (lowest $/M tokens)
        if [ -n "$cost_per_m" ]; then
            is_better=$(awk -v new="$cost_per_m" -v best="$best_cost_per_token" 'BEGIN {print (new < best) ? 1 : 0}')
            if [ "$is_better" -eq 1 ]; then
                best_cost_per_token=$cost_per_m
                best_batch_size=$batch_size
                best_log_path=$log_path
            fi
        fi
    else
        throughput="N/A"
        cost="N/A"
        cost_per_m="N/A"
        status="✗ FAILED"
    fi
    
    printf "%-12s %-15s %-15s %-20s %-25s\n" \
        "$batch_size" \
        "${throughput:-N/A}" \
        "${cost:-N/A}" \
        "${cost_per_m:-N/A}" \
        "$status"
done

echo "--------------------------------------------------------------------------------------------"
echo ""

# Announce the winner
if [ -n "$best_batch_size" ]; then
    echo "* Best batch size = $best_batch_size"
    echo "Cost per M tokens: \$${best_cost_per_token}"
    echo "Log file: ${best_log_path}"
    echo ""
    
    # Create a symlink to the best result for easy access (use relative path)
    ln -sf "batch_${best_batch_size}" "${config_dir}/output-${timestamp}/BEST_RESULT"
    echo "Best result directory: ${config_dir}/output-${timestamp}/BEST_RESULT -> batch_${best_batch_size}"
    echo ""
    
    # Show detailed results for best solution
    grep -A 5 "PERFORMANCE & COST METRICS:" "$best_log_path" | grep -v "^--" || true
    echo ""
else
    echo "❌ No successful solution found across any batch size!"
    exit 1
fi

echo "========================================================================"
echo "Batch sweep complete!"
echo "All results in: ${config_dir}/output-${timestamp}/"
echo "CSV results:    ${results_csv}"
echo "========================================================================"

