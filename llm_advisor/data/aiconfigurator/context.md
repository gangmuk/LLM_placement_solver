# AiConfigurator vLLM Sweep

## Source

AiConfigurator full vLLM performance sweep (`full_sweep_vllm.csv`).

## Coverage

- **Systems**: A100 SXM, H100 SXM, H200 SXM, B200 SXM, GB200 NVL72, L40S
- **Models**: 16 (dense + MoE + Mamba hybrid), including Meta-Llama 3.1 (8B/70B/405B), Qwen3 family, Nemotron variants
- **vLLM versions**: 0.12.0, 0.14.0, 0.14.1
- **Rows**: ~103,965

## Per-Operator Columns (`perop_*`)

These columns contain per-operator kernel timing data in milliseconds from vLLM's internal profiler. They are architecture-specific (sparse — not all columns apply to all models):

- `perop_mix_step_context_*` — prefill/mixed step timings (embedding, QKV/projection GEMMs, FFN, attention, AllReduce)
- `perop_genonly_step_generation_*` — decode-only step timings (same operators as above)
- `perop_scheduling_*` — step counts and scheduling latencies
- `perop_*_moe_*` — MoE-specific operators (router GEMM, pre/post dispatch)
- `perop_*_mamba_*` — Mamba state-space model operators (conv1d, SSM)
- `perop_*_mlp_*` — MLP-specific operators (for architectures with separate MLP blocks)

## Intentionally NaN Columns

These canonical columns are NaN for all aiconfigurator rows:

- **Cost**: `cloud`, `region`, `instance_type`, `price_per_instance_hour_usd`, `total_cost_usd`, `cost_per_1m_tokens_*`, `price_per_gpu_hour_usd` — AiConfigurator is hardware-only, no cloud billing
- **Latency percentiles**: `ttft_ms_p95/p99`, `tpot_ms_p95/p99`, `e2e_ms_p95/p99` — sweep only records median values
- **Phase split throughput**: `tokens_per_sec_prefill`, `tokens_per_sec_decode` — not split in top-level metrics
- **Request count**: `num_requests` — not recorded

## GB200 Topology Note

GB200 NVL72 has 36 Grace Blackwell Superchip nodes (each with 1 Grace CPU + 2 B200 GPUs), all connected via a unified NVLink 5.0 fabric. The concept of "node" is ambiguous in this topology, so `gpus_per_node` and `num_nodes` are NaN for GB200 rows.
