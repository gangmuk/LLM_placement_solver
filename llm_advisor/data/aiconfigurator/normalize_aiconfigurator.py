"""
Normalize AiConfigurator vLLM sweep data into canonical schema format.

Reads:  full_sweep_vllm.csv (103,965 rows × 116 cols)
Writes: llm_advisor/data/aiconfigurator/data.csv

Usage:
    python -m llm_advisor.data.aiconfigurator.normalize_aiconfigurator
    python -m llm_advisor.data.aiconfigurator.normalize_aiconfigurator --input /path/to/full_sweep_vllm.csv
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths — mirror build_canonical.py layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent            # llm_advisor/data/aiconfigurator/
DATA_DIR = SCRIPT_DIR.parent                            # llm_advisor/data/
LLM_ADVISOR_DIR = DATA_DIR.parent                       # llm_advisor/
PROJECT_ROOT = LLM_ADVISOR_DIR.parent                   # LLM_placement_solver/

# Add project root to path so we can import llm_advisor.*
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_advisor.gpu_specs import GPU_SPECS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_INPUT = Path("/home/orange/Desktop/tandemn/Tandemn-orca/full_sweep_vllm.csv")
DEFAULT_OUTPUT = SCRIPT_DIR / "data.csv"

CONFIG_CACHE_DIR = LLM_ADVISOR_DIR / ".model_cache"

HF_TOKEN: str | None = os.environ.get("HF_TOKEN")

# ---------------------------------------------------------------------------
# Canonical column order (copied from build_canonical.py)
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    "data_source", "data_source_type", "model_name", "model_architecture",
    "precision", "params_billion",
    "tp", "pp", "dp",
    "gpu_model", "gpu_count_total", "gpu_mem_gb", "num_nodes", "gpus_per_node",
    "interconnect",
    "cloud", "region", "instance_type", "price_per_instance_hour_usd",
    "runtime_stack",
    "task_type", "request_pattern", "num_requests", "max_num_seqs",
    "input_len_tokens_min", "input_len_tokens_max", "input_len_tokens_avg",
    "input_len_tokens_fixed",
    "output_len_tokens_min", "output_len_tokens_max", "output_len_tokens_avg",
    "output_len_tokens_fixed",
    "tokens_per_sec_total", "tokens_per_sec_per_gpu",
    "tokens_per_sec_prefill", "tokens_per_sec_decode",
    "ttft_ms_p50", "ttft_ms_p95", "ttft_ms_p99",
    "tpot_ms_p50", "tpot_ms_p95", "tpot_ms_p99",
    "e2e_ms_p50", "e2e_ms_p95", "e2e_ms_p99",
    "total_cost_usd", "cost_per_1m_tokens_total_usd",
    "cost_per_1m_tokens_prefill_usd", "cost_per_1m_tokens_decode_usd",
    "is_lmcache", "is_continuous_batching", "kv_offload_target",
    "cuda_graphs", "spec_decode",
    "prefill_decode_ratio", "batch_size",
    "model_config_json",
    "is_moe", "num_experts_active", "vocab_size", "attention_heads_per_kv_head",
    "model_size_gb", "params_per_gpu", "model_fits_single_gpu", "vram_headroom_gb",
    "gpu_bandwidth_gbps", "gpu_tflops_fp16", "gpu_generation",
    "bandwidth_per_param", "flops_per_param", "kv_heads_per_tp",
    "crosses_node_boundary",
    "price_per_gpu_hour_usd",
]

# vLLM-specific extra columns to keep (after canonical 73)
VLLM_EXTRAS = [
    "moe_tp", "moe_ep",
    "memory_used_gb", "power_w",
    "request_rate", "global_bs", "balance_score",
    "num_ctx_reqs", "num_gen_reqs",
    "ctx_tokens", "gen_tokens",
    "tokens_per_sec_per_user", "seq_per_sec", "seq_per_sec_per_gpu",
    "gemm", "kvcache", "fmha", "moe", "comm", "prefix",
]

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------
RENAMES = {
    "model":           "model_name",
    "bs":              "batch_size",
    "ttft":            "ttft_ms_p50",
    "tpot":            "tpot_ms_p50",
    "request_latency": "e2e_ms_p50",
    "tokens/s":        "tokens_per_sec_total",
    "tokens/s/gpu":    "tokens_per_sec_per_gpu",
    "num_total_gpus":  "gpu_count_total",
    "concurrency":     "max_num_seqs",
    # vLLM extras → canonical-friendly names
    "tokens/s/user":   "tokens_per_sec_per_user",
    "seq/s":           "seq_per_sec",
    "seq/s/gpu":       "seq_per_sec_per_gpu",
    "memory":          "memory_used_gb",
    # Fix column with a space in the name
    "perop_mix_step_context_attention (scaled)": "perop_mix_step_context_attention_scaled",
}

SYSTEM_TO_GPU = {
    "a100_sxm": "A100",
    "h100_sxm": "H100_SXM",
    "h200_sxm": "H200_SXM",
    "b200_sxm": "B200",
    "l40s":     "L40S",
    "gb200":    "GB200",
}

GEMM_TO_PRECISION = {
    "float16":   "fp16",
    "fp8":       "fp8",
    "fp8_block": "fp8",
    "nvfp4":     "fp4",
}

# GPUs per node for topology inference (standard DGX/HGX = 8)
# GB200 NVL72 is ambiguous — omitted
GPU_GPUS_PER_NODE = {
    "A100": 8, "H100_SXM": 8, "H200_SXM": 8,
    "B200": 8, "L40S": 8,
}


# ---------------------------------------------------------------------------
# HuggingFace helpers (copied from build_canonical.py for auth support)
# ---------------------------------------------------------------------------

def _fetch_config_with_auth(model_id: str) -> dict | None:
    """Fetch config.json from HuggingFace, using HF_TOKEN for gated models."""
    CONFIG_CACHE_DIR.mkdir(exist_ok=True)
    cache_key = hashlib.md5(model_id.encode()).hexdigest()
    cache_file = CONFIG_CACHE_DIR / f"{cache_key}.json"

    # Check disk cache first
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    # Build headers
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    urls = [
        f"https://huggingface.co/{model_id}/raw/main/config.json",
        f"https://huggingface.co/{model_id}/resolve/main/config.json",
    ]
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                config = resp.json()
                cache_file.write_text(json.dumps(config, indent=2))
                return config
        except Exception:
            continue

    return None


def _fetch_exact_params(model_id: str) -> float:
    """Fetch exact parameter count from HF Model Info API (safetensors.total).

    Returns params in billions, or NaN if unavailable.
    """
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        resp = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            safetensors = data.get("safetensors")
            if safetensors:
                total = safetensors.get("total")
                if total and total > 0:
                    return total / 1e9
    except Exception:
        pass
    return np.nan


def _parse_params_from_name(model_id: str) -> float:
    """Extract parameter count from model name (e.g. '235B' → 235.0)."""
    # Match patterns like "235B", "70B", "8B", "480B"
    # Prefer the first number-B pattern found
    m = re.search(r"(\d+)[Bb]", model_id)
    if m:
        return float(m.group(1))
    return np.nan


def _fetch_model_metadata(model_id: str) -> dict:
    """Fetch and extract all canonical model metadata fields.

    Returns a dict with keys matching canonical schema columns.
    """
    result = {
        "model_architecture": np.nan,
        "params_billion": np.nan,
        "is_moe": np.nan,
        "num_experts_active": np.nan,
        "vocab_size": np.nan,
        "attention_heads_per_kv_head": np.nan,
        "model_config_json": np.nan,
        "_num_key_value_heads": np.nan,  # private, used for kv_heads_per_tp
    }

    # 1. Fetch config.json (with auth for gated models)
    config = _fetch_config_with_auth(model_id)

    # 2. If 404, try stripping quantization suffix and retry
    if config is None:
        stripped = re.sub(
            r"[-_](FP8|FP4|BF16|INT4|NVFP4|GPTQ|AWQ)$", "",
            model_id, flags=re.IGNORECASE,
        )
        if stripped != model_id:
            print(f"  Retrying with stripped name: {stripped}")
            config = _fetch_config_with_auth(stripped)

    if config is None:
        print(f"  WARNING: Could not fetch config for {model_id}")
        # Still try to get params from name
        result["params_billion"] = _parse_params_from_name(model_id)
        return result

    # 3. Extract architecture class
    archs = config.get("architectures", [])
    result["model_architecture"] = archs[0] if archs else np.nan

    # 4. Get params_billion — safetensors API is ground truth
    params_b = _fetch_exact_params(model_id)
    if pd.isna(params_b):
        # Try stripped name for safetensors too
        stripped = re.sub(
            r"[-_](FP8|FP4|BF16|INT4|NVFP4|GPTQ|AWQ)$", "",
            model_id, flags=re.IGNORECASE,
        )
        if stripped != model_id:
            params_b = _fetch_exact_params(stripped)

    if pd.isna(params_b):
        # Fallback: parse from model name
        params_b = _parse_params_from_name(model_id)
        if pd.notna(params_b):
            print(f"  Using name-parsed params for {model_id}: {params_b:.0f}B")
        else:
            print(f"  WARNING: No param count for {model_id}")

    result["params_billion"] = params_b

    # 5. MoE detection — different HF configs use different key names
    num_experts = (
        config.get("num_local_experts")    # Mixtral, DeepSeek
        or config.get("num_experts")        # Qwen3-MoE
        or config.get("n_routed_experts")   # Nemotron-H
        or 0
    )
    result["is_moe"] = bool(num_experts > 0)
    if result["is_moe"]:
        result["num_experts_active"] = config.get("num_experts_per_tok", np.nan)

    # 6. Vocab size
    result["vocab_size"] = config.get("vocab_size", np.nan)

    # 7. GQA ratio
    n_heads = config.get("num_attention_heads")
    n_kv = config.get("num_key_value_heads")
    if n_heads and n_kv and n_kv > 0:
        result["attention_heads_per_kv_head"] = n_heads / n_kv
    result["_num_key_value_heads"] = n_kv if n_kv else np.nan

    # 8. Full config JSON
    result["model_config_json"] = json.dumps(config, separators=(",", ":"))

    return result


# ---------------------------------------------------------------------------
# Main normalization pipeline
# ---------------------------------------------------------------------------

def normalize(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Transform AiConfigurator vLLM CSV into canonical schema format."""

    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Shape: {df.shape}")

    # =======================================================================
    # Phase A: Column renames
    # =======================================================================
    df.rename(columns=RENAMES, inplace=True)

    # System → gpu_model
    df["gpu_model"] = df["system"].map(SYSTEM_TO_GPU)

    # Precision from gemm (keep original 'gemm' column as extra)
    df["precision"] = df["gemm"].map(GEMM_TO_PRECISION)

    # runtime_stack
    df["runtime_stack"] = "vllm " + df["version"].astype(str)

    # Input/output lengths — set all 4 variants to the fixed value
    df["input_len_tokens_fixed"] = df["isl"]
    df["input_len_tokens_min"] = df["isl"]
    df["input_len_tokens_max"] = df["isl"]
    df["input_len_tokens_avg"] = df["isl"]
    df["output_len_tokens_fixed"] = df["osl"]
    df["output_len_tokens_min"] = df["osl"]
    df["output_len_tokens_max"] = df["osl"]
    df["output_len_tokens_avg"] = df["osl"]

    # Drop original columns that are now renamed/replaced
    df.drop(columns=["system", "version", "isl", "osl", "backend", "parallel"],
            inplace=True, errors="ignore")

    # =======================================================================
    # Phase B: Static fields
    # =======================================================================
    df["data_source"] = "aiconfigurator"
    df["data_source_type"] = "measured"
    df["task_type"] = "batched"
    df["request_pattern"] = "offline_batch"
    df["is_continuous_batching"] = "True"
    df["is_lmcache"] = "None"
    df["cuda_graphs"] = "None"
    df["spec_decode"] = "None"
    df["kv_offload_target"] = "None"

    # NaN for unavailable canonical columns
    for col in [
        "cloud", "region", "instance_type", "price_per_instance_hour_usd",
        "num_requests",
        "tokens_per_sec_prefill", "tokens_per_sec_decode",
        "ttft_ms_p95", "ttft_ms_p99",
        "tpot_ms_p95", "tpot_ms_p99",
        "e2e_ms_p95", "e2e_ms_p99",
        "total_cost_usd", "cost_per_1m_tokens_total_usd",
        "cost_per_1m_tokens_prefill_usd", "cost_per_1m_tokens_decode_usd",
        "price_per_gpu_hour_usd",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # =======================================================================
    # Phase C: HuggingFace model metadata (MOST IMPORTANT)
    # =======================================================================
    unique_models = df["model_name"].unique()
    print(f"\nFetching HuggingFace metadata for {len(unique_models)} unique models...")

    model_metadata = {}
    for model_id in sorted(unique_models):
        print(f"  {model_id} ...")
        model_metadata[model_id] = _fetch_model_metadata(model_id)

    # Merge metadata onto dataframe
    meta_cols = [
        "model_architecture", "params_billion", "is_moe", "num_experts_active",
        "vocab_size", "attention_heads_per_kv_head", "model_config_json",
        "_num_key_value_heads",
    ]
    for col in meta_cols:
        df[col] = df["model_name"].map(lambda m, c=col: model_metadata.get(m, {}).get(c, np.nan))

    # =======================================================================
    # Phase D: GPU specs lookup
    # =======================================================================
    def _gpu_spec(gpu_model, field):
        spec = GPU_SPECS.get(gpu_model, {})
        return spec.get(field, np.nan)

    df["gpu_mem_gb"] = df["gpu_model"].map(lambda g: _gpu_spec(g, "vram_gb"))
    df["gpu_bandwidth_gbps"] = df["gpu_model"].map(lambda g: _gpu_spec(g, "memory_bandwidth_gbps"))
    df["gpu_tflops_fp16"] = df["gpu_model"].map(lambda g: _gpu_spec(g, "fp16_tflops"))
    df["gpu_generation"] = df["gpu_model"].map(lambda g: _gpu_spec(g, "architecture"))
    df["interconnect"] = df["gpu_model"].map(
        lambda g: "NVLink" if GPU_SPECS.get(g, {}).get("nvlink") else "PCIe"
    )

    # Topology: gpus_per_node and num_nodes
    df["gpus_per_node"] = df["gpu_model"].map(
        lambda g: GPU_GPUS_PER_NODE.get(g, np.nan)
    )
    df["num_nodes"] = np.where(
        df["gpus_per_node"].notna() & (df["gpus_per_node"] > 0),
        np.ceil(df["gpu_count_total"] / df["gpus_per_node"]),
        np.nan,
    )
    df["crosses_node_boundary"] = np.where(
        df["num_nodes"].notna(),
        df["num_nodes"] > 1,
        np.nan,
    )

    # =======================================================================
    # Phase E: Derived columns
    # =======================================================================
    # bytes_per_param: fp4→0.5, fp8→1, fp16→2
    bytes_per_param = df["precision"].map({"fp4": 0.5, "fp8": 1, "fp16": 2}).fillna(2)
    df["model_size_gb"] = df["params_billion"] * bytes_per_param

    gpu_count_valid = df["gpu_count_total"].where(df["gpu_count_total"] > 0)
    df["params_per_gpu"] = df["params_billion"] / gpu_count_valid

    df["model_fits_single_gpu"] = np.where(
        df["model_size_gb"].notna() & df["gpu_mem_gb"].notna(),
        df["model_size_gb"] <= df["gpu_mem_gb"],
        np.nan,
    )

    df["vram_headroom_gb"] = np.where(
        df["gpu_mem_gb"].notna() & df["gpu_count_total"].notna() & df["model_size_gb"].notna(),
        (df["gpu_mem_gb"] * df["gpu_count_total"]) - df["model_size_gb"],
        np.nan,
    )

    # prefill_decode_ratio
    osl_valid = df["output_len_tokens_fixed"].where(df["output_len_tokens_fixed"] > 0)
    df["prefill_decode_ratio"] = df["input_len_tokens_fixed"] / osl_valid

    # Efficiency ratios
    params_valid = df["params_billion"].where(df["params_billion"] > 0)
    df["bandwidth_per_param"] = df["gpu_bandwidth_gbps"] * df["tp"] / params_valid
    df["flops_per_param"] = df["gpu_tflops_fp16"] * df["tp"] / params_valid

    # kv_heads_per_tp
    tp_valid = df["tp"].where(df["tp"] > 0)
    df["kv_heads_per_tp"] = df["_num_key_value_heads"] / tp_valid

    # Drop private helper column
    df.drop(columns=["_num_key_value_heads"], inplace=True, errors="ignore")

    # =======================================================================
    # Phase F: Per-operator column cleanup
    # =======================================================================
    perop_cols = [c for c in df.columns if c.startswith("perop_")]
    for c in perop_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    all_nan_perops = [c for c in perop_cols if df[c].isna().all()]
    if all_nan_perops:
        print(f"\nDropping {len(all_nan_perops)} all-NaN perop columns")
        df.drop(columns=all_nan_perops, inplace=True)
    surviving_perops = sorted([c for c in df.columns if c.startswith("perop_")])

    # =======================================================================
    # Phase G: Output column ordering
    # =======================================================================
    # Ensure all canonical columns exist (fill missing with NaN)
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Filter vLLM extras to only those that exist
    extras = [c for c in VLLM_EXTRAS if c in df.columns]

    # Final column order: canonical 73 + vLLM extras + surviving perops
    final_cols = CANONICAL_COLUMNS + extras + surviving_perops
    # Remove any duplicates while preserving order
    seen = set()
    ordered = []
    for c in final_cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    df = df[ordered]

    # =======================================================================
    # Write output
    # =======================================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nWrote {df.shape[0]} rows × {df.shape[1]} cols → {output_path}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Canonical columns: {len(CANONICAL_COLUMNS)}")
    print(f"vLLM extras: {len(extras)}")
    print(f"Perop columns: {len(surviving_perops)}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Models: {df['model_name'].nunique()}")
    print(f"GPU types: {sorted(df['gpu_model'].unique())}")
    print(f"Precisions: {sorted(df['precision'].unique())}")
    print(f"MoE models: {df[df['is_moe'] == True]['model_name'].nunique()}")
    na_config = df["model_config_json"].isna().sum()
    if na_config > 0:
        print(f"WARNING: {na_config} rows have no model_config_json!")
    else:
        print(f"All rows have model_config_json ✓")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Normalize AiConfigurator vLLM data to canonical schema"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace API token (overrides HF_TOKEN env var)",
    )
    args = parser.parse_args()

    global HF_TOKEN
    if args.hf_token:
        HF_TOKEN = args.hf_token

    if not HF_TOKEN:
        print("WARNING: No HF_TOKEN set. Gated models (Meta-Llama) may fail to fetch.")

    normalize(args.input, args.output)


if __name__ == "__main__":
    main()
