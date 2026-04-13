"""
GPU specifications for performance reasoning.
These specs help the LLM understand hardware capabilities and make informed decisions.
"""

GPU_SPECS = {
    # NVIDIA A10G (AWS g5 instances)
    "A10G": {
        "name": "NVIDIA A10G",
        "architecture": "Ampere",
        "vram_gb": 24,
        "fp16_tflops": 125,  # Tensor Core
        "fp32_tflops": 31.2,
        "memory_bandwidth_gbps": 600,
        "nvlink": False,
        "pcie_gen": 4,
        "tdp_watts": 150,
        "aws_instance_prefix": "g5",
        "relative_performance": "entry-level",  # for LLM context
        "best_for": "small models, low batch sizes, cost-sensitive workloads",
    },

    # NVIDIA L4 (AWS g6 instances)
    "L4": {
        "name": "NVIDIA L4",
        "architecture": "Ada Lovelace",
        "vram_gb": 24,
        "fp16_tflops": 121,  # Tensor Core
        "fp32_tflops": 30.3,
        "memory_bandwidth_gbps": 300,
        "nvlink": False,
        "pcie_gen": 4,
        "tdp_watts": 72,
        "aws_instance_prefix": "g6",
        "relative_performance": "efficient",
        "best_for": "inference, power-efficient deployments",
    },

    # NVIDIA L40S (AWS g6e instances)
    "L40S": {
        "name": "NVIDIA L40S",
        "architecture": "Ada Lovelace",
        "vram_gb": 48,
        "fp16_tflops": 362,  # Tensor Core
        "fp32_tflops": 91.6,
        "memory_bandwidth_gbps": 864,
        "nvlink": False,
        "pcie_gen": 4,
        "tdp_watts": 350,
        "aws_instance_prefix": "g6e",
        "relative_performance": "mid-tier",
        "best_for": "medium models, good balance of performance and cost",
    },

    # NVIDIA A100 40GB (AWS p4d instances)
    "A100-40GB": {
        "name": "NVIDIA A100 40GB",
        "architecture": "Ampere",
        "vram_gb": 40,
        "fp16_tflops": 312,  # Tensor Core
        "fp32_tflops": 19.5,
        "memory_bandwidth_gbps": 1555,
        "nvlink": True,
        "nvlink_bandwidth_gbps": 600,
        "pcie_gen": 4,
        "tdp_watts": 400,
        "aws_instance_prefix": "p4d",
        "relative_performance": "high-end",
        "best_for": "large models, high throughput, training",
    },

    # NVIDIA A100 80GB (AWS p4de instances)
    "A100": {
        "name": "NVIDIA A100 80GB",
        "architecture": "Ampere",
        "vram_gb": 80,
        "fp16_tflops": 312,  # Tensor Core
        "fp32_tflops": 19.5,
        "memory_bandwidth_gbps": 2039,
        "nvlink": True,
        "nvlink_bandwidth_gbps": 600,
        "pcie_gen": 4,
        "tdp_watts": 400,
        "aws_instance_prefix": "p4de",
        "relative_performance": "high-end",
        "best_for": "large models, long contexts, high throughput",
    },

    # NVIDIA V100 (AWS p3 instances)
    "V100": {
        "name": "NVIDIA V100",
        "architecture": "Volta",
        "vram_gb": 32,
        "fp16_tflops": 125,  # Tensor Core
        "fp32_tflops": 15.7,
        "memory_bandwidth_gbps": 900,
        "nvlink": True,
        "nvlink_bandwidth_gbps": 300,
        "pcie_gen": 3,
        "tdp_watts": 300,
        "aws_instance_prefix": "p3",
        "relative_performance": "legacy high-end",
        "best_for": "older generation, still capable for medium models",
    },

    # NVIDIA H100 (AWS p5 instances)
    "H100": {
        "name": "NVIDIA H100",
        "architecture": "Hopper",
        "vram_gb": 80,
        "fp16_tflops": 989,  # Tensor Core with sparsity
        "fp32_tflops": 67,
        "memory_bandwidth_gbps": 3350,
        "nvlink": True,
        "nvlink_bandwidth_gbps": 900,
        "pcie_gen": 5,
        "tdp_watts": 700,
        "aws_instance_prefix": "p5",
        "relative_performance": "flagship",
        "best_for": "largest models, maximum throughput, cutting-edge performance",
    },

    # NVIDIA H100 SXM (DGX H100)
    "H100_SXM": {
        "name": "NVIDIA H100 SXM",
        "architecture": "Hopper",
        "vram_gb": 80,
        "fp16_tflops": 989,
        "fp32_tflops": 67,
        "memory_bandwidth_gbps": 3350,
        "nvlink": True,
        "nvlink_bandwidth_gbps": 900,
        "pcie_gen": 5,
        "tdp_watts": 700,
        "aws_instance_prefix": None,
        "relative_performance": "flagship",
        "best_for": "largest models, maximum throughput, DGX deployments",
    },

    # NVIDIA H200 (141 GB HBM3e)
    "H200": {
        "name": "NVIDIA H200",
        "architecture": "Hopper",
        "vram_gb": 141,
        "fp16_tflops": 989,
        "fp32_tflops": 67,
        "memory_bandwidth_gbps": 4800,
        "nvlink": True,
        "nvlink_bandwidth_gbps": 900,
        "pcie_gen": 5,
        "tdp_watts": 700,
        "aws_instance_prefix": None,
        "relative_performance": "flagship",
        "best_for": "largest models, maximum memory capacity, cutting-edge performance",
    },

    # NVIDIA H200 SXM (DGX H200)
    "H200_SXM": {
        "name": "NVIDIA H200 SXM",
        "architecture": "Hopper",
        "vram_gb": 141,
        "fp16_tflops": 989,
        "fp32_tflops": 67,
        "memory_bandwidth_gbps": 4800,
        "nvlink": True,
        "nvlink_bandwidth_gbps": 900,
        "pcie_gen": 5,
        "tdp_watts": 700,
        "aws_instance_prefix": None,
        "relative_performance": "flagship",
        "best_for": "largest models, maximum memory capacity, DGX deployments",
    },

    # NVIDIA B200 SXM (Blackwell, standalone)
    "B200": {
        "name": "NVIDIA B200 SXM",
        "architecture": "Blackwell",
        "vram_gb": 192,
        "fp16_tflops": 2250,  # Tensor Core dense; 4,500 with sparsity
        "fp32_tflops": 150,
        "memory_bandwidth_gbps": 8000,  # HBM3e, 8 TB/s
        "nvlink": True,
        "nvlink_bandwidth_gbps": 1800,  # NVLink 5.0
        "pcie_gen": 5,
        "tdp_watts": 1000,
        "aws_instance_prefix": None,
        "relative_performance": "next-gen flagship",
        "best_for": "largest models, FP4 quantization, ultra-high throughput",
    },

    # NVIDIA GB200 (Grace Blackwell Superchip, ships in NVL72 rack)
    # Per-GPU specs; same Blackwell die as B200 but higher power envelope in NVL72
    "GB200": {
        "name": "NVIDIA GB200 (Grace Blackwell Superchip)",
        "architecture": "Blackwell",
        "vram_gb": 192,
        "fp16_tflops": 2250,  # Same die as B200; conservative until NVIDIA disambiguates
        "fp32_tflops": 150,
        "memory_bandwidth_gbps": 8000,
        "nvlink": True,
        "nvlink_bandwidth_gbps": 1800,
        "pcie_gen": 6,  # Grace CPU uses PCIe 6 / NVLink-C2C
        "tdp_watts": 1200,  # Higher TDP per GPU in NVL72
        "aws_instance_prefix": None,
        "relative_performance": "next-gen flagship",
        "best_for": "NVL72 rack deployments, maximum scale inference",
    },

    # NVIDIA A40 (various cloud providers)
    "A40": {
        "name": "NVIDIA A40",
        "architecture": "Ampere",
        "vram_gb": 48,
        "fp16_tflops": 150,  # Tensor Core
        "fp32_tflops": 37.4,
        "memory_bandwidth_gbps": 696,
        "nvlink": True,
        "nvlink_bandwidth_gbps": 112.5,
        "pcie_gen": 4,
        "tdp_watts": 300,
        "aws_instance_prefix": None,
        "relative_performance": "mid-tier",
        "best_for": "workstation/datacenter, good memory capacity",
    },
}

# Model size requirements (approximate)
MODEL_REQUIREMENTS = {
    "7b": {
        "params_billions": 7,
        "fp16_size_gb": 14,
        "min_vram_inference_gb": 16,
        "recommended_tp": [1, 2],
    },
    "13b": {
        "params_billions": 13,
        "fp16_size_gb": 26,
        "min_vram_inference_gb": 28,
        "recommended_tp": [1, 2, 4],
    },
    "70b": {
        "params_billions": 70,
        "fp16_size_gb": 140,
        "min_vram_inference_gb": 150,
        "recommended_tp": [4, 8],
    },
    "405b": {
        "params_billions": 405,
        "fp16_size_gb": 810,
        "min_vram_inference_gb": 850,
        "recommended_tp": [8],
        "recommended_pp": [2, 4, 8],
    },
}


def get_gpu_spec(gpu_type: str) -> dict:
    """Get GPU specifications by type."""
    # Normalize the name
    gpu_type_upper = gpu_type.upper()

    # Direct match
    if gpu_type_upper in GPU_SPECS:
        return GPU_SPECS[gpu_type_upper]

    # Try partial match
    for key, spec in GPU_SPECS.items():
        if key in gpu_type_upper or gpu_type_upper in key:
            return spec

    return None


def estimate_model_size(model_name: str) -> dict:
    """Estimate model size requirements from model name."""
    model_lower = model_name.lower()

    if "405b" in model_lower:
        return MODEL_REQUIREMENTS["405b"]
    elif "70b" in model_lower:
        return MODEL_REQUIREMENTS["70b"]
    elif "13b" in model_lower:
        return MODEL_REQUIREMENTS["13b"]
    elif "7b" in model_lower or "8b" in model_lower:
        return MODEL_REQUIREMENTS["7b"]

    # Default to 70B for unknown large models
    return MODEL_REQUIREMENTS["70b"]


def format_gpu_specs_for_prompt(gpu_types: list) -> str:
    """Format GPU specs as a readable string for LLM prompt."""
    lines = ["## Available GPU Specifications\n"]

    for gpu_type in gpu_types:
        spec = get_gpu_spec(gpu_type)
        if spec:
            lines.append(f"### {spec['name']} ({gpu_type})")
            lines.append(f"- VRAM: {spec['vram_gb']} GB")
            lines.append(f"- FP16 TFLOPs: {spec['fp16_tflops']}")
            lines.append(f"- Memory Bandwidth: {spec['memory_bandwidth_gbps']} GB/s")
            lines.append(f"- NVLink: {'Yes' if spec['nvlink'] else 'No'}")
            lines.append(f"- Performance tier: {spec['relative_performance']}")
            lines.append(f"- Best for: {spec['best_for']}")
            lines.append("")

    return "\n".join(lines)
