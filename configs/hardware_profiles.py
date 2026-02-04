"""
HCDS 硬件配置档位
根据 GPU 显存自动选择最优配置
"""

from typing import Dict, Any, Optional


# ==================== Profile A: 高端单卡 ====================
# A100 / 4090 / A6000 Ada (24GB+)
HIGH_END_SINGLE = {
    "gpu_memory_min": 20000,  # MB
    "gpu_count": 1,
    "embedding": {
        "model": "intfloat/multilingual-e5-large",
        "batch_size": 256,
        "precision": "fp16",
        "num_workers": 8,
    },
    "clustering": {
        "pilot_ratio": 0.03,
        "pilot_max_samples": 500000,
        "faiss_index": "IVF8192,Flat",
        "faiss_nprobe": 128,
    },
    "compression": {
        "max_workers": 32,
        "batch_size": 100000,
    },
    "density": {
        "use_gpu": True,
        "batch_size": 100000,
    },
}


# ==================== Profile B: 中端单卡 ====================
# V100 / A6000 / RTX 4080 (14-20GB)
MID_SINGLE = {
    "gpu_memory_min": 14000,
    "gpu_memory_max": 20000,
    "gpu_count": 1,
    "embedding": {
        "model": "intfloat/multilingual-e5-large",
        "batch_size": 128,
        "precision": "fp16",
        "num_workers": 6,
    },
    "clustering": {
        "pilot_ratio": 0.02,
        "pilot_max_samples": 300000,
        "faiss_index": "IVF4096,Flat",
        "faiss_nprobe": 64,
    },
    "compression": {
        "max_workers": 24,
        "batch_size": 50000,
    },
    "density": {
        "use_gpu": True,
        "batch_size": 50000,
    },
}


# ==================== Profile C: 入门单卡 ====================
# RTX 3090 / 4080 / 3080 (8-14GB)
ENTRY_SINGLE = {
    "gpu_memory_min": 8000,
    "gpu_memory_max": 14000,
    "gpu_count": 1,
    "embedding": {
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "batch_size": 64,
        "precision": "fp16",
        "num_workers": 4,
        "gradient_checkpointing": True,
    },
    "clustering": {
        "pilot_ratio": 0.015,
        "pilot_max_samples": 200000,
        "faiss_index": "IVF2048,PQ32",
        "faiss_nprobe": 32,
    },
    "compression": {
        "max_workers": 16,
        "batch_size": 30000,
    },
    "density": {
        "use_gpu": True,
        "batch_size": 30000,
    },
}


# ==================== Profile D: 多卡并行 ====================
# 多 GPU 配置
MULTI_GPU = {
    "gpu_count_min": 2,
    "embedding": {
        "model": "intfloat/multilingual-e5-large",
        "batch_size_per_gpu": 128,
        "precision": "fp16",
        "strategy": "data_parallel",
        "num_workers_per_gpu": 4,
    },
    "clustering": {
        "pilot_ratio": 0.05,
        "pilot_max_samples": 1000000,
        "faiss_index": "IVF16384,Flat",
        "faiss_nprobe": 256,
        "use_multi_gpu_faiss": True,
    },
    "compression": {
        "max_workers": 64,
        "batch_size": 200000,
    },
    "density": {
        "use_gpu": True,
        "batch_size": 200000,
        "distribute_across_gpus": True,
    },
}


# ==================== Profile E: CPU Only ====================
# 无 GPU 或 GPU 显存不足
CPU_ONLY = {
    "gpu_count": 0,
    "embedding": {
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "batch_size": 32,
        "precision": "fp32",
        "num_workers": 4,
        "device": "cpu",
    },
    "clustering": {
        "pilot_ratio": 0.01,
        "pilot_max_samples": 100000,
        "faiss_index": "IVF1024,Flat",
        "faiss_nprobe": 16,
        "use_gpu_faiss": False,
    },
    "compression": {
        "max_workers": 8,
        "batch_size": 10000,
    },
    "density": {
        "use_gpu": False,
        "batch_size": 10000,
    },
}


# ==================== 资源估算参考 ====================
# 5M 样本场景下的资源消耗估算
RESOURCE_ESTIMATES = {
    "high_end_single": {
        "embedding_time": "~1.5 hours",
        "peak_gpu_memory": "~20GB",
        "peak_ram": "~32GB",
        "disk_space": "~25GB",
    },
    "mid_single": {
        "embedding_time": "~2.5 hours",
        "peak_gpu_memory": "~14GB",
        "peak_ram": "~32GB",
        "disk_space": "~25GB",
    },
    "entry_single": {
        "embedding_time": "~4 hours",
        "peak_gpu_memory": "~10GB",
        "peak_ram": "~24GB",
        "disk_space": "~10GB",
    },
    "multi_gpu_4x": {
        "embedding_time": "~25 minutes",
        "peak_gpu_memory": "~20GB per GPU",
        "peak_ram": "~64GB",
        "disk_space": "~25GB",
    },
}


# 所有 profiles 集合
PROFILES = {
    "high_end_single": HIGH_END_SINGLE,
    "mid_single": MID_SINGLE,
    "entry_single": ENTRY_SINGLE,
    "multi_gpu": MULTI_GPU,
    "cpu_only": CPU_ONLY,
}


def get_hardware_profile(
    gpu_memory_mb: Optional[int] = None,
    gpu_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    根据硬件配置获取对应的配置档位

    Args:
        gpu_memory_mb: GPU 显存 (MB)，None 则自动检测
        gpu_count: GPU 数量，None 则自动检测

    Returns:
        硬件配置档位字典
    """
    # 自动检测硬件
    if gpu_memory_mb is None or gpu_count is None:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = gpu_count or torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_memory_mb = gpu_memory_mb or (
                        torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                    )
            else:
                gpu_count = 0
                gpu_memory_mb = 0
        except ImportError:
            gpu_count = 0
            gpu_memory_mb = 0

    # 选择匹配的档位
    if gpu_count == 0:
        return CPU_ONLY
    elif gpu_count >= 2:
        return MULTI_GPU
    elif gpu_memory_mb >= 20000:
        return HIGH_END_SINGLE
    elif gpu_memory_mb >= 14000:
        return MID_SINGLE
    else:
        return ENTRY_SINGLE
