"""
硬件检测与并行配置生成
"""

import os
import psutil
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any


@dataclass
class ParallelConfig:
    """并行配置"""

    # 硬件信息
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 1)
    memory_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024**3))
    gpu_count: int = 0
    gpu_memory_gb: Optional[float] = None

    # 嵌入计算
    embedding_strategy: Literal["single", "data_parallel", "distributed"] = "single"
    embedding_num_gpus: int = 1
    embedding_batch_size: int = 64
    embedding_num_workers: int = 4

    # 聚类
    clustering_n_jobs: int = 1
    clustering_backend: Literal["joblib", "multiprocessing", "single"] = "joblib"

    # 压缩
    compression_max_workers: int = 1
    compression_backend: Literal["concurrent", "ray", "single"] = "concurrent"

    # 密度计算
    density_use_gpu: bool = False
    density_batch_size: int = 10000

    # 优先级
    priority_num_threads: int = 4
    priority_backend: Literal["threading", "single"] = "threading"


class ParallelDetector:
    """自动检测硬件并生成最优并行配置"""

    def __init__(self, user_overrides: Optional[Dict[str, Any]] = None):
        """
        初始化检测器

        Args:
            user_overrides: 用户覆盖配置
        """
        self.user_overrides = user_overrides or {}

    def detect(self) -> ParallelConfig:
        """检测硬件并返回配置"""
        config = ParallelConfig()

        # 检测 GPU
        config.gpu_count, config.gpu_memory_gb = self._detect_gpu()

        # 配置嵌入
        if config.gpu_count > 0:
            if config.gpu_count > 1:
                config.embedding_strategy = "data_parallel"
                config.embedding_num_gpus = config.gpu_count
                config.embedding_batch_size = 64 * config.gpu_count
            else:
                config.embedding_strategy = "single"
                config.embedding_num_gpus = 1
                config.embedding_batch_size = self._calc_embedding_batch(config.gpu_memory_gb)

            config.density_use_gpu = True
            config.density_batch_size = min(100000, int(config.gpu_memory_gb * 10000))
        else:
            config.embedding_strategy = "single"
            config.embedding_batch_size = 32
            config.density_use_gpu = False
            config.density_batch_size = 10000

        # CPU 配置
        safe_cpu_count = max(1, config.cpu_count - 2)

        config.clustering_n_jobs = min(safe_cpu_count, 16)
        config.compression_max_workers = min(safe_cpu_count, 32)
        config.embedding_num_workers = min(safe_cpu_count // 2, 8)
        config.priority_num_threads = min(safe_cpu_count, 8)

        # 应用用户覆盖
        config = self._apply_overrides(config)

        return config

    def _detect_gpu(self) -> tuple:
        """检测 GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    return gpu_count, gpu_memory_gb
        except ImportError:
            pass
        return 0, None

    def _calc_embedding_batch(self, gpu_memory_gb: float) -> int:
        """根据显存计算嵌入 batch size"""
        if gpu_memory_gb >= 20:
            return 256
        elif gpu_memory_gb >= 14:
            return 128
        elif gpu_memory_gb >= 8:
            return 64
        else:
            return 32

    def _apply_overrides(self, config: ParallelConfig) -> ParallelConfig:
        """应用用户覆盖"""
        for key, value in self.user_overrides.items():
            if value != "auto" and hasattr(config, key):
                setattr(config, key, value)
        return config


def detect_hardware() -> Dict[str, Any]:
    """
    检测硬件信息

    Returns:
        硬件信息字典
    """
    info = {
        "cpu_count": os.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
    }

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = []
            for i in range(info["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "id": i,
                    "name": props.name,
                    "memory_gb": props.total_memory / (1024**3),
                })
    except ImportError:
        info["cuda_available"] = False
        info["gpu_count"] = 0

    return info
