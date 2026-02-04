"""并行化模块"""

from hcds.parallel.detector import ParallelDetector, ParallelConfig, detect_hardware
from hcds.parallel.monitor import ResourceMonitor

__all__ = [
    "ParallelDetector",
    "ParallelConfig",
    "detect_hardware",
    "ResourceMonitor",
]
