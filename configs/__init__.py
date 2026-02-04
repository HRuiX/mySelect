"""
HCDS 配置模块
提供各种预设配置
"""

from configs.config import get_default_config
from configs.experiment_small import get_experiment_small_config

__all__ = [
    "get_default_config",
    "get_experiment_small_config",
]
