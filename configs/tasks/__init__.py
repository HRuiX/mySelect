"""
HCDS 任务配置模块
"""

from configs.tasks.math_reasoning import get_math_reasoning_config
from configs.tasks.instruction_tuning import get_instruction_tuning_config
from configs.tasks.code_generation import get_code_generation_config

__all__ = [
    "get_math_reasoning_config",
    "get_instruction_tuning_config",
    "get_code_generation_config",
]
