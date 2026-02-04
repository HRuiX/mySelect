"""
HCDS 指令微调任务配置
适用于 Alpaca, ShareGPT, LIMA 等通用指令数据集
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import get_default_config
from hcds.config.schema import (
    HCDSConfig,
    FieldMappingConfig,
    CorrectnessConfig,
    EntropyConfig,
    PriorityWeightsConfig,
    SelectionRatioConfig,
    RetirementConfig,
)


def get_instruction_tuning_config(data_path: str = "data/train.jsonl") -> HCDSConfig:
    """
    获取指令微调任务配置
    基于默认配置，覆盖指令微调特定参数

    Args:
        data_path: 数据文件路径

    Returns:
        HCDSConfig 实例
    """
    config = get_default_config(data_path=data_path)

    # 覆盖实验名称
    config.experiment.name = "hcds_instruction_tuning"

    # 数据配置
    config.data.field_mapping = FieldMappingConfig(
        instruction="instruction",
        input="input",
        output="output",
    )
    config.data.content_template = "{instruction}\n{input}"

    # 错误强度 - 指令微调以 Loss 为主
    config.error_intensity.weights = {"loss": 0.8, "correctness": 0.0, "entropy": 0.2}
    config.error_intensity.correctness = CorrectnessConfig(enabled=False)
    config.error_intensity.entropy = EntropyConfig(
        enabled=True,
        normalize_by_length=True,
    )

    # Thompson Sampling
    config.thompson_sampling.prior.strength = 2.0
    config.thompson_sampling.selection.ratio = 0.35

    # 优先级 - 平衡各维度
    config.priority.weights = PriorityWeightsConfig(
        difficulty=0.4,
        rarity=0.4,
        novelty_base=0.5,
    )
    config.priority.selection = SelectionRatioConfig(
        priority_ratio=0.7,
        rare_ratio=0.2,
        random_ratio=0.1,
    )

    # 退休 - 更宽松
    config.retirement = RetirementConfig(
        consecutive_threshold=4,
        error_threshold=0.15,
        revisit_probability=0.03,
    )

    return config
