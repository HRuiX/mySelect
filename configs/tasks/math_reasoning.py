"""
HCDS 数学推理任务配置
适用于 GSM8K, MATH, AQuA 等数学数据集
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import get_default_config
from hcds.config.schema import (
    HCDSConfig,
    ExperimentConfig,
    FieldMappingConfig,
    ErrorIntensityConfig,
    CorrectnessConfig,
    MathEvaluatorConfig,
    ThompsonSamplingConfig,
    TSPriorConfig,
    TSExplorationConfig,
    PriorityConfig,
    PriorityWeightsConfig,
    RetirementConfig,
)


def get_math_reasoning_config(data_path: str = "data/train.jsonl") -> HCDSConfig:
    """
    获取数学推理任务配置
    基于默认配置，覆盖数学任务特定参数

    Args:
        data_path: 数据文件路径

    Returns:
        HCDSConfig 实例
    """
    # 以默认配置为基础
    config = get_default_config(data_path=data_path)

    # 覆盖实验名称
    config.experiment.name = "hcds_math_reasoning"

    # 数据配置 - 数学数据集字段映射
    config.data.field_mapping = FieldMappingConfig(
        instruction="question",
        input=None,
        output="answer",
    )
    config.data.content_template = "{instruction}"

    # 错误强度 - 数学任务以 Correctness 为主
    config.error_intensity.weights = {"loss": 0.4, "correctness": 0.6, "entropy": 0.0}
    config.error_intensity.correctness = CorrectnessConfig(
        enabled=True,
        eval_interval=3,
        sample_ratio=0.15,
        evaluator="exact_match",
        math=MathEvaluatorConfig(
            answer_patterns=[
                r"#### ([-\d,\.]+)",
                r"\\boxed\{([^}]+)\}",
                r"The answer is[:\s]*([-\d,\.]+)",
                r"= ([-\d,\.]+)$",
            ],
            normalize_answer=True,
            remove_commas=True,
            to_float=True,
            tolerance=0.001,
        ),
    )

    # Thompson Sampling - 数学任务调整
    config.thompson_sampling.prior.strength = 3.0
    config.thompson_sampling.exploration.warmup_rounds = 3

    # 优先级 - 难度更重要
    config.priority.weights = PriorityWeightsConfig(
        difficulty=0.6,
        rarity=0.3,
        novelty_base=0.4,
    )
    config.priority.difficulty_smoothing = 0.6

    # 退休 - 数学题更容易过拟合
    config.retirement = RetirementConfig(
        consecutive_threshold=2,
        error_threshold=0.05,
        revisit_probability=0.08,
    )

    return config
