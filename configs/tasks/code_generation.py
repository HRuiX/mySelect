"""
HCDS 代码生成任务配置
适用于 CodeContests, HumanEval, MBPP 等代码数据集
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import get_default_config
from hcds.config.schema import (
    HCDSConfig,
    FieldMappingConfig,
    CorrectnessConfig,
    CodeEvaluatorConfig,
    PriorityWeightsConfig,
    RetirementConfig,
)


def get_code_generation_config(data_path: str = "data/train.jsonl") -> HCDSConfig:
    """
    获取代码生成任务配置
    基于默认配置，覆盖代码生成特定参数

    Args:
        data_path: 数据文件路径

    Returns:
        HCDSConfig 实例
    """
    config = get_default_config(data_path=data_path)

    # 覆盖实验名称
    config.experiment.name = "hcds_code_generation"

    # 数据配置
    config.data.field_mapping = FieldMappingConfig(
        instruction="prompt",
        input=None,
        output="solution",
    )
    config.data.content_template = "{instruction}"
    config.data.max_length = 8192  # 代码通常较长

    # 嵌入 - 代码需要更大模型
    config.embedding.model = "intfloat/multilingual-e5-large"
    config.embedding.max_length = 1024

    # 错误强度 - 代码任务 Correctness 最重要
    config.error_intensity.weights = {"loss": 0.3, "correctness": 0.7, "entropy": 0.0}
    config.error_intensity.correctness = CorrectnessConfig(
        enabled=True,
        eval_interval=5,
        sample_ratio=0.1,
        evaluator="test_exec",
        code=CodeEvaluatorConfig(
            timeout=10,
            memory_limit=512,
            test_cases_field="test_cases",
            hidden_test_ratio=0.5,
            sandbox=True,
            allowed_imports=[
                "math", "collections", "itertools", "functools",
                "heapq", "bisect", "re",
            ],
        ),
    )

    # Thompson Sampling
    config.thompson_sampling.prior.strength = 2.5
    config.thompson_sampling.exploration.warmup_rounds = 2

    # 优先级
    config.priority.weights = PriorityWeightsConfig(
        difficulty=0.6,
        rarity=0.3,
        novelty_base=0.4,
    )
    config.priority.difficulty_smoothing = 0.5

    # 退休
    config.retirement = RetirementConfig(
        consecutive_threshold=2,
        error_threshold=0.0,
        revisit_probability=0.1,
    )

    return config
