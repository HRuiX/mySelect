"""
HCDS MMLU/BBH Benchmark 任务配置
适用于 MMLU (Massive Multitask Language Understanding) 和 BBH (BIG-Bench Hard) 数据集
用于评估和微调 LLM 的多选题和推理能力
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import get_default_config
from hcds.config.schema import (
    HCDSConfig,
    ExperimentConfig,
    DataConfig,
    FieldMappingConfig,
    EmbeddingConfig,
    ErrorIntensityConfig,
    CorrectnessConfig,
    MathEvaluatorConfig,
    ThompsonSamplingConfig,
    TSPriorConfig,
    TSSelectionConfig,
    TSExplorationConfig,
    BudgetConfig,
    AllocationConfig,
    PriorityConfig,
    PriorityWeightsConfig,
    RetirementConfig,
    ClusterPriorWeightsConfig,
)


def get_mmlu_config(data_path: str = "data/mmlu_train.jsonl") -> HCDSConfig:
    """
    获取 MMLU 任务配置

    MMLU 数据集格式:
    - question: 问题文本
    - choices: 选项列表 [A, B, C, D]
    - answer: 正确答案 (A/B/C/D)
    - subject: 学科分类

    Args:
        data_path: 数据文件路径

    Returns:
        HCDSConfig 实例
    """
    config = get_default_config(data_path=data_path)

    # 实验配置
    config.experiment.name = "hcds_mmlu"
    config.experiment.output_dir = "outputs/mmlu/"

    # 数据配置 - MMLU 字段映射
    config.data.field_mapping = FieldMappingConfig(
        instruction="question",
        input="choices",  # 选项作为 input
        output="answer",
    )
    # 格式化模板: 将问题和选项组合
    config.data.content_template = "{instruction}\n{input}"
    config.data.max_length = 2048  # MMLU 题目通常较短

    # 嵌入配置 - 适合多选题
    config.embedding.content_strategy = "instruction_only"
    config.embedding.max_length = 512

    # 错误强度 - MMLU 以正确率为主
    config.error_intensity.weights = {"loss": 0.3, "correctness": 0.7, "entropy": 0.0}
    config.error_intensity.correctness = CorrectnessConfig(
        enabled=True,
        eval_interval=2,
        sample_ratio=0.2,
        evaluator="exact_match",
        math=MathEvaluatorConfig(
            answer_patterns=[
                r"^([A-D])$",  # 单独的选项字母
                r"[Aa]nswer[:\s]*([A-D])",
                r"[Tt]he answer is[:\s]*([A-D])",
                r"\(([A-D])\)",
            ],
            normalize_answer=True,
            remove_commas=False,
            to_float=False,
            tolerance=0.0,
        ),
    )

    # Thompson Sampling - MMLU 多学科需要更好的探索
    config.thompson_sampling.prior.strength = 2.5
    config.thompson_sampling.selection = TSSelectionConfig(
        num_clusters="auto",
        ratio=0.35,  # 选择更多簇以覆盖不同学科
    )
    config.thompson_sampling.exploration.warmup_rounds = 2
    config.thompson_sampling.exploration.min_samples_per_cluster = 5

    # 优先级 - 平衡难度和覆盖率
    config.priority.weights = PriorityWeightsConfig(
        difficulty=0.5,
        rarity=0.35,  # 稀有度较高以覆盖小众学科
        novelty_base=0.4,
    )
    config.priority.difficulty_smoothing = 0.65

    # 退休 - MMLU 题目较简单，适当放宽
    config.retirement = RetirementConfig(
        enabled=True,
        consecutive_threshold=3,
        error_threshold=0.08,
        revisit_probability=0.06,
    )

    # 预算配置
    config.budget = BudgetConfig(
        total_per_round=5000,  # MMLU 数据量大，每轮选取适量
        allocation=AllocationConfig(
            base_ratio=0.25,
            max_cluster_ratio=2.5,
        ),
    )

    # 簇先验权重 - 重视学科多样性
    config.cluster_prior_weights = ClusterPriorWeightsConfig(
        variance=0.2,
        global_distance=0.3,
        isolation=0.2,
        label_entropy=0.3,  # 更重视标签熵以覆盖不同学科
    )

    return config


def get_bbh_config(data_path: str = "data/bbh_train.jsonl") -> HCDSConfig:
    """
    获取 BBH (BIG-Bench Hard) 任务配置

    BBH 数据集格式:
    - input: 输入问题/任务描述
    - target: 目标答案
    - task: 任务类型

    Args:
        data_path: 数据文件路径

    Returns:
        HCDSConfig 实例
    """
    config = get_default_config(data_path=data_path)

    # 实验配置
    config.experiment.name = "hcds_bbh"
    config.experiment.output_dir = "outputs/bbh/"

    # 数据配置 - BBH 字段映射
    config.data.field_mapping = FieldMappingConfig(
        instruction="input",
        input=None,  # BBH 通常只有 input 和 target
        output="target",
    )
    config.data.content_template = "{instruction}"
    config.data.max_length = 4096  # BBH 推理任务可能较长

    # 嵌入配置
    config.embedding.content_strategy = "instruction_only"
    config.embedding.max_length = 768

    # 错误强度 - BBH 推理任务需要更精确的评估
    config.error_intensity.weights = {"loss": 0.35, "correctness": 0.65, "entropy": 0.0}
    config.error_intensity.correctness = CorrectnessConfig(
        enabled=True,
        eval_interval=3,
        sample_ratio=0.15,
        evaluator="exact_match",
        math=MathEvaluatorConfig(
            answer_patterns=[
                r"^(.+)$",  # BBH 答案格式多样
                r"[Aa]nswer[:\s]*(.+)",
                r"[Tt]he answer is[:\s]*(.+)",
                r"[Ff]inal answer[:\s]*(.+)",
            ],
            normalize_answer=True,
            remove_commas=True,
            to_float=False,
            tolerance=0.0,
        ),
    )

    # Thompson Sampling - BBH 需要更强的探索
    config.thompson_sampling.prior = TSPriorConfig(
        alpha_0=1.0,
        beta_0=1.0,
        strength=3.0,  # 更强的先验
    )
    config.thompson_sampling.selection = TSSelectionConfig(
        num_clusters="auto",
        ratio=0.4,  # BBH 任务多样，选择更多簇
    )
    config.thompson_sampling.exploration = TSExplorationConfig(
        warmup_rounds=3,
        min_samples_per_cluster=8,
    )

    # 优先级 - BBH 推理任务难度更重要
    config.priority.weights = PriorityWeightsConfig(
        difficulty=0.6,
        rarity=0.25,
        novelty_base=0.45,
    )
    config.priority.difficulty_smoothing = 0.55

    # 退休 - BBH 推理任务较难，收紧退休条件
    config.retirement = RetirementConfig(
        enabled=True,
        consecutive_threshold=4,  # 需要更多次低错误才退休
        error_threshold=0.05,
        revisit_probability=0.1,  # 更高的重访概率
    )

    # 预算配置
    config.budget = BudgetConfig(
        total_per_round=3000,  # BBH 数据量相对较小
        allocation=AllocationConfig(
            base_ratio=0.2,
            max_cluster_ratio=3.0,
        ),
    )

    # 簇先验权重 - 重视任务难度
    config.cluster_prior_weights = ClusterPriorWeightsConfig(
        variance=0.3,  # 更重视簇内方差
        global_distance=0.25,
        isolation=0.25,
        label_entropy=0.2,
    )

    return config


def get_mmlu_bbh_combined_config(data_path: str = "data/mmlu_bbh_train.jsonl") -> HCDSConfig:
    """
    获取 MMLU + BBH 联合训练配置

    混合数据集格式 (统一为):
    - question: 问题/任务描述
    - answer: 答案
    - source: 数据来源 (mmlu/bbh)
    - category: 类别/学科

    Args:
        data_path: 数据文件路径

    Returns:
        HCDSConfig 实例
    """
    config = get_default_config(data_path=data_path)

    # 实验配置
    config.experiment.name = "hcds_mmlu_bbh"
    config.experiment.output_dir = "outputs/mmlu_bbh/"

    # 数据配置 - 统一字段映射
    config.data.field_mapping = FieldMappingConfig(
        instruction="question",
        input=None,
        output="answer",
    )
    config.data.content_template = "{instruction}"
    config.data.max_length = 4096

    # 嵌入配置
    config.embedding.content_strategy = "instruction_only"
    config.embedding.max_length = 768

    # 错误强度 - 平衡配置
    config.error_intensity.weights = {"loss": 0.35, "correctness": 0.65, "entropy": 0.0}
    config.error_intensity.correctness = CorrectnessConfig(
        enabled=True,
        eval_interval=2,
        sample_ratio=0.15,
        evaluator="exact_match",
        math=MathEvaluatorConfig(
            answer_patterns=[
                r"^([A-D])$",
                r"[Aa]nswer[:\s]*(.+)",
                r"[Tt]he answer is[:\s]*(.+)",
            ],
            normalize_answer=True,
            remove_commas=True,
            to_float=False,
            tolerance=0.0,
        ),
    )

    # Thompson Sampling
    config.thompson_sampling.prior.strength = 2.8
    config.thompson_sampling.selection.ratio = 0.35
    config.thompson_sampling.exploration.warmup_rounds = 3

    # 优先级
    config.priority.weights = PriorityWeightsConfig(
        difficulty=0.55,
        rarity=0.3,
        novelty_base=0.45,
    )

    # 退休
    config.retirement = RetirementConfig(
        enabled=True,
        consecutive_threshold=3,
        error_threshold=0.06,
        revisit_probability=0.08,
    )

    # 预算
    config.budget.total_per_round = 8000

    return config
