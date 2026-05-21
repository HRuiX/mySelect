"""
BBH + Mistral-7B-v0.3 对比实验配置

使用 HuggingFace 加载所有数据和模型:
- 模型: mistralai/Mistral-7B-v0.3
- 训练数据: princeton-nlp/less_data
- 评估数据: maveriq/bigbenchhard (BBH)
- 数据选择比例: 0.5%
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hcds.config.schema import (
    HCDSConfig,
    ExperimentConfig,
    DataConfig,
    FieldMappingConfig,
    EmbeddingConfig,
    PCAConfig,
    EmbeddingStorageConfig,
    IncrementalConfig,
    ClusteringConfig,
    HDBSCANConfig,
    LargeScaleConfig,
    PilotSamplingConfig,
    FAISSConfig,
    SubdivisionConfig,
    NoiseHandlingConfig,
    CompressionConfig,
    ErrorIntensityConfig,
    NormalizationConfig,
    CorrectnessConfig,
    ThompsonSamplingConfig,
    TSPriorConfig,
    TSSelectionConfig,
    TSExplorationConfig,
    BudgetConfig,
    AllocationConfig,
    PriorityConfig,
    PriorityWeightsConfig,
    KNNConfig,
    SelectionRatioConfig,
    RetirementConfig,
    ParallelConfig,
    ParallelEmbeddingConfig,
    ParallelClusteringConfig,
    ParallelCompressionConfig,
    ParallelPriorityConfig,
    ResourceLimitsConfig,
    LoggingConfig,
    MetricsConfig,
    ClusterPriorWeightsConfig,
)
from dataclasses import dataclass, field
from typing import List


# ==================== HuggingFace 数据集路径 ====================

HUGGINGFACE_DATASETS = {
    # 训练数据 (LESS 数据集: Flan V2 + CoT + Dolly + OASST1, 约 270K 样本)
    # 注意: princeton-nlp/less_data 新版 datasets 库无法直接加载，使用本地解压后的文件
    "less_train": "data/less_train/train_all.jsonl",

    # 评估数据集
    "bbh": "hf://SaylorTwift/bbh",
    "mmlu": "hf://cais/mmlu",
    "tydiqa": "hf://google-research-datasets/tydiqa",
}

HUGGINGFACE_MODELS = {
    "mistral_7b_v0.3": "mistralai/Mistral-7B-v0.3",
    "mistral_7b_v0.1": "mistralai/Mistral-7B-v0.1",
    "llama2_7b": "meta-llama/Llama-2-7b-hf",
}


# ==================== BBH 0.5% 配置 ====================

def get_bbh_hf_config(
    selection_ratio: float = 0.005,  # 0.5%
    total_samples: int = 270000,
    num_rounds: int = 4,
) -> HCDSConfig:
    """
    获取 BBH + HuggingFace 的 HCDS 配置

    Args:
        selection_ratio: 数据选择比例 (0.005 = 0.5%)
        total_samples: 训练数据总量
        num_rounds: 采样轮数

    Returns:
        HCDSConfig 实例
    """
    # 计算每轮预算
    total_selected = int(total_samples * selection_ratio)
    budget_per_round = total_selected // num_rounds

    print(f"数据选择配置:")
    print(f"  - 总样本数: {total_samples:,}")
    print(f"  - 选择比例: {selection_ratio * 100:.1f}%")
    print(f"  - 总选择数: {total_selected:,}")
    print(f"  - 轮数: {num_rounds}")
    print(f"  - 每轮预算: {budget_per_round:,}")

    return HCDSConfig(
        # ==================== 实验配置 ====================
        experiment=ExperimentConfig(
            name="hcds_bbh_mistral7b_0.5pct",
            seed=42,
            total_rounds=num_rounds,
            output_dir="outputs/hcds_bbh_0.5pct/",
        ),

        # ==================== 数据配置 (本地 JSONL) ====================
        data=DataConfig(
            path=HUGGINGFACE_DATASETS["less_train"],
            format="jsonl",  # 使用本地 JSONL 文件
            field_mapping=FieldMappingConfig(
                instruction="messages",  # messages 格式会自动解析
                input=None,
                output="messages",
            ),
            content_template="{instruction}",
            max_length=2048,
            hf_split="train",
            hf_subset=None,
        ),

        # ==================== 嵌入配置 ====================
        embedding=EmbeddingConfig(
            model="intfloat/e5-large-v2",
            content_strategy="instruction_only",
            batch_size="auto",  # 自动检测 GPU 显存，A100 80GB 会用 1024
            max_length=512,
            normalize=True,
            precision="fp16",
            pca=PCAConfig(
                enabled=True,
                target_dim=256,
                variance_ratio=0.95,
            ),
            storage=EmbeddingStorageConfig(
                path="outputs/hcds_bbh_0.5pct/embeddings/",
                format="hdf5",
                compression="gzip",
            ),
            incremental=IncrementalConfig(
                enabled=True,
                checkpoint_interval=100000,  # 增大检查点间隔，大幅减少 I/O
                resume_from_latest=True,
            ),
        ),

        # ==================== 聚类配置 ====================
        clustering=ClusteringConfig(
            algorithm="hdbscan",
            hdbscan=HDBSCANConfig(
                min_cluster_size=50,
                min_samples=10,
                cluster_selection_method="eom",
                metric="euclidean",
            ),
            large_scale=LargeScaleConfig(
                threshold=100_000,  # 降低阈值，270K 样本会使用 Pilot+FAISS 模式
                pilot_sampling=PilotSamplingConfig(
                    ratio=0.05,
                    min_samples=20000,
                    max_samples=100000,
                    method="random",
                ),
                faiss=FAISSConfig(
                    index_type="IVF1024,Flat",
                    nprobe=32,
                    use_gpu=True,
                ),
                subdivision=SubdivisionConfig(
                    enabled=True,
                    max_cluster_size=50000,
                    target_sub_cluster_size=20000,
                    algorithm="minibatch_kmeans",
                ),
            ),
            noise_handling=NoiseHandlingConfig(
                strategy="nearest",
            ),
        ),

        # ==================== 压缩配置 ====================
        compression=CompressionConfig(
            max_representatives=2048,
            algorithm="fps",
            reference_set_size=512,
        ),

        # ==================== 错误强度配置 ====================
        error_intensity=ErrorIntensityConfig(
            weights={"loss": 0.4, "correctness": 0.6, "entropy": 0.0},
            normalization=NormalizationConfig(
                method="zscore",
                zscore_offset=0.5,
                running_momentum=0.99,
            ),
            correctness=CorrectnessConfig(
                enabled=True,
                eval_interval=1,
                sample_ratio=0.15,
                evaluator="exact_match",
            ),
        ),

        # ==================== Thompson Sampling ====================
        thompson_sampling=ThompsonSamplingConfig(
            prior=TSPriorConfig(
                alpha_0=1.0,
                beta_0=1.0,
                strength=2.0,
            ),
            selection=TSSelectionConfig(
                num_clusters="auto",
                ratio=0.3,
            ),
            exploration=TSExplorationConfig(
                warmup_rounds=1,
                min_samples_per_cluster=5,
            ),
        ),

        # ==================== 预算配置 ====================
        budget=BudgetConfig(
            total_per_round=budget_per_round,
            allocation=AllocationConfig(
                base_ratio=0.2,
                max_cluster_ratio=3.0,
            ),
        ),

        # ==================== 优先级配置 ====================
        priority=PriorityConfig(
            weights=PriorityWeightsConfig(
                difficulty=0.5,
                rarity=0.5,
                novelty_base=0.5,
            ),
            difficulty_smoothing=0.7,
            knn=KNNConfig(k=10, algorithm="auto"),
            selection=SelectionRatioConfig(
                priority_ratio=0.8,
                rare_ratio=0.15,
                random_ratio=0.05,
            ),
        ),

        # ==================== 退休配置 ====================
        retirement=RetirementConfig(
            enabled=True,
            consecutive_threshold=2,
            error_threshold=0.1,
            revisit_probability=0.05,
        ),

        # ==================== 并行配置 (A100 优化) ====================
        parallel=ParallelConfig(
            auto_detect=True,
            embedding=ParallelEmbeddingConfig(
                strategy="auto",
                num_gpus=1,
                num_workers=8,
                prefetch_factor=4,
                pin_memory=True,
            ),
            clustering=ParallelClusteringConfig(
                backend="joblib",
                n_jobs="auto",
            ),
            compression=ParallelCompressionConfig(
                backend="concurrent",
                max_workers="auto",
            ),
            priority=ParallelPriorityConfig(
                backend="threading",
                num_threads="auto",
            ),
            resources=ResourceLimitsConfig(
                max_cpu_percent=80,
                max_memory_percent=85,
                gpu_memory_fraction=0.9,
            ),
        ),

        # ==================== 日志配置 ====================
        logging=LoggingConfig(
            level="INFO",
            dir="outputs/hcds_bbh_0.5pct/logs/",
            to_file=True,
            to_console=True,
            metrics=MetricsConfig(
                log_interval=100,
                save_selection_history=True,
                tensorboard=True,
            ),
        ),

        # ==================== 簇先验权重 ====================
        cluster_prior_weights=ClusterPriorWeightsConfig(
            variance=0.25,
            global_distance=0.25,
            isolation=0.25,
            label_entropy=0.25,
        ),
    )


# ==================== 训练配置 (对齐 LESS) ====================

@dataclass
class MistralV03TrainingConfig:
    """
    Mistral-7B-v0.3 训练配置

    对齐 LESS 论文参数，针对 A100-80GB 优化
    """
    # 模型 (HuggingFace)
    model_name: str = "mistralai/Mistral-7B-v0.3"

    # LoRA (对齐 LESS)
    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 512
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # 训练参数 (对齐 LESS)
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: int = 4
    lr_scheduler_type: str = "linear"

    # 批次 (A100-80GB 优化: batch=2)
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16  # 有效 batch = 32

    # 序列长度
    max_seq_length: int = 2048

    # 精度
    bf16: bool = True
    fp16: bool = False

    # 输出
    output_dir: str = "outputs/hcds_bbh_0.5pct/model"
    save_strategy: str = "epoch"
    logging_steps: int = 1


# ==================== 默认实例 ====================

BBH_HF_CONFIG = get_bbh_hf_config()
TRAINING_CONFIG = MistralV03TrainingConfig()


if __name__ == "__main__":
    # 打印配置摘要
    config = get_bbh_hf_config()
    print("\n" + "=" * 60)
    print("HCDS BBH 配置摘要")
    print("=" * 60)
    print(f"数据源: {config.data.path}")
    print(f"每轮预算: {config.budget.total_per_round}")
    print(f"总轮数: {config.experiment.total_rounds}")
    print(f"输出目录: {config.experiment.output_dir}")
