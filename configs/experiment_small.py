"""
HCDS 预实验配置 (CPU模式，小规模数据)
适用于 1K-10K 样本的快速测试
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
    LoggingConfig,
    ClusterPriorWeightsConfig,
)


def get_experiment_small_config(data_path: str = "./data/sample_data.jsonl") -> HCDSConfig:
    """
    获取小规模实验配置

    Args:
        data_path: 数据文件路径

    Returns:
        HCDSConfig 实例
    """
    return HCDSConfig(
        # 实验配置
        experiment=ExperimentConfig(
            name="pilot_experiment",
            seed=42,
            total_rounds=5,
            output_dir="./output/experiment_small",
        ),

        # 数据配置
        data=DataConfig(
            path=data_path,
            format="jsonl",
            field_mapping=FieldMappingConfig(
                instruction="instruction",
                input=None,
                output="output",
            ),
            content_template="{instruction}",
            max_length=4096,
        ),

        # 嵌入配置 (CPU优化)
        embedding=EmbeddingConfig(
            model="paraphrase-multilingual-MiniLM-L12-v2",
            content_strategy="instruction_only",
            batch_size=32,
            max_length=128,
            normalize=True,
            precision="fp32",
            pca=PCAConfig(
                enabled=True,
                target_dim=128,
                variance_ratio=0.95,
            ),
            storage=EmbeddingStorageConfig(
                path="./output/experiment_small/embeddings",
                format="hdf5",
                compression="gzip",
            ),
            incremental=IncrementalConfig(
                enabled=True,
                checkpoint_interval=500,
            ),
        ),

        # 聚类配置
        clustering=ClusteringConfig(
            algorithm="hdbscan",
            hdbscan=HDBSCANConfig(
                min_cluster_size=20,
                min_samples=5,
                metric="euclidean",
                cluster_selection_method="eom",
            ),
            noise_handling=NoiseHandlingConfig(
                strategy="nearest",
            ),
            large_scale=LargeScaleConfig(
                threshold=50000,
                pilot_sampling=PilotSamplingConfig(
                    ratio=0.1,
                    min_samples=1000,
                    max_samples=10000,
                    method="random",
                ),
                faiss=FAISSConfig(
                    use_gpu=False,
                ),
                subdivision=SubdivisionConfig(
                    enabled=False,
                ),
            ),
        ),

        # 压缩配置
        compression=CompressionConfig(
            algorithm="fps",
            max_representatives=256,
            reference_set_size=128,
        ),

        # 错误强度配置
        error_intensity=ErrorIntensityConfig(
            weights={"loss": 0.4, "correctness": 0.6, "entropy": 0.0},
            normalization=NormalizationConfig(
                method="zscore",
                zscore_offset=0.5,
                running_momentum=0.99,
            ),
        ),

        # Thompson Sampling 配置
        thompson_sampling=ThompsonSamplingConfig(
            prior=TSPriorConfig(
                alpha_0=1.0,
                beta_0=1.0,
                strength=0.5,
            ),
            selection=TSSelectionConfig(
                num_clusters="auto",
                ratio=0.3,
            ),
            exploration=TSExplorationConfig(
                warmup_rounds=3,
            ),
        ),

        # 预算配置
        budget=BudgetConfig(
            total_per_round=100,
            allocation=AllocationConfig(
                base_ratio=0.3,
                max_cluster_ratio=1.5,
            ),
        ),

        # 优先级配置
        priority=PriorityConfig(
            weights=PriorityWeightsConfig(
                difficulty=0.5,
                rarity=0.3,
                novelty_base=0.2,
            ),
            difficulty_smoothing=0.7,
            knn=KNNConfig(
                k=5,
            ),
            selection=SelectionRatioConfig(
                priority_ratio=0.5,
                rare_ratio=0.3,
                random_ratio=0.2,
            ),
        ),

        # 样本退休配置
        retirement=RetirementConfig(
            enabled=True,
            consecutive_threshold=3,
            error_threshold=0.1,
            revisit_probability=0.05,
        ),

        # 并行配置
        parallel=ParallelConfig(
            auto_detect=True,
            embedding=ParallelEmbeddingConfig(
                strategy="single",
                num_gpus=0,
                num_workers=4,
            ),
            clustering=ParallelClusteringConfig(
                backend="single",
            ),
            compression=ParallelCompressionConfig(
                backend="concurrent",
                max_workers=4,
            ),
        ),

        # 日志配置
        logging=LoggingConfig(
            level="INFO",
            dir="./output/experiment_small/logs",
            to_file=True,
            to_console=True,
        ),

        # 簇先验权重
        cluster_prior_weights=ClusterPriorWeightsConfig(
            variance=0.3,
            global_distance=0.3,
            isolation=0.2,
            label_entropy=0.2,
        ),
    )
