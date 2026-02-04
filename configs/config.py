"""
HCDS 默认主配置
所有超参数集中管理
"""

import sys
from pathlib import Path

# 确保项目根目录在路径中
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
    MathEvaluatorConfig,
    CodeEvaluatorConfig,
    EntropyConfig,
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


def get_default_config(data_path: str = "data/train.jsonl") -> HCDSConfig:
    """
    获取默认配置

    Args:
        data_path: 数据文件路径

    Returns:
        HCDSConfig 实例
    """
    return HCDSConfig(
        # ==================== 实验配置 ====================
        experiment=ExperimentConfig(
            name="hcds_default",
            seed=42,
            total_rounds=10,
            output_dir="outputs/",
        ),

        # ==================== 数据配置 ====================
        data=DataConfig(
            path=data_path,
            format="jsonl",
            field_mapping=FieldMappingConfig(
                instruction="instruction",
                input="input",
                output="output",
            ),
            content_template="{instruction}\n{input}",
            max_samples=None,
            min_length=10,
            max_length=4096,
        ),

        # ==================== 嵌入配置 ====================
        embedding=EmbeddingConfig(
            model="intfloat/multilingual-e5-large",
            content_strategy="instruction_only",
            weights={"instruction": 0.7, "response": 0.3},
            batch_size="auto",
            max_length=512,
            normalize=True,
            precision="fp16",
            pca=PCAConfig(
                enabled=True,
                target_dim=256,
                variance_ratio=0.95,
            ),
            storage=EmbeddingStorageConfig(
                path="data/embeddings/",
                format="hdf5",
                compression="gzip",
            ),
            incremental=IncrementalConfig(
                enabled=True,
                checkpoint_interval=10000,
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
                threshold=1_000_000,
                pilot_sampling=PilotSamplingConfig(
                    ratio=0.02,
                    min_samples=50000,
                    max_samples=500000,
                    method="random",
                ),
                faiss=FAISSConfig(
                    index_type="IVF4096,Flat",
                    nprobe=64,
                    use_gpu=True,
                ),
                subdivision=SubdivisionConfig(
                    enabled=True,
                    max_cluster_size=100000,
                    target_sub_cluster_size=50000,
                    algorithm="minibatch_kmeans",
                ),
            ),
            noise_handling=NoiseHandlingConfig(
                strategy="nearest",
            ),
        ),

        # ==================== 簇内压缩配置 ====================
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
                eval_interval=5,
                sample_ratio=0.1,
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
                code=CodeEvaluatorConfig(
                    timeout=10,
                    test_cases_field="test_cases",
                ),
            ),
            entropy=EntropyConfig(
                enabled=False,
                normalize_by_length=True,
            ),
        ),

        # ==================== Thompson Sampling 配置 ====================
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
                warmup_rounds=2,
                min_samples_per_cluster=10,
            ),
        ),

        # ==================== 预算分配配置 ====================
        budget=BudgetConfig(
            total_per_round=10000,
            allocation=AllocationConfig(
                base_ratio=0.2,
                max_cluster_ratio=3.0,
            ),
        ),

        # ==================== 优先级计算配置 ====================
        priority=PriorityConfig(
            weights=PriorityWeightsConfig(
                difficulty=0.5,
                rarity=0.5,
                novelty_base=0.5,
            ),
            difficulty_smoothing=0.7,
            knn=KNNConfig(
                k=10,
                algorithm="auto",
            ),
            selection=SelectionRatioConfig(
                priority_ratio=0.8,
                rare_ratio=0.15,
                random_ratio=0.05,
            ),
        ),

        # ==================== 样本退休配置 ====================
        retirement=RetirementConfig(
            enabled=True,
            consecutive_threshold=3,
            error_threshold=0.1,
            revisit_probability=0.05,
        ),

        # ==================== 并行化配置 ====================
        parallel=ParallelConfig(
            auto_detect=True,
            embedding=ParallelEmbeddingConfig(
                strategy="auto",
                num_gpus="auto",
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
            dir="logs/",
            to_file=True,
            to_console=True,
            metrics=MetricsConfig(
                log_interval=100,
                save_selection_history=True,
                tensorboard=False,
            ),
        ),

        # ==================== 静态簇指标权重 ====================
        cluster_prior_weights=ClusterPriorWeightsConfig(
            variance=0.25,
            global_distance=0.25,
            isolation=0.25,
            label_entropy=0.25,
        ),
    )
