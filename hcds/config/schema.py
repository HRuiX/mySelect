"""
HCDS 配置模型定义 (Pydantic Schema)
所有超参数的类型定义与验证
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal, Dict, Union, Any
from enum import Enum
from pathlib import Path


# ==================== 枚举类型 ====================

class EmbeddingModelEnum(str, Enum):
    """预定义嵌入模型"""
    E5_LARGE = "intfloat/multilingual-e5-large"
    E5_BASE = "intfloat/multilingual-e5-base"
    MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
    OPENAI_SMALL = "text-embedding-3-small"
    OPENAI_LARGE = "text-embedding-3-large"


class ClusteringAlgorithm(str, Enum):
    """聚类算法"""
    HDBSCAN = "hdbscan"
    KMEANS = "kmeans"
    MINIBATCH_KMEANS = "minibatch_kmeans"


class NormalizationMethod(str, Enum):
    """归一化方法"""
    ZSCORE = "zscore"
    MINMAX = "minmax"
    PERCENTILE = "percentile"
    SIGMOID = "sigmoid"


class NoiseStrategy(str, Enum):
    """噪声处理策略"""
    NEAREST = "nearest"
    SEPARATE = "separate"
    DROP = "drop"


class CorrectnessEvaluator(str, Enum):
    """正确性评估器类型"""
    EXACT_MATCH = "exact_match"
    JUDGE_MODEL = "judge_model"
    TEST_EXEC = "test_exec"
    CUSTOM = "custom"


# ==================== 子配置模块 ====================

class ExperimentConfig(BaseModel):
    """实验配置"""
    name: str = "hcds_default"
    seed: int = 42
    total_rounds: int = Field(default=10, ge=1, description="总训练轮数 T")
    output_dir: str = "outputs/"


class FieldMappingConfig(BaseModel):
    """字段映射配置"""
    instruction: str = "instruction"
    input: Optional[str] = "input"
    output: str = "output"


class DataConfig(BaseModel):
    """数据加载配置"""
    path: str = Field(..., description="数据文件路径")
    format: Literal["jsonl", "json", "parquet", "huggingface"] = "jsonl"

    field_mapping: FieldMappingConfig = Field(default_factory=FieldMappingConfig)
    content_template: str = "{instruction}\n{input}"

    max_samples: Optional[int] = Field(default=None, ge=1)
    min_length: int = Field(default=10, ge=0)
    max_length: int = Field(default=4096, ge=1)

    @field_validator('path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        # 允许相对路径和绝对路径
        return v


class PCAConfig(BaseModel):
    """PCA 降维配置"""
    enabled: bool = True
    target_dim: Optional[int] = Field(default=256, ge=1)
    variance_ratio: float = Field(default=0.95, gt=0, le=1)


class EmbeddingStorageConfig(BaseModel):
    """嵌入存储配置"""
    path: str = "data/embeddings/"
    format: Literal["hdf5", "numpy", "mmap"] = "hdf5"
    compression: Optional[str] = "gzip"


class IncrementalConfig(BaseModel):
    """增量计算配置"""
    enabled: bool = True
    checkpoint_interval: int = Field(default=10000, ge=100)
    resume_from_latest: bool = True


class EmbeddingConfig(BaseModel):
    """嵌入计算配置"""
    model: str = "intfloat/multilingual-e5-large"

    content_strategy: Literal["instruction_only", "response_only", "full", "weighted"] = "instruction_only"
    weights: Dict[str, float] = Field(default={"instruction": 0.7, "response": 0.3})

    batch_size: Union[int, Literal["auto"]] = "auto"
    max_length: int = Field(default=512, ge=1)
    normalize: bool = True
    precision: Literal["fp16", "fp32", "bf16"] = "fp16"

    pca: PCAConfig = Field(default_factory=PCAConfig)
    storage: EmbeddingStorageConfig = Field(default_factory=EmbeddingStorageConfig)
    incremental: IncrementalConfig = Field(default_factory=IncrementalConfig)


class HDBSCANConfig(BaseModel):
    """HDBSCAN 参数"""
    min_cluster_size: int = Field(default=50, ge=2)
    min_samples: int = Field(default=10, ge=1)
    cluster_selection_method: Literal["eom", "leaf"] = "eom"
    metric: str = "euclidean"


class PilotSamplingConfig(BaseModel):
    """Pilot 采样配置"""
    ratio: float = Field(default=0.02, gt=0, le=1)
    min_samples: int = Field(default=50000, ge=1000)
    max_samples: int = Field(default=500000, ge=1000)
    method: Literal["random", "stratified"] = "random"


class FAISSConfig(BaseModel):
    """FAISS 配置"""
    index_type: str = "IVF4096,Flat"
    nprobe: int = Field(default=64, ge=1)
    use_gpu: bool = True


class SubdivisionConfig(BaseModel):
    """大簇细分配置"""
    enabled: bool = True
    max_cluster_size: int = Field(default=100000, ge=1000)
    target_sub_cluster_size: int = Field(default=50000, ge=100)
    algorithm: Literal["minibatch_kmeans", "kmeans"] = "minibatch_kmeans"


class LargeScaleConfig(BaseModel):
    """超大规模聚类配置"""
    threshold: int = Field(default=1_000_000, ge=10000)
    pilot_sampling: PilotSamplingConfig = Field(default_factory=PilotSamplingConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    subdivision: SubdivisionConfig = Field(default_factory=SubdivisionConfig)


class NoiseHandlingConfig(BaseModel):
    """噪声处理配置"""
    strategy: NoiseStrategy = NoiseStrategy.NEAREST


class ClusteringConfig(BaseModel):
    """聚类配置"""
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.HDBSCAN
    hdbscan: HDBSCANConfig = Field(default_factory=HDBSCANConfig)
    large_scale: LargeScaleConfig = Field(default_factory=LargeScaleConfig)
    noise_handling: NoiseHandlingConfig = Field(default_factory=NoiseHandlingConfig)


class CompressionConfig(BaseModel):
    """簇内压缩配置"""
    max_representatives: int = Field(default=2048, ge=10)
    algorithm: Literal["fps", "kmeans", "random"] = "fps"
    reference_set_size: int = Field(default=512, ge=10)


class NormalizationConfig(BaseModel):
    """归一化配置"""
    method: NormalizationMethod = NormalizationMethod.ZSCORE
    zscore_offset: float = Field(default=0.5, ge=0, le=1)
    running_momentum: float = Field(default=0.99, gt=0, lt=1)


class MathEvaluatorConfig(BaseModel):
    """数学评估器配置"""
    answer_patterns: List[str] = Field(default=[
        r"#### ([-\d,\.]+)",
        r"\\boxed\{([^}]+)\}",
        r"The answer is[:\s]*([-\d,\.]+)",
        r"= ([-\d,\.]+)$"
    ])
    normalize_answer: bool = True
    remove_commas: bool = True
    to_float: bool = True
    tolerance: float = Field(default=0.001, ge=0)


class CodeEvaluatorConfig(BaseModel):
    """代码评估器配置"""
    timeout: int = Field(default=10, ge=1)
    memory_limit: int = Field(default=512, ge=64)
    test_cases_field: str = "test_cases"
    hidden_test_ratio: float = Field(default=0.5, ge=0, le=1)
    sandbox: bool = True
    allowed_imports: List[str] = Field(default=[
        "math", "collections", "itertools", "functools",
        "heapq", "bisect", "re"
    ])


class CorrectnessConfig(BaseModel):
    """正确性评估配置"""
    enabled: bool = True
    eval_interval: int = Field(default=5, ge=1)
    sample_ratio: float = Field(default=0.1, gt=0, le=1)
    evaluator: CorrectnessEvaluator = CorrectnessEvaluator.EXACT_MATCH
    math: MathEvaluatorConfig = Field(default_factory=MathEvaluatorConfig)
    code: CodeEvaluatorConfig = Field(default_factory=CodeEvaluatorConfig)


class EntropyConfig(BaseModel):
    """Entropy 配置"""
    enabled: bool = False
    normalize_by_length: bool = True


class ErrorIntensityConfig(BaseModel):
    """错误强度计算配置"""
    weights: Dict[str, float] = Field(default={"loss": 0.4, "correctness": 0.6, "entropy": 0.0})
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
    entropy: EntropyConfig = Field(default_factory=EntropyConfig)

    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            return {k: val/total for k, val in v.items()}
        return v


class TSPriorConfig(BaseModel):
    """Thompson Sampling 先验配置"""
    alpha_0: float = Field(default=1.0, gt=0)
    beta_0: float = Field(default=1.0, gt=0)
    strength: float = Field(default=2.0, ge=0)


class TSSelectionConfig(BaseModel):
    """Thompson Sampling 选择配置"""
    num_clusters: Union[int, Literal["auto"]] = "auto"
    ratio: float = Field(default=0.3, gt=0, le=1)


class TSExplorationConfig(BaseModel):
    """Thompson Sampling 探索配置"""
    warmup_rounds: int = Field(default=2, ge=0)
    min_samples_per_cluster: int = Field(default=10, ge=1)


class ThompsonSamplingConfig(BaseModel):
    """Thompson Sampling 配置"""
    prior: TSPriorConfig = Field(default_factory=TSPriorConfig)
    selection: TSSelectionConfig = Field(default_factory=TSSelectionConfig)
    exploration: TSExplorationConfig = Field(default_factory=TSExplorationConfig)


class AllocationConfig(BaseModel):
    """预算分配策略配置"""
    base_ratio: float = Field(default=0.2, ge=0, le=1)
    max_cluster_ratio: float = Field(default=3.0, ge=1)


class BudgetConfig(BaseModel):
    """预算配置"""
    total_per_round: int = Field(default=10000, ge=1)
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)


class PriorityWeightsConfig(BaseModel):
    """优先级权重配置"""
    difficulty: float = Field(default=0.5, ge=0, le=1)
    rarity: float = Field(default=0.5, ge=0, le=1)
    novelty_base: float = Field(default=0.5, ge=0, le=1)


class KNNConfig(BaseModel):
    """kNN 配置"""
    k: int = Field(default=10, ge=1)
    algorithm: Literal["auto", "brute", "ball_tree", "kd_tree"] = "auto"


class SelectionRatioConfig(BaseModel):
    """选择比例配置"""
    priority_ratio: float = Field(default=0.8, ge=0, le=1)
    rare_ratio: float = Field(default=0.15, ge=0, le=1)
    random_ratio: float = Field(default=0.05, ge=0, le=1)

    @model_validator(mode='after')
    def validate_ratios(self) -> 'SelectionRatioConfig':
        total = self.priority_ratio + self.rare_ratio + self.random_ratio
        if abs(total - 1.0) > 0.01:
            self.priority_ratio /= total
            self.rare_ratio /= total
            self.random_ratio /= total
        return self


class PriorityConfig(BaseModel):
    """优先级计算配置"""
    weights: PriorityWeightsConfig = Field(default_factory=PriorityWeightsConfig)
    difficulty_smoothing: float = Field(default=0.7, ge=0, lt=1)
    knn: KNNConfig = Field(default_factory=KNNConfig)
    selection: SelectionRatioConfig = Field(default_factory=SelectionRatioConfig)


class RetirementConfig(BaseModel):
    """样本退休配置"""
    enabled: bool = True
    consecutive_threshold: int = Field(default=3, ge=1)
    error_threshold: float = Field(default=0.1, ge=0, le=1)
    revisit_probability: float = Field(default=0.05, ge=0, le=1)


class ParallelEmbeddingConfig(BaseModel):
    """嵌入并行配置"""
    strategy: Literal["auto", "data_parallel", "distributed", "single"] = "auto"
    num_gpus: Union[int, Literal["auto"]] = "auto"
    num_workers: int = Field(default=8, ge=0)
    prefetch_factor: int = Field(default=4, ge=1)
    pin_memory: bool = True


class ParallelClusteringConfig(BaseModel):
    """聚类并行配置"""
    backend: Literal["joblib", "multiprocessing", "single"] = "joblib"
    n_jobs: Union[int, Literal["auto"]] = "auto"


class ParallelCompressionConfig(BaseModel):
    """压缩并行配置"""
    backend: Literal["concurrent", "ray", "single"] = "concurrent"
    max_workers: Union[int, Literal["auto"]] = "auto"


class ParallelPriorityConfig(BaseModel):
    """优先级并行配置"""
    backend: Literal["threading", "single"] = "threading"
    num_threads: Union[int, Literal["auto"]] = "auto"


class ResourceLimitsConfig(BaseModel):
    """资源限制配置"""
    max_cpu_percent: float = Field(default=80, ge=0, le=100)
    max_memory_percent: float = Field(default=85, ge=0, le=100)
    gpu_memory_fraction: float = Field(default=0.9, ge=0, le=1)


class ParallelConfig(BaseModel):
    """并行配置"""
    auto_detect: bool = True
    embedding: ParallelEmbeddingConfig = Field(default_factory=ParallelEmbeddingConfig)
    clustering: ParallelClusteringConfig = Field(default_factory=ParallelClusteringConfig)
    compression: ParallelCompressionConfig = Field(default_factory=ParallelCompressionConfig)
    priority: ParallelPriorityConfig = Field(default_factory=ParallelPriorityConfig)
    resources: ResourceLimitsConfig = Field(default_factory=ResourceLimitsConfig)


class MetricsConfig(BaseModel):
    """指标记录配置"""
    log_interval: int = Field(default=100, ge=1)
    save_selection_history: bool = True
    tensorboard: bool = False


class LoggingConfig(BaseModel):
    """日志配置"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dir: str = "logs/"
    to_file: bool = True
    to_console: bool = True
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


class ClusterPriorWeightsConfig(BaseModel):
    """簇先验权重配置"""
    variance: float = Field(default=0.25, ge=0)
    global_distance: float = Field(default=0.25, ge=0)
    isolation: float = Field(default=0.25, ge=0)
    label_entropy: float = Field(default=0.25, ge=0)

    @model_validator(mode='after')
    def validate_weights(self) -> 'ClusterPriorWeightsConfig':
        total = self.variance + self.global_distance + self.isolation + self.label_entropy
        if total > 0 and abs(total - 1.0) > 0.01:
            self.variance /= total
            self.global_distance /= total
            self.isolation /= total
            self.label_entropy /= total
        return self


# ==================== 主配置 ====================

class HCDSConfig(BaseModel):
    """HCDS 主配置"""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    error_intensity: ErrorIntensityConfig = Field(default_factory=ErrorIntensityConfig)
    thompson_sampling: ThompsonSamplingConfig = Field(default_factory=ThompsonSamplingConfig)
    budget: BudgetConfig
    priority: PriorityConfig = Field(default_factory=PriorityConfig)
    retirement: RetirementConfig = Field(default_factory=RetirementConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cluster_prior_weights: ClusterPriorWeightsConfig = Field(default_factory=ClusterPriorWeightsConfig)

    model_config = {"use_enum_values": True, "extra": "allow"}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump()

    def save(self, path: Union[str, Path]) -> None:
        """保存配置到 YAML 文件"""
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
