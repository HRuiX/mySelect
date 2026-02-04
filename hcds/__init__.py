"""
HCDS: Hierarchical Clustering + Dynamic Sampling
基于语义聚类与动态反馈的分层训练数据选择框架

主要模块:
- config: 配置管理
- data: 数据加载与存储
- embedding: 嵌入计算
- clustering: 聚类与压缩
- sampling: Thompson Sampling 与采样策略
- feedback: 错误强度计算与后验更新
- parallel: 并行化支持
- metrics: 指标计算
- utils: 工具函数
"""

__version__ = "0.1.0"
__author__ = "HCDS Team"

from hcds.config import HCDSConfig, load_config
from hcds.core import HCDSPipeline

__all__ = [
    "HCDSConfig",
    "load_config",
    "HCDSPipeline",
    "__version__",
]
