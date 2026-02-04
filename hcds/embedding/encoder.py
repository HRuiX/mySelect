"""
嵌入编码器抽象基类与工厂
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

from hcds.config.schema import EmbeddingConfig


class Encoder(ABC):
    """嵌入编码器抽象基类"""

    @abstractmethod
    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        将文本编码为嵌入向量

        Args:
            texts: 文本列表
            batch_size: 批大小
            show_progress: 是否显示进度条

        Returns:
            嵌入矩阵 [N, D]
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """嵌入维度"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        pass

    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 归一化"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # 避免除零
        return embeddings / norms


class EncoderFactory:
    """编码器工厂"""

    @staticmethod
    def create(config: EmbeddingConfig, device: Optional[str] = None) -> Encoder:
        """
        创建编码器

        Args:
            config: 嵌入配置
            device: 设备 (cuda/cpu)

        Returns:
            Encoder 实例
        """
        model_name = config.model

        # OpenAI API 模型
        if model_name.startswith("text-embedding"):
            from hcds.embedding.api_encoder import OpenAIEncoder
            return OpenAIEncoder(
                model_name=model_name,
                batch_size=config.batch_size if config.batch_size != "auto" else 100
            )

        # Sentence-Transformers 模型
        from hcds.embedding.sentence_transformer import SentenceTransformerEncoder
        return SentenceTransformerEncoder(
            model_name=model_name,
            device=device,
            max_length=config.max_length,
            normalize=config.normalize,
            precision=config.precision
        )

    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """
        获取模型信息

        Args:
            model_name: 模型名称

        Returns:
            模型信息字典
        """
        model_info = {
            "intfloat/multilingual-e5-large": {
                "dim": 1024,
                "max_length": 512,
                "multilingual": True,
                "type": "sentence-transformers"
            },
            "intfloat/multilingual-e5-base": {
                "dim": 768,
                "max_length": 512,
                "multilingual": True,
                "type": "sentence-transformers"
            },
            "paraphrase-multilingual-MiniLM-L12-v2": {
                "dim": 384,
                "max_length": 128,
                "multilingual": True,
                "type": "sentence-transformers"
            },
            "text-embedding-3-small": {
                "dim": 1536,
                "max_length": 8191,
                "multilingual": True,
                "type": "openai"
            },
            "text-embedding-3-large": {
                "dim": 3072,
                "max_length": 8191,
                "multilingual": True,
                "type": "openai"
            },
        }

        return model_info.get(model_name, {
            "dim": None,
            "max_length": 512,
            "multilingual": False,
            "type": "unknown"
        })
