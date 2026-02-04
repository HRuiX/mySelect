"""
OpenAI API 编码器实现
"""

import os
import time
from typing import List, Optional
import numpy as np
from tqdm import tqdm


class OpenAIEncoder:
    """OpenAI API 嵌入编码器"""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化编码器

        Args:
            model_name: 模型名称
            api_key: API 密钥 (None 则从环境变量读取)
            batch_size: 批大小 (API 限制)
            max_retries: 最大重试次数
            retry_delay: 重试延迟 (秒)
        """
        self._model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._batch_size = min(batch_size, 2048)  # API 限制
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._client = None
        self._embedding_dim = None

        # 模型维度映射
        self._dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def _init_client(self):
        """初始化 OpenAI 客户端"""
        if self._client is not None:
            return

        if not self._api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量或传入 api_key 参数")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        self._client = OpenAI(api_key=self._api_key)
        self._embedding_dim = self._dim_map.get(self._model_name, 1536)

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        编码文本

        Args:
            texts: 文本列表
            batch_size: 批大小
            show_progress: 是否显示进度条

        Returns:
            嵌入矩阵 [N, D]
        """
        self._init_client()

        if batch_size is None:
            batch_size = self._batch_size

        all_embeddings = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="调用 OpenAI API")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            batch_embeddings = self._encode_batch_with_retry(batch_texts)
            all_embeddings.append(batch_embeddings)

        result = np.vstack(all_embeddings).astype(np.float32)

        # L2 归一化
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        result = result / norms

        return result

    def _encode_batch_with_retry(self, texts: List[str]) -> np.ndarray:
        """带重试的批量编码"""
        for attempt in range(self._max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self._model_name,
                    input=texts
                )
                embeddings = [item.embedding for item in response.data]
                return np.array(embeddings)

            except Exception as e:
                if attempt < self._max_retries - 1:
                    print(f"API 调用失败 (尝试 {attempt + 1}/{self._max_retries}): {e}")
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    raise

    @property
    def embedding_dim(self) -> int:
        """嵌入维度"""
        if self._embedding_dim is None:
            self._init_client()
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """模型名称"""
        return self._model_name

    def estimate_cost(self, n_tokens: int) -> float:
        """
        估算 API 成本

        Args:
            n_tokens: token 数量

        Returns:
            美元成本
        """
        # 2024 年定价
        price_per_million = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10,
        }

        price = price_per_million.get(self._model_name, 0.02)
        return (n_tokens / 1_000_000) * price
