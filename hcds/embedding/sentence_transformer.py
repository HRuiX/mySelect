"""
Sentence-Transformers 编码器实现
支持本地模型和 GPU 加速
"""

from typing import List, Optional, Literal
import numpy as np
from tqdm import tqdm


class SentenceTransformerEncoder:
    """Sentence-Transformers 编码器"""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: Optional[str] = None,
        max_length: int = 512,
        normalize: bool = True,
        precision: Literal["fp16", "fp32", "bf16"] = "fp16"
    ):
        """
        初始化编码器

        Args:
            model_name: 模型名称或路径
            device: 设备 (None 自动选择, "cuda", "cpu")
            max_length: 最大序列长度
            normalize: 是否 L2 归一化
            precision: 精度
        """
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._normalize = normalize
        self._precision = precision

        self._model = None
        self._embedding_dim = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        # 确定设备
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"加载嵌入模型: {self._model_name}")
        print(f"设备: {self._device}")

        self._model = SentenceTransformer(self._model_name, device=self._device)

        # 设置精度
        if self._device == "cuda":
            if self._precision == "fp16":
                self._model.half()
            elif self._precision == "bf16":
                self._model = self._model.to(torch.bfloat16)

        # 获取嵌入维度
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        print(f"嵌入维度: {self._embedding_dim}")

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
        self._load_model()

        if batch_size is None:
            batch_size = self._auto_batch_size()

        # 预处理文本 (E5 模型需要添加前缀)
        if "e5" in self._model_name.lower():
            texts = [f"query: {t}" if len(t) < 200 else f"passage: {t}" for t in texts]

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True
        )

        return embeddings.astype(np.float32)

    def encode_batched(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        分批编码 (内存友好)

        Args:
            texts: 文本列表
            batch_size: 批大小
            show_progress: 是否显示进度条

        Returns:
            嵌入矩阵 [N, D]
        """
        self._load_model()

        all_embeddings = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="编码嵌入")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            # 预处理
            if "e5" in self._model_name.lower():
                batch_texts = [f"query: {t}" if len(t) < 200 else f"passage: {t}" for t in batch_texts]

            batch_embeddings = self._model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                normalize_embeddings=self._normalize,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def _auto_batch_size(self) -> int:
        """根据 GPU 显存自动确定批大小"""
        try:
            import torch
            if not torch.cuda.is_available():
                return 32

            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            if gpu_memory_gb >= 20:
                return 256
            elif gpu_memory_gb >= 14:
                return 128
            elif gpu_memory_gb >= 8:
                return 64
            else:
                return 32
        except:
            return 32

    @property
    def embedding_dim(self) -> int:
        """嵌入维度"""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """模型名称"""
        return self._model_name

    @property
    def device(self) -> str:
        """当前设备"""
        return self._device


class ParallelSentenceTransformerEncoder(SentenceTransformerEncoder):
    """多 GPU 并行编码器"""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device_ids: Optional[List[int]] = None,
        max_length: int = 512,
        normalize: bool = True,
        precision: Literal["fp16", "fp32", "bf16"] = "fp16"
    ):
        super().__init__(model_name, None, max_length, normalize, precision)
        self._device_ids = device_ids
        self._pool = None

    def _load_model(self):
        """加载多 GPU 模型"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        # 确定设备列表
        if self._device_ids is None:
            self._device_ids = list(range(torch.cuda.device_count()))

        if len(self._device_ids) == 0:
            # 回退到单 GPU/CPU
            super()._load_model()
            return

        print(f"加载嵌入模型: {self._model_name}")
        print(f"使用 GPU: {self._device_ids}")

        self._model = SentenceTransformer(self._model_name)

        # 获取嵌入维度
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

        # 启动多 GPU 池
        self._pool = self._model.start_multi_process_pool(target_devices=self._device_ids)

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """多 GPU 并行编码"""
        self._load_model()

        if self._pool is None:
            # 回退到单 GPU
            return super().encode(texts, batch_size, show_progress)

        if batch_size is None:
            batch_size = 64 * len(self._device_ids)

        # 预处理
        if "e5" in self._model_name.lower():
            texts = [f"query: {t}" if len(t) < 200 else f"passage: {t}" for t in texts]

        embeddings = self._model.encode_multi_process(
            texts,
            self._pool,
            batch_size=batch_size,
            normalize_embeddings=self._normalize
        )

        return embeddings.astype(np.float32)

    def __del__(self):
        """清理多进程池"""
        if self._pool is not None and self._model is not None:
            self._model.stop_multi_process_pool(self._pool)
