"""
增量嵌入计算器
支持检查点、断点续传、增量更新
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
import numpy as np
from tqdm import tqdm

from hcds.embedding.encoder import Encoder, EncoderFactory
from hcds.data.storage import EmbeddingStorage
from hcds.config.schema import EmbeddingConfig


class IncrementalEmbeddingComputer:
    """增量嵌入计算器"""

    def __init__(
        self,
        config: EmbeddingConfig,
        encoder: Optional[Encoder] = None,
        storage: Optional[EmbeddingStorage] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None
    ):
        """
        初始化增量计算器

        Args:
            config: 嵌入配置
            encoder: 编码器 (None 则根据配置创建)
            storage: 存储管理器 (None 则根据配置创建)
            checkpoint_dir: 检查点目录
        """
        self.config = config

        # 编码器
        if encoder is None:
            encoder = EncoderFactory.create(config)
        self.encoder = encoder

        # 存储
        if storage is None:
            storage = EmbeddingStorage(
                path=config.storage.path,
                format=config.storage.format,
                compression=config.storage.compression
            )
        self.storage = storage

        # 检查点
        self.checkpoint_dir = Path(checkpoint_dir or config.storage.path) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._processed_ids: Set[str] = set()
        self._checkpoint_interval = config.incremental.checkpoint_interval

    def compute(
        self,
        texts: List[str],
        sample_ids: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        skip_existing: bool = True
    ) -> np.ndarray:
        """
        计算嵌入 (支持增量)

        Args:
            texts: 文本列表
            sample_ids: 样本 ID 列表
            batch_size: 批大小
            show_progress: 是否显示进度条
            skip_existing: 是否跳过已存在的嵌入

        Returns:
            嵌入矩阵 [N, D]
        """
        if len(texts) != len(sample_ids):
            raise ValueError("texts 和 sample_ids 长度不匹配")

        # 加载已处理的 ID
        self._load_processed_ids()

        # 过滤需要计算的样本
        if skip_existing and self._processed_ids:
            new_indices = [
                i for i, sid in enumerate(sample_ids)
                if sid not in self._processed_ids
            ]
            if len(new_indices) == 0:
                print("所有样本已有嵌入，跳过计算")
                return self._load_all_embeddings(sample_ids)

            new_texts = [texts[i] for i in new_indices]
            new_ids = [sample_ids[i] for i in new_indices]

            print(f"跳过 {len(self._processed_ids)} 个已处理样本，计算 {len(new_texts)} 个新样本")
        else:
            new_texts = texts
            new_ids = sample_ids

        # 计算批大小
        if batch_size is None:
            batch_size = self.config.batch_size
            if batch_size == "auto":
                batch_size = self._auto_batch_size()

        # 分批计算
        all_embeddings = []
        all_ids = []
        n_batches = (len(new_texts) + batch_size - 1) // batch_size

        iterator = range(0, len(new_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="计算嵌入")

        for batch_idx, start_idx in enumerate(iterator):
            end_idx = min(start_idx + batch_size, len(new_texts))
            batch_texts = new_texts[start_idx:end_idx]
            batch_ids = new_ids[start_idx:end_idx]

            # 编码
            batch_embeddings = self.encoder.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress=False
            )

            all_embeddings.append(batch_embeddings)
            all_ids.extend(batch_ids)
            self._processed_ids.update(batch_ids)

            # 检查点
            if self.config.incremental.enabled:
                processed_count = start_idx + len(batch_texts)
                if processed_count % self._checkpoint_interval < batch_size:
                    self._save_checkpoint(
                        np.vstack(all_embeddings),
                        all_ids,
                        processed_count
                    )

        new_embeddings = np.vstack(all_embeddings)

        # 保存最终结果
        self._save_final(new_embeddings, new_ids, sample_ids)

        # 返回完整嵌入 (包括之前已存在的)
        return self._load_all_embeddings(sample_ids)

    def _auto_batch_size(self) -> int:
        """自动确定批大小"""
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

    def _load_processed_ids(self) -> None:
        """加载已处理的样本 ID"""
        # 从检查点加载
        checkpoint_path = self.checkpoint_dir / "processed_ids.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                self._processed_ids = set(json.load(f))

        # 从存储加载
        if self.storage.exists():
            _, existing_ids, _ = self.storage.load()
            self._processed_ids.update(existing_ids)

    def _save_checkpoint(
        self,
        embeddings: np.ndarray,
        sample_ids: List[str],
        processed_count: int
    ) -> None:
        """保存检查点"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{processed_count}.npz"

        np.savez_compressed(
            checkpoint_file,
            embeddings=embeddings,
            sample_ids=np.array(sample_ids, dtype=object)
        )

        # 保存已处理 ID
        with open(self.checkpoint_dir / "processed_ids.json", 'w') as f:
            json.dump(list(self._processed_ids), f)

        print(f"检查点已保存: {processed_count} 个样本")

    def _save_final(
        self,
        new_embeddings: np.ndarray,
        new_ids: List[str],
        all_sample_ids: List[str]
    ) -> None:
        """保存最终结果"""
        if self.storage.exists():
            # 追加模式
            self.storage.append(new_embeddings, new_ids)
        else:
            # 新建
            metadata = {
                "model": self.encoder.model_name,
                "dim": self.encoder.embedding_dim,
                "count": len(new_ids)
            }
            self.storage.save(new_embeddings, new_ids, metadata)

        # 清理检查点
        self._cleanup_checkpoints()

    def _load_all_embeddings(self, sample_ids: List[str]) -> np.ndarray:
        """按顺序加载所有嵌入"""
        if not self.storage.exists():
            raise ValueError("存储中没有嵌入数据")

        return self.storage.get_embeddings(sample_ids)

    def _cleanup_checkpoints(self) -> None:
        """清理检查点文件"""
        for f in self.checkpoint_dir.glob("checkpoint_*.npz"):
            f.unlink()

    def resume_from_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        从最新检查点恢复

        Returns:
            检查点数据或 None
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.npz"))
        if not checkpoints:
            return None

        # 找最新的检查点
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))

        data = np.load(latest, allow_pickle=True)
        return {
            "embeddings": data["embeddings"],
            "sample_ids": data["sample_ids"].tolist(),
            "processed_count": int(latest.stem.split('_')[1])
        }


class PCAReducer:
    """PCA 降维器"""

    def __init__(
        self,
        target_dim: Optional[int] = None,
        variance_ratio: float = 0.95
    ):
        """
        初始化 PCA 降维器

        Args:
            target_dim: 目标维度 (优先)
            variance_ratio: 保留的方差比例
        """
        self.target_dim = target_dim
        self.variance_ratio = variance_ratio

        self._pca = None
        self._actual_dim = None

    def fit(self, embeddings: np.ndarray) -> 'PCAReducer':
        """
        拟合 PCA

        Args:
            embeddings: 嵌入矩阵 [N, D]

        Returns:
            self
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("请安装 scikit-learn: pip install scikit-learn")

        if self.target_dim is not None:
            n_components = min(self.target_dim, embeddings.shape[1])
        else:
            n_components = self.variance_ratio

        self._pca = PCA(n_components=n_components)
        self._pca.fit(embeddings)

        self._actual_dim = self._pca.n_components_

        print(f"PCA 降维: {embeddings.shape[1]} -> {self._actual_dim}")
        if hasattr(self._pca, 'explained_variance_ratio_'):
            total_var = sum(self._pca.explained_variance_ratio_)
            print(f"保留方差比例: {total_var:.4f}")

        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        降维

        Args:
            embeddings: 嵌入矩阵 [N, D]

        Returns:
            降维后的嵌入 [N, target_dim]
        """
        if self._pca is None:
            raise ValueError("请先调用 fit() 方法")

        return self._pca.transform(embeddings).astype(np.float32)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """拟合并降维"""
        self.fit(embeddings)
        return self.transform(embeddings)

    @property
    def output_dim(self) -> int:
        """输出维度"""
        return self._actual_dim

    def save(self, path: Union[str, Path]) -> None:
        """保存 PCA 模型"""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self._pca, f)

    def load(self, path: Union[str, Path]) -> 'PCAReducer':
        """加载 PCA 模型"""
        import pickle
        with open(path, 'rb') as f:
            self._pca = pickle.load(f)
        self._actual_dim = self._pca.n_components_
        return self
