"""
嵌入向量与聚类结果存储
支持 HDF5, NumPy, Memory-mapped 格式
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ClusterInfo:
    """簇信息"""
    id: int
    sample_ids: List[str]
    representative_ids: List[str]
    reference_ids: List[str]
    centroid: Optional[np.ndarray] = None

    # 静态指标
    variance: float = 0.0
    global_distance: float = 0.0
    isolation: float = 0.0
    label_entropy: float = 0.0
    prior_score: float = 0.5

    # Beta 后验参数
    alpha: float = 1.0
    beta: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (不含 centroid)"""
        return {
            "id": self.id,
            "sample_ids": self.sample_ids,
            "representative_ids": self.representative_ids,
            "reference_ids": self.reference_ids,
            "variance": self.variance,
            "global_distance": self.global_distance,
            "isolation": self.isolation,
            "label_entropy": self.label_entropy,
            "prior_score": self.prior_score,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], centroid: Optional[np.ndarray] = None) -> 'ClusterInfo':
        """从字典创建"""
        return cls(
            id=data["id"],
            sample_ids=data["sample_ids"],
            representative_ids=data["representative_ids"],
            reference_ids=data["reference_ids"],
            centroid=centroid,
            variance=data.get("variance", 0.0),
            global_distance=data.get("global_distance", 0.0),
            isolation=data.get("isolation", 0.0),
            label_entropy=data.get("label_entropy", 0.0),
            prior_score=data.get("prior_score", 0.5),
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
        )


class EmbeddingStorage:
    """嵌入向量存储管理器"""

    def __init__(
        self,
        path: Union[str, Path],
        format: str = "hdf5",
        compression: Optional[str] = "gzip"
    ):
        """
        初始化存储管理器

        Args:
            path: 存储目录
            format: 存储格式 (hdf5, numpy, mmap)
            compression: 压缩方式 (仅 hdf5 支持)
        """
        self.path = Path(path)
        self.format = format
        self.compression = compression

        self.path.mkdir(parents=True, exist_ok=True)

        self._embeddings: Optional[np.ndarray] = None
        self._id_to_idx: Optional[Dict[str, int]] = None
        self._idx_to_id: Optional[List[str]] = None

    def save(
        self,
        embeddings: np.ndarray,
        sample_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存嵌入向量

        Args:
            embeddings: 嵌入矩阵 [N, D]
            sample_ids: 样本 ID 列表
            metadata: 元数据
        """
        if len(embeddings) != len(sample_ids):
            raise ValueError(f"嵌入数量 ({len(embeddings)}) 与 ID 数量 ({len(sample_ids)}) 不匹配")

        if self.format == "hdf5":
            self._save_hdf5(embeddings, sample_ids, metadata)
        elif self.format == "numpy":
            self._save_numpy(embeddings, sample_ids, metadata)
        elif self.format == "mmap":
            self._save_mmap(embeddings, sample_ids, metadata)
        else:
            raise ValueError(f"不支持的格式: {self.format}")

    def load(self) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        加载嵌入向量

        Returns:
            (embeddings, sample_ids, metadata)
        """
        if self.format == "hdf5":
            return self._load_hdf5()
        elif self.format == "numpy":
            return self._load_numpy()
        elif self.format == "mmap":
            return self._load_mmap()
        else:
            raise ValueError(f"不支持的格式: {self.format}")

    def _save_hdf5(
        self,
        embeddings: np.ndarray,
        sample_ids: List[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """保存为 HDF5 格式"""
        try:
            import h5py
        except ImportError:
            raise ImportError("请安装 h5py: pip install h5py")

        file_path = self.path / "embeddings.h5"

        with h5py.File(file_path, 'w') as f:
            # 保存嵌入向量
            if self.compression:
                f.create_dataset(
                    "embeddings",
                    data=embeddings,
                    compression=self.compression,
                    chunks=True
                )
            else:
                f.create_dataset("embeddings", data=embeddings)

            # 保存 ID 列表
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("sample_ids", data=sample_ids, dtype=dt)

            # 保存元数据
            if metadata:
                f.attrs["metadata"] = json.dumps(metadata)

    def _load_hdf5(self) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """加载 HDF5 格式"""
        try:
            import h5py
        except ImportError:
            raise ImportError("请安装 h5py: pip install h5py")

        file_path = self.path / "embeddings.h5"

        with h5py.File(file_path, 'r') as f:
            embeddings = f["embeddings"][:]
            sample_ids = [s.decode() if isinstance(s, bytes) else s for s in f["sample_ids"][:]]

            metadata = {}
            if "metadata" in f.attrs:
                metadata = json.loads(f.attrs["metadata"])

        return embeddings, sample_ids, metadata

    def _save_numpy(
        self,
        embeddings: np.ndarray,
        sample_ids: List[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """保存为 NumPy 格式"""
        np.save(self.path / "embeddings.npy", embeddings)

        with open(self.path / "sample_ids.json", 'w') as f:
            json.dump(sample_ids, f)

        if metadata:
            with open(self.path / "metadata.json", 'w') as f:
                json.dump(metadata, f)

    def _load_numpy(self) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """加载 NumPy 格式"""
        embeddings = np.load(self.path / "embeddings.npy")

        with open(self.path / "sample_ids.json", 'r') as f:
            sample_ids = json.load(f)

        metadata = {}
        metadata_path = self.path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return embeddings, sample_ids, metadata

    def _save_mmap(
        self,
        embeddings: np.ndarray,
        sample_ids: List[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """保存为 Memory-mapped 格式"""
        mmap_path = self.path / "embeddings.mmap"

        # 保存形状信息
        shape_info = {"shape": embeddings.shape, "dtype": str(embeddings.dtype)}
        with open(self.path / "mmap_info.json", 'w') as f:
            json.dump(shape_info, f)

        # 创建 mmap 文件
        fp = np.memmap(mmap_path, dtype=embeddings.dtype, mode='w+', shape=embeddings.shape)
        fp[:] = embeddings[:]
        fp.flush()
        del fp

        # 保存 ID 和元数据
        with open(self.path / "sample_ids.json", 'w') as f:
            json.dump(sample_ids, f)

        if metadata:
            with open(self.path / "metadata.json", 'w') as f:
                json.dump(metadata, f)

    def _load_mmap(self) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """加载 Memory-mapped 格式"""
        with open(self.path / "mmap_info.json", 'r') as f:
            shape_info = json.load(f)

        shape = tuple(shape_info["shape"])
        dtype = np.dtype(shape_info["dtype"])

        embeddings = np.memmap(
            self.path / "embeddings.mmap",
            dtype=dtype,
            mode='r',
            shape=shape
        )

        with open(self.path / "sample_ids.json", 'r') as f:
            sample_ids = json.load(f)

        metadata = {}
        metadata_path = self.path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return embeddings, sample_ids, metadata

    def get_embedding(self, sample_id: str) -> Optional[np.ndarray]:
        """获取单个样本的嵌入向量"""
        if self._embeddings is None or self._id_to_idx is None:
            embeddings, sample_ids, _ = self.load()
            self._embeddings = embeddings
            self._id_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}
            self._idx_to_id = sample_ids

        idx = self._id_to_idx.get(sample_id)
        if idx is None:
            return None
        return self._embeddings[idx]

    def get_embeddings(self, sample_ids: List[str]) -> np.ndarray:
        """获取多个样本的嵌入向量"""
        if self._embeddings is None or self._id_to_idx is None:
            embeddings, sids, _ = self.load()
            self._embeddings = embeddings
            self._id_to_idx = {sid: idx for idx, sid in enumerate(sids)}
            self._idx_to_id = sids

        indices = [self._id_to_idx[sid] for sid in sample_ids if sid in self._id_to_idx]
        return self._embeddings[indices]

    def exists(self) -> bool:
        """检查存储是否存在"""
        if self.format == "hdf5":
            return (self.path / "embeddings.h5").exists()
        elif self.format == "numpy":
            return (self.path / "embeddings.npy").exists()
        elif self.format == "mmap":
            return (self.path / "embeddings.mmap").exists()
        return False

    def append(
        self,
        embeddings: np.ndarray,
        sample_ids: List[str]
    ) -> None:
        """
        追加嵌入向量 (增量更新)

        Args:
            embeddings: 新嵌入矩阵
            sample_ids: 新样本 ID 列表
        """
        if not self.exists():
            self.save(embeddings, sample_ids)
            return

        # 加载现有数据
        existing_emb, existing_ids, metadata = self.load()

        # 合并
        new_emb = np.vstack([existing_emb, embeddings])
        new_ids = existing_ids + sample_ids

        # 保存
        self.save(new_emb, new_ids, metadata)


class ClusterStorage:
    """聚类结果存储管理器"""

    def __init__(self, path: Union[str, Path]):
        """
        初始化存储管理器

        Args:
            path: 存储目录
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self._clusters: Optional[Dict[int, ClusterInfo]] = None

    def save(
        self,
        clusters: Dict[int, ClusterInfo],
        centroids: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存聚类结果

        Args:
            clusters: 簇信息字典
            centroids: 簇中心矩阵 [M, D]
            metadata: 元数据
        """
        # 保存簇信息 (不含 centroid)
        cluster_data = {cid: c.to_dict() for cid, c in clusters.items()}
        with open(self.path / "clusters.json", 'w', encoding='utf-8') as f:
            json.dump(cluster_data, f, ensure_ascii=False, indent=2)

        # 保存簇中心
        if centroids is not None:
            np.save(self.path / "centroids.npy", centroids)

        # 保存元数据
        if metadata:
            with open(self.path / "cluster_metadata.json", 'w') as f:
                json.dump(metadata, f)

    def load(self) -> Tuple[Dict[int, ClusterInfo], Optional[np.ndarray], Dict[str, Any]]:
        """
        加载聚类结果

        Returns:
            (clusters, centroids, metadata)
        """
        # 加载簇信息
        with open(self.path / "clusters.json", 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)

        # 加载簇中心
        centroids = None
        centroid_path = self.path / "centroids.npy"
        if centroid_path.exists():
            centroids = np.load(centroid_path)

        # 创建 ClusterInfo 对象
        clusters = {}
        for cid_str, data in cluster_data.items():
            cid = int(cid_str)
            centroid = centroids[cid] if centroids is not None and cid < len(centroids) else None
            clusters[cid] = ClusterInfo.from_dict(data, centroid)

        # 加载元数据
        metadata = {}
        metadata_path = self.path / "cluster_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return clusters, centroids, metadata

    def exists(self) -> bool:
        """检查存储是否存在"""
        return (self.path / "clusters.json").exists()

    def update_cluster(self, cluster_id: int, **updates) -> None:
        """
        更新单个簇的属性

        Args:
            cluster_id: 簇 ID
            **updates: 要更新的属性
        """
        if self._clusters is None:
            self._clusters, _, _ = self.load()

        if cluster_id not in self._clusters:
            raise KeyError(f"簇不存在: {cluster_id}")

        cluster = self._clusters[cluster_id]
        for key, value in updates.items():
            if hasattr(cluster, key):
                setattr(cluster, key, value)

    def save_state(self) -> None:
        """保存当前状态"""
        if self._clusters is not None:
            centroids = None
            centroid_path = self.path / "centroids.npy"
            if centroid_path.exists():
                centroids = np.load(centroid_path)
            self.save(self._clusters, centroids)

    def get_cluster(self, cluster_id: int) -> Optional[ClusterInfo]:
        """获取单个簇信息"""
        if self._clusters is None:
            self._clusters, _, _ = self.load()
        return self._clusters.get(cluster_id)

    def get_all_clusters(self) -> Dict[int, ClusterInfo]:
        """获取所有簇信息"""
        if self._clusters is None:
            self._clusters, _, _ = self.load()
        return self._clusters
