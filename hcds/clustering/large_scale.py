"""
超大规模聚类 (>1M 样本)
采用 Pilot 采样 + HDBSCAN + FAISS ANN 分配
"""

from typing import Optional, Dict, List, Tuple
import numpy as np
from tqdm import tqdm

from hcds.clustering.base import Clusterer, ClusterResult
from hcds.clustering.hdbscan_cluster import HDBSCANClusterer, MiniBatchKMeansClusterer
from hcds.config.schema import ClusteringConfig, LargeScaleConfig


class LargeScaleClusterer(Clusterer):
    """超大规模聚类器"""

    def __init__(
        self,
        config: LargeScaleConfig,
        hdbscan_config: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        初始化超大规模聚类器

        Args:
            config: 超大规模配置
            hdbscan_config: HDBSCAN 配置
            verbose: 是否输出详细信息
        """
        self.config = config
        self.hdbscan_config = hdbscan_config or {}
        self.verbose = verbose

        self._centroids = None
        self._faiss_index = None
        self._cluster_sizes = None

    @classmethod
    def from_config(cls, config: ClusteringConfig) -> 'LargeScaleClusterer':
        """从配置创建"""
        return cls(
            config=config.large_scale,
            hdbscan_config={
                "min_cluster_size": config.hdbscan.min_cluster_size,
                "min_samples": config.hdbscan.min_samples,
                "metric": config.hdbscan.metric,
                "cluster_selection_method": config.hdbscan.cluster_selection_method,
                "noise_strategy": config.noise_handling.strategy.value
            }
        )

    def fit(self, embeddings: np.ndarray) -> ClusterResult:
        """
        执行超大规模聚类

        Args:
            embeddings: 嵌入矩阵 [N, D]

        Returns:
            ClusterResult 实例
        """
        n_samples = len(embeddings)

        if self.verbose:
            print(f"超大规模聚类: {n_samples:,} 个样本")

        # Step 1: Pilot 采样
        pilot_indices, pilot_embeddings = self._pilot_sampling(embeddings)

        if self.verbose:
            print(f"Step 1: Pilot 采样完成 - {len(pilot_indices):,} 个样本")

        # Step 2: 对 Pilot 集执行 HDBSCAN
        pilot_labels, pilot_centroids = self._cluster_pilot(pilot_embeddings)

        if self.verbose:
            print(f"Step 2: Pilot 聚类完成 - {len(pilot_centroids)} 个簇")

        # Step 3: 构建 FAISS 索引
        self._build_faiss_index(pilot_centroids)

        if self.verbose:
            print(f"Step 3: FAISS 索引构建完成")

        # Step 4: 全量分配
        all_labels = self._assign_all(embeddings, pilot_indices, pilot_labels)

        if self.verbose:
            print(f"Step 4: 全量分配完成")

        # Step 5: 大簇细分 (可选)
        if self.config.subdivision.enabled:
            all_labels, pilot_centroids = self._subdivide_large_clusters(
                embeddings, all_labels, pilot_centroids
            )
            if self.verbose:
                print(f"Step 5: 大簇细分完成 - 最终 {len(np.unique(all_labels))} 个簇")

        # 重新计算中心和大小
        self._centroids, self._cluster_sizes = self.compute_centroids(embeddings, all_labels)

        return ClusterResult(
            labels=all_labels,
            n_clusters=len(self._cluster_sizes),
            centroids=self._centroids,
            cluster_sizes=self._cluster_sizes
        )

    def _pilot_sampling(
        self,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pilot 采样

        Returns:
            (采样索引, 采样嵌入)
        """
        n_samples = len(embeddings)
        pilot_config = self.config.pilot_sampling

        # 计算采样数量
        n_pilot = int(n_samples * pilot_config.ratio)
        n_pilot = max(n_pilot, pilot_config.min_samples)
        n_pilot = min(n_pilot, pilot_config.max_samples, n_samples)

        if pilot_config.method == "random":
            indices = np.random.choice(n_samples, n_pilot, replace=False)
        else:
            # stratified 需要标签，这里回退到随机
            indices = np.random.choice(n_samples, n_pilot, replace=False)

        indices = np.sort(indices)
        return indices, embeddings[indices]

    def _cluster_pilot(
        self,
        pilot_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对 Pilot 集聚类

        Returns:
            (标签, 簇中心)
        """
        clusterer = HDBSCANClusterer(**self.hdbscan_config)
        result = clusterer.fit(pilot_embeddings)
        return result.labels, result.centroids

    def _build_faiss_index(self, centroids: np.ndarray) -> None:
        """构建 FAISS 索引"""
        try:
            import faiss
        except ImportError:
            raise ImportError("请安装 faiss: pip install faiss-cpu 或 faiss-gpu")

        dim = centroids.shape[1]
        faiss_config = self.config.faiss

        # 归一化中心 (用于内积搜索)
        centroids_normalized = centroids / (
            np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
        )

        # 创建索引
        if faiss_config.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.IndexFlatIP(dim)
                self._faiss_index = faiss.index_cpu_to_gpu(res, 0, index)
            except:
                print("警告: GPU FAISS 不可用，使用 CPU")
                self._faiss_index = faiss.IndexFlatIP(dim)
        else:
            self._faiss_index = faiss.IndexFlatIP(dim)

        self._faiss_index.add(centroids_normalized.astype(np.float32))
        self._centroids = centroids

    def _assign_all(
        self,
        embeddings: np.ndarray,
        pilot_indices: np.ndarray,
        pilot_labels: np.ndarray
    ) -> np.ndarray:
        """
        全量分配

        Args:
            embeddings: 全量嵌入
            pilot_indices: Pilot 样本索引
            pilot_labels: Pilot 样本标签

        Returns:
            全量标签
        """
        n_samples = len(embeddings)
        all_labels = np.full(n_samples, -1, dtype=np.int32)

        # 先填入 Pilot 标签
        all_labels[pilot_indices] = pilot_labels

        # 找出需要分配的样本
        non_pilot_mask = np.ones(n_samples, dtype=bool)
        non_pilot_mask[pilot_indices] = False
        non_pilot_indices = np.where(non_pilot_mask)[0]

        if len(non_pilot_indices) == 0:
            return all_labels

        # 批量 ANN 分配
        batch_size = 50000
        n_batches = (len(non_pilot_indices) + batch_size - 1) // batch_size

        iterator = range(0, len(non_pilot_indices), batch_size)
        if self.verbose:
            iterator = tqdm(iterator, total=n_batches, desc="全量分配")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(non_pilot_indices))
            batch_indices = non_pilot_indices[start_idx:end_idx]
            batch_embeddings = embeddings[batch_indices]

            # 归一化
            batch_normalized = batch_embeddings / (
                np.linalg.norm(batch_embeddings, axis=1, keepdims=True) + 1e-12
            )

            # FAISS 搜索
            _, labels = self._faiss_index.search(
                batch_normalized.astype(np.float32), 1
            )
            all_labels[batch_indices] = labels.flatten()

        return all_labels

    def _subdivide_large_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        细分大簇

        Returns:
            (新标签, 新中心)
        """
        max_size = self.config.subdivision.max_cluster_size
        target_size = self.config.subdivision.target_sub_cluster_size

        new_labels = labels.copy()
        new_centroids_list = list(centroids)
        next_label = len(centroids)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]

        for label in unique_labels:
            mask = labels == label
            cluster_size = mask.sum()

            if cluster_size <= max_size:
                continue

            # 需要细分
            n_sub = max(2, cluster_size // target_size)
            cluster_embeddings = embeddings[mask]
            cluster_indices = np.where(mask)[0]

            if self.verbose:
                print(f"细分簇 {label}: {cluster_size:,} -> {n_sub} 子簇")

            # Mini-Batch KMeans 细分
            sub_clusterer = MiniBatchKMeansClusterer(
                n_clusters=n_sub,
                batch_size=min(1024, cluster_size // 10)
            )
            sub_result = sub_clusterer.fit(cluster_embeddings)

            # 更新标签 (保留第一个子簇使用原标签)
            for sub_label in range(n_sub):
                sub_mask = sub_result.labels == sub_label
                sub_indices = cluster_indices[sub_mask]

                if sub_label == 0:
                    # 保留原标签
                    new_centroids_list[label] = sub_result.centroids[0]
                else:
                    # 新标签
                    new_labels[sub_indices] = next_label
                    new_centroids_list.append(sub_result.centroids[sub_label])
                    next_label += 1

        return new_labels, np.array(new_centroids_list)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """预测新样本的簇标签"""
        if self._faiss_index is None:
            raise ValueError("请先调用 fit() 方法")

        # 归一化
        embeddings_normalized = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        )

        _, labels = self._faiss_index.search(
            embeddings_normalized.astype(np.float32), 1
        )
        return labels.flatten()


def create_clusterer(
    config: ClusteringConfig,
    n_samples: int
) -> Clusterer:
    """
    根据配置和样本数量创建合适的聚类器

    Args:
        config: 聚类配置
        n_samples: 样本数量

    Returns:
        Clusterer 实例
    """
    if n_samples >= config.large_scale.threshold:
        print(f"样本数 {n_samples:,} >= 阈值 {config.large_scale.threshold:,}，使用超大规模聚类")
        return LargeScaleClusterer.from_config(config)
    else:
        print(f"样本数 {n_samples:,} < 阈值，使用标准 HDBSCAN")
        return HDBSCANClusterer.from_config(config)
