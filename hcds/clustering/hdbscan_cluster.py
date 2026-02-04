"""
HDBSCAN 聚类器实现
"""

from typing import Optional, Literal
import numpy as np

from hcds.clustering.base import Clusterer, ClusterResult
from hcds.config.schema import ClusteringConfig


class HDBSCANClusterer(Clusterer):
    """HDBSCAN 聚类器"""

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        metric: str = "euclidean",
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        noise_strategy: Literal["nearest", "separate", "drop"] = "nearest"
    ):
        """
        初始化 HDBSCAN 聚类器

        Args:
            min_cluster_size: 最小簇大小
            min_samples: 核心点所需的最小邻居数
            metric: 距离度量
            cluster_selection_method: 簇选择方法
            noise_strategy: 噪声处理策略
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.noise_strategy = noise_strategy

        self._clusterer = None
        self._centroids = None

    @classmethod
    def from_config(cls, config: ClusteringConfig) -> 'HDBSCANClusterer':
        """从配置创建"""
        return cls(
            min_cluster_size=config.hdbscan.min_cluster_size,
            min_samples=config.hdbscan.min_samples,
            metric=config.hdbscan.metric,
            cluster_selection_method=config.hdbscan.cluster_selection_method,
            noise_strategy=config.noise_handling.strategy.value
        )

    def fit(self, embeddings: np.ndarray) -> ClusterResult:
        """
        执行 HDBSCAN 聚类

        Args:
            embeddings: 嵌入矩阵 [N, D]

        Returns:
            ClusterResult 实例
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError("请安装 hdbscan: pip install hdbscan")

        print(f"执行 HDBSCAN 聚类: {len(embeddings)} 个样本")

        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True
        )

        labels = self._clusterer.fit_predict(embeddings)

        # 处理噪声点
        noise_mask = labels == -1
        n_noise = noise_mask.sum()

        if n_noise > 0:
            print(f"检测到 {n_noise} 个噪声点 ({100*n_noise/len(labels):.2f}%)")
            labels = self._handle_noise(embeddings, labels)

        # 计算簇中心
        centroids, cluster_sizes = self.compute_centroids(embeddings, labels)
        self._centroids = centroids

        n_clusters = len(cluster_sizes)
        print(f"聚类完成: {n_clusters} 个簇")

        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            centroids=centroids,
            noise_mask=noise_mask,
            cluster_sizes=cluster_sizes
        )

    def _handle_noise(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """处理噪声点"""
        noise_mask = labels == -1

        if self.noise_strategy == "drop":
            # 保持 -1 标签，后续可选择过滤
            return labels

        elif self.noise_strategy == "separate":
            # 将噪声点作为独立簇
            max_label = labels.max()
            labels[noise_mask] = max_label + 1
            return labels

        elif self.noise_strategy == "nearest":
            # 分配到最近簇
            valid_labels = labels[~noise_mask]
            valid_embeddings = embeddings[~noise_mask]

            # 计算每个簇的中心
            unique_labels = np.unique(valid_labels)
            centroids = np.array([
                valid_embeddings[valid_labels == l].mean(axis=0)
                for l in unique_labels
            ])

            # 为每个噪声点找最近的簇
            noise_embeddings = embeddings[noise_mask]
            for i, emb in enumerate(noise_embeddings):
                distances = np.linalg.norm(centroids - emb, axis=1)
                nearest_label = unique_labels[np.argmin(distances)]
                labels[np.where(noise_mask)[0][i]] = nearest_label

            return labels

        return labels

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        预测新样本的簇标签

        Args:
            embeddings: 嵌入矩阵 [N, D]

        Returns:
            簇标签 [N]
        """
        if self._clusterer is None:
            raise ValueError("请先调用 fit() 方法")

        try:
            import hdbscan
            labels, _ = hdbscan.approximate_predict(self._clusterer, embeddings)
        except:
            # 回退到最近中心
            if self._centroids is None:
                raise ValueError("无法预测：没有簇中心")

            labels = []
            for emb in embeddings:
                distances = np.linalg.norm(self._centroids - emb, axis=1)
                labels.append(np.argmin(distances))
            labels = np.array(labels)

        return labels


class KMeansClusterer(Clusterer):
    """KMeans 聚类器"""

    def __init__(
        self,
        n_clusters: int = 100,
        max_iter: int = 300,
        n_init: int = 10,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self._kmeans = None

    def fit(self, embeddings: np.ndarray) -> ClusterResult:
        """执行 KMeans 聚类"""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("请安装 scikit-learn: pip install scikit-learn")

        print(f"执行 KMeans 聚类: {len(embeddings)} 个样本, K={self.n_clusters}")

        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state
        )

        labels = self._kmeans.fit_predict(embeddings)
        centroids = self._kmeans.cluster_centers_

        cluster_sizes = {}
        for label in range(self.n_clusters):
            cluster_sizes[label] = int((labels == label).sum())

        return ClusterResult(
            labels=labels,
            n_clusters=self.n_clusters,
            centroids=centroids,
            cluster_sizes=cluster_sizes
        )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """预测新样本的簇标签"""
        if self._kmeans is None:
            raise ValueError("请先调用 fit() 方法")
        return self._kmeans.predict(embeddings)


class MiniBatchKMeansClusterer(Clusterer):
    """Mini-Batch KMeans 聚类器 (适用于大规模数据)"""

    def __init__(
        self,
        n_clusters: int = 100,
        batch_size: int = 1024,
        max_iter: int = 100,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state

        self._kmeans = None

    def fit(self, embeddings: np.ndarray) -> ClusterResult:
        """执行 Mini-Batch KMeans 聚类"""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise ImportError("请安装 scikit-learn: pip install scikit-learn")

        print(f"执行 Mini-Batch KMeans 聚类: {len(embeddings)} 个样本, K={self.n_clusters}")

        self._kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        labels = self._kmeans.fit_predict(embeddings)
        centroids = self._kmeans.cluster_centers_

        cluster_sizes = {}
        for label in range(self.n_clusters):
            cluster_sizes[label] = int((labels == label).sum())

        return ClusterResult(
            labels=labels,
            n_clusters=self.n_clusters,
            centroids=centroids,
            cluster_sizes=cluster_sizes
        )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """预测新样本的簇标签"""
        if self._kmeans is None:
            raise ValueError("请先调用 fit() 方法")
        return self._kmeans.predict(embeddings)
