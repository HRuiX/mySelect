"""
聚类器抽象基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ClusterResult:
    """聚类结果"""
    labels: np.ndarray  # 每个样本的簇标签 [N]
    n_clusters: int     # 簇数量
    centroids: Optional[np.ndarray] = None  # 簇中心 [M, D]
    noise_mask: Optional[np.ndarray] = None  # 噪声标记 [N]

    # 统计信息
    cluster_sizes: Optional[Dict[int, int]] = None
    silhouette_score: Optional[float] = None


class Clusterer(ABC):
    """聚类器抽象基类"""

    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> ClusterResult:
        """
        执行聚类

        Args:
            embeddings: 嵌入矩阵 [N, D]

        Returns:
            ClusterResult 实例
        """
        pass

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        预测新样本的簇标签

        Args:
            embeddings: 嵌入矩阵 [N, D]

        Returns:
            簇标签 [N]
        """
        pass

    @staticmethod
    def compute_centroids(
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        计算簇中心

        Args:
            embeddings: 嵌入矩阵 [N, D]
            labels: 簇标签 [N]

        Returns:
            (簇中心矩阵, 簇大小字典)
        """
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]  # 排除噪声 (-1)

        centroids = []
        cluster_sizes = {}

        for label in sorted(unique_labels):
            mask = labels == label
            cluster_embeddings = embeddings[mask]
            centroid = cluster_embeddings.mean(axis=0)
            centroids.append(centroid)
            cluster_sizes[int(label)] = int(mask.sum())

        return np.array(centroids), cluster_sizes

    @staticmethod
    def compute_cluster_metrics(
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        计算簇级指标

        Args:
            embeddings: 嵌入矩阵 [N, D]
            labels: 簇标签 [N]
            centroids: 簇中心 [M, D]

        Returns:
            {cluster_id: {metric_name: value}}
        """
        metrics = {}

        # 全局中心
        global_centroid = embeddings.mean(axis=0)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]

        for label in unique_labels:
            label = int(label)
            mask = labels == label
            cluster_embs = embeddings[mask]
            centroid = centroids[label]

            # 簇内方差
            variance = np.mean(np.linalg.norm(cluster_embs - centroid, axis=1) ** 2)

            # 相对全局偏离 (余弦距离)
            global_dist = 1 - np.dot(centroid, global_centroid) / (
                np.linalg.norm(centroid) * np.linalg.norm(global_centroid) + 1e-12
            )

            # 簇间隔离度 (到最近其他簇中心的距离)
            if len(centroids) > 1:
                other_centroids = np.delete(centroids, label, axis=0)
                distances = 1 - np.dot(other_centroids, centroid) / (
                    np.linalg.norm(other_centroids, axis=1) * np.linalg.norm(centroid) + 1e-12
                )
                isolation = float(np.min(distances))
            else:
                isolation = 1.0

            metrics[label] = {
                "variance": float(variance),
                "global_distance": float(global_dist),
                "isolation": float(isolation),
                "size": int(mask.sum())
            }

        return metrics

    @staticmethod
    def normalize_metrics(
        metrics: Dict[int, Dict[str, float]],
        keys: List[str] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        归一化指标到 [0, 1]

        Args:
            metrics: 簇级指标字典
            keys: 要归一化的指标键

        Returns:
            归一化后的指标
        """
        if keys is None:
            keys = ["variance", "global_distance", "isolation"]

        result = {}

        # 收集所有值用于归一化
        all_values = {key: [] for key in keys}
        for cluster_metrics in metrics.values():
            for key in keys:
                if key in cluster_metrics:
                    all_values[key].append(cluster_metrics[key])

        # 计算 min/max
        ranges = {}
        for key in keys:
            values = all_values[key]
            if values:
                min_val, max_val = min(values), max(values)
                ranges[key] = (min_val, max_val - min_val + 1e-12)
            else:
                ranges[key] = (0, 1)

        # 归一化
        for cluster_id, cluster_metrics in metrics.items():
            result[cluster_id] = dict(cluster_metrics)
            for key in keys:
                if key in cluster_metrics:
                    min_val, range_val = ranges[key]
                    result[cluster_id][f"{key}_normalized"] = (
                        cluster_metrics[key] - min_val
                    ) / range_val

        return result
