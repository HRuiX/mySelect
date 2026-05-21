"""
簇级指标计算

性能优化:
- 使用 ThreadPoolExecutor 并行计算各簇指标
"""

from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os

from hcds.config.schema import ClusterPriorWeightsConfig


def _compute_single_cluster_metrics(
    label: int,
    mask: np.ndarray,
    embeddings: np.ndarray,
    centroids: np.ndarray,
    global_centroid: np.ndarray,
    global_norm: float
) -> Dict[str, float]:
    """
    计算单个簇的指标

    Args:
        label: 簇标签
        mask: 该簇的布尔掩码
        embeddings: 全量嵌入矩阵
        centroids: 所有簇中心
        global_centroid: 全局中心
        global_norm: 全局中心的范数

    Returns:
        该簇的指标字典
    """
    cluster_embs = embeddings[mask]

    if label < len(centroids):
        centroid = centroids[label]
    else:
        centroid = cluster_embs.mean(axis=0)

    centroid_norm = np.linalg.norm(centroid)

    # 1. 簇内方差
    variance = float(np.mean(np.linalg.norm(cluster_embs - centroid, axis=1) ** 2))

    # 2. 相对全局偏离 (余弦距离)
    if centroid_norm > 0 and global_norm > 0:
        global_distance = float(1 - np.dot(centroid, global_centroid) / (centroid_norm * global_norm))
    else:
        global_distance = 0.0

    # 3. 簇间隔离度
    if len(centroids) > 1:
        other_centroids = np.delete(centroids, min(label, len(centroids)-1), axis=0)
        other_norms = np.linalg.norm(other_centroids, axis=1)
        valid = other_norms > 0
        if valid.any():
            similarities = np.dot(other_centroids[valid], centroid) / (other_norms[valid] * centroid_norm + 1e-12)
            isolation = float(1 - np.max(similarities))
        else:
            isolation = 1.0
    else:
        isolation = 1.0

    return {
        "variance": variance,
        "global_distance": global_distance,
        "isolation": isolation,
        "size": int(mask.sum())
    }


def compute_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    max_workers: Optional[int] = None
) -> Dict[int, Dict[str, float]]:
    """
    计算簇级指标

    Args:
        embeddings: 嵌入矩阵 [N, D]
        labels: 簇标签 [N]
        centroids: 簇中心 [M, D]
        max_workers: 最大并行工作者数 (默认 CPU 核心数)

    Returns:
        {cluster_id: {variance, global_distance, isolation, size}}
    """
    metrics = {}

    # 全局中心
    global_centroid = embeddings.mean(axis=0)
    global_norm = np.linalg.norm(global_centroid)

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    # 簇数量较少时串行执行
    if len(unique_labels) < 4:
        for label in unique_labels:
            label = int(label)
            mask = labels == label
            metrics[label] = _compute_single_cluster_metrics(
                label, mask, embeddings, centroids, global_centroid, global_norm
            )
        return metrics

    # 使用线程池并行计算
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(unique_labels))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for label in unique_labels:
            label = int(label)
            mask = labels == label
            future = executor.submit(
                _compute_single_cluster_metrics,
                label, mask, embeddings, centroids, global_centroid, global_norm
            )
            futures[future] = label

        for future in as_completed(futures):
            label = futures[future]
            metrics[label] = future.result()

    return metrics


def normalize_metrics(
    metrics: Dict[int, Dict[str, float]],
    keys: List[str] = None
) -> Dict[int, Dict[str, float]]:
    """
    归一化指标到 [0, 1]

    Args:
        metrics: 簇级指标
        keys: 要归一化的键

    Returns:
        归一化后的指标
    """
    if keys is None:
        keys = ["variance", "global_distance", "isolation"]

    result = {}

    # 收集所有值
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
                result[cluster_id][f"{key}_norm"] = (cluster_metrics[key] - min_val) / range_val

    return result


def compute_prior_scores(
    metrics: Dict[int, Dict[str, float]],
    weights: ClusterPriorWeightsConfig
) -> Dict[int, float]:
    """
    计算簇的静态先验分数

    S_prior = λ1 * variance + λ2 * global_distance + λ3 * isolation + λ4 * entropy

    Args:
        metrics: 归一化后的簇级指标
        weights: 权重配置

    Returns:
        {cluster_id: prior_score}
    """
    # 确保指标已归一化
    if not any("_norm" in k for m in metrics.values() for k in m.keys()):
        metrics = normalize_metrics(metrics)

    prior_scores = {}

    for cluster_id, m in metrics.items():
        score = (
            weights.variance * m.get("variance_norm", 0) +
            weights.global_distance * m.get("global_distance_norm", 0) +
            weights.isolation * m.get("isolation_norm", 0) +
            weights.label_entropy * m.get("label_entropy_norm", 0)
        )
        prior_scores[cluster_id] = float(score)

    return prior_scores
