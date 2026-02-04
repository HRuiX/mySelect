"""
多样性度量
"""

from typing import List, Optional
import numpy as np


def compute_diversity(
    embeddings: np.ndarray,
    method: str = "avg_pairwise"
) -> float:
    """
    计算嵌入集合的多样性

    Args:
        embeddings: 嵌入矩阵 [N, D]
        method: 计算方法 (avg_pairwise, coverage, determinantal)

    Returns:
        多样性分数
    """
    if len(embeddings) < 2:
        return 0.0

    if method == "avg_pairwise":
        return _avg_pairwise_distance(embeddings)
    elif method == "coverage":
        return _coverage_score(embeddings)
    elif method == "determinantal":
        return _determinantal_diversity(embeddings)
    else:
        return _avg_pairwise_distance(embeddings)


def _avg_pairwise_distance(embeddings: np.ndarray) -> float:
    """平均成对距离"""
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-12)

    # 计算相似度矩阵
    sim_matrix = embeddings_norm @ embeddings_norm.T

    # 提取上三角 (排除对角线)
    n = len(embeddings)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_similarities = sim_matrix[upper_indices]

    # 转为距离
    pairwise_distances = 1 - pairwise_similarities

    return float(pairwise_distances.mean())


def _coverage_score(embeddings: np.ndarray, n_bins: int = 10) -> float:
    """
    覆盖度分数 (基于 PCA 空间的直方图)
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return _avg_pairwise_distance(embeddings)

    # PCA 降到 2D
    n_components = min(2, embeddings.shape[1], len(embeddings))
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)

    # 计算每个维度的覆盖
    coverage = 0.0
    for dim in range(reduced.shape[1]):
        values = reduced[:, dim]
        hist, _ = np.histogram(values, bins=n_bins)
        # 非空 bin 的比例
        coverage += (hist > 0).sum() / n_bins

    return coverage / reduced.shape[1]


def _determinantal_diversity(embeddings: np.ndarray) -> float:
    """
    行列式点过程 (DPP) 多样性
    基于核矩阵的行列式
    """
    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-12)

    # 核矩阵 (相似度矩阵)
    K = embeddings_norm @ embeddings_norm.T

    # 计算 log-det (数值稳定版本)
    try:
        sign, logdet = np.linalg.slogdet(K + 1e-6 * np.eye(len(K)))
        if sign > 0:
            # 归一化
            return logdet / len(embeddings)
    except:
        pass

    return 0.0


def compute_cluster_coverage(
    selected_cluster_ids: List[int],
    total_clusters: int
) -> float:
    """
    计算簇覆盖率

    Args:
        selected_cluster_ids: 选中的簇 ID
        total_clusters: 总簇数

    Returns:
        覆盖率 [0, 1]
    """
    unique_selected = len(set(selected_cluster_ids))
    return unique_selected / total_clusters if total_clusters > 0 else 0.0


def compute_selection_balance(
    cluster_selections: dict
) -> float:
    """
    计算选择的均衡性 (基于熵)

    Args:
        cluster_selections: {cluster_id: n_selected}

    Returns:
        归一化熵 [0, 1]
    """
    counts = list(cluster_selections.values())
    if not counts or sum(counts) == 0:
        return 0.0

    total = sum(counts)
    probs = np.array(counts) / total

    # 计算熵
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    # 归一化 (最大熵 = log(n))
    max_entropy = np.log(len(counts))
    if max_entropy > 0:
        return entropy / max_entropy

    return 0.0
