"""
样本级指标计算
"""

from typing import Dict, List, Optional
import numpy as np


def compute_rarity_scores(
    embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """
    计算稀有度分数

    Args:
        embeddings: 代表集嵌入 [N, D]
        reference_embeddings: 参照集嵌入 [M, D]
        k: kNN 的 k 值

    Returns:
        稀有度分数 [N] ∈ [0, 1]
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return _compute_rarity_simple(embeddings, reference_embeddings, k)

    k = min(k, len(reference_embeddings) - 1)
    if k < 1:
        return np.zeros(len(embeddings))

    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='auto')
    nn.fit(reference_embeddings)

    distances, _ = nn.kneighbors(embeddings)
    local_density = distances.mean(axis=1)

    # 归一化到 [0, 1]
    if local_density.max() > local_density.min():
        rarity = (local_density - local_density.min()) / (local_density.max() - local_density.min() + 1e-12)
    else:
        rarity = np.zeros_like(local_density)

    return rarity


def _compute_rarity_simple(
    embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int
) -> np.ndarray:
    """简化版稀有度计算"""
    densities = []
    k = min(k, len(reference_embeddings))

    for emb in embeddings:
        # 计算余弦相似度
        similarities = reference_embeddings @ emb
        similarities = similarities / (np.linalg.norm(reference_embeddings, axis=1) * np.linalg.norm(emb) + 1e-12)

        # 转为距离
        distances = 1 - similarities

        # 取 top-k 平均
        top_k_distances = np.sort(distances)[:k]
        densities.append(top_k_distances.mean())

    densities = np.array(densities)

    # 归一化
    if densities.max() > densities.min():
        return (densities - densities.min()) / (densities.max() - densities.min() + 1e-12)
    return np.zeros_like(densities)


def compute_novelty_scores(
    embeddings: np.ndarray,
    history_embeddings: Optional[np.ndarray]
) -> np.ndarray:
    """
    计算新颖度分数

    Args:
        embeddings: 候选嵌入 [N, D]
        history_embeddings: 历史已选嵌入 [H, D]

    Returns:
        新颖度分数 [N] ∈ [0, 1]
    """
    if history_embeddings is None or len(history_embeddings) == 0:
        return np.ones(len(embeddings))

    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='auto')
        nn.fit(history_embeddings)
        distances, _ = nn.kneighbors(embeddings)
        novelty = distances.flatten()

    except ImportError:
        # 简化版
        novelty = []
        for emb in embeddings:
            similarities = history_embeddings @ emb
            similarities = similarities / (np.linalg.norm(history_embeddings, axis=1) * np.linalg.norm(emb) + 1e-12)
            min_dist = 1 - similarities.max()
            novelty.append(min_dist)
        novelty = np.array(novelty)

    # 归一化
    if novelty.max() > novelty.min():
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-12)
    else:
        novelty = np.ones_like(novelty)

    return novelty
