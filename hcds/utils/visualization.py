"""
可视化工具
"""

from typing import Dict, List, Optional
import numpy as np


def plot_cluster_distribution(
    cluster_sizes: Dict[int, int],
    selected_clusters: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    绘制簇大小分布

    Args:
        cluster_sizes: {cluster_id: size}
        selected_clusters: 选中的簇 ID
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("请安装 matplotlib: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    cluster_ids = sorted(cluster_sizes.keys())
    sizes = [cluster_sizes[cid] for cid in cluster_ids]

    colors = ['steelblue'] * len(cluster_ids)
    if selected_clusters:
        for i, cid in enumerate(cluster_ids):
            if cid in selected_clusters:
                colors[i] = 'coral'

    ax.bar(range(len(cluster_ids)), sizes, color=colors)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Size')
    ax.set_title('Cluster Size Distribution')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_selection_history(
    history: List[Dict],
    save_path: Optional[str] = None
) -> None:
    """
    绘制选择历史

    Args:
        history: 选择历史列表
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("请安装 matplotlib: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rounds = range(1, len(history) + 1)

    # 1. 平均错误强度
    avg_errors = [h.get("avg_error", 0) for h in history]
    axes[0, 0].plot(rounds, avg_errors, 'b-o')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Average Error Intensity')
    axes[0, 0].set_title('Error Intensity Over Rounds')

    # 2. 簇覆盖率
    coverages = [h.get("cluster_coverage", 0) for h in history]
    axes[0, 1].plot(rounds, coverages, 'g-o')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Cluster Coverage')
    axes[0, 1].set_title('Cluster Coverage Over Rounds')

    # 3. 退休样本数
    retired = [h.get("n_retired", 0) for h in history]
    axes[1, 0].bar(rounds, retired, color='orange')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Newly Retired Samples')
    axes[1, 0].set_title('Sample Retirement Over Rounds')

    # 4. 多样性分数
    diversity = [h.get("diversity", 0) for h in history]
    axes[1, 1].plot(rounds, diversity, 'r-o')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Diversity Score')
    axes[1, 1].set_title('Selection Diversity Over Rounds')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    selected_indices: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    max_points: int = 5000
) -> None:
    """
    绘制嵌入空间 (t-SNE 降维)

    Args:
        embeddings: 嵌入矩阵 [N, D]
        labels: 簇标签 [N]
        selected_indices: 选中样本的索引
        save_path: 保存路径
        max_points: 最大绘制点数
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("请安装 matplotlib 和 scikit-learn")
        return

    # 采样
    if len(embeddings) > max_points:
        indices = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        if selected_indices is not None:
            selected_indices = np.intersect1d(selected_indices, indices)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)

    # 绘制
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        reduced[:, 0], reduced[:, 1],
        c=labels, cmap='tab20', alpha=0.5, s=10
    )

    if selected_indices is not None and len(selected_indices) > 0:
        ax.scatter(
            reduced[selected_indices, 0],
            reduced[selected_indices, 1],
            c='red', marker='x', s=30, label='Selected'
        )
        ax.legend()

    plt.colorbar(scatter, label='Cluster ID')
    ax.set_title('Embedding Space Visualization (t-SNE)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
