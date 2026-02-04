"""
簇内压缩：代表集与密度参照集
FPS (Farthest Point Sampling) 实现
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from hcds.config.schema import CompressionConfig


def fps_single_cluster(
    embeddings: np.ndarray,
    max_samples: int
) -> np.ndarray:
    """
    对单个簇执行 FPS 压缩

    Args:
        embeddings: 簇内嵌入 [N, D]
        max_samples: 最大代表数

    Returns:
        选中的索引
    """
    n = len(embeddings)

    if n <= max_samples:
        return np.arange(n)

    # Farthest Point Sampling
    selected = [0]  # 从第一个点开始
    distances = np.full(n, np.inf)

    for _ in range(max_samples - 1):
        last_selected = selected[-1]

        # 计算到最近已选点的距离
        new_distances = np.linalg.norm(
            embeddings - embeddings[last_selected], axis=1
        )
        distances = np.minimum(distances, new_distances)

        # 选择最远点
        distances[selected] = -1  # 排除已选点
        next_selected = np.argmax(distances)
        selected.append(next_selected)

    return np.array(selected)


def fps_single_cluster_cosine(
    embeddings: np.ndarray,
    max_samples: int
) -> np.ndarray:
    """
    FPS (余弦距离版本)

    Args:
        embeddings: L2 归一化的嵌入 [N, D]
        max_samples: 最大代表数

    Returns:
        选中的索引
    """
    n = len(embeddings)

    if n <= max_samples:
        return np.arange(n)

    # 使用余弦距离 = 1 - 内积 (对于归一化向量)
    selected = [0]
    distances = np.full(n, np.inf)

    for _ in range(max_samples - 1):
        last_selected = selected[-1]

        # 余弦距离
        similarities = embeddings @ embeddings[last_selected]
        new_distances = 1 - similarities

        distances = np.minimum(distances, new_distances)
        distances[selected] = -1

        next_selected = np.argmax(distances)
        selected.append(next_selected)

    return np.array(selected)


class FPSCompressor:
    """FPS 压缩器"""

    def __init__(
        self,
        max_samples: int = 2048,
        use_cosine: bool = True
    ):
        """
        初始化 FPS 压缩器

        Args:
            max_samples: 每簇最大代表数
            use_cosine: 是否使用余弦距离
        """
        self.max_samples = max_samples
        self.use_cosine = use_cosine

    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        """
        压缩单个簇

        Args:
            embeddings: 簇内嵌入 [N, D]

        Returns:
            选中的索引
        """
        if self.use_cosine:
            return fps_single_cluster_cosine(embeddings, self.max_samples)
        else:
            return fps_single_cluster(embeddings, self.max_samples)


class ClusterCompressor:
    """多簇并行压缩器"""

    def __init__(
        self,
        config: CompressionConfig,
        max_workers: Optional[int] = None
    ):
        """
        初始化压缩器

        Args:
            config: 压缩配置
            max_workers: 最大并行工作者数
        """
        self.config = config
        self.max_workers = max_workers

    def compress_all(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        show_progress: bool = True
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        压缩所有簇

        Args:
            embeddings: 全量嵌入 [N, D]
            labels: 簇标签 [N]
            show_progress: 是否显示进度条

        Returns:
            {cluster_id: {"representatives": indices, "references": indices}}
        """
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]

        results = {}

        if self.max_workers == 1:
            # 单进程
            iterator = unique_labels
            if show_progress:
                iterator = tqdm(iterator, desc="压缩簇")

            for label in iterator:
                cluster_mask = labels == label
                cluster_embeddings = embeddings[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]

                rep_local_indices, ref_local_indices = self._compress_cluster(
                    cluster_embeddings
                )

                results[int(label)] = {
                    "representatives": cluster_indices[rep_local_indices],
                    "references": cluster_indices[ref_local_indices]
                }
        else:
            # 多进程
            results = self._compress_parallel(
                embeddings, labels, unique_labels, show_progress
            )

        return results

    def _compress_cluster(
        self,
        cluster_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        压缩单个簇

        Returns:
            (代表集局部索引, 参照集局部索引)
        """
        n = len(cluster_embeddings)

        # 代表集
        if self.config.algorithm == "fps":
            rep_indices = fps_single_cluster_cosine(
                cluster_embeddings,
                self.config.max_representatives
            )
        elif self.config.algorithm == "kmeans":
            rep_indices = self._kmeans_select(
                cluster_embeddings,
                self.config.max_representatives
            )
        else:  # random
            rep_indices = np.random.choice(
                n,
                min(n, self.config.max_representatives),
                replace=False
            )

        # 参照集 (随机采样)
        ref_size = min(n, self.config.reference_set_size)
        ref_indices = np.random.choice(n, ref_size, replace=False)

        return rep_indices, ref_indices

    def _kmeans_select(
        self,
        embeddings: np.ndarray,
        n_select: int
    ) -> np.ndarray:
        """使用 KMeans 选择代表点"""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            # 回退到随机
            return np.random.choice(len(embeddings), n_select, replace=False)

        if len(embeddings) <= n_select:
            return np.arange(len(embeddings))

        kmeans = MiniBatchKMeans(n_clusters=n_select, n_init=1, random_state=42)
        kmeans.fit(embeddings)

        # 选择每个簇中心最近的点
        selected = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(embeddings - center, axis=1)
            nearest = np.argmin(distances)
            selected.append(nearest)

        return np.array(selected)

    def _compress_parallel(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        unique_labels: np.ndarray,
        show_progress: bool
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """多进程压缩"""
        # 准备任务数据
        tasks = []
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            tasks.append((int(label), cluster_embeddings, cluster_indices))

        results = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    _compress_task,
                    task[1],  # embeddings
                    self.config.max_representatives,
                    self.config.reference_set_size,
                    self.config.algorithm
                ): (task[0], task[2])  # (label, indices)
                for task in tasks
            }

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="压缩簇")

            for future in iterator:
                label, cluster_indices = futures[future]
                rep_local, ref_local = future.result()

                results[label] = {
                    "representatives": cluster_indices[rep_local],
                    "references": cluster_indices[ref_local]
                }

        return results


def _compress_task(
    embeddings: np.ndarray,
    max_rep: int,
    ref_size: int,
    algorithm: str
) -> Tuple[np.ndarray, np.ndarray]:
    """并行任务函数"""
    n = len(embeddings)

    # 代表集
    if algorithm == "fps":
        rep_indices = fps_single_cluster_cosine(embeddings, max_rep)
    else:
        rep_indices = np.random.choice(n, min(n, max_rep), replace=False)

    # 参照集
    ref_indices = np.random.choice(n, min(n, ref_size), replace=False)

    return rep_indices, ref_indices


def compute_local_density(
    embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """
    计算局部密度 (用于稀有度计算)

    Args:
        embeddings: 要计算密度的嵌入 [N, D]
        reference_embeddings: 参照集嵌入 [M, D]
        k: kNN 的 k 值

    Returns:
        局部密度分数 [N] (越大越稀疏)
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        raise ImportError("请安装 scikit-learn: pip install scikit-learn")

    k = min(k, len(reference_embeddings) - 1)
    if k < 1:
        return np.zeros(len(embeddings))

    # 使用余弦距离
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='auto')
    nn.fit(reference_embeddings)

    distances, _ = nn.kneighbors(embeddings)

    # 平均 kNN 距离作为稀疏度
    local_density = distances.mean(axis=1)

    return local_density
