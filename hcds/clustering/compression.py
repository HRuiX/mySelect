"""
簇内压缩：代表集与密度参照集
FPS (Farthest Point Sampling) 实现

性能优化:
- Numba JIT 加速 FPS 循环 (5-10x 加速)
- FAISS 加速 kNN 搜索 (5-100x 加速)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from hcds.config.schema import CompressionConfig

# ==================== Numba JIT 加速 ====================
# 尝试导入 Numba，不可用时使用原 Python 实现
_NUMBA_AVAILABLE = False
try:
    import numba
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    pass


if _NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=True)
    def _fps_cosine_numba(embeddings: np.ndarray, max_samples: int) -> np.ndarray:
        """
        Numba 加速的 FPS (余弦距离版本)

        Args:
            embeddings: L2 归一化的嵌入 [N, D]
            max_samples: 最大代表数

        Returns:
            选中的索引
        """
        n = embeddings.shape[0]

        if n <= max_samples:
            return np.arange(n)

        selected = np.empty(max_samples, dtype=np.int64)
        selected[0] = 0
        distances = np.full(n, np.inf, dtype=np.float64)

        for i in range(1, max_samples):
            last_selected = selected[i - 1]
            last_emb = embeddings[last_selected]

            # 并行计算余弦距离
            for j in prange(n):
                # 内积 (对于归一化向量，1 - 内积 = 余弦距离)
                dot_product = 0.0
                for k in range(embeddings.shape[1]):
                    dot_product += embeddings[j, k] * last_emb[k]
                new_dist = 1.0 - dot_product

                if new_dist < distances[j]:
                    distances[j] = new_dist

            # 将已选点距离设为负数
            for s in range(i):
                distances[selected[s]] = -1.0

            # 选择最远点
            max_dist = -2.0
            max_idx = 0
            for j in range(n):
                if distances[j] > max_dist:
                    max_dist = distances[j]
                    max_idx = j

            selected[i] = max_idx

        return selected


# ==================== FAISS 加速 ====================
_FAISS_AVAILABLE = False
_FAISS_GPU_AVAILABLE = False
try:
    import faiss
    _FAISS_AVAILABLE = True
    # 检查 GPU 支持
    if faiss.get_num_gpus() > 0:
        _FAISS_GPU_AVAILABLE = True
        print(f"FAISS GPU 加速可用: {faiss.get_num_gpus()} 个 GPU")
except ImportError:
    pass


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
    max_samples: int,
    use_numba: bool = True
) -> np.ndarray:
    """
    FPS (余弦距离版本)

    Args:
        embeddings: L2 归一化的嵌入 [N, D]
        max_samples: 最大代表数
        use_numba: 是否使用 Numba 加速 (默认 True)

    Returns:
        选中的索引
    """
    n = len(embeddings)

    if n <= max_samples:
        return np.arange(n)

    # 优先使用 Numba 加速版本
    if use_numba and _NUMBA_AVAILABLE:
        # 确保输入是连续数组
        embeddings_contiguous = np.ascontiguousarray(embeddings, dtype=np.float64)
        return _fps_cosine_numba(embeddings_contiguous, max_samples)

    # 原 Python 实现
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
    k: int = 10,
    use_faiss: bool = True
) -> np.ndarray:
    """
    计算局部密度 (用于稀有度计算)

    Args:
        embeddings: 要计算密度的嵌入 [N, D]
        reference_embeddings: 参照集嵌入 [M, D]
        k: kNN 的 k 值
        use_faiss: 是否使用 FAISS 加速 (默认 True)

    Returns:
        局部密度分数 [N] (越大越稀疏)
    """
    k = min(k, len(reference_embeddings) - 1)
    if k < 1:
        return np.zeros(len(embeddings))

    # 优先使用 FAISS
    if use_faiss and _FAISS_AVAILABLE:
        return _compute_local_density_faiss(embeddings, reference_embeddings, k)

    # 回退到 sklearn
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        raise ImportError("请安装 scikit-learn: pip install scikit-learn")

    # 使用余弦距离
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='auto')
    nn.fit(reference_embeddings)

    distances, _ = nn.kneighbors(embeddings)

    # 平均 kNN 距离作为稀疏度
    local_density = distances.mean(axis=1)

    return local_density


def _compute_local_density_faiss(
    embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    k: int
) -> np.ndarray:
    """
    使用 FAISS 计算局部密度

    对于余弦相似度，使用归一化向量的内积 (IndexFlatIP)
    """
    import faiss

    dim = embeddings.shape[1]

    # 归一化向量 (内积 = 余弦相似度)
    embeddings_norm = embeddings.astype(np.float32)
    reference_norm = reference_embeddings.astype(np.float32)

    # L2 归一化
    faiss.normalize_L2(embeddings_norm)
    faiss.normalize_L2(reference_norm)

    # 创建内积索引
    index = faiss.IndexFlatIP(dim)

    # 如果有 GPU 且数据量较大，使用 GPU 加速
    if _FAISS_GPU_AVAILABLE and len(reference_embeddings) > 10000:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(reference_norm)

    # 搜索 k 近邻 (返回相似度)
    similarities, _ = index.search(embeddings_norm, k)

    # 转换为距离: distance = 1 - similarity
    distances = 1.0 - similarities

    # 平均 kNN 距离作为稀疏度
    local_density = distances.mean(axis=1)

    return local_density
