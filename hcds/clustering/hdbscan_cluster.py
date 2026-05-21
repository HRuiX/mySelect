"""
HDBSCAN 聚类器实现
支持 GPU 加速 (cuML) 和 CPU 回退
"""

from typing import Optional, Literal
import numpy as np
from tqdm import tqdm

from hcds.clustering.base import Clusterer, ClusterResult
from hcds.config.schema import ClusteringConfig

# 检测 cuML GPU 支持
_CUML_AVAILABLE = False
try:
    import cuml
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    import cupy as cp
    _CUML_AVAILABLE = True
except ImportError:
    pass


def _check_cuda_available() -> bool:
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except:
        pass
    return False


class HDBSCANClusterer(Clusterer):
    """HDBSCAN 聚类器 (支持 GPU 加速)"""

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        metric: str = "euclidean",
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        noise_strategy: Literal["nearest", "separate", "drop"] = "nearest",
        use_gpu: bool = True
    ):
        """
        初始化 HDBSCAN 聚类器

        Args:
            min_cluster_size: 最小簇大小
            min_samples: 核心点所需的最小邻居数
            metric: 距离度量
            cluster_selection_method: 簇选择方法
            noise_strategy: 噪声处理策略
            use_gpu: 是否使用 GPU 加速 (需要 cuML)
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.noise_strategy = noise_strategy
        self.use_gpu = use_gpu and _CUML_AVAILABLE and _check_cuda_available()

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
            noise_strategy=config.noise_handling.strategy.value,
            use_gpu=True  # 默认尝试使用 GPU
        )

    def fit(self, embeddings: np.ndarray) -> ClusterResult:
        """
        执行 HDBSCAN 聚类

        Args:
            embeddings: 嵌入矩阵 [N, D]

        Returns:
            ClusterResult 实例
        """
        n_samples = len(embeddings)
        print(f"执行 HDBSCAN 聚类: {n_samples:,} 个样本")

        if self.use_gpu:
            labels = self._fit_gpu(embeddings)
        else:
            labels = self._fit_cpu(embeddings)

        # 处理噪声点
        noise_mask = labels == -1
        n_noise = noise_mask.sum()

        if n_noise > 0:
            print(f"检测到 {n_noise:,} 个噪声点 ({100*n_noise/len(labels):.2f}%)")
            print("正在处理噪声点...")
            labels = self._handle_noise(embeddings, labels)

        # 计算簇中心
        print("正在计算簇中心...")
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

    def _fit_gpu(self, embeddings: np.ndarray) -> np.ndarray:
        """使用 cuML GPU 加速的 HDBSCAN"""
        import cupy as cp
        from cuml.cluster import HDBSCAN as cuHDBSCAN

        print("使用 cuML GPU 加速 HDBSCAN")

        # 显示 GPU 信息
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_properties(0).name
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU: {gpu_name}, 显存: {gpu_mem:.1f} GB")
        except:
            pass

        embeddings_gpu = None
        try:
            # 将数据转移到 GPU
            print("正在将数据转移到 GPU...")
            embeddings_gpu = cp.asarray(embeddings.astype(np.float32))

            # cuML HDBSCAN 配置
            # 注意: cuML 的 metric 支持有限,主要支持 euclidean 和 l2
            metric = "euclidean" if self.metric in ["euclidean", "l2", "cosine"] else "euclidean"

            self._clusterer = cuHDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=metric,
                cluster_selection_method=self.cluster_selection_method,
                prediction_data=True,
                verbose=True
            )

            print("正在执行 GPU HDBSCAN 聚类...")
            labels = self._clusterer.fit_predict(embeddings_gpu)

            # 将结果转回 CPU
            labels = cp.asnumpy(labels)

            return labels

        except KeyboardInterrupt:
            print("\n聚类被中断，正在清理 GPU 资源...")
            raise

        finally:
            # 清理 GPU 内存
            if embeddings_gpu is not None:
                del embeddings_gpu
            cp.get_default_memory_pool().free_all_blocks()
            print("GPU 内存已清理")

    def _fit_cpu(self, embeddings: np.ndarray) -> np.ndarray:
        """使用 CPU 版本的 HDBSCAN"""
        try:
            import hdbscan
        except ImportError:
            raise ImportError("请安装 hdbscan: pip install hdbscan")

        n_samples = len(embeddings)
        print("使用 CPU 版本 HDBSCAN")

        # 警告大数据集
        if n_samples > 50000:
            estimated_hours = (n_samples / 50000) ** 2 * 0.5  # 粗略估计
            print(f"⚠️ 警告: {n_samples:,} 个样本使用 CPU HDBSCAN 可能需要 {estimated_hours:.1f}+ 小时")
            print("建议: 1) 安装 cuML GPU 加速: conda install -c rapidsai cuml")
            print("      2) 或降低 large_scale.threshold 使用 Pilot+FAISS 模式")

        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,
            core_dist_n_jobs=-1  # 使用所有 CPU 核心
        )

        print("正在执行 CPU HDBSCAN 聚类...")
        labels = self._clusterer.fit_predict(embeddings)

        return labels

    def _handle_noise(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """处理噪声点 (向量化实现, 带进度条)"""
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
            # 向量化分配到最近簇
            valid_labels = labels[~noise_mask]
            valid_embeddings = embeddings[~noise_mask]

            # 计算每个簇的中心
            unique_labels = np.unique(valid_labels)
            print(f"计算 {len(unique_labels)} 个簇的中心...")

            centroids = np.array([
                valid_embeddings[valid_labels == l].mean(axis=0)
                for l in tqdm(unique_labels, desc="计算簇中心")
            ])

            # 向量化计算: 所有噪声点到所有中心的距离
            noise_embeddings = embeddings[noise_mask]
            noise_indices = np.where(noise_mask)[0]

            if len(noise_embeddings) > 0 and len(centroids) > 0:
                print(f"为 {len(noise_embeddings):,} 个噪声点分配簇...")

                # 分批处理以节省内存
                batch_size = 10000
                n_batches = (len(noise_embeddings) + batch_size - 1) // batch_size

                for batch_idx in tqdm(range(n_batches), desc="分配噪声点"):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(noise_embeddings))

                    batch_embeddings = noise_embeddings[start_idx:end_idx]
                    batch_noise_indices = noise_indices[start_idx:end_idx]

                    # 使用广播计算距离矩阵 [batch_size, N_centroids]
                    all_distances = np.linalg.norm(
                        batch_embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :],
                        axis=2
                    )
                    # 找到每个噪声点的最近簇
                    nearest_indices = np.argmin(all_distances, axis=1)
                    nearest_labels = unique_labels[nearest_indices]

                    # 批量赋值
                    labels[batch_noise_indices] = nearest_labels

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
