"""
HCDS 核心 Pipeline
整合离线预处理与在线采样
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import json

from hcds.config import HCDSConfig, load_config
from hcds.data import DataLoader, Sample, EmbeddingStorage, ClusterStorage, ClusterInfo
from hcds.embedding import EncoderFactory, IncrementalEmbeddingComputer, PCAReducer
from hcds.clustering import create_clusterer, ClusterCompressor
from hcds.sampling import ThompsonSampler, BudgetAllocator, PrioritySampler, RetirementManager
from hcds.feedback import ErrorIntensityComputer, PosteriorUpdater
from hcds.metrics import compute_cluster_metrics, compute_prior_scores, compute_diversity
from hcds.parallel import ParallelDetector
from hcds.utils import setup_logger, get_logger, CheckpointManager


class HCDSPipeline:
    """HCDS 主 Pipeline"""

    def __init__(
        self,
        config: HCDSConfig,
        resume_from: Optional[str] = None
    ):
        """
        初始化 Pipeline

        Args:
            config: HCDS 配置
            resume_from: 检查点路径 (用于恢复)
        """
        self.config = config

        # 设置日志
        self.logger = setup_logger(
            name="hcds",
            level=config.logging.level,
            log_dir=config.logging.dir,
            log_to_file=config.logging.to_file,
            log_to_console=config.logging.to_console
        )

        # 设置随机种子
        np.random.seed(config.experiment.seed)

        # 初始化组件
        self._init_components()

        # 状态
        self._current_round = 0
        self._selection_history = []

        # 恢复检查点
        if resume_from:
            self._load_checkpoint(resume_from)

    def _init_components(self):
        """初始化各组件"""
        # 检测硬件
        self.parallel_config = ParallelDetector().detect()
        self.logger.info(f"检测到 {self.parallel_config.gpu_count} 个 GPU")

        # 数据加载器
        self.data_loader = DataLoader(self.config.data)

        # 嵌入存储
        self.embedding_storage = EmbeddingStorage(
            path=self.config.embedding.storage.path,
            format=self.config.embedding.storage.format
        )

        # 聚类存储
        self.cluster_storage = ClusterStorage(
            path=Path(self.config.experiment.output_dir) / "clusters"
        )

        # 检查点管理
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(self.config.experiment.output_dir) / "checkpoints"
        )

        # 延迟初始化的组件
        self._encoder = None
        self._thompson_sampler = None
        self._budget_allocator = None
        self._priority_sampler = None
        self._retirement_manager = None
        self._error_computer = None
        self._posterior_updater = None

    # ==================== 离线阶段 ====================

    def run_offline(self) -> Dict[str, Any]:
        """
        执行离线预处理

        Returns:
            离线统计信息
        """
        self.logger.info("=" * 50)
        self.logger.info("开始离线预处理")
        self.logger.info("=" * 50)

        # Step 1: 加载数据
        self.logger.info("Step 1: 加载数据")
        samples = self.data_loader.load()
        sample_ids = list(samples.keys())
        n_samples = len(sample_ids)
        self.logger.info(f"加载 {n_samples:,} 个样本")

        # Step 2: 计算嵌入
        self.logger.info("Step 2: 计算嵌入")
        embeddings = self._compute_embeddings(samples, sample_ids)
        self.logger.info(f"嵌入维度: {embeddings.shape}")

        # Step 3: PCA 降维 (可选)
        if self.config.embedding.pca.enabled:
            self.logger.info("Step 3: PCA 降维")
            embeddings = self._apply_pca(embeddings)
            self.logger.info(f"降维后维度: {embeddings.shape}")

        # Step 4: 聚类
        self.logger.info("Step 4: 聚类")
        labels, centroids = self._cluster(embeddings)
        n_clusters = len(np.unique(labels[labels >= 0]))
        self.logger.info(f"得到 {n_clusters} 个簇")

        # Step 5: 计算簇指标
        self.logger.info("Step 5: 计算簇级指标")
        cluster_metrics = compute_cluster_metrics(embeddings, labels, centroids)
        prior_scores = compute_prior_scores(cluster_metrics, self.config.cluster_prior_weights)

        # Step 6: 簇内压缩
        self.logger.info("Step 6: 簇内压缩")
        compression_result = self._compress_clusters(embeddings, labels)

        # Step 7: 保存结果
        self.logger.info("Step 7: 保存离线结果")
        self._save_offline_results(
            samples, sample_ids, embeddings, labels, centroids,
            cluster_metrics, prior_scores, compression_result
        )

        stats = {
            "n_samples": n_samples,
            "n_clusters": n_clusters,
            "embedding_dim": embeddings.shape[1],
            "cluster_sizes": {int(k): v["size"] for k, v in cluster_metrics.items()},
            "prior_scores": prior_scores
        }

        self.logger.info("离线预处理完成")
        self.logger.info(f"统计: {json.dumps(stats, indent=2)}")

        return stats

    def _compute_embeddings(
        self,
        samples: Dict[str, Sample],
        sample_ids: List[str]
    ) -> np.ndarray:
        """计算嵌入"""
        # 检查是否已存在
        if self.embedding_storage.exists():
            self.logger.info("发现已有嵌入，尝试加载")
            embeddings, stored_ids, _ = self.embedding_storage.load()

            # 检查是否匹配
            if set(stored_ids) == set(sample_ids):
                self.logger.info("嵌入已存在且匹配，跳过计算")
                # 按 sample_ids 顺序重排
                id_to_idx = {sid: idx for idx, sid in enumerate(stored_ids)}
                indices = [id_to_idx[sid] for sid in sample_ids]
                return embeddings[indices]

        # 创建编码器
        if self._encoder is None:
            self._encoder = EncoderFactory.create(self.config.embedding)

        # 获取文本
        texts = [samples[sid].get_text(self.config.data.content_template) for sid in sample_ids]

        # 增量计算
        computer = IncrementalEmbeddingComputer(
            config=self.config.embedding,
            encoder=self._encoder,
            storage=self.embedding_storage
        )

        embeddings = computer.compute(
            texts, sample_ids,
            batch_size=self.parallel_config.embedding_batch_size,
            show_progress=True
        )

        return embeddings

    def _apply_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """应用 PCA 降维"""
        pca_config = self.config.embedding.pca
        reducer = PCAReducer(
            target_dim=pca_config.target_dim,
            variance_ratio=pca_config.variance_ratio
        )
        return reducer.fit_transform(embeddings)

    def _cluster(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """执行聚类"""
        from hcds.clustering.large_scale import create_clusterer

        clusterer = create_clusterer(self.config.clustering, len(embeddings))
        result = clusterer.fit(embeddings)

        return result.labels, result.centroids

    def _compress_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """簇内压缩"""
        compressor = ClusterCompressor(
            config=self.config.compression,
            max_workers=self.parallel_config.compression_max_workers
        )

        return compressor.compress_all(embeddings, labels, show_progress=True)

    def _save_offline_results(
        self,
        samples: Dict[str, Sample],
        sample_ids: List[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        cluster_metrics: Dict,
        prior_scores: Dict[int, float],
        compression_result: Dict
    ):
        """保存离线结果"""
        # 更新样本信息
        for i, sid in enumerate(sample_ids):
            samples[sid].cluster_id = int(labels[i])

        # 构建簇信息
        clusters = {}
        for cluster_id in np.unique(labels):
            if cluster_id < 0:
                continue

            cluster_id = int(cluster_id)
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_sample_ids = [sample_ids[i] for i in cluster_indices]

            # 代表集和参照集
            comp = compression_result.get(cluster_id, {})
            rep_indices = comp.get("representatives", cluster_indices[:min(100, len(cluster_indices))])
            ref_indices = comp.get("references", cluster_indices[:min(512, len(cluster_indices))])

            rep_ids = [sample_ids[i] for i in rep_indices]
            ref_ids = [sample_ids[i] for i in ref_indices]

            # 标记代表样本
            for sid in rep_ids:
                samples[sid].is_representative = True

            metrics = cluster_metrics.get(cluster_id, {})

            clusters[cluster_id] = ClusterInfo(
                id=cluster_id,
                sample_ids=cluster_sample_ids,
                representative_ids=rep_ids,
                reference_ids=ref_ids,
                centroid=centroids[cluster_id] if cluster_id < len(centroids) else None,
                variance=metrics.get("variance", 0),
                global_distance=metrics.get("global_distance", 0),
                isolation=metrics.get("isolation", 0),
                prior_score=prior_scores.get(cluster_id, 0.5)
            )

        # 保存
        self.cluster_storage.save(clusters, centroids)
        self.data_loader.save_state(Path(self.config.experiment.output_dir) / "samples_state.json")

    # ==================== 在线阶段 ====================

    def run_online_round(self, round_num: Optional[int] = None) -> Dict[str, Any]:
        """
        执行单轮在线采样

        Args:
            round_num: 轮次 (None 则自动递增)

        Returns:
            本轮统计信息
        """
        if round_num is None:
            round_num = self._current_round + 1

        self._current_round = round_num
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"在线采样 - 第 {round_num} 轮")
        self.logger.info(f"{'='*50}")

        # 确保组件已初始化
        self._ensure_online_components()

        # 加载数据
        clusters, centroids, _ = self.cluster_storage.load()
        embeddings, sample_ids, _ = self.embedding_storage.load()
        id_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}

        # Stage 1: 簇选择
        self.logger.info("Stage 1: Thompson Sampling 簇选择")
        selected_clusters = self._thompson_sampler.select()
        self.logger.info(f"选中 {len(selected_clusters)} 个簇: {selected_clusters[:10]}...")

        # Stage 2: 预算分配
        self.logger.info("Stage 2: 预算分配")
        difficulty_weights = self._thompson_sampler.get_difficulty_weights()
        cluster_capacities = {
            cid: len(clusters[cid].representative_ids)
            for cid in selected_clusters
        }
        budget_allocation = self._budget_allocator.allocate(
            selected_clusters, difficulty_weights, cluster_capacities
        )
        self.logger.info(f"预算分配: {dict(list(budget_allocation.items())[:5])}...")

        # Stage 3: 簇内采样
        self.logger.info("Stage 3: 簇内优先级采样")
        selected_samples = {}
        all_selected_ids = []

        for cluster_id in selected_clusters:
            n_select = budget_allocation.get(cluster_id, 0)
            if n_select <= 0:
                continue

            cluster = clusters[cluster_id]
            rep_ids = cluster.representative_ids
            ref_ids = cluster.reference_ids

            # 获取嵌入
            rep_indices = [id_to_idx[sid] for sid in rep_ids if sid in id_to_idx]
            ref_indices = [id_to_idx[sid] for sid in ref_ids if sid in id_to_idx]

            rep_embeddings = embeddings[rep_indices]
            ref_embeddings = embeddings[ref_indices]

            # 获取难度
            difficulties = np.array([
                self.data_loader.get_sample(sid).difficulty
                for sid in rep_ids
            ])

            # 历史嵌入 (用于新颖度)
            history_embeddings = self._get_history_embeddings(embeddings, id_to_idx)

            # 活跃样本过滤
            excluded_ids = self._retirement_manager.get_retired_ids() if self._retirement_manager else set()

            # 采样
            selected_ids, priorities = self._priority_sampler.sample(
                n_select=n_select,
                sample_ids=rep_ids,
                embeddings=rep_embeddings,
                reference_embeddings=ref_embeddings,
                difficulties=difficulties,
                history_embeddings=history_embeddings,
                excluded_ids=excluded_ids
            )

            selected_samples[cluster_id] = selected_ids
            all_selected_ids.extend(selected_ids)

        self.logger.info(f"总共选中 {len(all_selected_ids)} 个样本")

        # 计算多样性
        selected_indices = [id_to_idx[sid] for sid in all_selected_ids if sid in id_to_idx]
        selected_embeddings = embeddings[selected_indices]
        diversity = compute_diversity(selected_embeddings)

        # 记录历史
        round_stats = {
            "round": round_num,
            "n_selected": len(all_selected_ids),
            "n_clusters_selected": len(selected_clusters),
            "cluster_coverage": len(selected_clusters) / len(clusters),
            "diversity": diversity,
            "budget_allocation": budget_allocation,
            "selected_clusters": selected_clusters
        }

        self._selection_history.append(round_stats)

        # 更新历史选中集
        self._update_history(all_selected_ids)

        # 保存检查点
        self._save_checkpoint(round_num)

        self.logger.info(f"第 {round_num} 轮完成: {len(all_selected_ids)} 样本, 多样性={diversity:.4f}")

        return {
            "selected_sample_ids": all_selected_ids,
            "selected_samples_by_cluster": selected_samples,
            "stats": round_stats
        }

    def update_feedback(
        self,
        sample_errors: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        根据训练反馈更新模型

        Args:
            sample_errors: {sample_id: error_intensity}

        Returns:
            更新统计
        """
        self._ensure_online_components()

        # 按簇聚合
        clusters, _, _ = self.cluster_storage.load()
        cluster_errors = {}

        for sid, error in sample_errors.items():
            sample = self.data_loader.get_sample(sid)
            if sample and sample.cluster_id is not None:
                cid = sample.cluster_id
                if cid not in cluster_errors:
                    cluster_errors[cid] = []
                cluster_errors[cid].append(error)

        # 更新 Thompson Sampling 后验
        self._thompson_sampler.update_batch(cluster_errors)

        # 更新样本难度
        for sid, error in sample_errors.items():
            sample = self.data_loader.get_sample(sid)
            if sample:
                from hcds.sampling.priority import update_sample_difficulty
                new_difficulty = update_sample_difficulty(
                    sample.difficulty, error,
                    self.config.priority.difficulty_smoothing
                )
                self.data_loader.update_sample(sid, difficulty=new_difficulty)

        # 更新退休状态
        newly_retired = self._retirement_manager.update_batch(sample_errors)

        # 更新历史统计
        if self._selection_history:
            self._selection_history[-1]["avg_error"] = np.mean(list(sample_errors.values()))
            self._selection_history[-1]["n_retired"] = len(newly_retired)

        stats = {
            "n_updated": len(sample_errors),
            "avg_error": np.mean(list(sample_errors.values())),
            "newly_retired": len(newly_retired),
            "total_retired": len(self._retirement_manager.get_retired_ids()),
            "cluster_stats": {
                cid: {"mean": np.mean(errs), "n": len(errs)}
                for cid, errs in cluster_errors.items()
            }
        }

        self.logger.info(f"反馈更新: {stats['n_updated']} 样本, "
                        f"平均错误={stats['avg_error']:.4f}, "
                        f"新退休={stats['newly_retired']}")

        return stats

    def _ensure_online_components(self):
        """确保在线组件已初始化"""
        if self._thompson_sampler is not None:
            return

        # 加载簇信息
        clusters, _, _ = self.cluster_storage.load()
        n_clusters = len(clusters)

        # 先验分数
        prior_scores = {cid: c.prior_score for cid, c in clusters.items()}

        # Thompson Sampler
        self._thompson_sampler = ThompsonSampler(
            n_clusters=n_clusters,
            config=self.config.thompson_sampling,
            prior_scores=prior_scores
        )

        # Budget Allocator
        self._budget_allocator = BudgetAllocator(self.config.budget)

        # Priority Sampler
        self._priority_sampler = PrioritySampler(self.config.priority)

        # Retirement Manager
        self._retirement_manager = RetirementManager(self.config.retirement)

        # Error Computer
        self._error_computer = ErrorIntensityComputer(self.config.error_intensity)

        # 历史选中集
        self._history_selected_ids = set()

    def _get_history_embeddings(
        self,
        embeddings: np.ndarray,
        id_to_idx: Dict[str, int]
    ) -> Optional[np.ndarray]:
        """获取历史选中样本的嵌入"""
        if not hasattr(self, '_history_selected_ids') or not self._history_selected_ids:
            return None

        indices = [id_to_idx[sid] for sid in self._history_selected_ids if sid in id_to_idx]
        if not indices:
            return None

        return embeddings[indices]

    def _update_history(self, selected_ids: List[str]):
        """更新历史选中集"""
        if not hasattr(self, '_history_selected_ids'):
            self._history_selected_ids = set()
        self._history_selected_ids.update(selected_ids)

    def _save_checkpoint(self, round_num: int):
        """保存检查点"""
        state = {
            "round": round_num,
            "thompson_state": self._thompson_sampler.save_state() if self._thompson_sampler else None,
            "retirement_state": self._retirement_manager.save_state() if self._retirement_manager else None,
            "error_computer_state": self._error_computer.save_state() if self._error_computer else None,
            "history_selected_ids": list(self._history_selected_ids) if hasattr(self, '_history_selected_ids') else [],
            "selection_history": self._selection_history
        }
        self.checkpoint_manager.save(state, round_num)

    def _load_checkpoint(self, filepath: str):
        """加载检查点"""
        state = self.checkpoint_manager.load(filepath)

        self._current_round = state.get("_round", 0)
        self._selection_history = state.get("selection_history", [])
        self._history_selected_ids = set(state.get("history_selected_ids", []))

        # 延迟加载组件状态
        self._checkpoint_state = state

    def get_selection_history(self) -> List[Dict]:
        """获取选择历史"""
        return self._selection_history.copy()
