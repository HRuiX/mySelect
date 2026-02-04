"""
Thompson Sampling 实现
基于 Beta 后验的簇选择
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from hcds.config.schema import ThompsonSamplingConfig


@dataclass
class BetaPosterior:
    """Beta 分布后验"""
    alpha: float = 1.0
    beta: float = 1.0

    def sample(self) -> float:
        """从 Beta 分布采样"""
        return np.random.beta(self.alpha, self.beta)

    def mean(self) -> float:
        """后验均值"""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """后验方差"""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))

    def update(self, success: float, failure: float) -> None:
        """
        更新后验

        Args:
            success: 成功伪计数 (错误强度之和)
            failure: 失败伪计数 (1 - 错误强度之和)
        """
        self.alpha += success
        self.beta += failure

    def update_batch(self, error_intensities: List[float]) -> None:
        """
        批量更新

        Args:
            error_intensities: 错误强度列表 [0, 1]
        """
        for g in error_intensities:
            self.alpha += g
            self.beta += (1 - g)


class ThompsonSampler:
    """Thompson Sampling 簇选择器"""

    def __init__(
        self,
        n_clusters: int,
        config: ThompsonSamplingConfig,
        prior_scores: Optional[Dict[int, float]] = None
    ):
        """
        初始化 Thompson Sampler

        Args:
            n_clusters: 簇数量
            config: TS 配置
            prior_scores: 静态先验分数 {cluster_id: score}
        """
        self.n_clusters = n_clusters
        self.config = config

        # 初始化 Beta 后验
        self._posteriors: Dict[int, BetaPosterior] = {}
        self._init_posteriors(prior_scores)

        # 探索统计
        self._round = 0
        self._cluster_visit_count: Dict[int, int] = {i: 0 for i in range(n_clusters)}

    def _init_posteriors(self, prior_scores: Optional[Dict[int, float]]) -> None:
        """初始化 Beta 后验"""
        alpha_0 = self.config.prior.alpha_0
        beta_0 = self.config.prior.beta_0
        c = self.config.prior.strength

        for cluster_id in range(self.n_clusters):
            if prior_scores and cluster_id in prior_scores:
                # 用静态先验调整初始化
                score = prior_scores[cluster_id]
                alpha = alpha_0 + c * score
                beta = beta_0 + c * (1 - score)
            else:
                alpha = alpha_0
                beta = beta_0

            self._posteriors[cluster_id] = BetaPosterior(alpha=alpha, beta=beta)

    def select(
        self,
        n_select: Optional[int] = None,
        available_clusters: Optional[List[int]] = None
    ) -> List[int]:
        """
        选择本轮参与的簇

        Args:
            n_select: 选择的簇数量 (None 则使用配置)
            available_clusters: 可选簇列表 (None 则全部可选)

        Returns:
            选中的簇 ID 列表
        """
        self._round += 1

        if available_clusters is None:
            available_clusters = list(range(self.n_clusters))

        # 确定选择数量
        if n_select is None:
            if self.config.selection.num_clusters == "auto":
                n_select = max(1, int(len(available_clusters) * self.config.selection.ratio))
            else:
                n_select = self.config.selection.num_clusters

        n_select = min(n_select, len(available_clusters))

        # 探索阶段：确保每个簇至少被访问一次
        if self._round <= self.config.exploration.warmup_rounds:
            return self._exploration_select(available_clusters, n_select)

        # 正常 Thompson Sampling
        return self._thompson_select(available_clusters, n_select)

    def _exploration_select(
        self,
        available_clusters: List[int],
        n_select: int
    ) -> List[int]:
        """探索阶段选择"""
        # 优先选择未访问的簇
        unvisited = [
            c for c in available_clusters
            if self._cluster_visit_count[c] == 0
        ]

        if len(unvisited) >= n_select:
            selected = list(np.random.choice(unvisited, n_select, replace=False))
        else:
            # 未访问的全选，剩余用 TS 补充
            selected = unvisited.copy()
            remaining = [c for c in available_clusters if c not in selected]

            if remaining:
                n_remaining = n_select - len(selected)
                ts_selected = self._thompson_select(remaining, n_remaining)
                selected.extend(ts_selected)

        for c in selected:
            self._cluster_visit_count[c] += 1

        return selected

    def _thompson_select(
        self,
        available_clusters: List[int],
        n_select: int
    ) -> List[int]:
        """Thompson Sampling 选择"""
        # 采样
        samples = {
            c: self._posteriors[c].sample()
            for c in available_clusters
        }

        # 选择采样值最大的 K 个
        sorted_clusters = sorted(
            available_clusters,
            key=lambda c: samples[c],
            reverse=True
        )

        selected = sorted_clusters[:n_select]

        for c in selected:
            self._cluster_visit_count[c] += 1

        return selected

    def update(
        self,
        cluster_id: int,
        error_intensities: List[float]
    ) -> None:
        """
        更新单个簇的后验

        Args:
            cluster_id: 簇 ID
            error_intensities: 该簇样本的错误强度列表
        """
        if cluster_id not in self._posteriors:
            return

        self._posteriors[cluster_id].update_batch(error_intensities)

    def update_batch(
        self,
        cluster_errors: Dict[int, List[float]]
    ) -> None:
        """
        批量更新多个簇

        Args:
            cluster_errors: {cluster_id: [error_intensities]}
        """
        for cluster_id, errors in cluster_errors.items():
            self.update(cluster_id, errors)

    def get_cluster_stats(self) -> Dict[int, Dict[str, float]]:
        """
        获取所有簇的统计信息

        Returns:
            {cluster_id: {"alpha": ..., "beta": ..., "mean": ..., "visits": ...}}
        """
        stats = {}
        for cluster_id, posterior in self._posteriors.items():
            stats[cluster_id] = {
                "alpha": posterior.alpha,
                "beta": posterior.beta,
                "mean": posterior.mean(),
                "variance": posterior.variance(),
                "visits": self._cluster_visit_count.get(cluster_id, 0)
            }
        return stats

    def get_difficulty_weights(self) -> Dict[int, float]:
        """
        获取簇的困难度权重 (后验均值)

        Returns:
            {cluster_id: difficulty_weight}
        """
        return {
            cluster_id: posterior.mean()
            for cluster_id, posterior in self._posteriors.items()
        }

    def save_state(self) -> Dict:
        """保存状态"""
        return {
            "posteriors": {
                cid: {"alpha": p.alpha, "beta": p.beta}
                for cid, p in self._posteriors.items()
            },
            "round": self._round,
            "visit_count": self._cluster_visit_count
        }

    def load_state(self, state: Dict) -> None:
        """加载状态"""
        for cid, params in state["posteriors"].items():
            cid = int(cid)
            if cid not in self._posteriors:
                self._posteriors[cid] = BetaPosterior()
            self._posteriors[cid].alpha = params["alpha"]
            self._posteriors[cid].beta = params["beta"]

        self._round = state.get("round", 0)
        self._cluster_visit_count = {
            int(k): v for k, v in state.get("visit_count", {}).items()
        }
