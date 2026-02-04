"""
优先级计算与簇内采样
"""

from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass

from hcds.config.schema import PriorityConfig
from hcds.clustering.compression import compute_local_density


@dataclass
class SamplePriority:
    """样本优先级信息"""
    sample_id: str
    priority: float
    difficulty: float
    rarity: float
    novelty: float


class PriorityCalculator:
    """优先级计算器"""

    def __init__(self, config: PriorityConfig):
        """
        初始化优先级计算器

        Args:
            config: 优先级配置
        """
        self.config = config

        # 权重
        self.c = config.weights.difficulty      # 难度权重
        self.a = config.weights.rarity          # 稀有度权重
        self.b0 = config.weights.novelty_base   # 新颖度基础权重

        # kNN 参数
        self.k = config.knn.k

    def compute_rarity(
        self,
        embeddings: np.ndarray,
        reference_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        计算稀有度 (簇内局部密度)

        Args:
            embeddings: 代表集嵌入 [N, D]
            reference_embeddings: 参照集嵌入 [M, D]

        Returns:
            稀有度分数 [N] ∈ [0, 1]
        """
        # 计算局部密度
        local_density = compute_local_density(
            embeddings,
            reference_embeddings,
            k=self.k
        )

        # 归一化到 [0, 1]
        if local_density.max() > local_density.min():
            rarity = (local_density - local_density.min()) / (
                local_density.max() - local_density.min() + 1e-12
            )
        else:
            rarity = np.zeros_like(local_density)

        return rarity

    def compute_novelty(
        self,
        embeddings: np.ndarray,
        history_embeddings: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        计算新颖度 (相对历史已选集的最近距离)

        Args:
            embeddings: 候选嵌入 [N, D]
            history_embeddings: 历史已选嵌入 [H, D] (None 表示无历史)

        Returns:
            新颖度分数 [N] ∈ [0, 1]
        """
        if history_embeddings is None or len(history_embeddings) == 0:
            return np.ones(len(embeddings))

        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            # 简化版本
            return self._compute_novelty_simple(embeddings, history_embeddings)

        nn = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='auto')
        nn.fit(history_embeddings)

        distances, _ = nn.kneighbors(embeddings)
        novelty = distances.flatten()

        # 归一化
        if novelty.max() > novelty.min():
            novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-12)
        else:
            novelty = np.ones_like(novelty)

        return novelty

    def _compute_novelty_simple(
        self,
        embeddings: np.ndarray,
        history_embeddings: np.ndarray
    ) -> np.ndarray:
        """简化版新颖度计算"""
        novelty = []
        for emb in embeddings:
            # 计算到历史集的最小余弦距离
            similarities = history_embeddings @ emb
            min_dist = 1 - similarities.max()
            novelty.append(min_dist)

        novelty = np.array(novelty)

        # 归一化
        if novelty.max() > novelty.min():
            novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min() + 1e-12)
        else:
            novelty = np.ones_like(novelty)

        return novelty

    def compute_priority(
        self,
        difficulty: np.ndarray,
        rarity: np.ndarray,
        novelty: np.ndarray
    ) -> np.ndarray:
        """
        计算综合优先级

        Priority = c * D + (1-c) * (a * R + b * N)
        其中 b = b0 * (1 - D) (新颖度门控)

        Args:
            difficulty: 难度分数 [N]
            rarity: 稀有度分数 [N]
            novelty: 新颖度分数 [N]

        Returns:
            优先级分数 [N]
        """
        # 新颖度门控权重
        b = self.b0 * (1 - difficulty)

        # 综合优先级
        priority = (
            self.c * difficulty +
            (1 - self.c) * (self.a * rarity + b * novelty)
        )

        return priority


class PrioritySampler:
    """基于优先级的簇内采样器"""

    def __init__(
        self,
        config: PriorityConfig,
        priority_calculator: Optional[PriorityCalculator] = None
    ):
        """
        初始化采样器

        Args:
            config: 优先级配置
            priority_calculator: 优先级计算器
        """
        self.config = config
        self.calculator = priority_calculator or PriorityCalculator(config)

        # 选择比例
        self.priority_ratio = config.selection.priority_ratio
        self.rare_ratio = config.selection.rare_ratio
        self.random_ratio = config.selection.random_ratio

    def sample(
        self,
        n_select: int,
        sample_ids: List[str],
        embeddings: np.ndarray,
        reference_embeddings: np.ndarray,
        difficulties: np.ndarray,
        history_embeddings: Optional[np.ndarray] = None,
        excluded_ids: Optional[Set[str]] = None
    ) -> Tuple[List[str], List[SamplePriority]]:
        """
        执行簇内采样

        Args:
            n_select: 选择的样本数
            sample_ids: 候选样本 ID 列表
            embeddings: 候选嵌入 [N, D]
            reference_embeddings: 参照集嵌入 [M, D]
            difficulties: 样本难度 [N]
            history_embeddings: 历史已选嵌入 (用于新颖度)
            excluded_ids: 排除的样本 ID (如已退休样本)

        Returns:
            (选中的样本 ID 列表, 优先级信息列表)
        """
        n_total = len(sample_ids)

        if n_total <= n_select:
            # 全选
            priorities = [
                SamplePriority(
                    sample_id=sid,
                    priority=1.0,
                    difficulty=difficulties[i],
                    rarity=0.5,
                    novelty=1.0
                )
                for i, sid in enumerate(sample_ids)
            ]
            return sample_ids, priorities

        # 创建有效索引掩码
        if excluded_ids:
            valid_mask = np.array([sid not in excluded_ids for sid in sample_ids])
        else:
            valid_mask = np.ones(n_total, dtype=bool)

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) <= n_select:
            selected_ids = [sample_ids[i] for i in valid_indices]
            priorities = [
                SamplePriority(
                    sample_id=sample_ids[i],
                    priority=1.0,
                    difficulty=difficulties[i],
                    rarity=0.5,
                    novelty=1.0
                )
                for i in valid_indices
            ]
            return selected_ids, priorities

        # 计算各维度分数
        rarity = self.calculator.compute_rarity(
            embeddings[valid_indices],
            reference_embeddings
        )

        novelty = self.calculator.compute_novelty(
            embeddings[valid_indices],
            history_embeddings
        )

        valid_difficulties = difficulties[valid_indices]

        # 计算优先级
        priority = self.calculator.compute_priority(
            valid_difficulties, rarity, novelty
        )

        # 混合选择
        selected_local_indices = self._mixed_selection(
            n_select,
            priority,
            rarity,
            valid_difficulties
        )

        # 转换回原始索引
        selected_indices = valid_indices[selected_local_indices]

        # 构建结果
        selected_ids = [sample_ids[i] for i in selected_indices]
        priorities = [
            SamplePriority(
                sample_id=sample_ids[selected_indices[j]],
                priority=float(priority[selected_local_indices[j]]),
                difficulty=float(valid_difficulties[selected_local_indices[j]]),
                rarity=float(rarity[selected_local_indices[j]]),
                novelty=float(novelty[selected_local_indices[j]])
            )
            for j in range(len(selected_local_indices))
        ]

        return selected_ids, priorities

    def _mixed_selection(
        self,
        n_select: int,
        priority: np.ndarray,
        rarity: np.ndarray,
        difficulty: np.ndarray
    ) -> np.ndarray:
        """
        混合选择策略

        Returns:
            选中的局部索引
        """
        n_total = len(priority)

        # 分配数量
        n_priority = int(n_select * self.priority_ratio)
        n_rare = int(n_select * self.rare_ratio)
        n_random = n_select - n_priority - n_rare

        selected = set()

        # 1. 按优先级选择
        priority_order = np.argsort(priority)[::-1]
        for idx in priority_order:
            if len(selected) >= n_priority:
                break
            selected.add(idx)

        # 2. 选择高稀有度样本 (未被选中的)
        rare_order = np.argsort(rarity)[::-1]
        for idx in rare_order:
            if len(selected) >= n_priority + n_rare:
                break
            if idx not in selected:
                selected.add(idx)

        # 3. 随机选择 (未被选中的)
        remaining = [i for i in range(n_total) if i not in selected]
        if remaining and n_random > 0:
            random_selected = np.random.choice(
                remaining,
                min(n_random, len(remaining)),
                replace=False
            )
            selected.update(random_selected)

        return np.array(list(selected))


def update_sample_difficulty(
    current_difficulty: float,
    new_error_intensity: float,
    smoothing: float = 0.7
) -> float:
    """
    更新样本难度 (指数移动平均)

    Args:
        current_difficulty: 当前难度
        new_error_intensity: 新的错误强度
        smoothing: 平滑系数 η

    Returns:
        更新后的难度
    """
    return smoothing * current_difficulty + (1 - smoothing) * new_error_intensity
