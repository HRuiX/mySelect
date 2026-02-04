"""
预算分配器
将总预算分配到选中的簇
"""

from typing import List, Dict, Optional
import numpy as np

from hcds.config.schema import BudgetConfig


class BudgetAllocator:
    """预算分配器"""

    def __init__(self, config: BudgetConfig):
        """
        初始化预算分配器

        Args:
            config: 预算配置
        """
        self.config = config
        self.total_budget = config.total_per_round

    def allocate(
        self,
        selected_clusters: List[int],
        difficulty_weights: Dict[int, float],
        cluster_capacities: Dict[int, int]
    ) -> Dict[int, int]:
        """
        分配预算到各簇

        Args:
            selected_clusters: 选中的簇 ID 列表
            difficulty_weights: 簇的困难度权重 {cluster_id: weight}
            cluster_capacities: 簇的容量 (代表集大小) {cluster_id: capacity}

        Returns:
            预算分配 {cluster_id: n_samples}
        """
        K = len(selected_clusters)
        if K == 0:
            return {}

        B = self.total_budget
        base_ratio = self.config.allocation.base_ratio
        max_ratio = self.config.allocation.max_cluster_ratio

        # Step 1: 保底分配
        base_per_cluster = max(5, int(B * base_ratio / K))
        allocations = {}

        for cluster_id in selected_clusters:
            capacity = cluster_capacities.get(cluster_id, base_per_cluster)
            allocations[cluster_id] = min(capacity, base_per_cluster)

        # Step 2: 按困难度分配剩余预算
        remaining = B - sum(allocations.values())

        if remaining > 0:
            # 计算权重
            weights = {}
            for cluster_id in selected_clusters:
                w = difficulty_weights.get(cluster_id, 0.5)
                weights[cluster_id] = w

            total_weight = sum(weights.values()) + 1e-12

            # 按权重分配
            for cluster_id in selected_clusters:
                capacity = cluster_capacities.get(cluster_id, self.total_budget)
                current = allocations[cluster_id]

                # 计算额外分配
                extra = int(remaining * weights[cluster_id] / total_weight)

                # 考虑容量和上限
                max_allowed = int(max_ratio * B / K)
                new_allocation = min(
                    capacity,
                    current + extra,
                    max_allowed
                )
                allocations[cluster_id] = new_allocation

        # Step 3: 重分配未用完的预算
        allocations = self._redistribute_remainder(
            allocations, cluster_capacities, difficulty_weights
        )

        return allocations

    def _redistribute_remainder(
        self,
        allocations: Dict[int, int],
        capacities: Dict[int, int],
        weights: Dict[int, float]
    ) -> Dict[int, int]:
        """重新分配剩余预算"""
        B = self.total_budget
        max_ratio = self.config.allocation.max_cluster_ratio
        K = len(allocations)

        current_total = sum(allocations.values())
        remainder = B - current_total

        if remainder <= 0:
            return allocations

        # 找出还有容量的簇
        available = []
        for cluster_id, allocation in allocations.items():
            capacity = capacities.get(cluster_id, self.total_budget)
            max_allowed = int(max_ratio * B / K)

            if allocation < min(capacity, max_allowed):
                available.append((cluster_id, weights.get(cluster_id, 0.5)))

        # 按权重排序
        available.sort(key=lambda x: x[1], reverse=True)

        # 逐个补充
        for cluster_id, _ in available:
            if remainder <= 0:
                break

            capacity = capacities.get(cluster_id, self.total_budget)
            max_allowed = int(max_ratio * B / K)
            current = allocations[cluster_id]

            can_add = min(capacity, max_allowed) - current
            add = min(can_add, remainder)

            allocations[cluster_id] += add
            remainder -= add

        return allocations

    def set_budget(self, budget: int) -> None:
        """动态调整预算"""
        self.total_budget = budget


class AdaptiveBudgetAllocator(BudgetAllocator):
    """自适应预算分配器"""

    def __init__(
        self,
        config: BudgetConfig,
        explore_ratio: float = 0.2
    ):
        """
        初始化

        Args:
            config: 预算配置
            explore_ratio: 探索预算比例
        """
        super().__init__(config)
        self.explore_ratio = explore_ratio
        self._round = 0

    def allocate(
        self,
        selected_clusters: List[int],
        difficulty_weights: Dict[int, float],
        cluster_capacities: Dict[int, int],
        cluster_visit_counts: Optional[Dict[int, int]] = None
    ) -> Dict[int, int]:
        """
        自适应分配 (考虑探索)

        Args:
            selected_clusters: 选中的簇 ID 列表
            difficulty_weights: 困难度权重
            cluster_capacities: 簇容量
            cluster_visit_counts: 访问次数 (用于 UCB 风格探索)

        Returns:
            预算分配
        """
        self._round += 1

        if cluster_visit_counts is None:
            # 回退到基础分配
            return super().allocate(
                selected_clusters, difficulty_weights, cluster_capacities
            )

        # 分离探索和利用预算
        explore_budget = int(self.total_budget * self.explore_ratio)
        exploit_budget = self.total_budget - explore_budget

        K = len(selected_clusters)
        allocations = {}

        # 探索分配：优先给访问次数少的簇
        visit_counts = [cluster_visit_counts.get(c, 0) for c in selected_clusters]
        min_visits = min(visit_counts) if visit_counts else 0

        low_visit_clusters = [
            c for c in selected_clusters
            if cluster_visit_counts.get(c, 0) <= min_visits + 1
        ]

        explore_per_cluster = explore_budget // max(1, len(low_visit_clusters))
        for cluster_id in low_visit_clusters:
            capacity = cluster_capacities.get(cluster_id, explore_per_cluster)
            allocations[cluster_id] = min(capacity, explore_per_cluster)

        # 利用分配
        remaining_budget = exploit_budget + (explore_budget - sum(allocations.values()))
        remaining_clusters = [c for c in selected_clusters if c not in allocations]

        # 基于困难度的利用分配
        if remaining_clusters and remaining_budget > 0:
            total_weight = sum(
                difficulty_weights.get(c, 0.5) for c in remaining_clusters
            ) + 1e-12

            for cluster_id in remaining_clusters:
                w = difficulty_weights.get(cluster_id, 0.5)
                capacity = cluster_capacities.get(cluster_id, remaining_budget)

                allocation = int(remaining_budget * w / total_weight)
                allocations[cluster_id] = min(capacity, allocation)

        # 重分配
        return self._redistribute_remainder(
            allocations, cluster_capacities, difficulty_weights
        )
