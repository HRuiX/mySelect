"""
后验更新器
整合 Thompson Sampling 后验更新与样本难度更新
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from hcds.sampling.thompson import ThompsonSampler
from hcds.sampling.retirement import RetirementManager
from hcds.sampling.priority import update_sample_difficulty
from hcds.feedback.error_intensity import ErrorIntensityComputer


class PosteriorUpdater:
    """后验更新器"""

    def __init__(
        self,
        thompson_sampler: ThompsonSampler,
        retirement_manager: RetirementManager,
        error_computer: ErrorIntensityComputer,
        difficulty_smoothing: float = 0.7
    ):
        """
        初始化后验更新器

        Args:
            thompson_sampler: Thompson Sampling 采样器
            retirement_manager: 退休管理器
            error_computer: 错误强度计算器
            difficulty_smoothing: 难度平滑系数 η
        """
        self.thompson_sampler = thompson_sampler
        self.retirement_manager = retirement_manager
        self.error_computer = error_computer
        self.difficulty_smoothing = difficulty_smoothing

    def update(
        self,
        selected_samples: Dict[int, List[str]],
        error_intensities: Dict[str, float],
        current_difficulties: Dict[str, float]
    ) -> Dict[str, Dict]:
        """
        执行全面的反馈更新

        Args:
            selected_samples: {cluster_id: [sample_ids]}
            error_intensities: {sample_id: error_intensity}
            current_difficulties: {sample_id: current_difficulty}

        Returns:
            更新结果 {
                "updated_difficulties": {sample_id: new_difficulty},
                "newly_retired": set of sample_ids,
                "cluster_stats": {cluster_id: {mean_error, n_samples}}
            }
        """
        updated_difficulties = {}
        cluster_errors = {}

        # Step 1: 按簇聚合错误强度
        for cluster_id, sample_ids in selected_samples.items():
            errors = []
            for sid in sample_ids:
                if sid in error_intensities:
                    errors.append(error_intensities[sid])

            if errors:
                cluster_errors[cluster_id] = errors

        # Step 2: 更新 Thompson Sampling 后验
        self.thompson_sampler.update_batch(cluster_errors)

        # Step 3: 更新样本难度
        for sid, error in error_intensities.items():
            current_d = current_difficulties.get(sid, 0.5)
            new_d = update_sample_difficulty(
                current_d, error, self.difficulty_smoothing
            )
            updated_difficulties[sid] = new_d

        # Step 4: 执行退休机制
        newly_retired = self.retirement_manager.update_batch(error_intensities)

        # Step 5: 汇总统计
        cluster_stats = {}
        for cluster_id, errors in cluster_errors.items():
            cluster_stats[cluster_id] = {
                "mean_error": float(np.mean(errors)),
                "std_error": float(np.std(errors)),
                "n_samples": len(errors),
                "min_error": float(np.min(errors)),
                "max_error": float(np.max(errors)),
            }

        return {
            "updated_difficulties": updated_difficulties,
            "newly_retired": newly_retired,
            "cluster_stats": cluster_stats,
            "retirement_stats": self.retirement_manager.get_statistics()
        }
