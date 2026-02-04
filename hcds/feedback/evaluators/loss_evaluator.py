"""Loss 评估器"""

from typing import List
import numpy as np

from hcds.feedback.evaluators.base import BaseEvaluator


class LossEvaluator(BaseEvaluator):
    """基于 Loss 的评估器"""

    @property
    def name(self) -> str:
        return "loss"

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        losses: List[float] = None,
        **kwargs
    ) -> List[float]:
        """
        评估错误强度 (基于 loss)

        Args:
            predictions: 不使用
            references: 不使用
            losses: 每个样本的 loss

        Returns:
            归一化后的错误强度
        """
        if losses is None:
            raise ValueError("LossEvaluator 需要 losses 参数")

        losses = np.array(losses)

        # Z-score 归一化
        if losses.std() > 0:
            z_scores = (losses - losses.mean()) / losses.std()
            return np.clip(z_scores + 0.5, 0, 1).tolist()
        else:
            return [0.5] * len(losses)
