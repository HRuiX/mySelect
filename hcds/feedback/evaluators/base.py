"""评估器基类"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseEvaluator(ABC):
    """错误强度评估器基类"""

    @abstractmethod
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> List[float]:
        """
        评估错误强度

        Args:
            predictions: 模型预测
            references: 标准答案
            **kwargs: 其他参数

        Returns:
            错误强度列表 [0, 1]
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """评估器名称"""
        pass
