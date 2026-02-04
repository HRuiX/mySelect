"""数学评估器"""

import re
from typing import List, Optional
import numpy as np

from hcds.feedback.evaluators.base import BaseEvaluator
from hcds.config.schema import MathEvaluatorConfig


class MathEvaluator(BaseEvaluator):
    """数学答案正确性评估器"""

    def __init__(self, config: Optional[MathEvaluatorConfig] = None):
        self.config = config or MathEvaluatorConfig()

        # 编译正则模式
        self._patterns = [re.compile(p) for p in self.config.answer_patterns]

    @property
    def name(self) -> str:
        return "math_exact_match"

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> List[float]:
        """
        评估数学答案正确性

        Args:
            predictions: 模型生成的答案文本
            references: 标准答案文本

        Returns:
            错误强度 (0=正确, 1=错误)
        """
        results = []

        for pred, ref in zip(predictions, references):
            pred_answer = self._extract_answer(pred)
            ref_answer = self._extract_answer(ref)

            if pred_answer is None or ref_answer is None:
                results.append(1.0)  # 无法提取 -> 错误
                continue

            if self._compare_answers(pred_answer, ref_answer):
                results.append(0.0)  # 正确
            else:
                results.append(1.0)  # 错误

        return results

    def _extract_answer(self, text: str) -> Optional[str]:
        """
        从文本中提取答案

        按优先级尝试多种模式
        """
        text = text.strip()

        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)

        # 回退：尝试提取最后一个数字
        numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]

        return None

    def _compare_answers(self, pred: str, ref: str) -> bool:
        """比较两个答案是否一致"""
        # 清理
        pred = self._normalize_answer(pred)
        ref = self._normalize_answer(ref)

        # 直接字符串比较
        if pred == ref:
            return True

        # 数值比较
        try:
            pred_num = float(pred)
            ref_num = float(ref)
            return abs(pred_num - ref_num) <= self.config.tolerance
        except (ValueError, TypeError):
            pass

        return False

    def _normalize_answer(self, answer: str) -> str:
        """归一化答案"""
        if not self.config.normalize_answer:
            return answer

        # 移除逗号
        if self.config.remove_commas:
            answer = answer.replace(",", "")

        # 去除首尾空白
        answer = answer.strip()

        # 转为浮点数再转回字符串 (统一格式)
        if self.config.to_float:
            try:
                num = float(answer)
                if num == int(num):
                    answer = str(int(num))
                else:
                    answer = str(num)
            except ValueError:
                pass

        return answer
