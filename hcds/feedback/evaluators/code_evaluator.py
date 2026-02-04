"""代码执行评估器"""

import subprocess
import tempfile
import os
from typing import List, Optional, Dict, Any

from hcds.feedback.evaluators.base import BaseEvaluator
from hcds.config.schema import CodeEvaluatorConfig


class CodeEvaluator(BaseEvaluator):
    """代码执行正确性评估器"""

    def __init__(self, config: Optional[CodeEvaluatorConfig] = None):
        self.config = config or CodeEvaluatorConfig()

    @property
    def name(self) -> str:
        return "code_test_exec"

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        test_cases: Optional[List[List[Dict[str, Any]]]] = None,
        **kwargs
    ) -> List[float]:
        """
        评估代码正确性

        Args:
            predictions: 模型生成的代码
            references: 标准答案代码 (用于提取测试用例)
            test_cases: 测试用例列表

        Returns:
            错误强度 (0=全部通过, 1=全部失败, 中间值=部分通过)
        """
        results = []

        for i, pred_code in enumerate(predictions):
            if test_cases and i < len(test_cases):
                cases = test_cases[i]
            else:
                # 无测试用例，使用简单的执行检查
                cases = None

            error_intensity = self._evaluate_single(pred_code, cases)
            results.append(error_intensity)

        return results

    def _evaluate_single(
        self,
        code: str,
        test_cases: Optional[List[Dict[str, Any]]]
    ) -> float:
        """评估单个代码"""
        if test_cases is None:
            # 仅检查是否可以执行
            return 0.0 if self._can_execute(code) else 1.0

        # 执行测试用例
        n_passed = 0
        n_total = len(test_cases)

        for test in test_cases:
            if self._run_test(code, test):
                n_passed += 1

        if n_total == 0:
            return 0.5  # 无测试用例

        return 1.0 - (n_passed / n_total)

    def _can_execute(self, code: str) -> bool:
        """检查代码是否可执行"""
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                timeout=self.config.timeout,
                text=True
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    def _run_test(self, code: str, test_case: Dict[str, Any]) -> bool:
        """运行单个测试用例"""
        test_input = test_case.get("input", "")
        expected_output = str(test_case.get("output", "")).strip()

        # 构建执行脚本
        script = f"{code}\n\n# Test\nresult = {test_case.get('call', '')}\nprint(result)"

        try:
            result = subprocess.run(
                ["python", "-c", script],
                input=test_input,
                capture_output=True,
                timeout=self.config.timeout,
                text=True
            )

            if result.returncode != 0:
                return False

            actual_output = result.stdout.strip()
            return actual_output == expected_output

        except (subprocess.TimeoutExpired, Exception):
            return False
