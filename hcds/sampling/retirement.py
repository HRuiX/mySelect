"""
样本退休管理器
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
import numpy as np

from hcds.config.schema import RetirementConfig


@dataclass
class SampleRetirementStatus:
    """样本退休状态"""
    sample_id: str
    consecutive_low_error: int = 0
    retired: bool = False
    last_error_intensity: float = 0.5
    times_selected: int = 0


class RetirementManager:
    """样本退休管理器"""

    def __init__(self, config: RetirementConfig):
        """
        初始化退休管理器

        Args:
            config: 退休配置
        """
        self.config = config
        self.enabled = config.enabled

        # 阈值
        self.consecutive_threshold = config.consecutive_threshold
        self.error_threshold = config.error_threshold
        self.revisit_probability = config.revisit_probability

        # 样本状态
        self._status: Dict[str, SampleRetirementStatus] = {}

    def update(
        self,
        sample_id: str,
        error_intensity: float
    ) -> bool:
        """
        更新样本状态并判断是否退休

        Args:
            sample_id: 样本 ID
            error_intensity: 本轮错误强度

        Returns:
            是否应该退休
        """
        if not self.enabled:
            return False

        # 获取或创建状态
        if sample_id not in self._status:
            self._status[sample_id] = SampleRetirementStatus(sample_id=sample_id)

        status = self._status[sample_id]
        status.last_error_intensity = error_intensity
        status.times_selected += 1

        # 判断是否低错误
        if error_intensity < self.error_threshold:
            status.consecutive_low_error += 1
        else:
            status.consecutive_low_error = 0

        # 判断是否退休
        if status.consecutive_low_error >= self.consecutive_threshold:
            status.retired = True
            return True

        return False

    def update_batch(
        self,
        sample_errors: Dict[str, float]
    ) -> Set[str]:
        """
        批量更新

        Args:
            sample_errors: {sample_id: error_intensity}

        Returns:
            新退休的样本 ID 集合
        """
        newly_retired = set()

        for sample_id, error in sample_errors.items():
            if self.update(sample_id, error):
                newly_retired.add(sample_id)

        return newly_retired

    def should_revisit(self, sample_id: str) -> bool:
        """
        判断退休样本是否应该回访

        Args:
            sample_id: 样本 ID

        Returns:
            是否应该回访
        """
        if not self.enabled:
            return True

        status = self._status.get(sample_id)
        if status is None or not status.retired:
            return True

        # 概率回访
        return np.random.random() < self.revisit_probability

    def get_retired_ids(self) -> Set[str]:
        """获取所有退休样本 ID"""
        return {
            sid for sid, status in self._status.items()
            if status.retired
        }

    def get_active_ids(self, sample_ids: List[str]) -> List[str]:
        """
        从样本列表中过滤出活跃样本

        Args:
            sample_ids: 样本 ID 列表

        Returns:
            活跃样本 ID 列表
        """
        if not self.enabled:
            return sample_ids

        retired = self.get_retired_ids()

        active = []
        for sid in sample_ids:
            if sid not in retired:
                active.append(sid)
            elif self.should_revisit(sid):
                active.append(sid)

        return active

    def reset_sample(self, sample_id: str) -> None:
        """
        重置样本状态 (用于回访后重新激活)

        Args:
            sample_id: 样本 ID
        """
        if sample_id in self._status:
            status = self._status[sample_id]
            status.consecutive_low_error = 0
            # 不重置 retired，保留历史记录

    def unretire_sample(self, sample_id: str) -> None:
        """
        取消样本退休状态

        Args:
            sample_id: 样本 ID
        """
        if sample_id in self._status:
            status = self._status[sample_id]
            status.retired = False
            status.consecutive_low_error = 0

    def get_statistics(self) -> Dict[str, int]:
        """获取退休统计"""
        total = len(self._status)
        retired = sum(1 for s in self._status.values() if s.retired)

        return {
            "total_tracked": total,
            "retired": retired,
            "active": total - retired,
            "retirement_rate": retired / total if total > 0 else 0
        }

    def save_state(self) -> Dict:
        """保存状态"""
        return {
            sid: {
                "consecutive_low_error": s.consecutive_low_error,
                "retired": s.retired,
                "last_error_intensity": s.last_error_intensity,
                "times_selected": s.times_selected
            }
            for sid, s in self._status.items()
        }

    def load_state(self, state: Dict) -> None:
        """加载状态"""
        for sid, data in state.items():
            self._status[sid] = SampleRetirementStatus(
                sample_id=sid,
                consecutive_low_error=data.get("consecutive_low_error", 0),
                retired=data.get("retired", False),
                last_error_intensity=data.get("last_error_intensity", 0.5),
                times_selected=data.get("times_selected", 0)
            )
