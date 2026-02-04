"""
检查点管理
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        初始化检查点管理器

        Args:
            checkpoint_dir: 检查点目录
            max_checkpoints: 最大保留检查点数
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

    def save(
        self,
        state: Dict[str, Any],
        round_num: int,
        prefix: str = "checkpoint"
    ) -> Path:
        """
        保存检查点

        Args:
            state: 状态字典
            round_num: 轮次
            prefix: 文件前缀

        Returns:
            保存路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_round{round_num:04d}_{timestamp}.pkl"
        filepath = self.checkpoint_dir / filename

        # 分离 numpy 数组
        numpy_state = {}
        json_state = {}

        for key, value in state.items():
            if isinstance(value, np.ndarray):
                numpy_state[key] = value
            else:
                json_state[key] = value

        # 保存
        with open(filepath, 'wb') as f:
            pickle.dump({
                "json_state": json_state,
                "numpy_state": numpy_state,
                "round": round_num,
                "timestamp": timestamp
            }, f)

        # 清理旧检查点
        self._cleanup()

        return filepath

    def load(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            filepath: 检查点路径 (None 则加载最新)

        Returns:
            状态字典
        """
        if filepath is None:
            filepath = self.get_latest()

        if filepath is None:
            raise FileNotFoundError("没有可用的检查点")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # 合并状态
        state = data["json_state"]
        state.update(data["numpy_state"])
        state["_round"] = data["round"]
        state["_timestamp"] = data["timestamp"]

        return state

    def get_latest(self) -> Optional[Path]:
        """获取最新检查点"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        checkpoints = []
        for path in sorted(self.checkpoint_dir.glob("checkpoint_*.pkl")):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                checkpoints.append({
                    "path": path,
                    "round": data.get("round"),
                    "timestamp": data.get("timestamp")
                })
            except:
                continue
        return checkpoints

    def _cleanup(self) -> None:
        """清理旧检查点"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for old_ckpt in checkpoints[self.max_checkpoints:]:
            old_ckpt.unlink()
