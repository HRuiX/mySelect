"""
资源监控器
"""

import psutil
from threading import Thread
from queue import Queue
import time
from typing import Dict, List, Any, Optional


class ResourceMonitor:
    """运行时资源监控"""

    def __init__(
        self,
        max_cpu_percent: float = 80,
        max_memory_percent: float = 85,
        check_interval: float = 1.0
    ):
        """
        初始化监控器

        Args:
            max_cpu_percent: CPU 使用上限
            max_memory_percent: 内存使用上限
            check_interval: 检查间隔 (秒)
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval

        self._running = False
        self._thread = None
        self._alerts = Queue()
        self._history = []

    def start(self) -> None:
        """启动监控"""
        self._running = True
        self._thread = Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            stats = self.get_current_stats()
            self._history.append(stats)

            # 检查阈值
            if stats["cpu_percent"] > self.max_cpu_percent:
                self._alerts.put(("cpu", stats["cpu_percent"]))

            if stats["memory_percent"] > self.max_memory_percent:
                self._alerts.put(("memory", stats["memory_percent"]))

            if "gpu" in stats:
                for gpu in stats["gpu"]:
                    usage = gpu["memory_used_gb"] / gpu["memory_total_gb"]
                    if usage > 0.95:
                        self._alerts.put(("gpu", gpu["id"], usage))

            # 限制历史大小
            if len(self._history) > 1000:
                self._history = self._history[-500:]

            time.sleep(self.check_interval)

    def get_alerts(self) -> List[tuple]:
        """获取告警"""
        alerts = []
        while not self._alerts.empty():
            alerts.append(self._alerts.get_nowait())
        return alerts

    @staticmethod
    def get_current_stats() -> Dict[str, Any]:
        """获取当前资源状态"""
        stats = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        }

        try:
            import torch
            if torch.cuda.is_available():
                stats["gpu"] = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    stats["gpu"].append({
                        "id": i,
                        "name": props.name,
                        "memory_used_gb": torch.cuda.memory_allocated(i) / (1024**3),
                        "memory_total_gb": props.total_memory / (1024**3),
                    })
        except ImportError:
            pass

        return stats

    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录"""
        return self._history.copy()

    def get_summary(self) -> Dict[str, float]:
        """获取统计摘要"""
        if not self._history:
            return {}

        cpu_values = [h["cpu_percent"] for h in self._history]
        mem_values = [h["memory_percent"] for h in self._history]

        return {
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_max": max(cpu_values),
            "memory_avg": sum(mem_values) / len(mem_values),
            "memory_max": max(mem_values),
        }
