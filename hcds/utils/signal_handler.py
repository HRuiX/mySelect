"""
优雅退出信号处理
支持 Ctrl+C 中断时保存进度并清理资源
"""

import signal
import sys
import threading
from typing import Optional, Callable, List, Any
from contextlib import contextmanager


class GracefulShutdown:
    """
    优雅退出管理器

    用法:
        shutdown = GracefulShutdown()

        # 注册清理回调
        shutdown.register_cleanup(save_checkpoint)
        shutdown.register_cleanup(cleanup_gpu)

        # 在长时间运行的循环中检查
        for batch in batches:
            if shutdown.should_exit:
                print("收到中断信号，正在保存进度...")
                break
            process(batch)
    """

    _instance: Optional['GracefulShutdown'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._should_exit = False
        self._cleanup_callbacks: List[Callable] = []
        self._original_sigint = None
        self._original_sigterm = None
        self._interrupt_count = 0
        self._max_interrupts = 2  # 第二次 Ctrl+C 强制退出
        self._initialized = True

        # 注册信号处理
        self._register_signals()

    def _register_signals(self):
        """注册信号处理器"""
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        self._interrupt_count += 1
        signal_name = "SIGINT (Ctrl+C)" if signum == signal.SIGINT else "SIGTERM"

        if self._interrupt_count == 1:
            print(f"\n{'='*60}")
            print(f"收到 {signal_name} 信号，正在优雅退出...")
            print("再次按 Ctrl+C 强制退出")
            print(f"{'='*60}")
            self._should_exit = True
        else:
            print(f"\n强制退出...")
            self._run_cleanup()
            sys.exit(1)

    @property
    def should_exit(self) -> bool:
        """检查是否应该退出"""
        return self._should_exit

    def register_cleanup(self, callback: Callable, *args, **kwargs):
        """
        注册清理回调函数

        Args:
            callback: 清理函数
            *args, **kwargs: 传递给清理函数的参数
        """
        self._cleanup_callbacks.append((callback, args, kwargs))

    def unregister_cleanup(self, callback: Callable):
        """取消注册清理回调"""
        self._cleanup_callbacks = [
            (cb, args, kwargs)
            for cb, args, kwargs in self._cleanup_callbacks
            if cb != callback
        ]

    def _run_cleanup(self):
        """执行所有清理回调"""
        print("正在执行清理操作...")
        for callback, args, kwargs in reversed(self._cleanup_callbacks):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"清理回调执行失败: {e}")

    def exit_gracefully(self, message: str = "程序已优雅退出"):
        """
        手动触发优雅退出

        Args:
            message: 退出消息
        """
        print(f"\n{message}")
        self._run_cleanup()
        self.reset()

    def reset(self):
        """重置状态 (用于测试或重新初始化)"""
        self._should_exit = False
        self._interrupt_count = 0
        self._cleanup_callbacks.clear()

    def restore_signals(self):
        """恢复原始信号处理器"""
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)


@contextmanager
def interruptible_operation(
    description: str = "操作",
    on_interrupt: Optional[Callable] = None,
    save_progress: Optional[Callable] = None
):
    """
    可中断操作的上下文管理器

    用法:
        with interruptible_operation("计算嵌入", save_progress=save_embeddings):
            for batch in batches:
                process(batch)

    Args:
        description: 操作描述
        on_interrupt: 中断时的回调
        save_progress: 保存进度的回调
    """
    shutdown = GracefulShutdown()

    if save_progress:
        shutdown.register_cleanup(save_progress)

    try:
        yield shutdown
    except KeyboardInterrupt:
        print(f"\n{description} 被中断")
        if on_interrupt:
            on_interrupt()
        raise
    finally:
        if save_progress:
            shutdown.unregister_cleanup(save_progress)


def check_interrupt() -> bool:
    """
    快捷函数: 检查是否收到中断信号

    用法:
        for item in items:
            if check_interrupt():
                print("保存进度并退出...")
                break
            process(item)
    """
    return GracefulShutdown().should_exit


def register_cleanup(callback: Callable, *args, **kwargs):
    """
    快捷函数: 注册清理回调

    用法:
        register_cleanup(save_checkpoint, checkpoint_path)
    """
    GracefulShutdown().register_cleanup(callback, *args, **kwargs)


def cleanup_gpu_memory():
    """清理 GPU 显存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("已清理 PyTorch GPU 缓存")
    except ImportError:
        pass

    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        print("已清理 CuPy GPU 缓存")
    except ImportError:
        pass


# 全局单例
_shutdown_handler = None

def get_shutdown_handler() -> GracefulShutdown:
    """获取全局 GracefulShutdown 实例"""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdown()
    return _shutdown_handler
