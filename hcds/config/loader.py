"""
配置加载与合并工具
支持 Python 配置文件和 YAML 配置文件加载
"""

import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from copy import deepcopy

from hcds.config.schema import HCDSConfig


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典，override 覆盖 base

    Args:
        base: 基础配置字典
        override: 覆盖配置字典

    Returns:
        合并后的字典
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_py_config(
    config_path: Union[str, Path],
    config_func: str = "get_default_config",
    **kwargs
) -> HCDSConfig:
    """
    从 Python 配置文件加载配置

    Args:
        config_path: Python 配置文件路径
        config_func: 配置函数名，默认为 "get_default_config"
        **kwargs: 传递给配置函数的参数

    Returns:
        HCDSConfig 实例

    Examples:
        >>> config = load_py_config("configs/config.py")
        >>> config = load_py_config("configs/config.py", data_path="data/train.jsonl")
        >>> config = load_py_config(
        ...     "configs/tasks/math_reasoning.py",
        ...     config_func="get_math_reasoning_config"
        ... )
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    if not config_path.suffix == ".py":
        raise ValueError(f"不是 Python 文件: {config_path}")

    # 动态导入模块
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取配置函数
    if not hasattr(module, config_func):
        raise AttributeError(f"配置文件中未找到函数: {config_func}")

    get_config = getattr(module, config_func)

    if not callable(get_config):
        raise TypeError(f"{config_func} 不是可调用对象")

    # 调用配置函数
    return get_config(**kwargs)


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    cli_args: Optional[Dict[str, Any]] = None,
    config_func: str = None,
    **kwargs
) -> HCDSConfig:
    """
    加载并解析 HCDS 配置
    自动识别 Python (.py) 或 YAML (.yaml/.yml) 配置文件

    Args:
        config_path: 配置文件路径 (.py 或 .yaml)
        overrides: 字典形式的配置覆盖 (仅 YAML 模式)
        cli_args: 命令行参数覆盖 (仅 YAML 模式)
        config_func: Python 配置函数名 (仅 Python 模式)
        **kwargs: 传递给配置函数的额外参数 (仅 Python 模式)

    Returns:
        HCDSConfig 实例

    Examples:
        # Python 配置
        >>> config = load_config("configs/config.py")
        >>> config = load_config("configs/config.py", data_path="data/train.jsonl")

        # YAML 配置 (保持向后兼容)
        >>> config = load_config("configs/config.yaml")
        >>> config = load_config(
        ...     "configs/config.yaml",
        ...     overrides={"experiment": {"name": "my_exp"}}
        ... )
    """
    config_path = Path(config_path)

    if config_path.suffix == ".py":
        # Python 配置文件
        func_name = config_func or _infer_config_func(config_path)
        return load_py_config(config_path, config_func=func_name, **kwargs)
    elif config_path.suffix in (".yaml", ".yml"):
        # YAML 配置文件
        return _load_yaml_config(config_path, overrides, cli_args)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")


def _infer_config_func(config_path: Path) -> str:
    """
    根据配置文件路径推断配置函数名

    Args:
        config_path: 配置文件路径

    Returns:
        配置函数名
    """
    name = config_path.stem

    # 常见映射
    func_map = {
        "config": "get_default_config",
        "experiment_small": "get_experiment_small_config",
        "math_reasoning": "get_math_reasoning_config",
        "instruction_tuning": "get_instruction_tuning_config",
        "code_generation": "get_code_generation_config",
    }

    return func_map.get(name, f"get_{name}_config")


def _load_yaml_config(
    config_path: Path,
    overrides: Optional[Dict[str, Any]] = None,
    cli_args: Optional[Dict[str, Any]] = None
) -> HCDSConfig:
    """
    加载 YAML 配置文件 (保持向后兼容)

    Args:
        config_path: YAML 配置文件路径
        overrides: 字典形式的配置覆盖
        cli_args: 命令行参数覆盖

    Returns:
        HCDSConfig 实例
    """
    import yaml

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) or {}

    # 解析继承
    config_dict = _resolve_inheritance(config_dict, config_path.parent)

    # 应用字典覆盖
    if overrides:
        config_dict = deep_merge(config_dict, overrides)

    # 应用命令行参数覆盖
    if cli_args:
        config_dict = _apply_cli_overrides(config_dict, cli_args)

    return HCDSConfig(**config_dict)


def _resolve_inheritance(config_dict: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    """
    解析配置继承 (_base_ 字段)

    Args:
        config_dict: 配置字典
        config_dir: 配置文件所在目录

    Returns:
        解析继承后的配置字典
    """
    import yaml

    if "_base_" not in config_dict:
        return config_dict

    base_path = config_dict.pop("_base_")

    # 解析相对路径
    if not os.path.isabs(base_path):
        base_path = config_dir / base_path

    with open(base_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f) or {}

    # 递归解析基础配置的继承
    base_config = _resolve_inheritance(base_config, Path(base_path).parent)

    # 合并配置
    return deep_merge(base_config, config_dict)


def _apply_cli_overrides(config_dict: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    应用命令行参数覆盖

    Args:
        config_dict: 配置字典
        cli_args: 命令行参数字典，键为点分隔路径

    Returns:
        更新后的配置字典
    """
    result = deepcopy(config_dict)

    for key_path, value in cli_args.items():
        keys = key_path.split(".")
        current = result

        # 遍历到倒数第二层
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # 设置最终值
        final_key = keys[-1]
        value = _parse_cli_value(value)
        current[final_key] = value

    return result


def _parse_cli_value(value: Any) -> Any:
    """
    解析命令行参数值，尝试转换为适当的类型
    """
    if isinstance(value, str):
        # 布尔值
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        elif value.lower() == "null" or value.lower() == "none":
            return None

        # 数字
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # 列表 (逗号分隔)
        if "," in value and not value.startswith("["):
            return [_parse_cli_value(v.strip()) for v in value.split(",")]

    return value


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典

    Args:
        *configs: 配置字典序列，后面的覆盖前面的

    Returns:
        合并后的配置字典
    """
    result = {}
    for config in configs:
        if config:
            result = deep_merge(result, config)
    return result


def get_hardware_profile(
    gpu_memory_mb: Optional[int] = None,
    gpu_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    根据硬件配置获取对应的配置档位

    Args:
        gpu_memory_mb: GPU 显存 (MB)，None 则自动检测
        gpu_count: GPU 数量，None 则自动检测

    Returns:
        硬件配置档位字典
    """
    from configs.hardware_profiles import get_hardware_profile as _get_profile
    return _get_profile(gpu_memory_mb, gpu_count)


def create_config_from_task(
    task: str,
    data_path: str,
    budget: int = None,
    **kwargs
) -> HCDSConfig:
    """
    根据任务类型快速创建配置

    Args:
        task: 任务类型 ("math", "instruction", "code")
        data_path: 数据文件路径
        budget: 每轮预算 (可选)
        **kwargs: 其他覆盖参数

    Returns:
        HCDSConfig 实例
    """
    task_config_map = {
        "math": ("configs/tasks/math_reasoning.py", "get_math_reasoning_config"),
        "instruction": ("configs/tasks/instruction_tuning.py", "get_instruction_tuning_config"),
        "code": ("configs/tasks/code_generation.py", "get_code_generation_config"),
    }

    if task not in task_config_map:
        raise ValueError(f"未知任务类型: {task}，支持: {list(task_config_map.keys())}")

    config_path, config_func = task_config_map[task]

    config = load_py_config(config_path, config_func=config_func, data_path=data_path)

    # 覆盖预算
    if budget is not None:
        config.budget.total_per_round = budget

    return config
