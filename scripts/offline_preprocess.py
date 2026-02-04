#!/usr/bin/env python
"""
HCDS 离线预处理脚本
计算嵌入、聚类、压缩
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hcds.config import load_config, create_config_from_task
from hcds.core import HCDSPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="HCDS 离线预处理")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.py",
        help="配置文件路径 (.py 或 .yaml)"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["math", "instruction", "code", None],
        default=None,
        help="任务类型 (使用预设配置)"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help="数据文件路径 (覆盖配置)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录 (覆盖配置)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新计算 (忽略已有结果)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    if args.task:
        # 使用任务预设配置
        data_path = args.data_path or "data/train.jsonl"
        config = create_config_from_task(task=args.task, data_path=data_path)
    else:
        # 使用指定配置文件
        config_path = args.config
        if args.data_path:
            config = load_config(config_path, data_path=args.data_path)
        else:
            config = load_config(config_path)

    # 覆盖输出目录
    if args.output_dir:
        config.experiment.output_dir = args.output_dir

    print(f"配置文件: {args.config if not args.task else f'tasks/{args.task}'}")
    print(f"数据路径: {config.data.path}")
    print(f"输出目录: {config.experiment.output_dir}")

    # 创建 Pipeline
    pipeline = HCDSPipeline(config)

    # 执行离线预处理
    stats = pipeline.run_offline()

    print("\n" + "=" * 50)
    print("离线预处理完成!")
    print(f"样本数: {stats['n_samples']:,}")
    print(f"簇数: {stats['n_clusters']}")
    print(f"嵌入维度: {stats['embedding_dim']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
