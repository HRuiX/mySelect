#!/usr/bin/env python
"""
HCDS 训练主循环脚本
在线采样 + 训练集成示例
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hcds.config import load_config, create_config_from_task
from hcds.core import HCDSPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="HCDS 训练")

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
        "--rounds", "-r",
        type=int,
        default=10,
        help="训练轮数"
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="从检查点恢复"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行 (只采样不训练)"
    )

    return parser.parse_args()


def simulate_training(sample_ids: List[str]) -> Dict[str, float]:
    """
    模拟训练并返回错误强度

    实际使用时，这里应该调用真正的训练循环，
    并在训练前收集每个样本的 loss 或正确性

    Args:
        sample_ids: 选中的样本 ID

    Returns:
        {sample_id: error_intensity}
    """
    import numpy as np

    # 模拟：随机生成错误强度
    # 实际中应该是训练时的 loss 或评估的正确性
    errors = {}
    for sid in sample_ids:
        # 模拟：大多数样本错误强度中等，少数很高或很低
        error = np.clip(np.random.beta(2, 5), 0, 1)
        errors[sid] = error

    return errors


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

    config.experiment.total_rounds = args.rounds

    print(f"配置文件: {args.config if not args.task else f'tasks/{args.task}'}")
    print(f"训练轮数: {args.rounds}")
    print(f"每轮预算: {config.budget.total_per_round}")

    # 创建 Pipeline
    pipeline = HCDSPipeline(config, resume_from=args.resume)

    # 检查离线数据
    if not pipeline.cluster_storage.exists():
        print("错误: 请先运行离线预处理 (scripts/offline_preprocess.py)")
        sys.exit(1)

    # 训练循环
    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"第 {round_num}/{args.rounds} 轮")
        print(f"{'='*60}")

        # Step 1: 在线采样
        result = pipeline.run_online_round(round_num)
        selected_ids = result["selected_sample_ids"]

        print(f"选中 {len(selected_ids)} 个样本进行训练")

        if args.dry_run:
            print("[Dry Run] 跳过训练")
            # 模拟反馈
            errors = simulate_training(selected_ids)
        else:
            # 实际训练
            # ========================================
            # 在这里集成你的训练代码
            # ========================================
            #
            # 示例:
            # trainer = YourTrainer(config)
            # train_dataset = get_samples_by_ids(selected_ids)
            # losses = trainer.train_one_epoch(train_dataset)
            # errors = {sid: normalize_loss(loss) for sid, loss in zip(selected_ids, losses)}
            #
            # ========================================

            # 目前使用模拟
            errors = simulate_training(selected_ids)

        # Step 2: 更新反馈
        update_stats = pipeline.update_feedback(errors)

        print(f"平均错误强度: {update_stats['avg_error']:.4f}")
        print(f"新退休样本: {update_stats['newly_retired']}")
        print(f"累计退休: {update_stats['total_retired']}")

    # 训练完成
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")

    # 输出统计
    history = pipeline.get_selection_history()
    if history:
        avg_errors = [h.get("avg_error", 0) for h in history if "avg_error" in h]
        if avg_errors:
            print(f"平均错误强度趋势: {avg_errors[0]:.4f} -> {avg_errors[-1]:.4f}")

        diversities = [h.get("diversity", 0) for h in history]
        print(f"平均多样性: {sum(diversities)/len(diversities):.4f}")


if __name__ == "__main__":
    main()
