#!/usr/bin/env python
"""
HCDS 完整 Demo 脚本
生成数据 → 离线预处理 → 在线训练 → 输出分析
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def log(msg: str, file=None):
    """打印并写入文件"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if file:
        file.write(line + "\n")
        file.flush()


def main():
    # 创建输出目录
    output_dir = Path("./output/demo_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(output_dir / "demo_output.txt", "w", encoding="utf-8")

    log("=" * 60, log_file)
    log("HCDS 完整 Demo 开始", log_file)
    log("=" * 60, log_file)

    # ==================== Step 1: 加载配置 ====================
    log("\n>>> Step 1: 加载配置", log_file)

    from configs.experiment_small import get_experiment_small_config

    config = get_experiment_small_config(data_path="./data/demo_data.jsonl")
    config.experiment.output_dir = str(output_dir)
    config.experiment.total_rounds = 3
    config.budget.total_per_round = 50
    config.embedding.storage.path = str(output_dir / "embeddings")
    config.logging.dir = str(output_dir / "logs")

    log(f"  实验名称: {config.experiment.name}", log_file)
    log(f"  数据路径: {config.data.path}", log_file)
    log(f"  嵌入模型: {config.embedding.model}", log_file)
    log(f"  每轮预算: {config.budget.total_per_round}", log_file)
    log(f"  训练轮数: {config.experiment.total_rounds}", log_file)

    # ==================== Step 2: 创建 Pipeline ====================
    log("\n>>> Step 2: 创建 Pipeline", log_file)

    from hcds.core import HCDSPipeline

    pipeline = HCDSPipeline(config)
    log("  Pipeline 创建成功", log_file)

    # ==================== Step 3: 离线预处理 ====================
    log("\n>>> Step 3: 离线预处理", log_file)
    log("  开始计算嵌入、聚类、压缩...", log_file)

    stats = pipeline.run_offline()

    log(f"\n  [离线结果]", log_file)
    log(f"  样本总数: {stats['n_samples']}", log_file)
    log(f"  聚类数量: {stats['n_clusters']}", log_file)
    log(f"  嵌入维度: {stats['embedding_dim']}", log_file)

    log(f"\n  [簇大小分布]", log_file)
    cluster_sizes = stats.get('cluster_sizes', {})
    for cid, size in sorted(cluster_sizes.items()):
        log(f"    簇 {cid}: {size} 个样本", log_file)

    log(f"\n  [先验分数]", log_file)
    prior_scores = stats.get('prior_scores', {})
    for cid, score in sorted(prior_scores.items()):
        log(f"    簇 {cid}: {score:.4f}", log_file)

    # ==================== Step 4: 在线训练循环 ====================
    log("\n>>> Step 4: 在线训练循环", log_file)

    all_round_stats = []

    for round_num in range(1, config.experiment.total_rounds + 1):
        log(f"\n  {'='*50}", log_file)
        log(f"  第 {round_num} 轮", log_file)
        log(f"  {'='*50}", log_file)

        # Stage 1-3: 在线采样
        result = pipeline.run_online_round(round_num)
        selected_ids = result["selected_sample_ids"]
        round_stats = result["stats"]

        log(f"\n  [采样结果]", log_file)
        log(f"    选中样本数: {len(selected_ids)}", log_file)
        log(f"    选中簇数: {round_stats['n_clusters_selected']}", log_file)
        log(f"    簇覆盖率: {round_stats['cluster_coverage']:.2%}", log_file)
        log(f"    多样性: {round_stats['diversity']:.4f}", log_file)

        log(f"\n  [预算分配]", log_file)
        budget_allocation = round_stats.get('budget_allocation', {})
        for cid, budget in sorted(budget_allocation.items()):
            log(f"    簇 {cid}: {budget} 个样本", log_file)

        # 模拟训练：生成随机错误强度
        np.random.seed(42 + round_num)
        errors = {}
        for sid in selected_ids:
            # 模拟：不同类型样本有不同的错误分布
            if "math" in sid:
                error = np.clip(np.random.beta(3, 3), 0, 1)  # 中等难度
            elif "code" in sid:
                error = np.clip(np.random.beta(4, 2), 0, 1)  # 较难
            else:
                error = np.clip(np.random.beta(2, 4), 0, 1)  # 较简单
            errors[sid] = error

        log(f"\n  [模拟训练]", log_file)
        log(f"    平均错误强度: {np.mean(list(errors.values())):.4f}", log_file)
        log(f"    最高错误强度: {np.max(list(errors.values())):.4f}", log_file)
        log(f"    最低错误强度: {np.min(list(errors.values())):.4f}", log_file)

        # Stage 4: 反馈更新
        update_stats = pipeline.update_feedback(errors)

        log(f"\n  [反馈更新]", log_file)
        log(f"    更新样本数: {update_stats['n_updated']}", log_file)
        log(f"    平均错误: {update_stats['avg_error']:.4f}", log_file)
        log(f"    新退休样本: {update_stats['newly_retired']}", log_file)
        log(f"    累计退休: {update_stats['total_retired']}", log_file)

        round_stats["avg_error"] = update_stats["avg_error"]
        round_stats["newly_retired"] = update_stats["newly_retired"]
        round_stats["total_retired"] = update_stats["total_retired"]
        all_round_stats.append(round_stats)

    # ==================== Step 5: 输出汇总 ====================
    log("\n>>> Step 5: 训练汇总", log_file)
    log("=" * 60, log_file)

    log("\n  [轮次统计]", log_file)
    log(f"  {'轮次':<6} {'选中样本':<10} {'多样性':<10} {'平均错误':<10} {'退休数':<8}", log_file)
    log(f"  {'-'*44}", log_file)

    for i, rs in enumerate(all_round_stats):
        log(f"  {i+1:<6} {rs['n_selected']:<10} {rs['diversity']:<10.4f} {rs.get('avg_error', 0):<10.4f} {rs.get('total_retired', 0):<8}", log_file)

    # Thompson Sampling 状态
    log("\n  [Thompson Sampling 最终状态]", log_file)
    ts_stats = pipeline._thompson_sampler.get_cluster_stats()
    log(f"  {'簇ID':<6} {'α':<10} {'β':<10} {'后验均值':<12} {'访问次数':<8}", log_file)
    log(f"  {'-'*46}", log_file)
    for cid, s in sorted(ts_stats.items()):
        log(f"  {cid:<6} {s['alpha']:<10.2f} {s['beta']:<10.2f} {s['mean']:<12.4f} {s['visits']:<8}", log_file)

    # 保存详细结果到 JSON
    results = {
        "offline_stats": {
            "n_samples": stats["n_samples"],
            "n_clusters": stats["n_clusters"],
            "embedding_dim": stats["embedding_dim"],
            "cluster_sizes": {str(k): v for k, v in cluster_sizes.items()},
            "prior_scores": {str(k): v for k, v in prior_scores.items()},
        },
        "round_stats": all_round_stats,
        "thompson_sampling_final": {
            str(k): {
                "alpha": v["alpha"],
                "beta": v["beta"],
                "mean": v["mean"],
                "visits": v["visits"],
            }
            for k, v in ts_stats.items()
        },
    }

    with open(output_dir / "demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\n  结果已保存到: {output_dir / 'demo_results.json'}", log_file)
    log(f"  日志已保存到: {output_dir / 'demo_output.txt'}", log_file)

    log("\n" + "=" * 60, log_file)
    log("Demo 完成!", log_file)
    log("=" * 60, log_file)

    log_file.close()

    return str(output_dir / "demo_output.txt")


if __name__ == "__main__":
    output_path = main()
    print(f"\n完整日志已保存到: {output_path}")
