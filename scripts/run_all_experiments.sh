#!/bin/bash
# ============================================================
# 运行所有对比实验并汇总结果
# HCDS vs Random
# ============================================================

set -e

echo "============================================================"
echo "BBH + Mistral-7B-v0.3 数据选择对比实验"
echo "============================================================"
echo "配置:"
echo "  - 模型: mistralai/Mistral-7B-v0.3"
echo "  - 数据: princeton-nlp/less_data (270K)"
echo "  - 评估: lukaemon/bbh (BBH)"
echo "  - 选择比例: 0.5% (1350 样本)"
echo "============================================================"
echo ""

# 设置环境变量
export OUTPUT_DIR_HCDS="outputs/hcds_bbh_0.5pct"
export OUTPUT_DIR_RANDOM="outputs/random_bbh_0.5pct"
export SAMPLE_SIZE=1350

# 记录总时间
TOTAL_START=$(date +%s)

# 选择运行模式
MODE=${1:-"all"}  # all, hcds, random, compare

case $MODE in
    "hcds")
        echo "运行 HCDS 实验..."
        bash scripts/run_hcds_experiment.sh
        ;;
    "random")
        echo "运行 Random 实验..."
        bash scripts/run_random_experiment.sh
        ;;
    "all")
        echo "运行所有实验..."
        echo ""

        echo "[1/2] 运行 HCDS 实验..."
        bash scripts/run_hcds_experiment.sh

        echo ""
        echo "[2/2] 运行 Random 实验..."
        bash scripts/run_random_experiment.sh
        ;;
    "compare")
        echo "仅汇总结果..."
        ;;
    *)
        echo "用法: $0 [all|hcds|random|compare]"
        exit 1
        ;;
esac

# 汇总结果
echo ""
echo "============================================================"
echo "                      实验结果汇总"
echo "============================================================"

python << 'SUMMARY_SCRIPT'
import json
import os

results = {}

# 读取各实验结果
for method, dir_path in [
    ("HCDS", "outputs/hcds_bbh_0.5pct"),
    ("Random", "outputs/random_bbh_0.5pct"),
]:
    result_file = f"{dir_path}/eval_results.json"
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
            results[method] = {
                "accuracy": data.get("accuracy", 0),
                "correct": data.get("correct", 0),
                "total": data.get("total", 0),
            }
    else:
        results[method] = None

# 打印结果表格
print()
print("┌─────────────┬────────────┬─────────────┬───────────┐")
print("│    方法     │   准确率   │   正确/总数  │   提升    │")
print("├─────────────┼────────────┼─────────────┼───────────┤")

baseline_acc = results.get("Random", {}).get("accuracy", 0) if results.get("Random") else 0

for method in ["HCDS", "Random"]:
    if results.get(method):
        r = results[method]
        acc = r["accuracy"]
        correct = r["correct"]
        total = r["total"]

        if method == "Random":
            improvement = "-"
        else:
            if baseline_acc > 0:
                improvement = f"+{acc - baseline_acc:.2f}%"
            else:
                improvement = "N/A"

        print(f"│ {method:^11} │ {acc:>8.2f}% │ {correct:>5}/{total:<5} │ {improvement:^9} │")
    else:
        print(f"│ {method:^11} │    N/A     │     N/A     │    N/A    │")

print("└─────────────┴────────────┴─────────────┴───────────┘")
print()

# 找出最佳方法
if all(results.get(m) for m in ["HCDS", "Random"]):
    best_method = max(["HCDS", "Random"], key=lambda m: results[m]["accuracy"])
    best_acc = results[best_method]["accuracy"]
    print(f"最佳方法: {best_method} (准确率: {best_acc:.2f}%)")

    # 计算相对于 Random 的提升
    random_acc = results["Random"]["accuracy"]
    improvement = results["HCDS"]["accuracy"] - random_acc
    print(f"HCDS 相对于 Random 提升: {improvement:+.2f}%")

# 保存汇总结果
summary = {
    "results": results,
    "config": {
        "model": "mistralai/Mistral-7B-v0.3",
        "dataset": "princeton-nlp/less_data",
        "eval_dataset": "lukaemon/bbh",
        "selection_ratio": 0.005,
        "sample_size": 1350,
    }
}

os.makedirs("outputs", exist_ok=True)
with open("outputs/comparison_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print()
print(f"完整结果保存至: outputs/comparison_summary.json")
SUMMARY_SCRIPT

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================================"
echo "总耗时: $((TOTAL_DURATION / 3600)) 小时 $(((TOTAL_DURATION % 3600) / 60)) 分 $((TOTAL_DURATION % 60)) 秒"
echo "============================================================"
