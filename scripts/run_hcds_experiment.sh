#!/bin/bash
# ============================================================
# HCDS: BBH + Mistral-7B-v0.3 + 0.5% 数据选择
# 使用 HCDS 动态采样方法
# ============================================================

set -e

# 配置
MODEL="mistralai/Mistral-7B-v0.3"
OUTPUT_DIR="outputs/hcds_bbh_0.5pct"
SELECTION_RATIO=0.005
TOTAL_SAMPLES=270000
NUM_ROUNDS=4
BUDGET_PER_ROUND=$((TOTAL_SAMPLES * 5 / 1000 / NUM_ROUNDS))  # 337

echo "============================================================"
echo "HCDS 数据选择实验"
echo "============================================================"
echo "模型: $MODEL"
echo "选择比例: 0.5% ($(($TOTAL_SAMPLES * 5 / 1000)) 样本)"
echo "轮数: $NUM_ROUNDS"
echo "每轮预算: $BUDGET_PER_ROUND"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 记录开始时间
START_TIME=$(date +%s)

# Step 1: 运行 HCDS 数据选择 + 微调
echo ""
echo "[Step 1/3] 运行 HCDS 数据选择 + 微调..."
python scripts/finetune_mistral.py \
    --task bbh \
    --data-path "data/less_train/train_all.jsonl" \
    --model $MODEL \
    --rounds $NUM_ROUNDS \
    --budget $BUDGET_PER_ROUND \
    --batch-size 8 \
    --grad-accum 4 \
    --lora-r 128 \
    --lora-alpha 512 \
    --lora-dropout 0.1 \
    --lr 2e-5 \
    --epochs 4 \
    --weight-decay 0.0 \
    --max-seq-len 2048 \
    --output-dir $OUTPUT_DIR

# Step 2: 评估模型
echo ""
echo "[Step 2/3] 评估模型在 BBH 上的性能..."
python << 'EVAL_SCRIPT'
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import os

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
ADAPTER_PATH = os.environ.get("OUTPUT_DIR", "outputs/hcds_bbh_0.5pct") + "/final"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/hcds_bbh_0.5pct")

print(f"Loading model from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 加载 LoRA adapter
if os.path.exists(ADAPTER_PATH):
    print(f"Loading adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
else:
    print(f"Warning: Adapter not found at {ADAPTER_PATH}, using base model")

model.eval()

# 加载 BBH 测试数据
print("Loading BBH test data from lukaemon/bbh...")
from datasets import concatenate_datasets
bbh_subsets = [
    'boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa',
    'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton',
    'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects',
    'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting',
    'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names',
    'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences',
    'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting'
]
bbh_datasets = []
for subset in bbh_subsets:
    try:
        ds = load_dataset("lukaemon/bbh", subset, split="test")
        bbh_datasets.append(ds)
    except Exception as e:
        print(f"  Warning: Could not load {subset}: {e}")
bbh = concatenate_datasets(bbh_datasets)

# 批量推理配置
BATCH_SIZE = 32  # 增加显存占用

# 左侧padding对生成任务更友好
tokenizer.padding_side = "left"

print(f"Evaluating on {len(bbh)} samples with batch_size={BATCH_SIZE}...")
correct = 0
total = 0
results = []

# 准备所有数据
all_prompts = []
all_targets = []
all_inputs = []
for sample in bbh:
    input_text = sample.get('input', sample.get('question', ''))
    target = sample.get('target', sample.get('answer', ''))
    prompt = f"### Input:\n{input_text}\n\n### Response:"
    all_prompts.append(prompt)
    all_targets.append(target)
    all_inputs.append(input_text)

# 批量推理
from math import ceil
num_batches = ceil(len(all_prompts) / BATCH_SIZE)

for batch_idx in tqdm(range(num_batches), desc="Batch inference"):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(all_prompts))

    batch_prompts = all_prompts[start_idx:end_idx]
    batch_targets = all_targets[start_idx:end_idx]
    batch_inputs = all_inputs[start_idx:end_idx]

    # 批量tokenize
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_lengths = inputs['attention_mask'].sum(dim=1)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 解码每个样本
    for i, (output, target, input_text, input_len) in enumerate(zip(outputs, batch_targets, batch_inputs, input_lengths)):
        response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()

        # 简单的精确匹配
        is_correct = target.lower().strip() in response.lower() or response.lower().strip() in target.lower()
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "input": input_text[:100],
            "target": target,
            "prediction": response[:100],
            "correct": is_correct
        })

accuracy = correct / total * 100
print(f"\n{'='*60}")
print(f"HCDS 评估结果")
print(f"{'='*60}")
print(f"正确: {correct}/{total}")
print(f"准确率: {accuracy:.2f}%")
print(f"{'='*60}")

# 保存结果
with open(f"{OUTPUT_DIR}/eval_results.json", "w") as f:
    json.dump({
        "method": "HCDS",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "selection_ratio": 0.005,
        "samples": results[:10]  # 保存前10个样本
    }, f, indent=2)

print(f"结果保存至: {OUTPUT_DIR}/eval_results.json")
EVAL_SCRIPT

# Step 3: 输出总结
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "HCDS 实验完成"
echo "============================================================"
echo "总耗时: $((DURATION / 60)) 分 $((DURATION % 60)) 秒"
echo "结果目录: $OUTPUT_DIR"
echo ""

# 显示评估结果
if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
    echo "评估结果:"
    python -c "import json; r=json.load(open('$OUTPUT_DIR/eval_results.json')); print(f'  准确率: {r[\"accuracy\"]:.2f}%')"
fi

echo "============================================================"
