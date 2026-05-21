#!/bin/bash
# ============================================================
# Random: BBH + Mistral-7B-v0.3 + 0.5% 随机数据选择
# 作为对照组 (Baseline)
# ============================================================

set -e

# 配置
MODEL="mistralai/Mistral-7B-v0.3"
OUTPUT_DIR="outputs/random_bbh_0.5pct"
SAMPLE_SIZE=1350  # 270K * 0.5% = 1350

echo "============================================================"
echo "Random 随机选择实验 (Baseline)"
echo "============================================================"
echo "模型: $MODEL"
echo "选择样本数: $SAMPLE_SIZE (0.5%)"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/data

# 记录开始时间
START_TIME=$(date +%s)

# Step 1: 随机选择数据
echo ""
echo "[Step 1/3] 从 HuggingFace 随机选择数据..."
python << 'PREP_SCRIPT'
import numpy as np
import json
import os
from datasets import load_dataset

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/random_bbh_0.5pct")
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "1350"))
SEED = 42

print("Loading LESS training data from local JSONL...")
import json
less_data = []
with open("data/less_train/train_all.jsonl", "r") as f:
    for line in f:
        less_data.append(json.loads(line))
total_samples = len(less_data)
print(f"  Total samples: {total_samples}")

print(f"Randomly selecting {SAMPLE_SIZE} samples (seed={SEED})...")
np.random.seed(SEED)
selected_indices = np.random.choice(total_samples, size=SAMPLE_SIZE, replace=False)
np.save(f'{OUTPUT_DIR}/data/selected_indices.npy', selected_indices)
print(f"  Saved: {OUTPUT_DIR}/data/selected_indices.npy")

print("Preparing selected samples...")
selected_samples = []
for i in selected_indices:
    sample = less_data[int(i)]
    # 支持 messages 格式
    if 'messages' in sample:
        input_text = next((m['content'] for m in sample['messages'] if m['role'] == 'user'), '')
        output_text = next((m['content'] for m in sample['messages'] if m['role'] == 'assistant'), '')
    else:
        input_text = sample.get('input', sample.get('instruction', ''))
        output_text = sample.get('output', sample.get('target', ''))
    text = f"### Input:\n{input_text}\n\n### Response:\n{output_text}"
    selected_samples.append({"text": text, "index": int(i)})

with open(f'{OUTPUT_DIR}/data/selected_samples.jsonl', 'w') as f:
    for s in selected_samples:
        f.write(json.dumps(s) + '\n')

print(f"  Saved {len(selected_samples)} samples to {OUTPUT_DIR}/data/selected_samples.jsonl")
print("Data preparation complete!")
PREP_SCRIPT

# Step 2: 微调模型
echo ""
echo "[Step 2/3] 微调 Mistral-7B-v0.3..."
python << 'TRAIN_SCRIPT'
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/random_bbh_0.5pct")
MODEL_NAME = "mistralai/Mistral-7B-v0.3"

print(f"Loading selected samples...")
selected_samples = []
with open(f'{OUTPUT_DIR}/data/selected_samples.jsonl', 'r') as f:
    for line in f:
        selected_samples.append(json.loads(line))
print(f"  {len(selected_samples)} samples loaded")

print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Applying LoRA...")
lora_config = LoraConfig(
    r=128,
    lora_alpha=512,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Tokenizing data...")
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=2048):
        self.encodings = tokenizer(
            [t["text"] for t in texts],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    def __len__(self):
        return len(self.encodings.input_ids)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

dataset = TextDataset(selected_samples, tokenizer)

print("Starting training...")
training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.0,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"Model saved to {OUTPUT_DIR}/final")
TRAIN_SCRIPT

# Step 3: 评估模型
echo ""
echo "[Step 3/3] 评估模型在 BBH 上的性能..."
python << 'EVAL_SCRIPT'
import torch
import json
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/random_bbh_0.5pct")
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
ADAPTER_PATH = f"{OUTPUT_DIR}/final"

print(f"Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)

if os.path.exists(ADAPTER_PATH):
    print(f"Loading adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()

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
for sample in bbh:
    input_text = sample.get('input', sample.get('question', ''))
    target = sample.get('target', sample.get('answer', ''))
    prompt = f"### Input:\n{input_text}\n\n### Response:"
    all_prompts.append(prompt)
    all_targets.append(target)

# 批量推理
from math import ceil
num_batches = ceil(len(all_prompts) / BATCH_SIZE)

for batch_idx in tqdm(range(num_batches), desc="Batch inference"):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(all_prompts))

    batch_prompts = all_prompts[start_idx:end_idx]
    batch_targets = all_targets[start_idx:end_idx]

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
            pad_token_id=tokenizer.pad_token_id
        )

    # 解码每个样本
    for i, (output, target, input_len) in enumerate(zip(outputs, batch_targets, input_lengths)):
        response = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()

        is_correct = target.lower().strip() in response.lower() or response.lower().strip() in target.lower()
        if is_correct:
            correct += 1
        total += 1
        results.append({"target": target, "prediction": response[:100], "correct": is_correct})

accuracy = correct / total * 100
print(f"\n{'='*60}")
print(f"Random 评估结果")
print(f"{'='*60}")
print(f"正确: {correct}/{total}")
print(f"准确率: {accuracy:.2f}%")
print(f"{'='*60}")

with open(f"{OUTPUT_DIR}/eval_results.json", "w") as f:
    json.dump({"method": "Random", "accuracy": accuracy, "correct": correct, "total": total, "samples": results[:10]}, f, indent=2)
EVAL_SCRIPT

# 输出总结
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Random 随机选择实验完成"
echo "============================================================"
echo "总耗时: $((DURATION / 60)) 分 $((DURATION % 60)) 秒"
echo "结果目录: $OUTPUT_DIR"
echo ""

if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
    echo "评估结果:"
    python -c "import json; r=json.load(open('$OUTPUT_DIR/eval_results.json')); print(f'  准确率: {r[\"accuracy\"]:.2f}%')"
fi

echo "============================================================"
