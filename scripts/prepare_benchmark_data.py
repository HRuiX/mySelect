#!/usr/bin/env python
"""
下载并预处理 MMLU 和 BBH 数据集
转换为 HCDS 框架兼容的 JSONL 格式

执行方式:
    # 下载 MMLU 数据集
    python scripts/prepare_benchmark_data.py --dataset mmlu --output data/mmlu.jsonl

    # 下载 BBH 数据集
    python scripts/prepare_benchmark_data.py --dataset bbh --output data/bbh.jsonl

    # 下载两个数据集并合并
    python scripts/prepare_benchmark_data.py --dataset mmlu_bbh --output data/mmlu_bbh.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_mmlu() -> List[Dict[str, Any]]:
    """
    下载 MMLU 数据集
    
    Returns:
        样本列表
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets: pip install datasets")
    
    print("下载 MMLU 数据集...")
    
    samples = []
    
    # MMLU 包含多个学科子集
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions"
    ]
    
    sample_id = 0
    for subject in subjects:
        try:
            # 尝试加载数据集
            dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
            
            for item in dataset:
                question = item["question"]
                choices = item["choices"]
                answer_idx = item["answer"]
                
                # 转换答案索引为字母
                answer = chr(65 + answer_idx)  # 0->A, 1->B, etc.
                
                sample = {
                    "id": f"mmlu_{sample_id}",
                    "question": question,
                    "choices": json.dumps(choices),  # 存储为 JSON 字符串
                    "answer": answer,
                    "subject": subject,
                    "source": "mmlu"
                }
                samples.append(sample)
                sample_id += 1
                
        except Exception as e:
            print(f"  跳过 {subject}: {e}")
            continue
        
        print(f"  {subject}: {len(dataset)} 样本")
    
    print(f"MMLU 总计: {len(samples)} 样本")
    return samples


def download_bbh() -> List[Dict[str, Any]]:
    """
    下载 BBH (BIG-Bench Hard) 数据集
    
    Returns:
        样本列表
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets: pip install datasets")
    
    print("下载 BBH 数据集...")
    
    samples = []
    
    # BBH 包含 23 个任务
    tasks = [
        "boolean_expressions", "causal_judgement", "date_understanding",
        "disambiguation_qa", "dyck_languages", "formal_fallacies",
        "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
        "logical_deduction_seven_objects", "logical_deduction_three_objects",
        "movie_recommendation", "multistep_arithmetic_two", "navigate",
        "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
        "ruin_names", "salient_translation_error_detection", "snarks",
        "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
        "web_of_lies", "word_sorting"
    ]
    
    sample_id = 0
    for task in tasks:
        try:
            dataset = load_dataset("lukaemon/bbh", task, split="test", trust_remote_code=True)
            
            for item in dataset:
                input_text = item["input"]
                target = item["target"]
                
                sample = {
                    "id": f"bbh_{sample_id}",
                    "input": input_text,
                    "target": target,
                    "task": task,
                    "source": "bbh"
                }
                samples.append(sample)
                sample_id += 1
                
        except Exception as e:
            print(f"  跳过 {task}: {e}")
            continue
        
        print(f"  {task}: {len(dataset)} 样本")
    
    print(f"BBH 总计: {len(samples)} 样本")
    return samples


def convert_to_hcds_format(samples: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """
    转换为 HCDS 框架兼容格式
    
    Args:
        samples: 原始样本列表
        source: 数据源 (mmlu/bbh)
    
    Returns:
        转换后的样本列表
    """
    converted = []
    
    for sample in samples:
        if source == "mmlu" or sample.get("source") == "mmlu":
            # MMLU 格式转换
            converted_sample = {
                "id": sample["id"],
                "question": sample["question"],
                "choices": sample.get("choices", ""),
                "answer": sample["answer"],
                "subject": sample.get("subject", ""),
                "source": "mmlu"
            }
        elif source == "bbh" or sample.get("source") == "bbh":
            # BBH 格式转换
            converted_sample = {
                "id": sample["id"],
                "input": sample["input"],
                "target": sample["target"],
                "task": sample.get("task", ""),
                "source": "bbh"
            }
        else:
            # 通用格式
            converted_sample = sample
        
        converted.append(converted_sample)
    
    return converted


def save_jsonl(samples: List[Dict[str, Any]], output_path: str):
    """
    保存为 JSONL 格式
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"已保存 {len(samples)} 个样本到: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="下载并预处理 MMLU/BBH 数据集")
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mmlu", "bbh", "mmlu_bbh"],
        required=True,
        help="要下载的数据集: mmlu, bbh, 或 mmlu_bbh (联合)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出文件路径 (JSONL 格式)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每个数据集最大样本数 (用于快速测试)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    all_samples = []
    
    if args.dataset in ["mmlu", "mmlu_bbh"]:
        mmlu_samples = download_mmlu()
        if args.max_samples:
            mmlu_samples = mmlu_samples[:args.max_samples]
        mmlu_converted = convert_to_hcds_format(mmlu_samples, "mmlu")
        all_samples.extend(mmlu_converted)
    
    if args.dataset in ["bbh", "mmlu_bbh"]:
        bbh_samples = download_bbh()
        if args.max_samples:
            bbh_samples = bbh_samples[:args.max_samples]
        bbh_converted = convert_to_hcds_format(bbh_samples, "bbh")
        all_samples.extend(bbh_converted)
    
    save_jsonl(all_samples, args.output)
    
    print(f"\n完成! 总计 {len(all_samples)} 个样本")
    print(f"输出文件: {args.output}")
    
    # 打印使用示例
    print("\n使用示例:")
    if args.dataset == "mmlu":
        print(f"  python scripts/finetune_mistral.py --task mmlu --data-path {args.output} --rounds 5 --dry-run")
    elif args.dataset == "bbh":
        print(f"  python scripts/finetune_mistral.py --task bbh --data-path {args.output} --rounds 5 --dry-run")
    else:
        print(f"  python scripts/finetune_mistral.py --task mmlu_bbh --data-path {args.output} --rounds 5 --dry-run")


if __name__ == "__main__":
    main()
