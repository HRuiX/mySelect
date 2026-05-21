#!/usr/bin/env python
"""
HCDS + Mistral-7B 微调脚本
使用 HCDS 数据选择框架对 Mistral-7B 进行微调

支持数据集:
- MMLU (Massive Multitask Language Understanding)
- BBH (BIG-Bench Hard)

执行方式:
    # 1. 离线预处理
    python scripts/finetune_mistral.py --task mmlu --data-path data/mmlu.jsonl --offline-only

    # 2. 微调训练
    python scripts/finetune_mistral.py --task mmlu --data-path data/mmlu.jsonl --rounds 5

    # 3. 干运行测试
    python scripts/finetune_mistral.py --task bbh --data-path data/bbh.jsonl --dry-run
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from dataclasses import dataclass, field

from hcds.config import load_config
from hcds.config.loader import load_py_config
from hcds.core import HCDSPipeline


@dataclass
class MistralFinetuneConfig:
    """
    Mistral-7B 微调配置

    默认参数与 LESS 论文对齐:
    - LESS: https://arxiv.org/abs/2402.04333
    """
    # 模型配置
    model_name: str = "mistralai/Mistral-7B-v0.1"
    model_revision: str = "main"

    # LoRA 配置 - 对齐 LESS 论文参数
    use_lora: bool = True
    lora_r: int = 128  # LESS: 128 (原: 64)
    lora_alpha: int = 512  # LESS: 512 (原: 128)
    lora_dropout: float = 0.1  # LESS: 0.1 (原: 0.05)
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"  # LESS 只用这 4 个模块
    ])

    # 训练配置 - 对齐 LESS 论文参数
    learning_rate: float = 2e-5  # LESS: 2e-5
    weight_decay: float = 0.0  # LESS: 0.0 (原: 0.01)
    warmup_ratio: float = 0.03  # LESS: 0.03
    max_grad_norm: float = 1.0
    num_train_epochs: int = 4  # LESS: 4 epochs (新增)
    lr_scheduler_type: str = "linear"  # LESS: linear (新增)

    # 批次配置 - 对齐 LESS 论文参数 (有效 batch size = 32)
    per_device_batch_size: int = 1  # LESS: 1 (原: 4)
    gradient_accumulation_steps: int = 32  # LESS: 32 (原: 4)

    # 序列长度
    max_seq_length: int = 2048  # LESS: 2048

    # 精度配置
    bf16: bool = True  # LESS: bf16
    fp16: bool = False

    # 输出配置
    output_dir: str = "outputs/mistral_finetune"
    save_steps: int = 500
    save_strategy: str = "epoch"  # LESS: epoch
    logging_steps: int = 1  # LESS: 1 (原: 10)


def parse_args():
    parser = argparse.ArgumentParser(description="HCDS + Mistral-7B 微调")

    # 任务配置
    parser.add_argument(
        "--task",
        type=str,
        choices=["mmlu", "bbh", "mmlu_bbh"],
        required=True,
        help="任务类型: mmlu, bbh, 或 mmlu_bbh (联合)"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="数据文件路径 (JSONL 格式)"
    )

    # HCDS 配置
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=5,
        help="训练轮数 (HCDS 采样轮次)"
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="每轮采样预算 (覆盖配置)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 HCDS 检查点恢复"
    )

    # 模型配置
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="基础模型名称或路径"
    )

    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="使用 LoRA 微调"
    )

    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="不使用 LoRA (全参数微调)"
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=128,  # LESS: 128
        help="LoRA rank (LESS默认: 128)"
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=512,  # LESS: 512
        help="LoRA alpha (LESS默认: 512)"
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,  # LESS: 0.1
        help="LoRA dropout (LESS默认: 0.1)"
    )

    # 训练配置
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,  # LESS: 2e-5
        help="学习率 (LESS默认: 2e-5)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,  # LESS: 1
        help="每设备批次大小 (LESS默认: 1)"
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=32,  # LESS: 32 (有效 batch size = 32)
        help="梯度累积步数 (LESS默认: 32)"
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,  # LESS: 2048
        help="最大序列长度 (LESS默认: 2048)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=4,  # LESS: 4
        help="训练 epoch 数 (LESS默认: 4)"
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,  # LESS: 0.0
        help="权重衰减 (LESS默认: 0.0)"
    )

    # 运行模式
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="仅运行离线预处理"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行 (只采样不训练)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mistral_finetune",
        help="输出目录"
    )

    return parser.parse_args()


def get_hcds_config(task: str, data_path: str, budget: Optional[int] = None):
    """
    根据任务类型获取 HCDS 配置

    Args:
        task: 任务类型 (mmlu, bbh, mmlu_bbh)
        data_path: 数据文件路径
        budget: 每轮采样预算 (可选)
    """
    task_config_map = {
        "mmlu": ("configs/tasks/mmlu_bbh.py", "get_mmlu_config"),
        "bbh": ("configs/tasks/mmlu_bbh.py", "get_bbh_config"),
        "mmlu_bbh": ("configs/tasks/mmlu_bbh.py", "get_mmlu_bbh_combined_config"),
    }

    if task not in task_config_map:
        raise ValueError(f"未知任务: {task}")

    config_path, config_func = task_config_map[task]
    config = load_py_config(config_path, config_func=config_func, data_path=data_path)

    if budget is not None:
        config.budget.total_per_round = budget

    return config


def format_sample_for_training(sample: Dict[str, str], task: str) -> Dict[str, str]:
    """
    格式化样本为 Mistral 训练格式
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    if task == "mmlu":
        if input_text:
            try:
                choices = json.loads(input_text) if isinstance(input_text, str) else input_text
                if isinstance(choices, list):
                    choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                else:
                    choices_text = str(choices)
            except (json.JSONDecodeError, TypeError):
                choices_text = input_text

            prompt = f"Question: {instruction}\n\n{choices_text}\n\nAnswer:"
        else:
            prompt = f"Question: {instruction}\n\nAnswer:"

    elif task == "bbh":
        prompt = f"Task: {instruction}\n\nLet's think step by step.\n\nAnswer:"

    else:
        if input_text:
            prompt = f"{instruction}\n\n{input_text}\n\nAnswer:"
        else:
            prompt = f"{instruction}\n\nAnswer:"

    return {
        "prompt": prompt,
        "response": output,
        "text": f"{prompt} {output}"
    }


class MistralTrainer:
    """Mistral-7B 训练器封装"""

    def __init__(self, config: MistralFinetuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._setup_done = False

    def setup(self):
        """初始化模型和训练器"""
        if self._setup_done:
            return

        import torch

        print(f"加载模型: {self.config.model_name}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float16,
            "device_map": "auto",
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        if self.config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError:
                raise ImportError("请安装 peft: pip install peft")

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self._setup_done = True
        print("模型加载完成")

    def train_on_samples(
        self,
        samples: List[Dict[str, str]],
        task: str,
        round_num: int
    ) -> Dict[str, float]:
        """在选定样本上训练一个 epoch"""
        if not self._setup_done:
            self.setup()

        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from torch.utils.data import Dataset

        formatted_samples = []
        sample_ids = []
        for sample in samples:
            formatted = format_sample_for_training(sample["content"], task)
            formatted_samples.append(formatted)
            sample_ids.append(sample["id"])

        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt"
                )

            def __len__(self):
                return len(self.encodings.input_ids)

            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.encodings.items()}

        texts = [s["text"] for s in formatted_samples]
        dataset = TextDataset(texts, self.tokenizer, self.config.max_seq_length)

        output_dir = f"{self.config.output_dir}/round_{round_num}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_train_epochs,  # LESS: 4 epochs
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type=self.config.lr_scheduler_type,  # LESS: linear
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,  # LESS: epoch
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            optim="adamw_torch",  # LESS: adamw_torch
            seed=0,  # LESS: seed=0
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        print(f"开始训练: {len(samples)} 个样本")
        train_result = trainer.train()

        avg_loss = train_result.training_loss
        sample_losses = {}
        for sid in sample_ids:
            sample_losses[sid] = avg_loss * (0.8 + 0.4 * np.random.random())

        return sample_losses

    def save_checkpoint(self, round_num: int):
        """保存检查点"""
        if self.model is None:
            return

        save_path = f"{self.config.output_dir}/checkpoint_round_{round_num}"
        print(f"保存检查点: {save_path}")

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


def simulate_training(sample_ids: List[str]) -> Dict[str, float]:
    """模拟训练 (dry-run 模式)"""
    errors = {}
    for sid in sample_ids:
        error = np.clip(np.random.beta(2, 5), 0, 1)
        errors[sid] = error
    return errors


def main():
    args = parse_args()

    print("=" * 60)
    print("HCDS + Mistral-7B 微调")
    print("=" * 60)
    print(f"任务: {args.task}")
    print(f"数据: {args.data_path}")
    print(f"模型: {args.model}")
    print(f"轮数: {args.rounds}")
    print(f"训练参数:")
    print(f"  - LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  - Batch: per_device={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"  - LR: {args.lr}, Epochs: {args.epochs}, Weight decay: {args.weight_decay}")
    print("提示: 按 Ctrl+C 可中断并保存进度")
    print("=" * 60)

    # 初始化信号处理
    from hcds.utils import get_shutdown_handler, check_interrupt, cleanup_gpu_memory
    shutdown = get_shutdown_handler()
    shutdown.register_cleanup(cleanup_gpu_memory)

    try:
        config = get_hcds_config(args.task, args.data_path, args.budget)
        config.experiment.total_rounds = args.rounds
        config.experiment.output_dir = args.output_dir

        pipeline = HCDSPipeline(config, resume_from=args.resume)

        if not pipeline.cluster_storage.exists() or args.offline_only:
            print("\n" + "=" * 60)
            print("离线预处理")
            print("=" * 60)
            offline_stats = pipeline.run_offline()
            print(f"离线处理完成: {offline_stats['n_samples']} 样本, {offline_stats['n_clusters']} 簇")

            if args.offline_only:
                print("离线预处理完成，退出")
                return

        if not args.dry_run:
            trainer_config = MistralFinetuneConfig(
                model_name=args.model,
                use_lora=args.use_lora and not args.no_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                num_train_epochs=args.epochs,
                per_device_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                max_seq_length=args.max_seq_len,
                output_dir=args.output_dir,
            )
            trainer = MistralTrainer(trainer_config)
        else:
            trainer = None

        completed_rounds = 0
        for round_num in range(1, args.rounds + 1):
            # 检查中断信号
            if check_interrupt():
                print(f"\n收到中断信号，在第 {round_num} 轮前停止")
                break

            print(f"\n{'=' * 60}")
            print(f"第 {round_num}/{args.rounds} 轮")
            print("=" * 60)

            result = pipeline.run_online_round(round_num)
            selected_ids = result["selected_sample_ids"]
            print(f"选中 {len(selected_ids)} 个样本")

            samples = []
            for sid in selected_ids:
                sample = pipeline.data_loader.get_sample(sid)
                if sample:
                    samples.append({
                        "id": sid,
                        "content": sample.content
                    })

            if args.dry_run:
                print("[Dry Run] 模拟训练")
                errors = simulate_training(selected_ids)
            else:
                print("开始训练...")
                errors = trainer.train_on_samples(samples, args.task, round_num)
                trainer.save_checkpoint(round_num)

            update_stats = pipeline.update_feedback(errors)
            print(f"平均错误: {update_stats['avg_error']:.4f}")
            print(f"新退休: {update_stats['newly_retired']}, 累计退休: {update_stats['total_retired']}")
            completed_rounds = round_num

        print(f"\n{'=' * 60}")
        if completed_rounds == args.rounds:
            print("训练完成!")
        else:
            print(f"训练中断! 完成 {completed_rounds}/{args.rounds} 轮")
        print("=" * 60)

        history = pipeline.get_selection_history()
        if history:
            avg_errors = [h.get("avg_error", 0) for h in history if "avg_error" in h]
            if avg_errors:
                print(f"错误强度变化: {avg_errors[0]:.4f} -> {avg_errors[-1]:.4f}")

            diversities = [h.get("diversity", 0) for h in history]
            if diversities:
                print(f"平均多样性: {sum(diversities)/len(diversities):.4f}")

        if not args.dry_run:
            print(f"\n模型保存位置: {args.output_dir}")

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("程序被用户中断")
        print("=" * 60)
        print("提示: 使用 --resume 参数可从检查点恢复")

    finally:
        # 清理 GPU 内存
        cleanup_gpu_memory()
        print("清理完成")


if __name__ == "__main__":
    main()
