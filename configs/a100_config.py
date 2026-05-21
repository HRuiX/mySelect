"""
单卡 A100 优化配置

针对单张 A100 (40GB/80GB) 优化的配置参数
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class A100SingleGPUConfig:
    """
    单张 A100 优化配置

    显存估算 (Mistral-7B + LoRA + bf16):
    - 模型: ~14 GB
    - LoRA + 优化器: ~1.5 GB
    - 激活值 + 梯度: ~3-5 GB
    - 总计: ~18-20 GB (40GB A100 足够)
    """

    # ==================== A100 40GB 推荐配置 ====================
    # 与 LESS 论文完全对齐

    # 模型
    model_name: str = "mistralai/Mistral-7B-v0.1"

    # LoRA (对齐 LESS)
    lora_r: int = 128
    lora_alpha: int = 512
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # 训练参数 (对齐 LESS)
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: int = 4
    lr_scheduler_type: str = "linear"

    # 批次配置 (对齐 LESS: 有效 batch size = 32)
    per_device_batch_size: int = 1  # A100 40GB 安全值
    gradient_accumulation_steps: int = 32  # 有效 batch = 32

    # 序列长度
    max_seq_length: int = 2048

    # 精度
    bf16: bool = True
    fp16: bool = False

    # 显存优化
    gradient_checkpointing: bool = False  # A100 不需要
    optim: str = "adamw_torch"  # 或 "adamw_torch_fused" 更快


@dataclass
class A100_80GB_Config(A100SingleGPUConfig):
    """
    A100 80GB 可以使用更大的 batch size 加速训练
    """
    per_device_batch_size: int = 2  # 可以增加
    gradient_accumulation_steps: int = 16  # 保持有效 batch = 32


@dataclass
class A100_40GB_LargeSeq_Config(A100SingleGPUConfig):
    """
    A100 40GB + 更长序列 (需要梯度检查点)
    """
    max_seq_length: int = 4096
    gradient_checkpointing: bool = True  # 启用以节省显存
    per_device_batch_size: int = 1


# ==================== 快速参考 ====================

CONFIGS = {
    "a100_40gb": A100SingleGPUConfig(),
    "a100_80gb": A100_80GB_Config(),
    "a100_40gb_long_seq": A100_40GB_LargeSeq_Config(),
}


def print_config_summary(config_name: str = "a100_40gb"):
    """打印配置摘要"""
    config = CONFIGS.get(config_name, A100SingleGPUConfig())

    effective_batch = config.per_device_batch_size * config.gradient_accumulation_steps

    print(f"""
{'='*60}
配置: {config_name}
{'='*60}
模型: {config.model_name}
LoRA: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}
{'='*60}
训练参数:
  - Learning rate: {config.learning_rate}
  - Epochs: {config.num_train_epochs}
  - Per-device batch: {config.per_device_batch_size}
  - Gradient accumulation: {config.gradient_accumulation_steps}
  - 有效 batch size: {effective_batch}
  - Max seq length: {config.max_seq_length}
  - Precision: {'bf16' if config.bf16 else 'fp16' if config.fp16 else 'fp32'}
{'='*60}
预估显存: ~{18 if config.per_device_batch_size == 1 else 25} GB
{'='*60}
""")


if __name__ == "__main__":
    import sys
    config_name = sys.argv[1] if len(sys.argv) > 1 else "a100_40gb"
    print_config_summary(config_name)
