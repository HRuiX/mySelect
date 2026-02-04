# HCDS: Hierarchical Clustering + Dynamic Sampling

基于语义聚类与动态反馈的分层训练数据选择框架

> 适用场景: LLM 推理、数学、指令微调、代码生成

## 核心优势

| 传统方法的问题 | HCDS 的解决方案 |
|---|---|
| 逐样本评分，计算开销 O(N) | 离线压缩 + 在线仅处理候选池 |
| 一次性选集，策略静态 | Thompson Sampling 动态调整 |
| 高分样本聚集少数模式 | 多簇覆盖 + 稀有度 + 多样性 |

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        HCDS Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────── 离线阶段 (一次性) ────────────────────┐  │
│  │  Raw Data → Embedding → Clustering → Compression          │  │
│  │              ↓            ↓            ↓                   │  │
│  │           Encoder      HDBSCAN      FPS/kCenter            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌──────────────────── 在线阶段 (每轮迭代) ──────────────────┐  │
│  │  Stage1: Thompson Sampling → 选择 K 个簇                   │  │
│  │  Stage2: Budget Allocation → 分配预算到各簇                │  │
│  │  Stage3: Priority Sampling → 簇内选样 (R+N+D)             │  │
│  │  Stage4: Feedback Update  → 更新后验 + 难度 + 退休        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 离线预处理

```bash
# 使用默认配置
python scripts/offline_preprocess.py --config configs/config.yaml --data-path data/train.jsonl

# 使用任务预设
python scripts/offline_preprocess.py --task math --data-path data/gsm8k.jsonl
```

### 在线训练

```bash
# 干运行 (模拟训练)
python scripts/run_training.py --config configs/config.yaml --rounds 10 --dry-run

# 实际训练
python scripts/run_training.py --config configs/config.yaml --rounds 10
```

### 集成到现有训练代码

```python
from hcds import HCDSPipeline, load_config

# 加载配置
config = load_config("configs/config.yaml")
pipeline = HCDSPipeline(config)

# 离线预处理 (仅需一次)
pipeline.run_offline()

# 训练循环
for round_num in range(1, T + 1):
    # 获取本轮训练子集
    result = pipeline.run_online_round(round_num)
    selected_ids = result["selected_sample_ids"]

    # 用 selected_ids 训练模型...
    losses = your_training_function(selected_ids)

    # 反馈更新
    sample_errors = {sid: loss for sid, loss in zip(selected_ids, losses)}
    pipeline.update_feedback(sample_errors)
```

## 项目结构

```
mySelect/
├── configs/                    # 配置文件
│   ├── config.py               # 主配置
│   ├── hardware_profiles.py    # 硬件自适应配置
│   └── tasks/                  # 任务预设
├── hcds/                       # 核心库
│   ├── config/                 # 配置管理
│   ├── data/                   # 数据加载
│   ├── embedding/              # 嵌入计算
│   ├── clustering/             # 聚类与压缩
│   ├── sampling/               # 采样策略
│   ├── feedback/               # 反馈计算
│   ├── parallel/               # 并行化支持
│   ├── metrics/                # 指标计算
│   ├── utils/                  # 工具函数
│   └── core.py                 # 主 Pipeline
├── scripts/                    # 运行脚本
├── data/                       # 数据目录
└── docs/                       # 详细文档
```

## 文档

- [设计文档](docs/HCDS_Design.md) - 详细的模块设计与超参说明
- [执行流程](docs/EXECUTION_FLOW.md) - Pipeline 执行流程详解
- [论文笔记](docs/HCDS_Paper.md) - 相关论文与理论基础

## License

MIT
