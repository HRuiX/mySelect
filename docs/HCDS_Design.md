# HCDS: 基于语义聚类与动态反馈的分层训练数据选择

**Hierarchical Clustering + Dynamic Sampling**

> 版本: 0.1.0 | 适用场景: LLM 推理、数学、指令微调、代码生成

---

## 1. 项目概述

HCDS 是一个面向大规模 LLM 训练的数据选择框架，通过**离线语义聚类与压缩**和**在线动态反馈采样**两阶段实现高效、动态、覆盖友好的数据选择。

### 核心优势

| 传统方法的问题 | HCDS 的解决方案 |
|---|---|
| 逐样本评分，计算开销 O(N) | 离线压缩 + 在线仅处理候选池 |
| 一次性选集，策略静态 | Thompson Sampling 动态调整 |
| 高分样本聚集少数模式 | 多簇覆盖 + 稀有度 + 多样性 |

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        HCDS Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────── 离线阶段 (一次性) ────────────────────┐  │
│  │  Raw Data → Embedding → Clustering → Compression          │  │
│  │              ↓            ↓            ↓                   │  │
│  │           Encoder      HDBSCAN      FPS/kCenter            │  │
│  │          (E5-large)  (+FAISS ANN)  + 参照集               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌──────────────────── 在线阶段 (每轮迭代) ──────────────────┐  │
│  │  Stage1: Thompson Sampling → 选择 K 个簇                   │  │
│  │  Stage2: Budget Allocation → 分配预算到各簇                │  │
│  │  Stage3: Priority Sampling → 簇内选样 (R+N+D)             │  │
│  │  Stage4: Feedback Update  → 更新后验 + 难度 + 退休        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 快速开始

### 2.1 安装

```bash
cd mySelect
pip install -r requirements.txt
```

### 2.2 离线预处理

```bash
# 使用默认配置
python scripts/offline_preprocess.py --config configs/config.yaml --data-path data/train.jsonl

# 使用任务预设
python scripts/offline_preprocess.py --task math --data-path data/gsm8k.jsonl
```

### 2.3 在线训练

```bash
# 干运行 (模拟训练)
python scripts/run_training.py --config configs/config.yaml --rounds 10 --dry-run

# 实际训练
python scripts/run_training.py --config configs/config.yaml --rounds 10
```

### 2.4 集成到现有训练代码

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

---

## 3. 模块设计

### 3.1 目录结构

```
mySelect/
├── configs/                    # 配置文件 (所有超参数集中管理)
│   ├── config.yaml             # 主配置
│   ├── hardware_profiles.yaml  # 硬件自适应配置
│   └── tasks/                  # 任务预设
│       ├── math_reasoning.yaml
│       ├── instruction_tuning.yaml
│       └── code_generation.yaml
├── hcds/                       # 核心库
│   ├── config/                 # 配置管理 (Pydantic Schema + Loader)
│   ├── data/                   # 数据加载 (多格式支持)
│   ├── embedding/              # 嵌入计算 (ST + API + 增量)
│   ├── clustering/             # 聚类 (HDBSCAN + 大规模 + 压缩)
│   ├── sampling/               # 采样 (TS + Budget + Priority + Retirement)
│   ├── feedback/               # 反馈 (三层错误强度 + 评估器)
│   ├── parallel/               # 并行化 (自动检测 + 监控)
│   ├── metrics/                # 指标 (簇级 + 样本级 + 多样性)
│   ├── utils/                  # 工具 (日志 + 检查点 + 可视化)
│   └── core.py                 # 主 Pipeline
└── scripts/                    # 运行脚本
```

### 3.2 模块依赖关系

```
config ←── 所有模块共用
  ↓
data ←── embedding ←── clustering ←── sampling ←── feedback
  ↓                                      ↓
parallel (横切关注点)                    metrics
  ↓
utils (基础设施)
  ↓
core.py (整合所有模块)
```

---

## 4. 关键设计决策与依据

### 4.1 错误强度：三层体系

| 层级 | 信号 | 开销 | 适用场景 |
|---|---|---|---|
| Layer 1: Loss | 训练时 cross-entropy loss | 零额外开销 | 所有任务 |
| Layer 2: Correctness | 答案正确性检查 | 周期性评估 (K=5轮, 10%抽样) | 数学/代码 |
| Layer 3: Entropy | Token 概率分布熵 | 轻量 | 指令微调 |

**归一化公式**: `g_loss = clip((loss - μ) / σ + 0.5, 0, 1)` (Z-score + offset)

**文献依据**:
- Loss-based: [Not All Samples Are Created Equal (Katharopoulos & Fleuret, 2018)](https://arxiv.org/abs/1803.00942)
- Correctness: [Training Verifiers to Solve Math (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168)
- Entropy: [Deep Bayesian Active Learning (Gal et al., 2017)](https://arxiv.org/abs/1703.02910)

**推荐权重**:

| 任务 | Loss | Correctness | Entropy |
|---|---|---|---|
| 指令微调 | 0.8 | 0 | 0.2 |
| 数学推理 | 0.4 | 0.6 | 0 |
| 代码生成 | 0.3 | 0.7 | 0 |

### 4.2 嵌入编码

- **主选模型**: `intfloat/multilingual-e5-large` (1024 维, 中英多语言)
- **轻量备选**: `paraphrase-multilingual-MiniLM-L12-v2` (384 维, 低显存场景)
- **编码策略**: Instruction-only (按问题类型聚类，非答案模式)
- **PCA 降维**: 保留 95% 方差，降至 256 维加速后续计算

**文献依据**: [E5 (Wang et al., 2022)](https://arxiv.org/abs/2212.03533)

### 4.3 超大规模聚类 (>1M 样本)

标准 HDBSCAN 的 O(N^2) 空间复杂度在超大规模数据上不可行。采用分层近似方案：

```
Step 1: 随机采样 2% 样本作为 Pilot Set
Step 2: 对 Pilot Set 执行 HDBSCAN
Step 3: FAISS 索引簇中心
Step 4: 全量样本 ANN 分配到最近簇
Step 5: 大簇 (>100K) Mini-Batch KMeans 细分
```

**文献依据**:
- [CURE (Guha et al., 1998)](https://dl.acm.org/doi/10.1145/276305.276312) - 采样聚类
- [FAISS (Johnson et al., 2019)](https://arxiv.org/abs/1702.08734) - 大规模向量检索

### 4.4 Thompson Sampling 簇选择

使用 Beta-Bernoulli 模型，以错误强度作为连续奖励信号：

- **先验**: `Beta(α₀ + c·S_prior, β₀ + c·(1-S_prior))`
- **更新**: 每个样本贡献 `(g_i, 1-g_i)` 伪计数
- **前 2 轮系统探索**: 确保每个簇至少被采样一次

### 4.5 优先级函数 (含新颖度门控)

```
Priority = c·D + (1-c)·(a·R + b·N)
其中 b = b₀·(1 - D)  ← 新颖度门控：困难样本不被新颖度抑制
```

默认参数: `c=0.5, a=0.5, b₀=0.5`

### 4.6 并行化策略

| 阶段 | 并行方式 | 实现 |
|---|---|---|
| 嵌入计算 | GPU DataParallel | PyTorch / SentenceTransformers |
| 聚类 | CPU 多进程 | joblib |
| FPS 压缩 | 簇间多进程 | ProcessPoolExecutor |
| kNN 密度 | GPU 批量查询 | FAISS |
| 优先级计算 | 多线程 | ThreadPoolExecutor |
| 资源监控 | 守护线程 | psutil |

支持自动硬件检测 + 用户可覆盖配置。

---

## 5. 配置系统

### 5.1 配置继承

```yaml
# configs/tasks/math_reasoning.yaml
_base_: "../config.yaml"  # 继承主配置

error_intensity:
  weights:
    loss: 0.4
    correctness: 0.6
```

### 5.2 命令行覆盖

```bash
python scripts/run_training.py --config configs/config.yaml \
  --override embedding.batch_size=256 \
  --override budget.total_per_round=5000
```

### 5.3 硬件自适应

系统自动检测 GPU 并选择最优配置：

| 配置档位 | GPU 显存 | 嵌入模型 | Batch Size |
|---|---|---|---|
| A: 高端单卡 | 24GB+ | E5-large | 256 |
| B: 中端单卡 | 16GB | E5-large | 128 |
| C: 入门单卡 | 8-12GB | MiniLM | 64 |
| D: 多卡并行 | N×24GB | E5-large | 128×N |
| E: CPU Only | - | MiniLM | 32 |

---

## 6. 默认超参总览

| 参数 | 默认值 | 说明 | 配置路径 |
|---|---|---|---|
| `m_max` | 2048 | 每簇最大代表数 | `compression.max_representatives` |
| `m_ref` | 512 | 密度参照集大小 | `compression.reference_set_size` |
| `k` | 10 | kNN 的 k 值 | `priority.knn.k` |
| `α₀, β₀` | 1.0, 1.0 | Beta 先验 | `thompson_sampling.prior.alpha_0/beta_0` |
| `c` (先验强度) | 2.0 | 静态先验调节 | `thompson_sampling.prior.strength` |
| `η` | 0.7 | 难度平滑系数 | `priority.difficulty_smoothing` |
| `c` (Priority) | 0.5 | 难度 vs 覆盖权重 | `priority.weights.difficulty` |
| `a, b₀` | 0.5, 0.5 | 稀有度/新颖度权重 | `priority.weights.rarity/novelty_base` |
| `q` (退休) | 3 | 连续低错误次数 | `retirement.consecutive_threshold` |
| `τ` (退休) | 0.1 | 错误强度阈值 | `retirement.error_threshold` |
| 回访概率 | 5% | 退休样本回访率 | `retirement.revisit_probability` |

---

## 7. Ablation 实验设计

### 7.1 错误强度权重 (实验 A)

| ID | Loss | Correctness | Entropy | 验证点 |
|---|---|---|---|---|
| A1 | 1.0 | 0 | 0 | Loss-only baseline |
| A2 | 0 | 1.0 | 0 | Correctness-only |
| A5 | 0.4 | 0.6 | 0 | 我们的数学配置 |
| A6 | 0.5 | 0.5 | 0 | 等权重对照 |

### 7.2 归一化方法 (实验 B)

| ID | 方法 | 公式 |
|---|---|---|
| B1 | Z-score | `clip((x-μ)/σ + 0.5, 0, 1)` |
| B2 | Min-Max | `(x-min)/(max-min)` |
| B4 | Sigmoid | `1/(1+exp(-(x-μ)/τ))` |

### 7.3 评估频率 (实验 C)

| ID | K (间隔) | 抽样比 | 额外开销 |
|---|---|---|---|
| C1 | 1 | 10% | 30% |
| C3 | 5 | 10% | 6% |
| C4 | 10 | 10% | 3% |

---

## 8. 复杂度分析

### 离线阶段

| 步骤 | 复杂度 | 5M 样本估算 |
|---|---|---|
| 嵌入计算 | O(N) | 1.5-4h (GPU) |
| Pilot 采样 | O(N) | <1min |
| HDBSCAN (Pilot) | O(S log S) | 5-10min |
| FAISS 全量分配 | O(N log M) | 10-20min |
| FPS 压缩 | O(ΣC_j · m_max) | 10min |

### 在线每轮

| 步骤 | 复杂度 | 说明 |
|---|---|---|
| Stage1: TS 采样 | O(M) | 极快 |
| Stage2: 预算分配 | O(K) | 极快 |
| Stage3: 优先级采样 | O(Σ hat{C}_j log hat{C}_j) | hat{C}_j ≤ m_max |
| Stage4: 反馈更新 | O(B) | 与训练前向融合 |

---

## 9. 与原始论文设计的差异

基于 LLM 场景的特殊需求，我们对原始 HCDS 设计做了以下调整：

| 原始设计 | 我们的调整 | 原因 |
|---|---|---|
| 二元错误率 g_i ∈ {0,1} | 三层连续信号 g_i ∈ [0,1] | LLM 任务需要更丰富的反馈信号 |
| 单一编码器 | 多编码器支持 (ST + API) | 适配不同资源和语言需求 |
| 标准 HDBSCAN | Pilot+HDBSCAN+FAISS 分层 | 支持 >5M 样本 |
| 无增量支持 | 嵌入增量计算 + 检查点 | 工程鲁棒性 |
| 固定硬件假设 | 4 档自适应配置 | 适配不同 GPU 环境 |

---

## 10. 参考文献

1. Bengio, Y. et al. (2009). Curriculum Learning. *ICML*.
2. Kumar, M. P. et al. (2010). Self-Paced Learning. *NeurIPS*.
3. Katharopoulos, A. & Fleuret, F. (2018). Not All Samples Are Created Equal. *ICML*.
4. Coleman, C. et al. (2020). Selection via Proxy. *ICLR*.
5. Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*.
6. Wang, X. et al. (2023). Self-Consistency Improves Chain of Thought Reasoning. *ICLR*.
7. Lightman, H. et al. (2023). Let's Verify Step by Step. *arXiv:2305.20050*.
8. Reimers, N. & Gurevych, I. (2019). Sentence-BERT. *EMNLP*.
9. Wang, L. et al. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training. *arXiv:2212.03533*.
10. Johnson, J. et al. (2019). Billion-scale similarity search with GPUs. *IEEE TBD*.
11. Guha, S. et al. (1998). CURE: An Efficient Clustering Algorithm. *SIGMOD*.
12. Sculley, D. (2010). Web-Scale K-Means Clustering. *WWW*.
13. Gal, Y. et al. (2017). Deep Bayesian Active Learning with Image Data. *ICML*.
14. Sener, O. & Savarese, S. (2018). Active Learning for Convolutional Neural Networks: A Core-Set Approach. *ICLR*.
