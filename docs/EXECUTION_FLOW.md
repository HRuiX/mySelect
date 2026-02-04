# HCDS 方法论文档

## 1. 项目目标

HCDS (Hierarchical Clustering + Dynamic Sampling) 是一个**面向大规模 LLM 训练的数据选择框架**。

### 1.1 要解决的问题

| 传统方法的问题 | HCDS 的解决方案 |
|---|---|
| **计算开销大**：逐样本评分，复杂度 O(N) | 离线压缩 + 在线仅处理代表集 |
| **策略静态**：一次性选集，无法根据训练状态调整 | Thompson Sampling 动态调整簇权重 |
| **覆盖不足**：高分样本聚集在少数语义模式 | 多簇覆盖 + 稀有度 + 新颖度机制 |
| **过拟合风险**：简单样本被反复训练 | 样本退休机制，连续低错误自动退休 |

### 1.2 核心思想

将数据选择分为两阶段：

1. **离线阶段**：语义聚类 + 簇内压缩，将 N 个样本压缩为 K 个簇 × m 个代表样本
2. **在线阶段**：Thompson Sampling 动态选簇 → 预算分配 → 优先级采样 → 反馈更新

---

## 2. 离线阶段：语义聚类与压缩

### 2.1 嵌入计算

**目的**：将文本样本映射到语义向量空间，使语义相似的样本距离相近。

**方法**：使用预训练的 Sentence Transformer 模型（默认 `intfloat/multilingual-e5-large`）对样本的 instruction 部分编码。

**实现位置**：`hcds/embedding/sentence_transformer.py::SentenceTransformerEncoder.encode_batch()`

**可选 PCA 降维**：

将高维嵌入（如 1024 维）降到目标维度（如 256 维），保留 95% 方差，加速后续聚类计算。

**实现位置**：`hcds/embedding/incremental.py::PCAReducer.fit_transform()`

---

### 2.2 HDBSCAN 聚类

**目的**：将样本按语义相似度分成若干簇，每个簇代表一种"语义模式"或"任务类型"。

**为什么选择 HDBSCAN**：

- 不需要预设簇数量 K
- 能发现任意形状的簇
- 自动识别噪声点
- 适合语义空间中不均匀分布的数据

**实现位置**：`hcds/clustering/hdbscan_cluster.py::HDBSCANClusterer.fit()`

**关键参数**：

| 参数 | 含义 | 默认值 |
|-----|------|--------|
| `min_cluster_size` | 最小簇大小，小于此值的聚合会被视为噪声 | 50 |
| `min_samples` | 核心点邻域最小样本数，影响密度阈值 | 10 |
| `cluster_selection_method` | 簇选择方法，`eom` 更稳定 | "eom" |

---

### 2.3 簇级指标计算

**目的**：为每个簇计算"静态先验分数"，用于初始化 Thompson Sampling 的 Beta 分布。

**计算的三个指标**：

#### 2.3.1 簇内方差 (Variance)

衡量簇内样本的离散程度。方差大表示簇内样本多样性高，可能包含更多有价值的学习信号。

$$\text{Variance}_k = \frac{1}{|C_k|} \sum_{x \in C_k} \|x - \mu_k\|^2$$

其中 $\mu_k$ 是簇 k 的中心。

#### 2.3.2 全局偏离度 (Global Distance)

衡量簇中心相对于全局中心的偏离程度（余弦距离）。偏离大的簇代表"长尾"或"边缘"语义模式。

$$\text{GlobalDist}_k = 1 - \frac{\mu_k \cdot \mu_{\text{global}}}{\|\mu_k\| \cdot \|\mu_{\text{global}}\|}$$

#### 2.3.3 簇间隔离度 (Isolation)

衡量该簇与其他簇的分离程度。隔离度高表示簇的语义独特，值得重点采样。

$$\text{Isolation}_k = 1 - \max_{j \neq k} \text{CosineSim}(\mu_k, \mu_j)$$

**静态先验分数**：

$$S_{\text{prior}}^{(k)} = \lambda_1 \cdot \hat{V}_k + \lambda_2 \cdot \hat{G}_k + \lambda_3 \cdot \hat{I}_k$$

其中 $\hat{V}, \hat{G}, \hat{I}$ 是归一化到 [0,1] 的指标值，$\lambda_1 + \lambda_2 + \lambda_3 = 1$（默认各 0.25）。

**实现位置**：`hcds/metrics/cluster_metrics.py::compute_cluster_metrics()` 和 `compute_prior_scores()`

---

### 2.4 簇内压缩 (FPS)

**目的**：每个簇可能有数万样本，直接在线处理计算量大。用 FPS 选出能代表整个簇语义分布的"代表集"。

**Farthest Point Sampling (最远点采样) 算法**：

1. 初始化：选择第一个点作为第一个代表
2. 迭代：每次选择距离已选代表集**最远**的点加入代表集
3. 终止：达到最大代表数 $m_{\max}$

$$x_{\text{next}} = \arg\max_{x \notin S} \min_{s \in S} d(x, s)$$

**为什么用 FPS**：

- 保证代表集**最大化覆盖**整个簇的语义空间
- 比随机采样或 KMeans 中心更好地保留边缘样本
- 时间复杂度 O(nm)，可接受

**实现位置**：`hcds/clustering/compression.py::fps_single_cluster_cosine()`

**关键参数**：

| 参数 | 含义 | 默认值 |
|-----|------|--------|
| `max_representatives` | 每簇最大代表数 $m_{\max}$ | 2048 |
| `reference_set_size` | 密度参照集大小（用于稀有度计算）| 512 |

---

## 3. 在线阶段：动态采样

每轮训练前执行以下四个 Stage。

### 3.1 Stage 1: Thompson Sampling 簇选择

**目的**：在 K 个簇中选择本轮参与训练的 K' 个簇，平衡**探索**（尝试新簇）和**利用**（选择高价值簇）。

**方法**：为每个簇维护一个 Beta 分布 $\text{Beta}(\alpha_k, \beta_k)$，表示对该簇"训练价值"的置信度。

#### 3.1.1 Beta 分布初始化

$$\alpha_k^{(0)} = \alpha_0 + c \cdot S_{\text{prior}}^{(k)}$$
$$\beta_k^{(0)} = \beta_0 + c \cdot (1 - S_{\text{prior}}^{(k)})$$

其中：
- $\alpha_0, \beta_0 = 1.0$：Beta 分布基础参数（无信息先验）
- $c = 2.0$：静态先验强度，控制初始先验的影响力
- $S_{\text{prior}}^{(k)}$：离线计算的簇先验分数

**参数含义**：先验分数高的簇，初始 $\alpha$ 更大，更可能被采样。

#### 3.1.2 Thompson Sampling 选择过程

1. 为每个簇从其 Beta 分布采样：$\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$
2. 选择采样值最大的 K' 个簇

$$\text{Selected} = \text{TopK}(\{\theta_k\}_{k=1}^K)$$

**为什么用 Thompson Sampling**：

- 自动平衡探索与利用：不确定性大的簇（方差大）有机会被探索
- 收敛保证：随着训练进行，会收敛到真正高价值的簇
- 适应性强：能动态适应模型训练状态的变化

**实现位置**：`hcds/sampling/thompson.py::ThompsonSampler.select()`

**关键参数**：

| 参数 | 含义 | 默认值 |
|-----|------|--------|
| `alpha_0`, `beta_0` | Beta 先验初始参数 | 1.0 |
| `strength` (c) | 静态先验强度 | 2.0 |
| `selection.ratio` | 每轮选择的簇比例 | 0.3 |
| `warmup_rounds` | 探索期轮数（确保每簇至少被访问一次）| 2 |

---

### 3.2 Stage 2: 预算分配

**目的**：将总预算 B 分配到 K' 个选中的簇。困难的簇（错误强度高）应该获得更多预算。

**分配公式**：

1. **保底分配**：每个簇至少获得 $B_{\text{base}} = B \cdot r_{\text{base}} / K'$

2. **按困难度分配剩余预算**：

$$B_k = B_{\text{base}} + (B - K' \cdot B_{\text{base}}) \cdot \frac{w_k}{\sum_j w_j}$$

其中 $w_k = E[\theta_k]$ 是簇 k 的 Beta 后验均值，代表该簇的"困难度权重"。

3. **上限约束**：单簇最大预算不超过 $\rho \cdot B / K'$，防止过度集中

**实现位置**：`hcds/sampling/budget.py::BudgetAllocator.allocate()`

**关键参数**：

| 参数 | 含义 | 默认值 |
|-----|------|--------|
| `total_per_round` (B) | 每轮总预算 | 10000 |
| `base_ratio` ($r_{\text{base}}$) | 保底分配比例 | 0.2 |
| `max_cluster_ratio` ($\rho$) | 单簇最大预算倍数 | 3.0 |

---

### 3.3 Stage 3: 簇内优先级采样

**目的**：在每个簇的代表集中，选择最有价值的样本。综合考虑**难度**、**稀有度**、**新颖度**三个维度。

#### 3.3.1 优先级计算公式

$$P_i = c \cdot D_i + (1-c) \cdot (a \cdot R_i + b_i \cdot N_i)$$

其中：

- $D_i \in [0,1]$：难度分数（根据历史错误强度计算）
- $R_i \in [0,1]$：稀有度分数（kNN 局部密度）
- $N_i \in [0,1]$：新颖度分数（与历史已选集的距离）
- $c = 0.5$：难度权重
- $a = 0.5$：稀有度权重
- $b_i = b_0 \cdot (1 - D_i)$：新颖度权重（门控机制）

**门控机制解释**：难度高的样本（$D_i$ 大），新颖度权重 $b_i$ 变小。这是因为困难样本本身就值得重复训练，不需要额外强调新颖性。

#### 3.3.2 稀有度计算

稀有度衡量样本在簇内的"边缘程度"。使用 kNN 平均距离作为局部密度的倒数：

$$R_i = \text{Normalize}\left(\frac{1}{k}\sum_{j \in \text{kNN}(i)} d(x_i, x_j)\right)$$

距离大 → 局部稀疏 → 稀有度高

**实现位置**：`hcds/clustering/compression.py::compute_local_density()`

#### 3.3.3 新颖度计算

新颖度衡量样本与**历史已选集**的距离。避免重复选择语义相似的样本：

$$N_i = \text{Normalize}\left(\min_{h \in \text{History}} d(x_i, h)\right)$$

距离大 → 与历史不同 → 新颖度高

**实现位置**：`hcds/sampling/priority.py::PriorityCalculator.compute_novelty()`

#### 3.3.4 混合选择策略

不完全按优先级选择，而是混合三种策略：

| 策略 | 比例 | 目的 |
|-----|------|------|
| 按优先级选择 | 80% | 主要选择高价值样本 |
| 按稀有度选择 | 15% | 确保长尾覆盖 |
| 随机选择 | 5% | 保持探索性 |

**实现位置**：`hcds/sampling/priority.py::PrioritySampler._mixed_selection()`

**关键参数**：

| 参数 | 含义 | 默认值 |
|-----|------|--------|
| `weights.difficulty` (c) | 难度权重 | 0.5 |
| `weights.rarity` (a) | 稀有度权重 | 0.5 |
| `weights.novelty_base` ($b_0$) | 新颖度基础权重 | 0.5 |
| `knn.k` | kNN 的 k 值 | 10 |
| `selection.priority_ratio` | 优先级选择比例 | 0.8 |
| `selection.rare_ratio` | 稀有度选择比例 | 0.15 |

---

### 3.4 Stage 4: 反馈更新

训练完成后，根据每个样本的错误强度更新系统状态。

#### 3.4.1 错误强度计算

三层信号融合：

$$G_i = w_L \cdot \hat{L}_i + w_C \cdot C_i + w_E \cdot \hat{E}_i$$

| 层级 | 信号 | 含义 | 默认权重 |
|-----|------|------|---------|
| Layer 1 | $\hat{L}_i$ | 归一化的训练 Loss | 0.4 |
| Layer 2 | $C_i$ | 正确性（0=对,1=错）| 0.6 |
| Layer 3 | $\hat{E}_i$ | 预测熵 | 0.0 |

**Loss 归一化 (Z-score)**：

$$\hat{L}_i = \text{clip}\left(\frac{L_i - \mu}{\sigma} + 0.5, 0, 1\right)$$

使用在线 running statistics 更新 $\mu, \sigma$。

**实现位置**：`hcds/feedback/error_intensity.py::ErrorIntensityComputer.compute()`

#### 3.4.2 Thompson Sampling 后验更新

将错误强度作为"成功"信号更新 Beta 分布：

$$\alpha_k \leftarrow \alpha_k + \sum_{i \in C_k} G_i$$
$$\beta_k \leftarrow \beta_k + \sum_{i \in C_k} (1 - G_i)$$

错误强度高的簇 → $\alpha$ 增加更多 → 后验均值上升 → 下一轮更可能被选中

**实现位置**：`hcds/sampling/thompson.py::ThompsonSampler.update_batch()`

#### 3.4.3 样本难度更新

使用指数移动平均更新样本难度：

$$D_i^{(t+1)} = \eta \cdot D_i^{(t)} + (1-\eta) \cdot G_i$$

其中 $\eta = 0.7$ 是平滑系数。平滑可以避免单次波动导致难度估计不稳定。

**实现位置**：`hcds/sampling/priority.py::update_sample_difficulty()`

#### 3.4.4 样本退休机制

**目的**：已经学会的样本（连续多轮错误强度低）应该退休，避免过拟合和计算浪费。

**退休条件**：连续 $q$ 轮错误强度 $< \tau$

$$\text{Retire}_i = \mathbb{1}\left[\forall t \in [T-q+1, T]: G_i^{(t)} < \tau\right]$$

**回访机制**：退休样本有小概率 $p_{\text{revisit}}$ 被重新采样，避免永久遗忘。

**实现位置**：`hcds/sampling/retirement.py::RetirementManager.update_batch()`

**关键参数**：

| 参数 | 含义 | 默认值 |
|-----|------|--------|
| `consecutive_threshold` (q) | 连续低错误轮数 | 3 |
| `error_threshold` ($\tau$) | 错误强度阈值 | 0.1 |
| `revisit_probability` ($p_{\text{revisit}}$) | 回访概率 | 0.05 |

---

## 4. 完整执行流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                           离线阶段 (一次性)                           │
├─────────────────────────────────────────────────────────────────────┤
│  1. 加载数据           DataLoader.load()                             │
│  2. 计算嵌入           SentenceTransformerEncoder.encode_batch()     │
│  3. PCA 降维           PCAReducer.fit_transform()                    │
│  4. HDBSCAN 聚类       HDBSCANClusterer.fit()                        │
│  5. 计算簇指标         compute_cluster_metrics()                      │
│  6. 计算先验分数       compute_prior_scores()                         │
│  7. FPS 压缩           ClusterCompressor.compress_all()              │
│  8. 保存结果           ClusterStorage.save()                         │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        在线阶段 (每轮迭代)                            │
├─────────────────────────────────────────────────────────────────────┤
│  for round in 1..T:                                                 │
│                                                                     │
│    [Stage 1] Thompson Sampling 选簇                                  │
│              ThompsonSampler.select()                               │
│              → 选出 K' 个簇                                          │
│                                                                     │
│    [Stage 2] 预算分配                                                │
│              BudgetAllocator.allocate()                             │
│              → 为每个簇分配样本数                                     │
│                                                                     │
│    [Stage 3] 簇内优先级采样                                          │
│              PrioritySampler.sample()                               │
│              → 选出 B 个样本                                         │
│                                                                     │
│    [训练] 用选中的样本训练模型                                        │
│           your_training_function(selected_ids)                      │
│                                                                     │
│    [Stage 4] 反馈更新                                                │
│              ErrorIntensityComputer.compute()                       │
│              ThompsonSampler.update_batch()                         │
│              update_sample_difficulty()                              │
│              RetirementManager.update_batch()                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 核心函数对照表

| 功能 | 函数位置 | 核心公式/算法 |
|-----|---------|--------------|
| 嵌入编码 | `embedding/sentence_transformer.py::encode_batch()` | Transformer 编码 |
| PCA 降维 | `embedding/incremental.py::PCAReducer.fit_transform()` | SVD 分解 |
| HDBSCAN 聚类 | `clustering/hdbscan_cluster.py::fit()` | HDBSCAN |
| 簇指标计算 | `metrics/cluster_metrics.py::compute_cluster_metrics()` | 方差、余弦距离 |
| 先验分数 | `metrics/cluster_metrics.py::compute_prior_scores()` | $S = \lambda_1 V + \lambda_2 G + \lambda_3 I$ |
| FPS 压缩 | `clustering/compression.py::fps_single_cluster_cosine()` | 最远点采样 |
| Thompson Sampling | `sampling/thompson.py::select()` | $\theta \sim \text{Beta}(\alpha, \beta)$ |
| 预算分配 | `sampling/budget.py::allocate()` | 按困难度权重分配 |
| 优先级计算 | `sampling/priority.py::compute_priority()` | $P = cD + (1-c)(aR + bN)$ |
| 稀有度计算 | `clustering/compression.py::compute_local_density()` | kNN 平均距离 |
| 新颖度计算 | `sampling/priority.py::compute_novelty()` | 最近历史距离 |
| 错误强度 | `feedback/error_intensity.py::compute()` | $G = w_L L + w_C C + w_E E$ |
| 后验更新 | `sampling/thompson.py::update_batch()` | $\alpha += G, \beta += 1-G$ |
| 难度更新 | `sampling/priority.py::update_sample_difficulty()` | $D' = \eta D + (1-\eta) G$ |
| 退休判断 | `sampling/retirement.py::update()` | 连续 q 次 $G < \tau$ |

---

## 6. 参数调优建议

### 6.1 离线阶段

| 场景 | 调整建议 |
|-----|---------|
| 样本量 > 100万 | 启用大规模模式，pilot sampling |
| 簇太少 | 降低 `min_cluster_size` |
| 簇太多/碎片化 | 提高 `min_cluster_size`，或合并相似簇 |
| 代表集覆盖不足 | 提高 `max_representatives` |

### 6.2 在线阶段

| 场景 | 调整建议 |
|-----|---------|
| 探索不足（总是选同样的簇）| 增加 `warmup_rounds`，降低 `prior.strength` |
| 利用不足（随机性太大）| 提高 `prior.strength`，降低 `selection.random_ratio` |
| 难度样本训练不够 | 提高 `weights.difficulty` |
| 长尾覆盖不足 | 提高 `selection.rare_ratio` |
| 退休太快 | 降低 `consecutive_threshold`，提高 `error_threshold` |
| 退休太慢 | 提高 `consecutive_threshold`，降低 `error_threshold` |
