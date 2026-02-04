"""
错误强度计算器
三层体系: Loss + Correctness + Entropy
"""

from typing import Dict, List, Optional, Any
import numpy as np

from hcds.config.schema import ErrorIntensityConfig, NormalizationMethod


class RunningStatistics:
    """在线 running statistics (用于 Z-score 归一化)"""

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, values: np.ndarray) -> None:
        """更新统计量"""
        batch_mean = values.mean()
        batch_var = values.var()

        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1 - self.momentum) * batch_var

        self.count += len(values)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Z-score 归一化"""
        std = np.sqrt(self.var + 1e-12)
        return (values - self.mean) / std


class ErrorIntensityComputer:
    """错误强度计算器"""

    def __init__(self, config: ErrorIntensityConfig):
        """
        初始化计算器

        Args:
            config: 错误强度配置
        """
        self.config = config

        # 权重
        self.w_loss = config.weights.get("loss", 0.4)
        self.w_correctness = config.weights.get("correctness", 0.6)
        self.w_entropy = config.weights.get("entropy", 0.0)

        # Loss 归一化
        self.normalization = config.normalization.method
        self.zscore_offset = config.normalization.zscore_offset
        self._loss_stats = RunningStatistics(momentum=config.normalization.running_momentum)

        # 归一化器
        self._normalizer = self._create_normalizer()

    def _create_normalizer(self):
        """创建归一化函数"""
        if self.normalization == NormalizationMethod.ZSCORE:
            return self._normalize_zscore
        elif self.normalization == NormalizationMethod.MINMAX:
            return self._normalize_minmax
        elif self.normalization == NormalizationMethod.PERCENTILE:
            return self._normalize_percentile
        elif self.normalization == NormalizationMethod.SIGMOID:
            return self._normalize_sigmoid
        else:
            return self._normalize_zscore

    def compute(
        self,
        losses: Optional[np.ndarray] = None,
        correctness: Optional[np.ndarray] = None,
        entropies: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        计算综合错误强度

        Args:
            losses: 训练 loss [N]
            correctness: 正确性分数 [N] (0=正确, 1=错误)
            entropies: 预测熵 [N]
            n_samples: 样本数量 (用于检查)

        Returns:
            错误强度 [N] ∈ [0, 1]
        """
        # 确定样本数
        if n_samples is None:
            for arr in [losses, correctness, entropies]:
                if arr is not None:
                    n_samples = len(arr)
                    break
        if n_samples is None:
            raise ValueError("至少需要提供一种信号")

        result = np.zeros(n_samples)
        total_weight = 0.0

        # Layer 1: Loss
        if losses is not None and self.w_loss > 0:
            normalized_loss = self._normalize_loss(losses)
            result += self.w_loss * normalized_loss
            total_weight += self.w_loss

        # Layer 2: Correctness
        if correctness is not None and self.w_correctness > 0:
            correctness = np.clip(correctness, 0, 1)
            result += self.w_correctness * correctness
            total_weight += self.w_correctness

        # Layer 3: Entropy
        if entropies is not None and self.w_entropy > 0:
            normalized_entropy = self._normalize_entropy(entropies)
            result += self.w_entropy * normalized_entropy
            total_weight += self.w_entropy

        # 归一化权重
        if total_weight > 0:
            result /= total_weight

        return np.clip(result, 0, 1)

    def _normalize_loss(self, losses: np.ndarray) -> np.ndarray:
        """归一化 loss"""
        # 更新 running statistics
        self._loss_stats.update(losses)

        return self._normalizer(losses)

    def _normalize_zscore(self, values: np.ndarray) -> np.ndarray:
        """Z-score 归一化: clip((x - μ) / σ + offset, 0, 1)"""
        z_scores = self._loss_stats.normalize(values)
        return np.clip(z_scores + self.zscore_offset, 0, 1)

    def _normalize_minmax(self, values: np.ndarray) -> np.ndarray:
        """Min-Max 归一化"""
        v_min, v_max = values.min(), values.max()
        if v_max > v_min:
            return (values - v_min) / (v_max - v_min)
        return np.zeros_like(values)

    def _normalize_percentile(self, values: np.ndarray) -> np.ndarray:
        """百分位归一化"""
        ranks = np.argsort(np.argsort(values)).astype(float)
        return ranks / (len(values) - 1 + 1e-12)

    def _normalize_sigmoid(self, values: np.ndarray) -> np.ndarray:
        """Sigmoid 归一化"""
        z_scores = self._loss_stats.normalize(values)
        return 1 / (1 + np.exp(-z_scores))

    def _normalize_entropy(self, entropies: np.ndarray) -> np.ndarray:
        """归一化 entropy"""
        if entropies.max() > entropies.min():
            return (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-12)
        return np.zeros_like(entropies)

    def compute_from_model_output(
        self,
        logits: Any,
        labels: Any,
        sample_losses: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        从模型输出计算错误强度

        Args:
            logits: 模型输出的 logits (PyTorch tensor)
            labels: 标签 (PyTorch tensor)
            sample_losses: 每个样本的 loss (如果已计算)

        Returns:
            错误强度 [N]
        """
        import torch

        losses = None
        entropies = None

        # 计算 per-sample loss
        if sample_losses is not None:
            losses = sample_losses
        elif logits is not None and labels is not None:
            with torch.no_grad():
                # Cross-entropy loss per sample
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                if logits.dim() == 3:
                    # Sequence-level: [B, T, V] -> mean over T
                    B, T, V = logits.shape
                    flat_logits = logits.view(-1, V)
                    flat_labels = labels.view(-1)
                    per_token_loss = loss_fn(flat_logits, flat_labels)
                    losses = per_token_loss.view(B, T).mean(dim=1).cpu().numpy()
                else:
                    losses = loss_fn(logits, labels).cpu().numpy()

        # 计算 entropy
        if self.w_entropy > 0 and logits is not None:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                token_entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                if token_entropy.dim() == 2:
                    entropies = token_entropy.mean(dim=1).cpu().numpy()
                else:
                    entropies = token_entropy.cpu().numpy()

        return self.compute(losses=losses, entropies=entropies)

    def save_state(self) -> Dict:
        """保存状态"""
        return {
            "loss_stats": {
                "mean": self._loss_stats.mean,
                "var": self._loss_stats.var,
                "count": self._loss_stats.count
            }
        }

    def load_state(self, state: Dict) -> None:
        """加载状态"""
        if "loss_stats" in state:
            self._loss_stats.mean = state["loss_stats"]["mean"]
            self._loss_stats.var = state["loss_stats"]["var"]
            self._loss_stats.count = state["loss_stats"]["count"]
