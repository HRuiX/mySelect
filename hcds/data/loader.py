"""
统一数据加载接口
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator, Union
from dataclasses import dataclass, field
import numpy as np

from hcds.data.formats import (
    DataFormat, detect_format, get_parser, ParsedSample
)
from hcds.config.schema import DataConfig


@dataclass
class Sample:
    """训练样本数据结构"""
    id: str
    content: Dict[str, str]  # {"instruction": ..., "input": ..., "output": ...}

    # 嵌入向量 (离线计算后填充)
    embedding: Optional[np.ndarray] = None

    # 聚类信息 (离线计算后填充)
    cluster_id: Optional[int] = None
    is_representative: bool = False

    # 动态属性 (在线更新)
    difficulty: float = 0.5
    rarity: float = 0.5
    novelty: float = 1.0

    # 统计信息
    times_selected: int = 0
    last_error_intensity: float = 0.5
    consecutive_low_error: int = 0
    retired: bool = False

    # 原始数据 (可选保留)
    raw: Optional[Dict[str, Any]] = None

    def get_text(self, template: str = "{instruction}\n{input}") -> str:
        """根据模板获取文本内容"""
        return template.format(**self.content).strip()

    def get_full_text(self) -> str:
        """获取完整文本 (含输出)"""
        return f"{self.content.get('instruction', '')}\n{self.content.get('input', '')}\n{self.content.get('output', '')}".strip()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (不含嵌入向量)"""
        return {
            "id": self.id,
            "content": self.content,
            "cluster_id": self.cluster_id,
            "is_representative": self.is_representative,
            "difficulty": self.difficulty,
            "rarity": self.rarity,
            "novelty": self.novelty,
            "times_selected": self.times_selected,
            "last_error_intensity": self.last_error_intensity,
            "consecutive_low_error": self.consecutive_low_error,
            "retired": self.retired,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Sample':
        """从字典创建"""
        return cls(
            id=data["id"],
            content=data["content"],
            cluster_id=data.get("cluster_id"),
            is_representative=data.get("is_representative", False),
            difficulty=data.get("difficulty", 0.5),
            rarity=data.get("rarity", 0.5),
            novelty=data.get("novelty", 1.0),
            times_selected=data.get("times_selected", 0),
            last_error_intensity=data.get("last_error_intensity", 0.5),
            consecutive_low_error=data.get("consecutive_low_error", 0),
            retired=data.get("retired", False),
        )


class DataLoader:
    """统一数据加载器"""

    def __init__(
        self,
        config: DataConfig,
        cache_in_memory: bool = True,
        verbose: bool = True
    ):
        """
        初始化数据加载器

        Args:
            config: 数据配置
            cache_in_memory: 是否缓存到内存
            verbose: 是否输出详细信息
        """
        self.config = config
        self.cache_in_memory = cache_in_memory
        self.verbose = verbose

        self._samples: Optional[Dict[str, Sample]] = None
        self._sample_ids: Optional[List[str]] = None

        # 字段映射
        self._field_mapping = {
            "instruction": config.field_mapping.instruction,
            "input": config.field_mapping.input,
            "output": config.field_mapping.output,
        }
        # 移除 None 值
        self._field_mapping = {k: v for k, v in self._field_mapping.items() if v}

    def load(self) -> Dict[str, Sample]:
        """
        加载数据

        Returns:
            样本字典 {sample_id: Sample}
        """
        if self._samples is not None:
            return self._samples

        if self.verbose:
            print(f"正在加载数据: {self.config.path}")

        # 获取解析器
        format_type = DataFormat(self.config.format)
        parser = get_parser(
            self.config.path,
            format=format_type,
            field_mapping=self._field_mapping
        )

        samples = {}
        sample_ids = []

        for idx, parsed in enumerate(parser):
            # 长度过滤
            content_len = len(parsed.instruction) + len(parsed.input) + len(parsed.output)
            if content_len < self.config.min_length:
                continue
            if content_len > self.config.max_length:
                continue

            # 创建 Sample
            sample = Sample(
                id=parsed.id,
                content={
                    "instruction": parsed.instruction,
                    "input": parsed.input,
                    "output": parsed.output,
                },
                raw=parsed.raw if self.cache_in_memory else None,
            )

            samples[sample.id] = sample
            sample_ids.append(sample.id)

            # 最大样本数限制
            if self.config.max_samples and len(samples) >= self.config.max_samples:
                break

        if self.verbose:
            print(f"加载完成: {len(samples)} 个样本")

        if self.cache_in_memory:
            self._samples = samples
            self._sample_ids = sample_ids

        return samples

    def get_sample(self, sample_id: str) -> Optional[Sample]:
        """获取单个样本"""
        if self._samples is None:
            self.load()
        return self._samples.get(sample_id)

    def get_samples(self, sample_ids: List[str]) -> List[Sample]:
        """获取多个样本"""
        if self._samples is None:
            self.load()
        return [self._samples[sid] for sid in sample_ids if sid in self._samples]

    def get_all_ids(self) -> List[str]:
        """获取所有样本 ID"""
        if self._sample_ids is None:
            self.load()
        return self._sample_ids

    def get_texts(
        self,
        sample_ids: Optional[List[str]] = None,
        template: Optional[str] = None
    ) -> List[str]:
        """
        获取样本文本

        Args:
            sample_ids: 样本 ID 列表，None 则获取全部
            template: 内容模板

        Returns:
            文本列表
        """
        if self._samples is None:
            self.load()

        if template is None:
            template = self.config.content_template

        if sample_ids is None:
            sample_ids = self._sample_ids

        return [self._samples[sid].get_text(template) for sid in sample_ids]

    def __len__(self) -> int:
        if self._samples is None:
            self.load()
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        if self._samples is None:
            self.load()
        for sid in self._sample_ids:
            yield self._samples[sid]

    def save_state(self, path: Union[str, Path]) -> None:
        """
        保存样本状态 (不含嵌入向量)

        Args:
            path: 保存路径
        """
        if self._samples is None:
            raise ValueError("没有已加载的数据")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "sample_ids": self._sample_ids,
            "samples": {sid: s.to_dict() for sid, s in self._samples.items()},
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self, path: Union[str, Path]) -> None:
        """
        加载样本状态

        Args:
            path: 状态文件路径
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"状态文件不存在: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self._sample_ids = state["sample_ids"]
        self._samples = {
            sid: Sample.from_dict(data)
            for sid, data in state["samples"].items()
        }

    def update_sample(self, sample_id: str, **updates) -> None:
        """
        更新样本属性

        Args:
            sample_id: 样本 ID
            **updates: 要更新的属性
        """
        if self._samples is None:
            self.load()

        if sample_id not in self._samples:
            raise KeyError(f"样本不存在: {sample_id}")

        sample = self._samples[sample_id]
        for key, value in updates.items():
            if hasattr(sample, key):
                setattr(sample, key, value)

    def batch_update(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """
        批量更新样本

        Args:
            updates: {sample_id: {attr: value, ...}, ...}
        """
        for sample_id, sample_updates in updates.items():
            self.update_sample(sample_id, **sample_updates)

    def get_active_samples(self) -> List[Sample]:
        """获取未退休的活跃样本"""
        if self._samples is None:
            self.load()
        return [s for s in self._samples.values() if not s.retired]

    def get_cluster_samples(self, cluster_id: int) -> List[Sample]:
        """获取指定簇的样本"""
        if self._samples is None:
            self.load()
        return [s for s in self._samples.values() if s.cluster_id == cluster_id]

    def get_representative_samples(self, cluster_id: Optional[int] = None) -> List[Sample]:
        """获取代表样本"""
        if self._samples is None:
            self.load()

        samples = self._samples.values()
        if cluster_id is not None:
            samples = [s for s in samples if s.cluster_id == cluster_id]

        return [s for s in samples if s.is_representative]
