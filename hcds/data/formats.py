"""
多格式数据解析器
支持 JSONL, JSON, Parquet, HuggingFace 等格式
"""

import json
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator, Union
from dataclasses import dataclass


class DataFormat(Enum):
    """数据格式枚举"""
    JSONL = "jsonl"
    JSON = "json"
    PARQUET = "parquet"
    HUGGINGFACE = "huggingface"
    CSV = "csv"


# 常见字段名映射
FIELD_ALIASES = {
    "instruction": ["instruction", "prompt", "question", "query", "input_text", "user", "human"],
    "input": ["input", "context", "additional_context", "system"],
    "output": ["output", "response", "answer", "completion", "assistant", "target", "label"],
}


def detect_format(path: Union[str, Path]) -> DataFormat:
    """
    根据文件扩展名检测数据格式

    Args:
        path: 文件路径

    Returns:
        DataFormat 枚举值
    """
    path = Path(path)
    suffix = path.suffix.lower()

    format_map = {
        ".jsonl": DataFormat.JSONL,
        ".json": DataFormat.JSON,
        ".parquet": DataFormat.PARQUET,
        ".csv": DataFormat.CSV,
    }

    # 检查是否是 HuggingFace 数据集
    if not path.exists() and "/" in str(path):
        return DataFormat.HUGGINGFACE

    return format_map.get(suffix, DataFormat.JSONL)


def normalize_field_name(field: str, field_type: str) -> Optional[str]:
    """
    将字段名归一化到标准字段名

    Args:
        field: 原始字段名
        field_type: 目标字段类型 ("instruction", "input", "output")

    Returns:
        如果匹配则返回原始字段名，否则返回 None
    """
    aliases = FIELD_ALIASES.get(field_type, [])
    field_lower = field.lower()

    for alias in aliases:
        if field_lower == alias.lower():
            return field

    return None


def auto_detect_fields(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    自动检测字段映射

    Args:
        sample: 样本字典

    Returns:
        字段映射 {"instruction": "actual_field_name", ...}
    """
    mapping = {}

    for field_type in ["instruction", "input", "output"]:
        for field_name in sample.keys():
            if normalize_field_name(field_name, field_type):
                mapping[field_type] = field_name
                break

    return mapping


@dataclass
class ParsedSample:
    """解析后的样本"""
    id: str
    instruction: str
    input: str
    output: str
    raw: Dict[str, Any]

    def get_content(self, template: str = "{instruction}\n{input}") -> str:
        """根据模板获取内容"""
        content = template.format(
            instruction=self.instruction,
            input=self.input,
            output=self.output
        )
        # 清理多余空白
        content = re.sub(r'\n+', '\n', content).strip()
        return content


def parse_sample(
    raw: Dict[str, Any],
    field_mapping: Dict[str, str],
    sample_id: Optional[str] = None
) -> ParsedSample:
    """
    解析原始样本数据

    Args:
        raw: 原始样本字典
        field_mapping: 字段映射
        sample_id: 样本 ID

    Returns:
        ParsedSample 实例
    """
    instruction_field = field_mapping.get("instruction", "instruction")
    input_field = field_mapping.get("input")
    output_field = field_mapping.get("output", "output")

    instruction = str(raw.get(instruction_field, ""))
    input_text = str(raw.get(input_field, "")) if input_field else ""
    output = str(raw.get(output_field, ""))

    # 生成 ID
    if sample_id is None:
        if "id" in raw:
            sample_id = str(raw["id"])
        elif "_id" in raw:
            sample_id = str(raw["_id"])
        else:
            # 使用内容哈希
            import hashlib
            content = f"{instruction}{input_text}{output}"
            sample_id = hashlib.md5(content.encode()).hexdigest()[:12]

    return ParsedSample(
        id=sample_id,
        instruction=instruction,
        input=input_text,
        output=output,
        raw=raw
    )


class FormatParser:
    """格式解析器基类"""

    def __init__(self, path: Union[str, Path], field_mapping: Optional[Dict[str, str]] = None):
        self.path = Path(path)
        self.field_mapping = field_mapping
        self._auto_detected = False

    def _ensure_field_mapping(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """确保字段映射存在"""
        if self.field_mapping is None:
            self.field_mapping = auto_detect_fields(sample)
            self._auto_detected = True
        return self.field_mapping

    def __iter__(self) -> Iterator[ParsedSample]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class JSONLParser(FormatParser):
    """JSONL 格式解析器"""

    def __iter__(self) -> Iterator[ParsedSample]:
        with open(self.path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    self._ensure_field_mapping(raw)
                    yield parse_sample(raw, self.field_mapping, sample_id=str(idx))
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {idx+1} 行 JSON 解析失败: {e}")
                    continue

    def __len__(self) -> int:
        count = 0
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


class JSONParser(FormatParser):
    """JSON 格式解析器 (数组或对象)"""

    def __init__(self, path: Union[str, Path], field_mapping: Optional[Dict[str, str]] = None):
        super().__init__(path, field_mapping)
        self._data = None

    def _load_data(self) -> List[Dict[str, Any]]:
        if self._data is None:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 支持数组和带有 data 字段的对象
            if isinstance(data, list):
                self._data = data
            elif isinstance(data, dict):
                if "data" in data:
                    self._data = data["data"]
                elif "samples" in data:
                    self._data = data["samples"]
                elif "examples" in data:
                    self._data = data["examples"]
                else:
                    self._data = [data]
            else:
                raise ValueError(f"不支持的 JSON 结构: {type(data)}")

        return self._data

    def __iter__(self) -> Iterator[ParsedSample]:
        data = self._load_data()
        for idx, raw in enumerate(data):
            self._ensure_field_mapping(raw)
            yield parse_sample(raw, self.field_mapping, sample_id=str(idx))

    def __len__(self) -> int:
        return len(self._load_data())


class ParquetParser(FormatParser):
    """Parquet 格式解析器"""

    def __init__(self, path: Union[str, Path], field_mapping: Optional[Dict[str, str]] = None):
        super().__init__(path, field_mapping)
        self._df = None

    def _load_data(self):
        if self._df is None:
            try:
                import pandas as pd
                self._df = pd.read_parquet(self.path)
            except ImportError:
                raise ImportError("请安装 pandas 和 pyarrow: pip install pandas pyarrow")
        return self._df

    def __iter__(self) -> Iterator[ParsedSample]:
        df = self._load_data()
        for idx, row in df.iterrows():
            raw = row.to_dict()
            self._ensure_field_mapping(raw)
            yield parse_sample(raw, self.field_mapping, sample_id=str(idx))

    def __len__(self) -> int:
        return len(self._load_data())


class HuggingFaceParser(FormatParser):
    """HuggingFace 数据集解析器"""

    def __init__(
        self,
        path: str,
        field_mapping: Optional[Dict[str, str]] = None,
        split: str = "train",
        streaming: bool = False
    ):
        super().__init__(path, field_mapping)
        self.split = split
        self.streaming = streaming
        self._dataset = None

    def _load_data(self):
        if self._dataset is None:
            try:
                from datasets import load_dataset
                self._dataset = load_dataset(
                    str(self.path),
                    split=self.split,
                    streaming=self.streaming
                )
            except ImportError:
                raise ImportError("请安装 datasets: pip install datasets")
        return self._dataset

    def __iter__(self) -> Iterator[ParsedSample]:
        dataset = self._load_data()
        for idx, raw in enumerate(dataset):
            self._ensure_field_mapping(raw)
            yield parse_sample(raw, self.field_mapping, sample_id=str(idx))

    def __len__(self) -> int:
        if self.streaming:
            raise ValueError("流式数据集无法获取长度")
        dataset = self._load_data()
        return len(dataset)


class CSVParser(FormatParser):
    """CSV 格式解析器"""

    def __init__(self, path: Union[str, Path], field_mapping: Optional[Dict[str, str]] = None):
        super().__init__(path, field_mapping)
        self._df = None

    def _load_data(self):
        if self._df is None:
            try:
                import pandas as pd
                self._df = pd.read_csv(self.path)
            except ImportError:
                raise ImportError("请安装 pandas: pip install pandas")
        return self._df

    def __iter__(self) -> Iterator[ParsedSample]:
        df = self._load_data()
        for idx, row in df.iterrows():
            raw = row.to_dict()
            self._ensure_field_mapping(raw)
            yield parse_sample(raw, self.field_mapping, sample_id=str(idx))

    def __len__(self) -> int:
        return len(self._load_data())


def get_parser(
    path: Union[str, Path],
    format: Optional[DataFormat] = None,
    field_mapping: Optional[Dict[str, str]] = None,
    **kwargs
) -> FormatParser:
    """
    获取对应格式的解析器

    Args:
        path: 数据路径
        format: 数据格式，None 则自动检测
        field_mapping: 字段映射
        **kwargs: 其他参数

    Returns:
        FormatParser 实例
    """
    if format is None:
        format = detect_format(path)

    parser_map = {
        DataFormat.JSONL: JSONLParser,
        DataFormat.JSON: JSONParser,
        DataFormat.PARQUET: ParquetParser,
        DataFormat.HUGGINGFACE: HuggingFaceParser,
        DataFormat.CSV: CSVParser,
    }

    parser_class = parser_map.get(format)
    if parser_class is None:
        raise ValueError(f"不支持的数据格式: {format}")

    return parser_class(path, field_mapping, **kwargs)
