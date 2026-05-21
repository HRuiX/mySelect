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
    根据文件路径检测数据格式

    支持:
    - 本地文件: 根据扩展名判断
    - HuggingFace 数据集: hf:// 前缀或包含 / 的非本地路径

    Args:
        path: 文件路径或数据集名称

    Returns:
        DataFormat 枚举值
    """
    path_str = str(path)

    # 检查是否是 HuggingFace 数据集
    if path_str.startswith("hf://"):
        return DataFormat.HUGGINGFACE

    # 检查是否是 HuggingFace 风格的路径 (如 "username/dataset")
    if "/" in path_str and not Path(path_str).exists():
        # 排除本地相对路径
        if not path_str.startswith("./") and not path_str.startswith("../"):
            return DataFormat.HUGGINGFACE

    path = Path(path_str)
    suffix = path.suffix.lower()

    format_map = {
        ".jsonl": DataFormat.JSONL,
        ".json": DataFormat.JSON,
        ".parquet": DataFormat.PARQUET,
        ".csv": DataFormat.CSV,
    }

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
    # 检查是否是 messages 格式 (如 LESS 数据集)
    if "messages" in raw and isinstance(raw["messages"], list):
        # 从 messages 中提取 user/assistant 内容
        instruction = ""
        output = ""
        for msg in raw["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                instruction = content
            elif role == "assistant":
                output = content
        input_text = ""
    else:
        # 标准字段映射
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
    """
    HuggingFace 数据集解析器

    支持多种路径格式:
    - "dataset_name" (如 "maveriq/bigbenchhard")
    - "hf://dataset_name" (带前缀)
    - "hf://dataset_name:config" (指定配置/子集)
    - "hf://dataset_name:config:split" (指定配置和分割)

    常用数据集:
    - BBH: "maveriq/bigbenchhard" 或 "lukaemon/bbh"
    - MMLU: "cais/mmlu"
    - LESS 数据: "princeton-nlp/less_data"
    - TydiQA: "google-research-datasets/tydiqa"
    """

    def __init__(
        self,
        path: str,
        field_mapping: Optional[Dict[str, str]] = None,
        split: str = "train",
        subset: Optional[str] = None,
        streaming: bool = False,
        trust_remote_code: bool = True,
    ):
        # 解析路径格式
        path, parsed_subset, parsed_split = self._parse_hf_path(path)
        super().__init__(path, field_mapping)

        self.split = parsed_split or split
        self.subset = parsed_subset or subset
        self.streaming = streaming
        self.trust_remote_code = trust_remote_code
        self._dataset = None
        self._length = None

    @staticmethod
    def _parse_hf_path(path: str) -> tuple:
        """
        解析 HuggingFace 路径

        支持格式:
        - "dataset_name"
        - "hf://dataset_name"
        - "hf://dataset_name:subset"
        - "hf://dataset_name:subset:split"

        Returns:
            (dataset_name, subset, split)
        """
        # 移除 hf:// 前缀
        if path.startswith("hf://"):
            path = path[5:]

        parts = path.split(":")
        dataset_name = parts[0]
        subset = parts[1] if len(parts) > 1 else None
        split = parts[2] if len(parts) > 2 else None

        return dataset_name, subset, split

    def _load_data(self):
        if self._dataset is None:
            try:
                from datasets import load_dataset, concatenate_datasets
            except ImportError:
                raise ImportError("请安装 datasets: pip install datasets")

            dataset_path = str(self.path)
            print(f"从 HuggingFace 加载数据集: {dataset_path}")
            if self.subset:
                print(f"  子集/配置: {self.subset}")
            print(f"  分割: {self.split}")

            try:
                # 尝试加载指定子集
                if self.subset:
                    self._dataset = load_dataset(
                        dataset_path,
                        self.subset,
                        split=self.split,
                        streaming=self.streaming,
                        trust_remote_code=self.trust_remote_code,
                    )
                else:
                    # 尝试直接加载
                    try:
                        self._dataset = load_dataset(
                            dataset_path,
                            split=self.split,
                            streaming=self.streaming,
                            trust_remote_code=self.trust_remote_code,
                        )
                    except ValueError as e:
                        # 可能需要指定配置，尝试加载所有配置并合并
                        if "Config name is missing" in str(e) or "specify a configuration" in str(e):
                            print(f"  数据集需要指定配置，尝试加载所有配置...")
                            self._dataset = self._load_all_configs(dataset_path)
                        else:
                            raise

                if not self.streaming:
                    self._length = len(self._dataset)
                    print(f"  加载完成: {self._length} 个样本")

            except Exception as e:
                print(f"加载 HuggingFace 数据集失败: {e}")
                raise

        return self._dataset

    def _load_all_configs(self, dataset_path: str):
        """加载所有配置并合并"""
        from datasets import load_dataset, concatenate_datasets, get_dataset_config_names

        try:
            configs = get_dataset_config_names(dataset_path, trust_remote_code=self.trust_remote_code)
            print(f"  发现 {len(configs)} 个配置: {configs[:5]}{'...' if len(configs) > 5 else ''}")

            datasets = []
            for config in configs:
                try:
                    ds = load_dataset(
                        dataset_path,
                        config,
                        split=self.split,
                        trust_remote_code=self.trust_remote_code,
                    )
                    # 添加配置名作为列
                    ds = ds.add_column("_config", [config] * len(ds))
                    datasets.append(ds)
                except Exception as e:
                    print(f"  跳过配置 {config}: {e}")
                    continue

            if not datasets:
                raise ValueError(f"无法加载任何配置: {dataset_path}")

            return concatenate_datasets(datasets)

        except Exception as e:
            print(f"  无法获取配置列表: {e}")
            raise

    def __iter__(self) -> Iterator[ParsedSample]:
        dataset = self._load_data()
        for idx, raw in enumerate(dataset):
            # 将 HF dataset 行转换为字典
            if hasattr(raw, 'items'):
                raw_dict = dict(raw)
            else:
                raw_dict = raw

            self._ensure_field_mapping(raw_dict)
            yield parse_sample(raw_dict, self.field_mapping, sample_id=str(idx))

    def __len__(self) -> int:
        if self.streaming:
            raise ValueError("流式数据集无法获取长度")
        if self._length is None:
            self._load_data()
        return self._length

    def get_dataset(self):
        """直接获取 HuggingFace Dataset 对象"""
        return self._load_data()


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
    split: str = "train",
    subset: Optional[str] = None,
    streaming: bool = False,
    **kwargs
) -> FormatParser:
    """
    获取对应格式的解析器

    Args:
        path: 数据路径 (本地文件或 HuggingFace 数据集名)
        format: 数据格式，None 则自动检测
        field_mapping: 字段映射
        split: 数据分割 (用于 HuggingFace 数据集)
        subset: 子集/配置名 (用于 HuggingFace 数据集)
        streaming: 是否流式加载 (用于 HuggingFace 数据集)
        **kwargs: 其他参数

    Returns:
        FormatParser 实例

    Examples:
        # 本地 JSONL 文件
        parser = get_parser("data/train.jsonl")

        # HuggingFace 数据集
        parser = get_parser("maveriq/bigbenchhard", split="train")

        # 带子集的 HuggingFace 数据集
        parser = get_parser("hf://cais/mmlu:college_physics", split="test")

        # 使用字段映射
        parser = get_parser(
            "princeton-nlp/less_data",
            field_mapping={"instruction": "input", "output": "target"}
        )
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

    # HuggingFace 解析器需要额外参数
    if format == DataFormat.HUGGINGFACE:
        return parser_class(
            path,
            field_mapping,
            split=split,
            subset=subset,
            streaming=streaming,
            **kwargs
        )

    return parser_class(path, field_mapping, **kwargs)
