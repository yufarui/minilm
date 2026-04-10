"""按 ``kind`` 选择预训练或 SFT 预处理策略（共享去重/文本质量实现）。"""

from __future__ import annotations

from src.preprocess.strategies.base import PreprocessStrategy
from src.preprocess.strategies.pretrain import PretrainPreprocessStrategy
from src.preprocess.strategies.sft import SftPreprocessStrategy

__all__ = [
    "PreprocessStrategy",
    "PretrainPreprocessStrategy",
    "SftPreprocessStrategy",
]
