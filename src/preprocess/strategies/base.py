"""预处理策略：预训练与 SFT 共用同一入口协议。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PreprocessStrategy(Protocol):
    """由任务 YAML 的 ``kind`` 选择具体实现。"""

    def run(self, input_path: Path, output_path: Path) -> Any:
        """执行流水线并返回可 ``to_json_dict`` 或可直接日志打印的统计对象。"""
        ...
