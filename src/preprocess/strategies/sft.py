"""SFT 对话 JSONL（conversations）策略：委托 ``SftPreprocessPipeline``。"""

from __future__ import annotations

from pathlib import Path

from src.preprocess.strategies.sft_pipeline import SftPipelineConfig, SftPreprocessPipeline
from src.preprocess.stats_types import SftPreprocessStats


class SftPreprocessStrategy:
    def __init__(self, cfg: SftPipelineConfig) -> None:
        self._pipe = SftPreprocessPipeline(cfg)

    def run(self, input_path: Path, output_path: Path) -> SftPreprocessStats:
        return self._pipe.run(input_path, output_path)
