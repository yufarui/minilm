"""预训练 JSONL（text 字段）策略：委托现有 ``PreprocessPipeline``。"""

from __future__ import annotations

from pathlib import Path

from src.preprocess.strategies.pipeline import PreprocessPipeline, PreprocessPipelineConfig
from src.preprocess.stats_types import PreprocessPipelineStats


class PretrainPreprocessStrategy:
    def __init__(self, cfg: PreprocessPipelineConfig) -> None:
        self._pipe = PreprocessPipeline(cfg)

    def run(self, input_path: Path, output_path: Path) -> PreprocessPipelineStats:
        return self._pipe.run(input_path, output_path)
