from types import SimpleNamespace

import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class NoLenEvalDataset(IterableDataset):
    def __iter__(self):
        yield {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "labels": torch.tensor([1, 0, 0], dtype=torch.long),
        }


class TinyMetricModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 4, device=input_ids.device)
        return SimpleNamespace(logits=logits + self.weight)


def stack_collator(features):
    return {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0]
    }


def test_eval_batch_metrics_accepts_iterable_dataset_without_len(monkeypatch):
    monkeypatch.setattr(TrainingDiagnosticsCallback, "_swanlab_log", staticmethod(lambda *_args, **_kwargs: None))
    logs = {}
    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=stack_collator,
        eval_dataset=NoLenEvalDataset(),
        every_n_steps=1,
    )

    callback._run_eval_batch_metrics(TinyMetricModel(), logs, step=1)

    assert logs["diag/eval_top1"] == 1.0
    assert "diag/eval_entropy" in logs
