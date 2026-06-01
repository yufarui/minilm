from types import SimpleNamespace

import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class _TinyIterableEvalDataset(IterableDataset):
    def __iter__(self):
        for offset in range(2):
            yield {
                "input_ids": torch.tensor([offset, offset + 1, offset + 2], dtype=torch.long),
                "labels": torch.tensor([offset, offset + 1, offset + 2], dtype=torch.long),
            }


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, 8, device=input_ids.device)
        return SimpleNamespace(logits=logits)


def _collate(features):
    return {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0]
    }


def test_eval_batch_metrics_accepts_iterable_dataset_without_len():
    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=_collate,
        eval_dataset=_TinyIterableEvalDataset(),
        every_n_steps=1,
        num_eval_batches=2,
    )
    model = _TinyModel()
    logs: dict[str, float] = {}

    callback._run_eval_batch_metrics(model, logs, step=1)

    assert "diag/eval_top1" in logs
    assert "diag/eval_entropy" in logs
    assert model.training
