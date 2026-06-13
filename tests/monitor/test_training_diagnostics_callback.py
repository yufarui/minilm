from types import SimpleNamespace

import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class _TinyEvalDataset(IterableDataset):
    def __iter__(self):
        yield {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3], dtype=torch.long),
        }
        yield {
            "input_ids": torch.tensor([1, 3, 4], dtype=torch.long),
            "labels": torch.tensor([1, 3, 4], dtype=torch.long),
        }


class _TinyEvalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 8, device=input_ids.device)
        return SimpleNamespace(logits=logits)


def _collate(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


def test_training_diagnostics_handles_iterable_eval_dataset_without_len():
    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=_collate,
        eval_dataset=_TinyEvalDataset(),
        every_n_steps=1,
    )
    logs = {}

    callback._run_eval_batch_metrics(_TinyEvalModel(), logs, step=1)

    assert "diag/eval_entropy" in logs
