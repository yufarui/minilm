from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class _LengthlessEvalDataset(IterableDataset):
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        yield from self.rows


class _ToyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, labels, **_kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, 4, device=input_ids.device)
        logits[..., 1] = 10.0
        return SimpleNamespace(logits=logits)


def _collate(features):
    return {
        "input_ids": torch.stack([feature["input_ids"] for feature in features]),
        "labels": torch.stack([feature["labels"] for feature in features]),
    }


def test_eval_batch_metrics_accepts_iterable_dataset_without_len():
    rows = [
        {
            "input_ids": torch.tensor([0, 1, 1], dtype=torch.long),
            "labels": torch.tensor([0, 1, 1], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([2, 1, 1], dtype=torch.long),
            "labels": torch.tensor([2, 1, 1], dtype=torch.long),
        },
    ]
    dataset = _LengthlessEvalDataset(rows)
    with pytest.raises(TypeError):
        len(dataset)

    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=_collate,
        eval_dataset=dataset,
        every_n_steps=1,
    )
    logs = {}

    callback._run_eval_batch_metrics(_ToyCausalModel(), logs, step=1)

    assert logs["diag/eval_top1"] == pytest.approx(1.0)
    assert "diag/eval_entropy" in logs


def test_eval_batch_metrics_accepts_empty_iterable_dataset_without_len():
    dataset = _LengthlessEvalDataset([])
    with pytest.raises(TypeError):
        len(dataset)

    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=_collate,
        eval_dataset=dataset,
        every_n_steps=1,
    )
    logs = {}

    callback._run_eval_batch_metrics(_ToyCausalModel(), logs, step=1)

    assert logs == {}
