from types import SimpleNamespace

import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class _EvalStream(IterableDataset):
    def __iter__(self):
        yield {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3], dtype=torch.long),
        }


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> SimpleNamespace:
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(
            batch_size,
            seq_len,
            8,
            dtype=torch.float32,
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits)


def _collate(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


def test_diagnostics_eval_batch_metrics_accepts_iterable_dataset_without_len():
    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=_collate,
        eval_dataset=_EvalStream(),
        every_n_steps=1,
        num_eval_batches=1,
    )
    logs = {}

    callback._run_eval_batch_metrics(_TinyModel(), logs, step=1)

    assert "diag/eval_top1" in logs
    assert "diag/eval_entropy" in logs
