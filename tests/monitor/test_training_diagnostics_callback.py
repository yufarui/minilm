from types import SimpleNamespace

import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class _StreamingEvalDataset(IterableDataset):
    def __iter__(self):
        yield {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3], dtype=torch.long),
        }


class _TinyDiagnosticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            8,
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits)


def _collate(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


def test_training_diagnostics_accepts_iterable_eval_dataset():
    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=_collate,
        eval_dataset=_StreamingEvalDataset(),
        every_n_steps=1,
        num_eval_batches=1,
    )
    logs = {}

    callback._run_eval_batch_metrics(_TinyDiagnosticModel(), logs, step=1)

    assert "diag/eval_top1" in logs
    assert "diag/eval_entropy" in logs
