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


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, labels=None):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 8, device=input_ids.device)
        logits[..., 2] = 1.0
        return SimpleNamespace(logits=logits)


def _collate(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


def test_diagnostics_eval_metrics_accept_streaming_dataset():
    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=_collate,
        eval_dataset=_StreamingEvalDataset(),
        every_n_steps=1,
    )
    model = _TinyModel().train()
    logs = {}

    callback._run_eval_batch_metrics(model, logs, step=1)

    assert "diag/eval_top1" in logs
    assert "diag/eval_entropy" in logs
    assert model.training
