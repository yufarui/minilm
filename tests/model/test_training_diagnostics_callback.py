from types import SimpleNamespace

import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class NoLenEvalDataset(IterableDataset):
    def __iter__(self):
        yield {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3], dtype=torch.long),
        }
        yield {
            "input_ids": torch.tensor([2, 3, 4], dtype=torch.long),
            "labels": torch.tensor([2, 3, 4], dtype=torch.long),
        }


class DummyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, 8, device=input_ids.device)
        return SimpleNamespace(logits=logits)


def _collate(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }


def test_training_diagnostics_supports_iterable_eval_dataset_without_len():
    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=_collate,
        eval_dataset=NoLenEvalDataset(),
        every_n_steps=1,
    )
    callback._swanlab_log = lambda *_args, **_kwargs: None

    logs = {}
    callback._run_eval_batch_metrics(DummyCausalLM(), logs, step=1)

    assert "diag/eval_top1" in logs
    assert "diag/eval_entropy" in logs
