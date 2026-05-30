from types import SimpleNamespace

import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class StreamingEvalDataset(IterableDataset):
    def __iter__(self):
        yield {
            "input_ids": torch.tensor([0, 1, 2], dtype=torch.long),
            "labels": torch.tensor([0, 1, 2], dtype=torch.long),
        }


def stack_collator(features):
    return {
        "input_ids": torch.stack([feature["input_ids"] for feature in features]),
        "labels": torch.stack([feature["labels"] for feature in features]),
    }


class TinyDiagnosticsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels=None):
        logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
        return SimpleNamespace(logits=logits)


def test_eval_batch_metrics_supports_streaming_dataset_without_len():
    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=stack_collator,
        eval_dataset=StreamingEvalDataset(),
        every_n_steps=1,
    )
    logs = {}

    callback._run_eval_batch_metrics(TinyDiagnosticsModel(), logs, step=1)

    assert "diag/eval_top1" in logs
    assert "diag/eval_entropy" in logs
