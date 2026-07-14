from types import SimpleNamespace

import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class _StreamingEvalDataset(IterableDataset):
    def __iter__(self):
        for _ in range(2):
            yield {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([0, 0, 0]),
            }


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels):
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(batch_size, sequence_length, 4, device=input_ids.device)
        return SimpleNamespace(logits=logits)


def _collate(features):
    return {
        key: torch.stack([feature[key] for feature in features])
        for key in features[0]
    }


def test_eval_metrics_support_streaming_dataset_without_length():
    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=_collate,
        eval_dataset=_StreamingEvalDataset(),
        num_eval_batches=1,
    )
    model = _Model()
    model.train()
    logs = {}

    callback._run_eval_batch_metrics(model, logs, step=1)

    assert logs["diag/eval_top1"] == 1.0
    assert "diag/eval_entropy" in logs
    assert model.training
