from types import SimpleNamespace

import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class TinyIterableEvalDataset(IterableDataset):
    def __iter__(self):
        yield {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels, **kwargs):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 8, device=input_ids.device)
        return SimpleNamespace(logits=logits)


def test_eval_batch_metrics_accepts_iterable_dataset_without_len():
    def collate(features):
        return {key: torch.stack([feature[key] for feature in features]) for key in features[0]}

    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=collate,
        eval_dataset=TinyIterableEvalDataset(),
        every_n_steps=1,
    )
    logs = {}

    callback._run_eval_batch_metrics(TinyModel(), logs, step=1)

    assert "diag/eval_top1" in logs
