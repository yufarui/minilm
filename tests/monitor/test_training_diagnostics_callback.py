from types import SimpleNamespace

import torch
from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import TrainingDiagnosticsCallback


class _EvalIterable(IterableDataset):
    def __iter__(self):
        for _ in range(3):
            yield {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "labels": torch.tensor([1, 2, 3], dtype=torch.long),
            }


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, labels=None, **_kwargs):
        logits = torch.zeros((*input_ids.shape, 8), device=input_ids.device)
        logits[:, :-1, :].scatter_(2, input_ids[:, 1:].unsqueeze(-1), 1.0)
        return SimpleNamespace(logits=logits)


def _collate(features):
    return {key: torch.stack([item[key] for item in features]) for key in features[0]}


def test_diagnostics_eval_metrics_accept_iterable_dataset_without_len() -> None:
    callback = TrainingDiagnosticsCallback(
        tokenizer=None,
        data_collator=_collate,
        eval_dataset=_EvalIterable(),
        every_n_steps=1,
    )
    logs: dict[str, float] = {}

    callback._run_eval_batch_metrics(_TinyModel(), logs, step=1)

    assert logs["diag/eval_top1"] == 1.0
    assert "diag/eval_entropy" in logs
