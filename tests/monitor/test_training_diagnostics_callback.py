from torch.utils.data import IterableDataset

from src.monitor.common.training_diagnostics_callback import _eval_probe_batch_size


class _StreamingDataset(IterableDataset):
    def __iter__(self):
        yield {"input_ids": [1], "labels": [1]}


def test_eval_probe_batch_size_handles_iterable_dataset_without_len():
    assert _eval_probe_batch_size(_StreamingDataset()) == 4


def test_eval_probe_batch_size_skips_empty_sized_dataset():
    assert _eval_probe_batch_size([]) is None


def test_eval_probe_batch_size_caps_sized_dataset_to_maximum():
    assert _eval_probe_batch_size([1, 2, 3, 4, 5], max_batch_size=3) == 3
