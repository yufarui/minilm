"""Unit tests for train/val split invariants (never empty the training set)."""

from __future__ import annotations

import pytest

from src.preprocess.split_dataset import split_pretrain_train_val, split_sft_train_and_eval_sets


def test_pretrain_split_rejects_empty_train():
    rows = [{"text": f"doc {i}"} for i in range(3)]
    with pytest.raises(ValueError, match="empty training set"):
        split_pretrain_train_val(rows, val_size=3, seed=0)
    with pytest.raises(ValueError, match="empty training set"):
        split_pretrain_train_val(rows, val_size=5000, seed=0)


def test_pretrain_split_keeps_train_when_val_smaller():
    rows = [{"text": f"doc {i}"} for i in range(10)]
    train_rows, val_rows = split_pretrain_train_val(rows, val_size=3, seed=42)
    assert len(train_rows) == 7
    assert len(val_rows) == 3
    assert {id(r) for r in train_rows}.isdisjoint({id(r) for r in val_rows})


def test_pretrain_split_zero_val_returns_all_train():
    rows = [{"text": "a"}, {"text": "b"}]
    train_rows, val_rows = split_pretrain_train_val(rows, val_size=0, seed=0)
    assert train_rows == rows
    assert val_rows == []


def _multi_turn(i: int) -> dict:
    return {
        "id": i,
        "conversations": [
            {"role": "user", "content": f"q{i}-1"},
            {"role": "assistant", "content": f"a{i}-1"},
            {"role": "user", "content": f"q{i}-2"},
            {"role": "assistant", "content": f"a{i}-2"},
        ],
    }


def test_sft_split_rejects_when_val_buckets_consume_all_rows():
    # Default-style sizes on a mini all-multi-turn corpus.
    rows = [_multi_turn(i) for i in range(20)]
    with pytest.raises(ValueError, match="empty training set"):
        split_sft_train_and_eval_sets(
            rows,
            tool_val_size=1000,
            multi_turn_val_size=5000,
            seed=42,
        )


def test_sft_split_keeps_train_when_budget_allows():
    rows = [_multi_turn(i) for i in range(20)]
    train_rows, tool_rows, multi_rows = split_sft_train_and_eval_sets(
        rows,
        tool_val_size=0,
        multi_turn_val_size=5,
        seed=42,
    )
    assert len(tool_rows) == 0
    assert len(multi_rows) == 5
    assert len(train_rows) == 15
