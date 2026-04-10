"""Distributed helpers for training callbacks."""

from __future__ import annotations

import os

import torch.distributed as dist


def is_main_process() -> bool:
    """
    True for single-process runs or global rank 0 when distributed is initialized.

    Used so SwanLab / ``logger`` / expensive diagnostic forwards run only once.
    """
    if not dist.is_available():
        return True
    if dist.is_initialized():
        return dist.get_rank() == 0
    # e.g. deepspeed launcher set WORLD_SIZE before init in some phases
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return True
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return rank == 0
