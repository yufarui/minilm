"""On-disk format detection / cleanup for pretrain token corpora."""

from __future__ import annotations

import shutil
from pathlib import Path


def resolve_pretrain_source(data_path: str | Path) -> tuple[str, list[Path]]:
    """Decide which on-disk format ``PreTrainDataset`` should stream.

    Returns ``(source, parquet_files)`` where ``source`` is one of
    ``parquet-shards`` | ``arrow`` | ``parquet`` | ``jsonl``.

    Parquet shard directories take precedence over HuggingFace ``save_to_disk``
    markers. Older ``to_arrow`` wrote HF Arrow (``state.json`` + ``*.arrow``);
    the current writer emits ``part-*.parquet`` and a lightweight
    ``dataset_info.json`` without removing leftover ``state.json``. Preferring
    Arrow whenever ``state.json`` exists would silently train on the obsolete
    corpus after a rebuild.
    """
    p = Path(data_path)
    parquet_files = sorted(p.glob("*.parquet")) if p.is_dir() else []
    has_hf_arrow = (
        p.is_dir()
        and (p / "dataset_info.json").exists()
        and (p / "state.json").exists()
    )
    if parquet_files:
        return "parquet-shards", parquet_files
    if has_hf_arrow:
        return "arrow", []
    if p.is_file() and p.suffix == ".parquet":
        return "parquet", []
    return "jsonl", []


def remove_legacy_hf_disk_artifacts(out_dir: Path) -> list[str]:
    """Drop leftovers from older ``datasets.save_to_disk`` outputs.

    Returns paths (as strings) that were removed.
    """
    removed: list[str] = []
    for name in ("state.json", "dataset_dict.json"):
        path = out_dir / name
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    for arrow in out_dir.glob("*.arrow"):
        arrow.unlink()
        removed.append(str(arrow))
    for child in list(out_dir.iterdir()):
        if child.is_dir() and any(child.glob("*.arrow")):
            shutil.rmtree(child)
            removed.append(str(child))
    return removed
