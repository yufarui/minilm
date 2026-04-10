"""精确去重（哈希）、近似去重（MinHash+LSH）、文档级辅助。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass
class ExactDedupConfig:
    algorithm: str = "sha1"


@dataclass
class NearDedupConfig:
    enabled: bool = True
    threshold: float = 0.88
    num_perm: int = 128
    shingle_size: int = 5


def content_fingerprint(text: str, algorithm: str = "sha1") -> str:
    raw = text.encode("utf-8", errors="ignore")
    if algorithm.lower() == "md5":
        return hashlib.md5(raw).hexdigest()
    return hashlib.sha1(raw).hexdigest()


def _shingles(text: str, n: int) -> list[bytes]:
    toks = re.findall(r"\S+", text.lower())
    if len(toks) < n:
        return [b" ".join(toks).encode("utf-8", errors="ignore")] if toks else []
    return [
        b" ".join(toks[i : i + n]).encode("utf-8", errors="ignore")
        for i in range(len(toks) - n + 1)
    ]


def iter_unique_by_exact_hash(
    texts: Iterable[str],
    seen: set[str] | None = None,
    algorithm: str = "sha1",
) -> Iterable[tuple[str, str]]:
    """Yield (text, hash_hex) for first occurrence of each content hash."""
    if seen is None:
        seen = set()
    for t in texts:
        h = content_fingerprint(t, algorithm=algorithm)
        if h in seen:
            continue
        seen.add(h)
        yield t, h


def near_dedup_mask(
    texts: list[str],
    cfg: NearDedupConfig,
) -> list[bool]:
    """
    对 texts 顺序扫描，若与已保留文档近似重复则标记 False。
    需要 datasketch。
    """
    if not cfg.enabled:
        return [True] * len(texts)
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as e:
        raise ImportError("近似去重需要 datasketch：pip install minilm[preprocess] 或 pip install datasketch") from e

    lsh = MinHashLSH(threshold=cfg.threshold, num_perm=cfg.num_perm)
    keep: list[bool] = []
    for i, text in enumerate(texts):
        m = MinHash(num_perm=cfg.num_perm)
        for sh in _shingles(text, cfg.shingle_size):
            m.update(sh)
        dup = len(lsh.query(m)) > 0
        if dup:
            keep.append(False)
            continue
        keep.append(True)
        lsh.insert(f"id_{i}", m)
    return keep


def filter_by_mask(texts: list[str], mask: list[bool]) -> list[str]:
    return [t for t, k in zip(texts, mask, strict=True) if k]
