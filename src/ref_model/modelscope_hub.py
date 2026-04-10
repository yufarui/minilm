"""ModelScope 模型解析与下载（返回本地目录路径）。"""

from __future__ import annotations

import os
from pathlib import Path

from src.util.path_util import resolve_under_project


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def resolve_model_dir(model_name: str) -> str:
    """将模型名解析为本地目录；优先本地路径，其次 ModelScope 下载。"""
    p = Path(model_name)
    candidate = p if p.is_absolute() else resolve_under_project(p)
    if (candidate / "config.json").is_file():
        return str(candidate.resolve())

    # 离线模式下禁止远端拉取，要求用户提供本地目录。
    if _env_truthy("MODELSCOPE_OFFLINE") or _env_truthy("HF_HUB_OFFLINE") or _env_truthy("TRANSFORMERS_OFFLINE"):
        raise RuntimeError(
            f"离线模式下无法从 ModelScope 下载模型 {model_name!r}。"
            "请将模型放到项目本地目录（含 config.json），并把配置中的 model_name 改为该路径。"
        )

    from modelscope.hub.snapshot_download import snapshot_download

    # 优先按原始 id 尝试；对无命名空间的常见短名做一次兼容尝试。
    candidates = [model_name]
    if "/" not in model_name:
        candidates.append(f"AI-ModelScope/{model_name}")

    last_error: Exception | None = None
    for rid in candidates:
        try:
            local_dir = snapshot_download(rid, repo_type="model")
            return str(Path(local_dir).resolve())
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue

    raise RuntimeError(
        f"无法从 ModelScope 下载模型 {model_name!r}。"
        f"已尝试: {candidates}。请确认模型 id 存在，或改为本地路径。"
    ) from last_error

