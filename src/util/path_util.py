"""仓库根目录下的路径解析（配置、数据、tokenizer 等相对路径）。"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """minilm 项目根目录（含 ``pyproject.toml``、``config/`` 的目录）。"""
    return Path(__file__).resolve().parents[2]


def resolve_under_project(maybe_relative: str | Path) -> Path:
    """相对路径按项目根解析；已是绝对路径则原样返回为 ``Path``。"""
    root = project_root()
    p = Path(maybe_relative)
    return p if p.is_absolute() else (root / p)
