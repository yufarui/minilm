"""Unicode 与换行规范化（后续长度/语言/token 步骤的公共入口）。"""

from __future__ import annotations

import unicodedata


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    s = "\n".join(lines)
    return s.strip()
