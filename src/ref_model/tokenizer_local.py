"""本地目录下的 HuggingFace ``AutoTokenizer``（按解析后的绝对路径单例）。"""

from __future__ import annotations

from pathlib import Path

from src.ref_model.registry import get_or_create
from src.util.path_util import resolve_under_project


def get_auto_tokenizer_local(
    pretrained: str | Path,
    *,
    trust_remote_code: bool = True,
    use_fast: bool | None = None,
):
    """
    相对路径按项目根解析；缓存键包含 ``trust_remote_code``、``use_fast``。

    用于预处理 YAML 中的 ``tokenizer_path_for_diagnostics``、tokenizer 评测等本地目录。
    """
    from transformers import AutoTokenizer

    resolved = resolve_under_project(Path(pretrained)).resolve()
    key = ("auto_tokenizer_local", str(resolved), trust_remote_code, use_fast)

    def load():
        kw: dict = {"trust_remote_code": trust_remote_code}
        if use_fast is not None:
            kw["use_fast"] = use_fast
        return AutoTokenizer.from_pretrained(str(resolved), **kw)

    return get_or_create(key, load)
