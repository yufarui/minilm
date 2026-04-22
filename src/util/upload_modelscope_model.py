from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any

from modelscope.hub.api import HubApi

from src.util.path_util import resolve_under_project


def _first_non_empty_env(names: list[str]) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _filtered_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return only keyword args accepted by callable signature."""
    sig = inspect.signature(callable_obj)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def upload_model_to_modelscope(
    repo_id: str,
    model_dir: str | Path,
    token: str | None = None,
    commit_message: str = "upload trained model",
    create_if_missing: bool = True,
) -> None:
    """Upload local trained model directory to ModelScope model hub."""
    local_dir = resolve_under_project(model_dir).resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"模型目录不存在: {local_dir}")

    config_file = local_dir / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json: {config_file}")

    auth_token = token or _first_non_empty_env(
        ["MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN", "MODEL_SCOPE_TOKEN"]
    )

    api = HubApi()
    if auth_token:
        api.login(auth_token)

    # Optional repo creation for first-time upload.
    if create_if_missing and hasattr(api, "create_model"):
        try:
            create_kwargs = _filtered_kwargs(api.create_model, {"model_id": repo_id})
            api.create_model(**create_kwargs)
        except Exception:
            # Repo may already exist or current SDK may handle auto-create during upload.
            pass

    if hasattr(api, "upload_folder"):
        upload_kwargs = _filtered_kwargs(
            api.upload_folder,
            {
                "repo_id": repo_id,
                "folder_path": str(local_dir),
                "repo_type": "model",
                "commit_message": commit_message,
            },
        )
        if not upload_kwargs:
            raise RuntimeError("当前 ModelScope SDK 的 upload_folder 签名无法识别。")
        api.upload_folder(**upload_kwargs)
        print(f"[OK] uploaded by upload_folder: {local_dir} -> {repo_id}")
        return

    if hasattr(api, "push_model"):
        push_kwargs = _filtered_kwargs(
            api.push_model,
            {
                "model_id": repo_id,
                "model_dir": str(local_dir),
                "commit_message": commit_message,
            },
        )
        if not push_kwargs:
            raise RuntimeError("当前 ModelScope SDK 的 push_model 签名无法识别。")
        api.push_model(**push_kwargs)
        print(f"[OK] uploaded by push_model: {local_dir} -> {repo_id}")
        return

    raise RuntimeError("当前 ModelScope SDK 不包含 upload_folder/push_model，无法上传。")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload trained model directory to ModelScope.")
    parser.add_argument("--repo-id", required=True, help="目标仓库 ID，例如: your_name/minilm-dpo")
    parser.add_argument("--model-dir", required=True, help="本地模型目录（可写相对项目根路径）")
    parser.add_argument(
        "--token",
        default=None,
        help="ModelScope 访问令牌；不传则尝试环境变量 MODELSCOPE_API_TOKEN/MODELSCOPE_TOKEN",
    )
    parser.add_argument(
        "--commit-message",
        default="upload trained model",
        help="提交信息（若 SDK 支持该参数）",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="不尝试先创建仓库（默认会尝试创建，已存在会自动忽略）",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    upload_model_to_modelscope(
        repo_id=args.repo_id,
        model_dir=args.model_dir,
        token=args.token,
        commit_message=args.commit_message,
        create_if_missing=not args.no_create,
    )


if __name__ == "__main__":
    main()
