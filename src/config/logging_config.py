"""
应用日志：默认写入项目根目录下的 ``logs/``，可选同时输出到控制台。

在程序入口尽早调用一次 ``setup_logging()``；各模块使用 ``logging.getLogger(__name__)`` 即可。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_project_root() -> Path:
    """``src/config/logging_config.py`` → 仓库根目录 ``minilm/``。"""
    return Path(__file__).resolve().parents[2]


def setup_logging(
        level: int = logging.INFO,
        log_dir: Path | str | None = None,
        log_filename: str = "minilm.log",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        console: bool = True,
        root: bool = True,
) -> logging.Logger:
    """
    配置日志。

    :param log_dir: 日志目录，默认 ``<项目根>/logs``
    :param root: True 时配置根 logger，全库 ``logging.getLogger(__name__)`` 生效；
                 False 时仅返回名为 ``minilm`` 的 logger。
    """
    base = Path(log_dir) if log_dir is not None else get_project_root() / "logs"
    base.mkdir(parents=True, exist_ok=True)
    log_path = base / log_filename

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger() if root else logging.getLogger("minilm")
    logger.setLevel(level)
    if not root:
        logger.propagate = False

    # 避免重复注册 handler（多次调用 setup_logging / 重载时）
    for h in list(logger.handlers):
        if isinstance(h, (RotatingFileHandler, logging.StreamHandler)) and getattr(
                h, "_minilm_configured", False
        ):
            logger.removeHandler(h)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    file_handler._minilm_configured = True
    logger.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(level)
        stream.setFormatter(fmt)
        stream._minilm_configured = True
        logger.addHandler(stream)

    return logger
