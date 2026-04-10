"""参考模型单例缓存（线程安全）。"""

from __future__ import annotations

import threading
from typing import Any, Callable, Hashable, TypeVar

T = TypeVar("T")

_lock = threading.RLock()
_store: dict[tuple[Any, ...], Any] = {}


def get_or_create(key: Hashable, factory: Callable[[], T]) -> T:
    """按不可变键懒加载单例；同一进程内复用同一实例。"""
    k = key if isinstance(key, tuple) else (key,)
    with _lock:
        if k not in _store:
            _store[k] = factory()
        return _store[k]  # type: ignore[return-value]


def clear_ref_model_cache() -> None:
    """测试或释放显存前可清空缓存（一般生产流水线无需调用）。"""
    with _lock:
        _store.clear()
