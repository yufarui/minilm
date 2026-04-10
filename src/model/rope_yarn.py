"""YaRN RoPE (https://arxiv.org/abs/2309.00071), aligned with HuggingFace `modeling_rope_utils._compute_yarn_parameters`."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch


def _get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_attention_factor(rope_scaling: Mapping[str, Any]) -> float:
    """mscale applied to rotated Q/K (HF applies the same factor to cos and sin)."""
    factor = float(rope_scaling["factor"])
    af = rope_scaling.get("attention_factor")
    if af is not None:
        return float(af)
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")
    if mscale is not None and mscale_all_dim is not None:
        return float(_get_mscale(factor, float(mscale)) / _get_mscale(factor, float(mscale_all_dim)))
    return float(_get_mscale(factor))


def _find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
    truncate: bool,
) -> tuple[float, float]:
    low = _find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _linear_ramp_factor(low: float, high: float, n: int, device: torch.device) -> torch.Tensor:
    if low == high:
        high = high + 0.001
    linear_func = (torch.arange(n, dtype=torch.float32, device=device) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def compute_yarn_inv_freq(
    rope_theta: float,
    head_dim: int,
    rope_scaling: Mapping[str, Any],
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")

    dim = head_dim
    base = float(rope_theta)
    factor = float(rope_scaling["factor"])
    original_max = int(rope_scaling["original_max_position_embeddings"])
    beta_fast = float(rope_scaling.get("beta_fast") or 32)
    beta_slow = float(rope_scaling.get("beta_slow") or 1)
    truncate = bool(rope_scaling.get("truncate", True))

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = _find_correction_range(beta_fast, beta_slow, dim, base, original_max, truncate)
    ramp = _linear_ramp_factor(low, high, dim // 2, device)
    inv_freq_extrapolation_factor = 1.0 - ramp

    inv_freq = (
        inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq
