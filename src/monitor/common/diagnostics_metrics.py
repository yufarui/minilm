"""标量诊断指标（注意力熵、各层 hidden 统计等）。"""

from __future__ import annotations

from typing import Dict, List

import torch


def summarize_hidden_states(hidden_states: tuple[torch.Tensor, ...]) -> Dict[str, float]:
    """对各层 decoder 输出 (B, S, D) 记录均值、标准差、RMS；并给出跨层 RMS 均值。"""
    out: Dict[str, float] = {}
    rms_list: List[torch.Tensor] = []
    for i, h in enumerate(hidden_states):
        if h is None:
            continue
        x = h.detach().float()
        out[f"hidden_mean/layer_{i}"] = float(x.mean().item())
        out[f"hidden_std/layer_{i}"] = float(x.std().item())
        rms = x.pow(2).mean().sqrt()
        out[f"hidden_rms/layer_{i}"] = float(rms.item())
        rms_list.append(rms)
    if rms_list:
        out["hidden_rms/mean_over_layers"] = float(torch.stack(rms_list).mean().item())
    return out


def attention_head_entropy(attn_weights: torch.Tensor, dim_keys: int = -1) -> torch.Tensor:
    """(B, H, Q, K) 已 softmax，对 key 维熵后在 batch/head/query 上平均。"""
    p = attn_weights.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=dim_keys)
    return ent.mean()


def summarize_attentions(attentions: tuple[torch.Tensor, ...]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ents: List[torch.Tensor] = []
    for i, aw in enumerate(attentions):
        if aw is None:
            continue
        e = attention_head_entropy(aw.detach())
        out[f"attn_entropy/layer_{i}"] = float(e.item())
        ents.append(e)
    if ents:
        out["attn_entropy/mean"] = float(torch.stack(ents).mean().item())
    return out
