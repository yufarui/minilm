from __future__ import annotations

import torch
import torch.nn as nn

from src.config.model_config import MiniLMConfig
from .mlp import MLP
from .moe_gate import MoeGate


class Moe(nn.Module):
    """
    单卡高效 MoE：

    - Top-K 路由
    - token → expert 分桶（排序实现）
    - expert 批处理
    - 加权融合
    - shared experts 融合
    """

    def __init__(self, config: MiniLMConfig):
        super().__init__()
        self.config = config
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        self.gate = MoeGate(config)

        self.routed_experts = nn.ModuleList(
            [MLP(config) for _ in range(self.n_routed)]
        )

        self.shared_experts = nn.ModuleList(
            [MLP(config) for _ in range(self.n_shared)]
        ) if self.n_shared > 0 else None

        self.aux_loss = torch.tensor(0.0)

    def forward(self, hidden_states):
        """
        hidden_states: [B, S, H]
        """
        identity = hidden_states
        B, S, H = hidden_states.shape

        # ===== gating =====
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        self.aux_loss = aux_loss

        x = hidden_states.reshape(-1, H)  # [N, H]
        N = x.shape[0]

        # ===== 展平 Top-K =====
        flat_expert = topk_idx.reshape(-1)  # [N*K]
        flat_weight = topk_weight.reshape(-1)  # [N*K]

        # ===== token index（重复 K 次）=====
        token_idx = torch.arange(N, device=x.device).unsqueeze(1).repeat(1, self.top_k).reshape(-1)

        # ===== 按 expert 排序（核心优化）=====
        sorted_idx = flat_expert.argsort()
        flat_expert = flat_expert[sorted_idx]
        flat_weight = flat_weight[sorted_idx]
        token_idx = token_idx[sorted_idx]

        # ===== 分桶执行 expert =====
        expert_outputs = torch.zeros_like(x)

        start = 0
        for expert_id in range(self.n_routed):
            end = start
            while end < flat_expert.shape[0] and flat_expert[end] == expert_id:
                end += 1
            if start == end:
                continue

            idx = token_idx[start:end]  # 属于该 expert 的 token
            w = flat_weight[start:end]  # 对应权重

            tokens = x[idx]  # [T, H]
            out = self.routed_experts[expert_id](tokens)  # [T, H]

            # 加权累加（scatter add）；autocast 下 out 可能与 expert_outputs（与 x 同 dtype）不一致
            scaled = out * w.unsqueeze(-1)
            expert_outputs.index_add_(0, idx, scaled.to(dtype=expert_outputs.dtype))
            start = end

        y = expert_outputs.view(B, S, H)

        # ===== shared experts =====
        if self.shared_experts is not None:
            shared_out = 0
            for expert in self.shared_experts:
                shared_out = shared_out + expert(identity)
            y = y + shared_out

        return y
