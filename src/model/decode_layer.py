import torch
import torch.nn as nn
from transformers import Cache

from .attention import Attention
from src.config.model_config import MiniLMConfig
from .mlp import MLP
from .moe import Moe
from .rms_norm import RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, config: MiniLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = Attention(config, layer_idx)

        self.mlp = MLP(config) if not config.moe_enable else Moe(config)

        self.input_norm = RMSNorm(config)
        self.attn_norm = RMSNorm(config)

    @staticmethod
    def _token_keep_mask(
            attention_mask: torch.Tensor | None,
            batch: int,
            seq_len: int,
    ) -> torch.Tensor | None:
        """从 attention_mask 推导非 pad token 掩码 [B, S]，供 MoE aux 使用。"""
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if attention_mask.shape != (batch, seq_len):
                return None
            return attention_mask.bool()
        if attention_mask.dim() == 4:
            # [B, H_or_1, Q, K]：pad query 整行屏蔽 → any(K) 为 False
            if attention_mask.shape[-2] != seq_len:
                return None
            return attention_mask.any(dim=-1).any(dim=1)
        return None

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: Cache | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)

        hidden_states, attn_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        if isinstance(self.mlp, Moe):
            token_mask = self._token_keep_mask(
                attention_mask, hidden_states.shape[0], hidden_states.shape[1]
            )
            hidden_states = self.mlp(hidden_states, token_mask=token_mask)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights
