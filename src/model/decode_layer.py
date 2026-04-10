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
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights
