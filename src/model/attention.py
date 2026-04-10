from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache

from src.config.model_config import MiniLMConfig
from .rope_yarn import yarn_attention_factor


class Attention(nn.Module):

    def __init__(self, config: MiniLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        rs = getattr(config, "rope_scaling", None)
        self.yarn_attn_factor = (
            yarn_attention_factor(rs) if isinstance(rs, dict) and rs.get("type") == "yarn" else 1.0
        )

        self.attn_dropout = config.attention_dropout

        # 统一取消偏置，减少参数，训练稳定
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.use_flash_attn = (hasattr(torch.nn.functional, 'scaled_dot_product_attention')
                               and config.use_flash_attention)

    def forward(self,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Cache] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch, seq_len, _ = hidden_states.shape
        hidden_shape = (batch, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.yarn_attn_factor != 1.0:
            query_states = query_states * self.yarn_attn_factor
            key_states = key_states * self.yarn_attn_factor
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.use_flash_attn:
            attn_output, attn_weights = self.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
            )
        else:
            attn_output, attn_weights = self.eager_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask
            )

        attn_output = attn_output.reshape(batch, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, num_repeat: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if num_repeat == 1:
            return hidden_states
        hidden_states = hidden_states.unsqueeze(dim=2)
        hidden_states = hidden_states.expand(batch, num_key_value_heads, num_repeat, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * num_repeat, slen, head_dim)

    def eager_attention_forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor]
    ):
        key_states = Attention.repeat_kv(key, self.num_key_value_groups)
        value_states = Attention.repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)

        # 需要注意softmax中精度大小控制
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, self.attn_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # b,s,a,d
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    def scaled_dot_product_attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
    ):
        key_states = Attention.repeat_kv(key, self.num_key_value_groups)
        value_states = Attention.repeat_kv(value, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            # 完全由attn_mask来控制掩码,训练时，模型也不一定为自回归
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    @staticmethod
    def rotate_half(x):
        # Split and rotate. Note that this function is different from e.g. Llama.
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rot_x = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return rot_x

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (Attention.rotate_half(q) * sin)
        k_embed = (k * cos) + (Attention.rotate_half(k) * sin)
        return q_embed, k_embed
