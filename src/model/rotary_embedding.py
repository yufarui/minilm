import torch
import torch.nn as nn
from transformers.utils.generic import maybe_autocast

from src.config.model_config import MiniLMConfig
from .rope_yarn import compute_yarn_inv_freq


class RotaryEmbedding(nn.Module):

    def __init__(self, config: MiniLMConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        rs = getattr(config, "rope_scaling", None)
        if isinstance(rs, dict) and rs.get("type") == "yarn":
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dev = torch.device(device) if device is not None else None
            inv_freq = compute_yarn_inv_freq(config.rope_theta, head_dim, rs, device=dev)
        else:
            inv_freq = self.compute_default_rope_params(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("origin_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_params(config, device):
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        # 默认除法会转换成float32
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.int64) / dim))

        return inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids: torch.Tensor):
        """
        :param x
        :param position_ids: [batch, pos_len] 注意在kv_cache中可以指定位置
        :return:
        """
        batch_size, pos_len = position_ids.shape

        inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(-1).float()
        inv_freq_expanded = inv_freq.expand(batch_size, -1, -1).to(x.device)

        position_ids_expanded = position_ids.unsqueeze(1).float()

        device_type = x.device.type

        # Force float32
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            # 注意这里还是保持了和公式一样的顺序
            emb = torch.stack([freqs, freqs], dim=-1).flatten(-2)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(x.dtype), sin.to(x.dtype)
