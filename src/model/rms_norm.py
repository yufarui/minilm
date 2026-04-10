import torch.nn as nn
import torch

from src.config.model_config import MiniLMConfig


class RMSNorm(nn.Module):
    def __init__(self, config: MiniLMConfig):
        super().__init__()
        self.config = config
        self.variance_epsilon = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
