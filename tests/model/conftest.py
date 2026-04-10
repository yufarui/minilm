import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from src.util.path_util import resolve_under_project
from src.config.model_config import MiniLMConfig

@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(42)


@pytest.fixture
def tokenizer_vocab_size() -> int:
    cfg_path = Path("tokenizer/minilm/tokenizer_config.json")
    cfg_path = resolve_under_project(cfg_path)
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    dec = data.get("added_tokens_decoder", {})
    max_added_id = max((int(k) for k in dec.keys()), default=-1)
    base_vocab = int(max_added_id + 1)
    return max(base_vocab, int(data.get("vocab_size", 0) or 0), 16027)


@pytest.fixture
def local_tokenizer():
    tok_path = resolve_under_project(Path("tokenizer/minilm"))
    return AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True)


@pytest.fixture
def tiny_config(tokenizer_vocab_size: int) -> MiniLMConfig:
    # 与 config/config.json 的关键 MoE 设定保持一致（默认开启 MoE）
    return MiniLMConfig(
        vocab_size=tokenizer_vocab_size,
        pad_token_id=0,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        max_position_embeddings=256,
        use_flash_attention=False,
        inference_rope_scaling=True,
        moe_enable=True,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=1,
        scoring_func="sigmoid",
        aux_loss_alpha=0.01,
        norm_topk_prob=True,
        seq_aux=True,
    )
