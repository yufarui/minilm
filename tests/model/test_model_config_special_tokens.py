import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_model_eos_token_id_matches_tokenizer() -> None:
    with (PROJECT_ROOT / "config/config.json").open(encoding="utf-8") as fp:
        model_config = json.load(fp)
    with (PROJECT_ROOT / "tokenizer/minilm/tokenizer_config.json").open(
        encoding="utf-8"
    ) as fp:
        tokenizer_config = json.load(fp)

    eos_token = tokenizer_config["eos_token"]
    eos_token_id = next(
        int(token_id)
        for token_id, token in tokenizer_config["added_tokens_decoder"].items()
        if token["content"] == eos_token
    )

    assert model_config["eos_token_id"] == eos_token_id
