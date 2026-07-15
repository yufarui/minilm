import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_dpo_evaluation_requires_an_evaluation_dataset() -> None:
    train_args = json.loads(
        (PROJECT_ROOT / "config/dpo/train_args.json").read_text(encoding="utf-8")
    )
    data_args = json.loads(
        (PROJECT_ROOT / "config/dpo/data_config.json").read_text(encoding="utf-8")
    )

    if data_args.get("eval_data_path"):
        return

    assert train_args.get("do_eval") is False
    assert train_args.get("eval_strategy") == "no"
