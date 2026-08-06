import torch

from src.model.model import MiniLmForCausalLM
from src.util.data_collator import TrainDataCollator


def test_position_ids_reset_on_pack_separator_and_padding(local_tokenizer):
    sep = local_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    collator = TrainDataCollator(local_tokenizer, ignore_index=-100)
    features = [
        {"input_ids": [10, 11, 12, sep, 13, 14, 15], "labels": [10, 11, 12, sep, 13, 14, 15]},
        {"input_ids": [10, 11, 12, sep, 13, 14, 15, 16, 17], "labels": [10, 11, 12, sep, -100, -100, 15, 16, 17]},
    ]
    batch = collator(features)
    pos = batch["position_ids"]
    expected = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 2, 0, 0],
            [0, 1, 2, 3, 0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    assert pos.shape == expected.shape
    assert torch.equal(pos, expected)


def test_attention_mask_shape_and_padding_block(local_tokenizer):
    collator = TrainDataCollator(local_tokenizer, ignore_index=-100)
    features = [
        {"input_ids": [7, 8, 9, 2, 10], "labels": [7, 8, 9, -100, 10]},
        {"input_ids": [3, 4], "labels": [3, 4]},
    ]
    batch = collator(features)
    mask = batch["attention_mask"]
    assert mask.shape == (2, 1, 5, 5)
    assert set(torch.unique(mask).tolist()).issubset({0, 1})
    assert mask[1, 0, :, 2:].sum().item() == 0


def test_attention_mask_prefix_mode_runs(local_tokenizer):
    sep = local_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    collator = TrainDataCollator(local_tokenizer, ignore_index=-100)
    features = [
        {"input_ids": [10, 11, 12, 13, sep, 14, 15, sep], "labels": [-100, -100, 12, 13, sep, 14, 15, sep]},
    ]
    batch = collator(features)
    mask = batch["attention_mask"]
    expected = torch.tensor(
        [
            [
                [
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                ]
            ]
        ],
        dtype=torch.long,
    )
    print("mask\n", mask)
    assert mask.shape == (1, 1, 8, 8)
    assert torch.equal(mask, expected)
    assert set(torch.unique(mask).tolist()).issubset({0, 1})


def test_attention_mask_segment_isolation_and_causality(local_tokenizer):
    sep = local_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    collator = TrainDataCollator(local_tokenizer, ignore_index=-100)
    features = [
        {
            "input_ids": [101, 102, sep, 201, 202, sep, 301],
            "labels": [101, 102, sep, 201, 202, sep, 301],
        }
    ]

    batch = collator(features)
    mask = batch["attention_mask"][0, 0]

    print(f"mask {mask.shape}\n", mask)

    # 段0: [101, 102, sep]，段1: [201, 202, sep]，段2: [301]
    # 段1中的token不能看见段0
    assert mask[3, :3].sum().item() == 0
    assert mask[4, :3].sum().item() == 0

    # 段2中的token不能看见段0和段1
    assert mask[6, :6].sum().item() == 0

    # 段内保持因果
    assert mask[4, 3].item() == 1
    assert mask[3, 4].item() == 0


def test_attention_mask_padding_queries_and_keys_are_blocked(local_tokenizer):
    sep = local_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    collator = TrainDataCollator(local_tokenizer, ignore_index=-100)
    features = [
        {"input_ids": [11, 12, sep, 21], "labels": [11, 12, sep, 21]},
        {"input_ids": [31, 32], "labels": [31, 32]},
    ]

    batch = collator(features)
    mask = batch["attention_mask"][1, 0]  # 第二条样本被padding到长度4

    print(f"mask {mask.shape}\n", mask)

    # padding 的 key 不可见
    assert mask[:, 2:].sum().item() == 0
    # padding 的 query 不可发起注意力
    assert mask[2:, :].sum().item() == 0


@torch.no_grad()
def test_collator_outputs_feed_model_forward(tiny_config, local_tokenizer):
    sep = local_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    collator = TrainDataCollator(
        local_tokenizer,
        ignore_index=-100,
    )
    features = [
        {
            "input_ids": [10, 11, 12, sep, 13, 14, 15],
            "labels": [10, 11, 12, sep, 13, 14, 15]
        },
        {
            "input_ids": [10, 11, 12, sep, 13, 14, 15, 16, 17, sep],
            "labels": [10, 11, 12, sep, -100, -100, 15, 16, 17, sep]
        },
    ]
    batch = collator(features)

    print("batch\n", batch)
    model = MiniLmForCausalLM(tiny_config).eval()
    out = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    assert out.logits.shape[:2] == batch["input_ids"].shape
    assert torch.isfinite(out.logits).all()


def test_sft_collator_keeps_context_across_literal_pack_sep(local_tokenizer):
    """SFT must not treat in-content <|endoftext|> as a pack boundary.

    With pack segments enabled, assistant labels after a literal sep cannot attend to
    the user question that precedes it — silent instruction-tuning corruption.
    """
    sep = local_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    # prompt tokens … sep … supervised assistant tokens (labels != -100)
    input_ids = [101, 102, 103, sep, 104, 201, 202, 203]
    labels = [-100, -100, -100, -100, -100, 201, 202, 203]
    features = [{"input_ids": input_ids, "labels": labels}]

    broken = TrainDataCollator(local_tokenizer, enable_pack_segments=True)(features)
    fixed = TrainDataCollator(local_tokenizer, enable_pack_segments=False)(features)

    broken_mask = broken["attention_mask"][0, 0]
    fixed_mask = fixed["attention_mask"][0, 0]
    asst0 = 5

    # Pack-segment mode isolates assistant from tokens before sep.
    assert broken_mask[asst0, 0].item() == 0
    assert broken_mask[asst0, 2].item() == 0
    # SFT mode keeps full causal context through the literal sep.
    assert fixed_mask[asst0, 0].item() == 1
    assert fixed_mask[asst0, 2].item() == 1
    assert fixed_mask[asst0, 3].item() == 1  # sep itself still visible causally
    assert fixed_mask[asst0, 4].item() == 1
    # Still causal: cannot attend future.
    assert fixed_mask[asst0, 6].item() == 0

    # Positions stay contiguous when pack segments are disabled.
    assert torch.equal(
        fixed["position_ids"][0],
        torch.arange(len(input_ids), dtype=torch.long),
    )


def test_sft_trainer_disables_pack_segments():
    import ast
    from pathlib import Path

    src = Path("src/trainer/train_full_sft.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "TrainDataCollator":
            continue
        for kw in node.keywords:
            if kw.arg == "enable_pack_segments" and isinstance(kw.value, ast.Constant):
                assert kw.value.value is False
                found = True
    assert found, "train_full_sft must construct TrainDataCollator(enable_pack_segments=False)"
