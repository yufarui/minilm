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
