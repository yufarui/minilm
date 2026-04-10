from .dataset_test_utils import DPO_JSONL, PRETRAIN_SCHEDULE_JSONL, write_jsonl


def test_build_dpo_mock_file() -> None:
    rows = []
    for i in range(10):
        prompt = f"标题：样本{i} 产品评论：这是一条用于 DPO 测试的评论。你认为是褒义还是贬义？"
        rows.append(
            {
                "chosen": [
                    {"content": prompt, "role": "user"},
                    {"content": f"这是第{i}条样本的较优回答，倾向给出完整判断。", "role": "assistant"},
                ],
                "rejected": [
                    {"content": prompt, "role": "user"},
                    {"content": "好", "role": "assistant"},
                ],
            }
        )
    write_jsonl(DPO_JSONL, rows)

    lines = DPO_JSONL.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10


def test_build_pretrain_schedule_input_file() -> None:
    rows = [
        {"text": "alpha " * 200},
        {"text": "beta " * 200},
    ]
    write_jsonl(PRETRAIN_SCHEDULE_JSONL, rows)
    assert PRETRAIN_SCHEDULE_JSONL.exists()
