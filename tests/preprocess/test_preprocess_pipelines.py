import json
from pathlib import Path

from src.preprocess.strategies.pipeline import PreprocessPipeline, PreprocessPipelineConfig
from src.preprocess.strategies.sft_pipeline import SftPipelineConfig, SftPreprocessPipeline

TEST_TMP_DIR = Path(__file__).parents[1] / "tmp" / "preprocess"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _test_case_dir(case_name: str) -> Path:
    case_dir = TEST_TMP_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def test_pretrain_pipeline_with_100_synthetic_rows():
    # 统一把测试中间文件写到 tests/tmp/preprocess 下，便于本地查看。
    case_dir = _test_case_dir("pretrain_pipeline")
    input_path = case_dir / "pretrain_input.jsonl"
    output_path = case_dir / "pretrain_output.jsonl"

    rows = []
    # 0-19: 长度不足（会在 basic clean 阶段被过滤）
    for i in range(20):
        rows.append({"id": i, "text": "short"})
    # 20-39: 可通过 basic clean，但 10 组重复文本（exact dedup 会删掉一半）
    for i in range(20, 40):
        dup_group = (i - 20) % 10
        rows.append({"id": i, "text": f"duplicate pretrain doc {dup_group} " + ("alpha " * 8)})
    # 40-99: 60 条唯一且有效文本
    for i in range(40, 100):
        rows.append({"id": i, "text": f"unique pretrain document {i} " + ("content " * 8)})

    _write_jsonl(input_path, rows)

    cfg = PreprocessPipelineConfig(
        basic=PreprocessPipelineConfig().basic,
        exact_dedup=True,
        near_dedup=PreprocessPipelineConfig().near_dedup,
        run_diagnostics=False,
    )
    cfg.basic.min_chars = 20
    cfg.basic.allowed_langs = []
    cfg.near_dedup.enabled = False

    stats = PreprocessPipeline(cfg).run(input_path, output_path)
    out_rows = _read_jsonl(output_path)

    assert stats.input_lines == 100
    assert stats.after_basic_clean == 80
    assert stats.after_exact_dedup == 70
    assert stats.after_near_dedup == 70
    assert stats.output_lines == 70
    assert len(out_rows) == 70
    assert len({r["text"] for r in out_rows}) == 70


def test_sft_pipeline_with_100_synthetic_rows():
    # 统一把测试中间文件写到 tests/tmp/preprocess 下，便于本地查看。
    case_dir = _test_case_dir("sft_pipeline")
    input_path = case_dir / "sft_input.jsonl"
    output_path = case_dir / "sft_output.jsonl"

    rows = []
    # 0-9: 空对话
    for i in range(10):
        rows.append({"id": i, "conversations": []})
    # 10-19: 角色顺序错误（assistant 开头）
    for i in range(10, 20):
        rows.append({"id": i, "conversations": [{"role": "assistant", "content": "wrong start role"}]})
    # 20-29: 拒答
    for i in range(20, 30):
        rows.append(
            {
                "id": i,
                "conversations": [
                    {"role": "user", "content": "Can you help me?"},
                    {"role": "assistant", "content": "I cannot answer this request."},
                ],
            }
        )
    # 30-39: 长度过短
    for i in range(30, 40):
        rows.append(
            {
                "id": i,
                "conversations": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ],
            }
        )
    # 40-59: 10 组重复对话（每组 2 条），可通过清洗但 exact dedup 会去重
    for i in range(40, 60):
        dup_group = (i - 40) % 10
        rows.append(
            {
                "id": i,
                "conversations": [
                    {"role": "user", "content": f"<think>duplicate sft question {dup_group}</think>"},
                    {"role": "assistant", "content": f"duplicate answer {dup_group} " + ("detail " * 5)},
                ],
            }
        )
    # 60-99: 40 条有效唯一对话，其中 1 条包含可修复的 tool_calls 字符串
    for i in range(60, 100):
        conv = [
            {"role": "user", "content": f"<think>unique user question {i}</think> " + ("context " * 4)},
            {"role": "assistant", "content": f"unique assistant answer {i} " + ("explain " * 6)},
        ]
        if i == 60:
            conv[1]["tool_calls"] = '[{"name":"search","arguments":{"q":"abc",}},]'
            conv.append({"role": "tool", "content": '{"result":"ok"}'})
            conv.append({"role": "assistant", "content": "tool result summarized " + ("done " * 4)})
        rows.append({"id": i, "conversations": conv})

    _write_jsonl(input_path, rows)

    cfg = SftPipelineConfig(
        strict_role_order=True,
        repair_tool_calls=True,
        normalize_markers={"<think>": "<reason>", "</think>": "</reason>"},
        filter_refuse_replies=True,
        min_chars=20,
        allowed_langs=[],
        exact_dedup=True,
        run_diagnostics=False,
    )
    cfg.near_dedup.enabled = False

    stats = SftPreprocessPipeline(cfg).run(input_path, output_path)
    out_rows = _read_jsonl(output_path)

    assert stats.input_lines == 100
    assert stats.skipped_empty_conversations == 10
    assert stats.skipped_role_order == 10
    assert stats.skipped_refuse_reply == 10
    assert stats.skipped_length == 10
    assert stats.after_exact_dedup == 50
    # 当前实现中，near dedup 关闭时 after_near_dedup 保持为清洗后值（去重前）
    assert stats.after_near_dedup == 60
    assert stats.output_lines == 50
    assert len(out_rows) == 50
    assert stats.markers_normalized_rows == 90
    assert stats.tool_calls_repaired >= 1
