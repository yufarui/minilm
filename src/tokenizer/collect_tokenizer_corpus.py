from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from src.ref_model import get_auto_tokenizer_local

"""
本文件执行脚本
python -m src.tokenizer.collect_tokenizer_corpus \
  --pretrain-jsonl data/pretrain_t2t_mini.jsonl \
  --sft-jsonl data/grpo/rlaif.jsonl \
  --tokenizer-path tokenizer/minilm \
  --output-path tokenizer/minilm/train_tokenizer.txt \
  --max-pretrain-rows 800000 \
  --max-sft-rows 200000 \
  --sft-tool-sample-ratio 0.2 \
  --seed 42
"""

SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是minilm，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是minilm，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are minilm, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are minilm, a small but useful language model.",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect tokenizer corpus from pretrain + SFT data."
    )
    parser.add_argument(
        "--pretrain-jsonl",
        type=str,
        default=None,
        help="Path to pretrain JSONL (expects field: text).",
    )
    parser.add_argument(
        "--sft-jsonl",
        type=str,
        default=None,
        help="Path to SFT JSONL (expects field: conversations).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer/minilm",
        help="Tokenizer dir used for apply_chat_template on SFT samples.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="tokenizer/minilm/train_tokenizer.txt",
        help="Output plain text file for tokenizer training.",
    )
    parser.add_argument(
        "--max-pretrain-rows",
        type=int,
        default=None,
        help="Maximum pretrain rows to collect.",
    )
    parser.add_argument(
        "--max-sft-rows",
        type=int,
        default=None,
        help="Maximum SFT rows to collect.",
    )
    parser.add_argument(
        "--sft-add-system-ratio",
        type=float,
        default=0.2,
        help="If conversation does not start with system, prepend one with this probability.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling behavior.",
    )
    parser.add_argument(
        "--sft-tool-sample-ratio",
        type=float,
        default=0.0,
        help="When max-sft-rows is set, reserve this ratio for tool-related SFT samples.",
    )
    return parser.parse_args()


def _safe_json_loads(line: str) -> dict[str, Any] | None:
    try:
        data = json.loads(line)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = _safe_json_loads(line)
            if data is not None:
                yield data


def _parse_tools_from_system(first_message: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw_tools = first_message.get("tools")
    if raw_tools is None:
        return None
    if isinstance(raw_tools, list):
        return raw_tools
    if isinstance(raw_tools, str):
        s = raw_tools.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else None
        except json.JSONDecodeError:
            return None
    return None


def _fill_assistant_tool_calls(conv: list[dict[str, Any]]) -> bool:
    for msg in conv:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if "tool_calls" not in msg:
            continue
        raw = msg["tool_calls"]
        if isinstance(raw, list):
            continue
        if not isinstance(raw, str):
            return False
        s = raw.strip()
        if not s:
            msg["tool_calls"] = []
            continue
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, list):
            return False
        msg["tool_calls"] = parsed
    return True


def _build_sft_text(
    tokenizer,
    conversations: list[dict[str, Any]],
    add_system_ratio: float,
    rng: random.Random,
) -> str | None:
    conv = copy.deepcopy(conversations)
    if not conv:
        return None

    first_message = conv[0]
    if first_message.get("role") == "system" and not first_message.get("content"):
        first_message["content"] = rng.choice(SYSTEM_PROMPTS)

    tools = None
    if first_message.get("role") == "system":
        tools = _parse_tools_from_system(first_message)

    if first_message.get("role") != "system" and rng.random() < add_system_ratio:
        conv.insert(0, {"role": "system", "content": rng.choice(SYSTEM_PROMPTS)})

    if not _fill_assistant_tool_calls(conv):
        return None

    try:
        text = tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=False,
            tokenize=False,
            tools=tools,
            open_think=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=False,
            tokenize=False,
            tools=tools,
        )
    if not isinstance(text, str):
        return None
    text = text.strip()
    return text if text else None


def _reservoir_offer(pool: list[str], item: str, seen: int, k: int, rng: random.Random) -> None:
    if k <= 0:
        return
    if len(pool) < k:
        pool.append(item)
        return
    j = rng.randint(1, seen)
    if j <= k:
        pool[j - 1] = item


def _is_tool_related_conversation(conversations: list[dict[str, Any]]) -> bool:
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in {"tool", "function"}:
            return True
        if role == "system" and msg.get("tools"):
            return True
        if role == "assistant":
            tc = msg.get("tool_calls")
            if isinstance(tc, list) and tc:
                return True
            if isinstance(tc, str) and tc.strip():
                return True
    return False


def collect_pretrain_texts(
    pretrain_jsonl: Path,
    rng: random.Random,
    max_rows: int | None = None,
) -> list[str]:
    results: list[str] = []
    seen = 0
    for data in _iter_jsonl(pretrain_jsonl):
        text = data.get("text", "")
        if not text:
            continue
        text = str(text).strip()
        if not text:
            continue
        if max_rows is None:
            results.append(text)
            continue
        seen += 1
        _reservoir_offer(results, text, seen=seen, k=max_rows, rng=rng)
    rng.shuffle(results)
    return results


def collect_sft_texts(
    sft_jsonl: Path,
    tokenizer,
    rng: random.Random,
    max_rows: int | None = None,
    add_system_ratio: float = 0.2,
    tool_sample_ratio: float = 0.0,
) -> list[str]:
    results: list[str] = []
    seen = 0
    tool_sample_ratio = max(0.0, min(1.0, tool_sample_ratio))
    tool_target = int(max_rows * tool_sample_ratio) if max_rows is not None else 0
    non_tool_target = (max_rows - tool_target) if max_rows is not None else 0
    tool_pool: list[str] = []
    non_tool_pool: list[str] = []
    tool_seen = 0
    non_tool_seen = 0

    for data in _iter_jsonl(sft_jsonl):
        conversations = data.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            continue
        text = _build_sft_text(
            tokenizer,
            conversations,
            add_system_ratio=add_system_ratio,
            rng=rng,
        )
        if text is None:
            continue
        if max_rows is None:
            results.append(text)
            continue

        if tool_target > 0:
            if _is_tool_related_conversation(conversations):
                tool_seen += 1
                _reservoir_offer(tool_pool, text, seen=tool_seen, k=max_rows, rng=rng)
            else:
                non_tool_seen += 1
                _reservoir_offer(non_tool_pool, text, seen=non_tool_seen, k=max_rows, rng=rng)
            continue

        seen += 1
        _reservoir_offer(results, text, seen=seen, k=max_rows, rng=rng)

    if max_rows is None:
        rng.shuffle(results)
        return results

    if tool_target > 0:
        tool_take = min(tool_target, len(tool_pool))
        non_tool_take = min(non_tool_target, len(non_tool_pool))

        if tool_take < tool_target:
            non_tool_take = min(len(non_tool_pool), non_tool_take + (tool_target - tool_take))
        if non_tool_take < non_tool_target:
            tool_take = min(len(tool_pool), tool_take + (non_tool_target - non_tool_take))

        rng.shuffle(tool_pool)
        rng.shuffle(non_tool_pool)
        results = tool_pool[:tool_take] + non_tool_pool[:non_tool_take]
        rng.shuffle(results)

    return results


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    pretrain_path = Path(args.pretrain_jsonl) if args.pretrain_jsonl else None
    sft_path = Path(args.sft_jsonl) if args.sft_jsonl else None
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pretrain_path is None and sft_path is None:
        raise ValueError("At least one of --pretrain-jsonl or --sft-jsonl must be provided.")

    tokenizer = None
    if sft_path is not None:
        tokenizer = get_auto_tokenizer_local(args.tokenizer_path, trust_remote_code=True)

    all_texts: list[str] = []

    if pretrain_path is not None:
        pretrain_texts = collect_pretrain_texts(
            pretrain_path,
            rng=rng,
            max_rows=args.max_pretrain_rows,
        )
        print(f"[collect] pretrain: {len(pretrain_texts)}")
        all_texts.extend(pretrain_texts)

    if sft_path is not None:
        sft_texts = collect_sft_texts(
            sft_path,
            tokenizer=tokenizer,
            rng=rng,
            max_rows=args.max_sft_rows,
            add_system_ratio=args.sft_add_system_ratio,
            tool_sample_ratio=args.sft_tool_sample_ratio,
        )
        print(f"[collect] sft(apply_chat_template): {len(sft_texts)}")
        all_texts.extend(sft_texts)

    rng.shuffle(all_texts)
    with output_path.open("w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text.replace("\n", "\\n") + "\n")

    print(f"[done] wrote {len(all_texts)} lines -> {output_path}")


if __name__ == "__main__":
    main()
