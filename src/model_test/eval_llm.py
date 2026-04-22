"""
使用已训练好的指令微调（SFT）checkpoint 批量测试，并将结果保存为 Excel。

示例:
uv run src/model_test/eval_llm.py\
  --checkpoint checkpoints/sft --latest \
  --prompts_file data/sft/test.txt \
  --output_excel outputs/sft_test_results.xlsx
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from openpyxl import Workbook

from src.model.model import MiniLmForCausalLM
from src.ref_model import get_auto_tokenizer_local
from src.util.path_util import resolve_under_project


def _find_latest_checkpoint(parent: Path) -> Path:
    """在目录下查找 checkpoint-<step>，返回 step 最大的子目录。"""
    best: tuple[int, Path] | None = None
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        matched = re.match(r"^checkpoint-(\d+)$", child.name)
        if not matched:
            continue
        step = int(matched.group(1))
        if best is None or step > best[0]:
            best = (step, child)
    if best is None:
        raise FileNotFoundError(
            f"在 {parent} 下未找到 checkpoint-<step> 子目录；请传入具体 checkpoint。"
        )
    return best[1]


@dataclass(slots=True)
class TestSample:
    sample_id: str
    prompt: str
    reference: str = ""


def _load_samples_from_file(path: Path) -> list[TestSample]:
    """
    暂时只支持.txt
    1) .txt: 每行一个 prompt
    """
    if not path.exists():
        raise FileNotFoundError(f"prompts_file 不存在: {path}")

    samples: list[TestSample] = []
    suffix = path.suffix.lower()

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            prompt = line.strip()
            if not prompt:
                continue
            samples.append(TestSample(sample_id=str(idx), prompt=prompt))

    if not samples:
        raise ValueError(f"未从 {path} 读取到有效测试样本。")
    return samples


@torch.no_grad()
def _generate_text(
        model: MiniLmForCausalLM,
        tokenizer,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        device: torch.device,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("chat_prompt", chat_prompt)
    encoded = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    print("attention_mask", attention_mask)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    generate_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_id,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["do_sample"] = False

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generate_kwargs,
    )
    # generate 返回的是 [prompt + new_tokens]，仅解码新增部分便于观察模型是否实际续写。
    new_token_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)


def _save_to_excel(
        rows: list[dict[str, str | int | float]],
        output_path: Path,
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "sft_test_results"

    headers = [
        "sample_id",
        "prompt",
        "reference",
        "model_output",
        "latency_sec",
    ]
    sheet.append(headers)

    for row in rows:
        sheet.append([row.get(h, "") for h in headers])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于指令微调（SFT）checkpoint 批量测试并导出 Excel。"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint 目录，或训练 output_dir（配合 --latest 自动选择最新）。",
    )
    parser.add_argument("--latest", action="store_true", help="自动选择最新 checkpoint-*。")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="tokenizer/minilm",
        help="与训练一致的 tokenizer 路径。",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="测试样本文件路径（.txt 或 .jsonl）。",
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        default="outputs/sft_test_results.xlsx",
        help="Excel 输出路径。",
    )
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--do_sample", action="store_true", help="开启采样；默认 greedy。")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    args = parser.parse_args()

    checkpoint = resolve_under_project(args.checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint 路径不存在或不是目录: {checkpoint}")
    if args.latest:
        checkpoint = _find_latest_checkpoint(checkpoint)

    prompts_file = resolve_under_project(args.prompts_file)
    output_excel = resolve_under_project(args.output_excel)

    samples = _load_samples_from_file(prompts_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, checkpoint={checkpoint}")
    print(f"loaded {len(samples)} test samples from {prompts_file}")

    tokenizer = get_auto_tokenizer_local(
        resolve_under_project(args.tokenizer_name_or_path),
        trust_remote_code=True,
    )
    model = MiniLmForCausalLM.from_pretrained(str(checkpoint))
    model.to(device)
    model.eval()

    result_rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        start = time.perf_counter()
        output_text = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=sample.prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        latency = round(time.perf_counter() - start, 4)

        result_rows.append(
            {
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "reference": sample.reference,
                "model_output": output_text,
                "latency_sec": latency,
            }
        )
        print(f"[{idx}/{len(samples)}] done, sample_id={sample.sample_id}, latency={latency}s")

    _save_to_excel(result_rows, output_excel)
    print(f"saved excel: {output_excel}")


if __name__ == "__main__":
    main()
