"""
从训练 checkpoint 加载因果 LM，测试自回归续写。

用法（在项目根目录）:
  python -m src.test --checkpoint checkpoints/pretrain/checkpoint-5000
  python -m src.test --checkpoint checkpoints/pretrain --latest
  python -m src.test --checkpoint checkpoints/pretrain/checkpoint-5000 --prompt "你好，"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from src.model.model import MiniLmForCausalLM
from src.ref_model import get_auto_tokenizer_local
from src.util.path_util import resolve_under_project


def _find_latest_checkpoint(parent: Path) -> Path:
    """在目录下查找 checkpoint-<step>，返回 step 最大的子目录。"""
    best: tuple[int, Path] | None = None
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"^checkpoint-(\d+)$", child.name)
        if not m:
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, child)
    if best is None:
        raise FileNotFoundError(
            f"在 {parent} 下未找到 checkpoint-<step> 子目录；请直接传入具体 checkpoint 路径。"
        )
    return best[1]


def load_model_and_tokenizer(
    checkpoint: Path,
    tokenizer_name_or_path: str,
    device: torch.device,
) -> tuple[MiniLmForCausalLM, object]:
    tok_path = resolve_under_project(tokenizer_name_or_path)
    tokenizer = get_auto_tokenizer_local(tok_path, trust_remote_code=True)
    model = MiniLmForCausalLM.from_pretrained(str(checkpoint))
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def continuation_generate(
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
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    gen_ids = model.generate(
        input_ids,
        attention_mask=None,
        **gen_kwargs,
    )
    full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return full_text


def main() -> None:
    parser = argparse.ArgumentParser(description="从 checkpoint 加载模型并测试自回归续写。")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint 目录（含 config.json 与权重），或训练 output_dir（需配合 --latest）",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="若 --checkpoint 为 output_dir，则自动选用其中 step 最大的 checkpoint-*",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="tokenizer/minilm",
        help="与训练一致的分词器路径",
    )
    parser.add_argument("--prompt", type=str, default="你好，", help="续写前缀")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true", help="采样；默认 greedy")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help=">1.0 抑制重复 token；1.0 表示关闭（与 Transformers generate 一致）",
    )
    args = parser.parse_args()

    ck = resolve_under_project(args.checkpoint)
    if not ck.is_dir():
        raise FileNotFoundError(f"checkpoint 路径不存在或不是目录: {ck}")

    if args.latest:
        ck = _find_latest_checkpoint(ck)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, checkpoint={ck}")

    model, tokenizer = load_model_and_tokenizer(
        ck, args.tokenizer_name_or_path, device
    )

    out = continuation_generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )

    print("--- prompt ---")
    print(args.prompt)
    print("--- continuation (full decoded sequence) ---")
    print(out)


if __name__ == "__main__":
    main()
