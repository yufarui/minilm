"""参考模型困惑度、可选 MLM 负对数似然（信息量 proxy）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
import torch


@dataclass
class Gpt2PplConfig:
    model_name: str = "gpt2"
    max_length: int = 1024
    device: str | None = None


@dataclass
class BertMlmNllConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    max_masks: int = 32
    device: str | None = None


def looks_like_code_or_table(text: str) -> bool:
    """低 PPL 但应保留：代码/表格等结构化文本的粗启发。"""
    if not text:
        return False
    n = max(len(text), 1)
    punct_code = sum(c in "{}[]();<>`" for c in text) / n
    if punct_code > 0.015:
        return True
    lines = text.splitlines()[:80]
    indent = sum(1 for ln in lines if ln.startswith(("    ", "\t"))) / max(len(lines), 1)
    if indent > 0.25:
        return True
    head = sum(
        1
        for ln in lines[:40]
        if ln.lstrip().startswith(
            ("def ", "class ", "import ", "from ", "function ", "const ", "let ", "var ", "#include")
        )
    )
    if head >= 2:
        return True
    if text.count("\t") > 5 and "\t" in text[:200]:
        return True
    return False


@torch.inference_mode()
def gpt2_perplexities(
    texts: list[str],
    cfg: Gpt2PplConfig,
    progress_every: int = 0,
) -> list[float]:
    """逐文档因果 LM 平均 NLL，返回 PPL = exp(nll)。"""
    from src.ref_model import get_causal_lm_reference

    ref = get_causal_lm_reference(cfg.model_name, cfg.device)
    tok, model = ref.tokenizer, ref.model
    dev = next(model.parameters()).device

    out: list[float] = []
    for idx, text in enumerate(texts):
        if progress_every and idx and idx % progress_every == 0:
            print(f"[PPL] {idx}/{len(texts)}", flush=True)
        enc = tok(text, return_tensors="pt", truncation=False, add_special_tokens=True)
        ids = enc["input_ids"][0]
        if ids.numel() < 2:
            out.append(float("nan"))
            continue

        max_len = cfg.max_length
        if ids.numel() > max_len:
            ids = ids[:max_len]

        chunk = ids.unsqueeze(0).to(dev, non_blocking=dev.type == "cuda")
        logits = model(chunk).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll = float(loss.item())
        total_count = int(shift_labels.numel())

        mean_nll = total_nll / max(total_count, 1)
        out.append(math.exp(mean_nll))
    return out


@torch.inference_mode()
def bert_mlm_mean_nll(
    texts: list[str],
    cfg: BertMlmNllConfig,
) -> list[float]:
    """随机掩码位置的平均 token NLL（越高越不自然/难预测，仅作相对分数）。"""
    from src.ref_model import get_masked_lm_reference

    ref = get_masked_lm_reference(cfg.model_name, cfg.device)
    tok, model = ref.tokenizer, ref.model
    device = next(model.parameters()).device

    results: list[float] = []
    for text in texts:
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_length,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        if input_ids.numel() < 3:
            results.append(float("nan"))
            continue

        ids = input_ids[0]
        special = tok.all_special_ids
        candidates = [i for i in range(ids.numel()) if int(ids[i].item()) not in special]
        if not candidates:
            results.append(float("nan"))
            continue
        k = min(cfg.max_masks, len(candidates))
        idx = torch.tensor(candidates[:k], device=device, dtype=torch.long)
        labels = ids.clone()
        masked = ids.clone()
        for j in idx:
            masked[j] = tok.mask_token_id

        logits = model(masked.unsqueeze(0), attention_mask=attn).logits[0]
        loss_sum = 0.0
        cnt = 0
        for j in idx:
            tgt = int(labels[j].item())
            loss_sum += float(
                torch.nn.functional.cross_entropy(
                    logits[j : j + 1],
                    labels[j : j + 1],
                    reduction="sum",
                ).item()
            )
            cnt += 1
        results.append(loss_sum / max(cnt, 1))
    return results


def percentile_bounds(values: list[float], low_pct: float, high_pct: float) -> tuple[float, float]:
    arr = [v for v in values if not math.isnan(v) and math.isfinite(v)]
    if not arr:
        return float("nan"), float("nan")
    t = torch.tensor(arr, dtype=torch.float64)
    lo = torch.quantile(t, float(low_pct) / 100.0).item()
    hi = torch.quantile(t, float(high_pct) / 100.0).item()
    return float(lo), float(hi)


def ppl_keep_mask(
    texts: list[str],
    ppls: list[float],
    low_pct: float,
    high_pct: float,
    keep_low_if_structured: bool = True,
    *,
    thresholds: tuple[float, float] | None = None,
) -> list[bool]:
    lo, hi = thresholds if thresholds is not None else percentile_bounds(ppls, low_pct, high_pct)
    keep: list[bool] = []
    for t, p in zip(texts, ppls, strict=True):
        if math.isnan(p) or not math.isfinite(p):
            keep.append(False)
            continue
        if p > hi:
            keep.append(False)
            continue
        if p < lo:
            if keep_low_if_structured and isinstance(t, str) and looks_like_code_or_table(t):
                keep.append(True)
            else:
                keep.append(False)
            continue
        keep.append(True)
    return keep
