from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, text, color="#f5f5f5", fontsize=10):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#444444",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def add_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.2,
            color="#333333",
        )
    )


def main() -> None:
    out_path = Path("image/minilm_moe_architecture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 1.0, "MiniLM Architecture with MoE Routing", ha="center", va="top", fontsize=18, weight="bold")

    add_box(ax, 0.39, 0.90, 0.22, 0.045, "Input: input_ids", "#f0f0f0")
    add_box(ax, 0.34, 0.81, 0.32, 0.055, "Token Embedding (vocab_size=16000, hidden_size=768)", "#d9f2ff")
    add_box(ax, 0.73, 0.81, 0.20, 0.06, "Rotary Embedding (RoPE)\ncos/sin -> all layers", "#e8f7ff", fontsize=9)

    add_box(ax, 0.22, 0.34, 0.56, 0.43, "", "#fafafa", fontsize=12)
    ax.text(0.08, 0.555, "Decoder Layer x8", fontsize=11, ha="left", va="center", color="#555555")
    add_box(ax, 0.30, 0.70, 0.40, 0.045, "RMSNorm (input_norm)", "#eeeeee")
    add_box(ax, 0.30, 0.61, 0.40, 0.07, "Self-Attention (GQA)\nq/k/v/o proj, heads=8, kv_heads=2", "#d9ecff")
    add_box(ax, 0.30, 0.54, 0.40, 0.04, "Residual Add", "#f5f5f5")
    add_box(ax, 0.30, 0.47, 0.40, 0.045, "RMSNorm (attn_norm)", "#eeeeee")
    add_box(ax, 0.30, 0.35, 0.40, 0.09, "MoE FFN (moe_enable=True)\nMoeGate Top-K(K=2) -> Routed Experts(4) + Shared(1)\nweighted combine + scatter add", "#ffe8cc", fontsize=9)
    add_box(ax, 0.30, 0.29, 0.40, 0.04, "Residual Add", "#f5f5f5")
    ax.text(0.80, 0.56, "FlashAttention/SDPA\n(when available)", fontsize=9, ha="left", va="center", color="#2b5d8a")
    ax.text(0.80, 0.41, "aux_loss", fontsize=10, ha="left", va="center", color="#a15c00")

    add_box(ax, 0.34, 0.20, 0.32, 0.055, "Final RMSNorm", "#eeeeee")
    add_box(ax, 0.34, 0.115, 0.32, 0.055, "LM Head Linear (768 -> 16000 logits)", "#d9ffd9")
    add_box(ax, 0.34, 0.03, 0.32, 0.055, "Output: logits / loss", "#d9ffd9")

    add_arrow(ax, 0.5, 0.90, 0.5, 0.865)
    add_arrow(ax, 0.5, 0.81, 0.5, 0.77)
    add_arrow(ax, 0.5, 0.77, 0.5, 0.745)
    add_arrow(ax, 0.5, 0.70, 0.5, 0.68)
    add_arrow(ax, 0.5, 0.61, 0.5, 0.58)
    add_arrow(ax, 0.5, 0.54, 0.5, 0.515)
    add_arrow(ax, 0.5, 0.47, 0.5, 0.44)
    add_arrow(ax, 0.5, 0.35, 0.5, 0.33)
    add_arrow(ax, 0.5, 0.29, 0.5, 0.255)
    add_arrow(ax, 0.5, 0.20, 0.5, 0.17)
    add_arrow(ax, 0.5, 0.115, 0.5, 0.085)

    add_arrow(ax, 0.73, 0.84, 0.74, 0.66)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
