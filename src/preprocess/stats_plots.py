"""预处理统计图：清理前后对比、阶段漏斗、剔除构成。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if TYPE_CHECKING:
    from src.preprocess.strategies.pipeline import PreprocessPipelineConfig
    from src.preprocess.stats_types import PreprocessPipelineStats


def _setup_cjk_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_preprocess_charts(
    stats: PreprocessPipelineStats,
    cfg: PreprocessPipelineConfig,
    plots_dir: str | Path,
) -> list[Path]:
    """
    写出多张 PNG：阶段柱状图、清理前后对比、剔除瀑布、读入构成堆叠条。
    返回已写入文件路径列表。
    """
    _setup_cjk_font()
    out = Path(plots_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    series = stats.stage_count_series()
    labels = [x[0] for x in series]
    counts = [x[1] for x in series]

    # 1) 各阶段保留量
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(range(len(labels)), counts, color="#2e7dd7", edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("文档条数")
    ax1.set_title("预处理各阶段保留量")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    for i, b in enumerate(bars):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h, f"{int(h):,}", ha="center", va="bottom", fontsize=8)
    fig1.tight_layout()
    p1 = out / "01_stage_retention.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    written.append(p1)

    # 2) 清理前 vs 清理后
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    before_after = [stats.input_lines, stats.output_lines]
    bx = ax2.bar(
        ["清理前（读入）", "清理后（写出）"],
        before_after,
        color=["#c44e52", "#55a868"],
        width=0.5,
        edgecolor="white",
        linewidth=0.5,
    )
    ax2.set_ylabel("文档条数")
    ax2.set_title("清理前后文档量对比")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    for b in bx:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h, f"{int(h):,}", ha="center", va="bottom", fontsize=10)
    rr = stats.retention_rate()
    ax2.text(
        0.5,
        0.02,
        f"保留率：{rr * 100:.2f}%",
        transform=ax2.transAxes,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig2.tight_layout()
    p2 = out / "02_before_after.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    written.append(p2)

    # 3) 漏斗曲线：各阶段保留量连线
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    xs = list(range(len(labels)))
    ax3.plot(xs, counts, "-o", color="#2e7dd7", linewidth=2.2, markersize=9, markerfacecolor="white", markeredgewidth=2)
    ax3.fill_between(xs, counts, alpha=0.15, color="#2e7dd7")
    ax3.set_xticks(xs)
    ax3.set_xticklabels(labels, rotation=22, ha="right")
    ax3.set_ylabel("保留文档条数")
    ax3.set_title("清理全过程：各阶段保留量（漏斗）")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax3.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig3.tight_layout()
    p3 = out / "03_funnel_curve.png"
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)
    written.append(p3)

    # 4) 读入条数的构成：最终保留 + 各原因剔除（单条水平堆叠）
    fig4, ax4 = plt.subplots(figsize=(10, 2.8))
    d_basic, d_exact, d_near, d_ppl = (d[1] for d in stats.drop_counts())
    kept = stats.output_lines
    parts = [kept, d_basic, d_exact, d_near, d_ppl]
    part_labels = ["写出保留", "基础清洗剔除", "精确重复剔除", "近似重复剔除", "PPL 剔除"]
    colors4 = ["#55a868", "#dd8452", "#c44e52", "#8172b3", "#937860"]
    left = 0
    total_in = stats.input_lines or 1
    for lab, w, c in zip(part_labels, parts, colors4, strict=True):
        if w <= 0:
            continue
        ax4.barh(0, w, left=left, height=0.45, label=f"{lab} ({w:,})", color=c, edgecolor="white")
        left += w
    ax4.set_xlim(0, total_in)
    ax4.set_yticks([])
    ax4.set_xlabel("文档条数（堆叠和 = 读入总量）")
    ax4.set_title("读入文档构成（清理前总量分解）")
    ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8)
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
    fig4.tight_layout()
    p4 = out / "04_composition_stacked.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    written.append(p4)

    # 5) 可选指标文本图（PPL 阈值、BERT MLM、主题数）
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    ax5.axis("off")
    lines = [
        f"读入：{stats.input_lines:,} 条",
        f"写出：{stats.output_lines:,} 条",
        f"保留率：{stats.retention_rate() * 100:.2f}%",
    ]
    if cfg.ppl_enable and stats.ppl_threshold_low is not None and stats.ppl_threshold_high is not None:
        lines.append(
            f"PPL 分位阈值（低/高）：{stats.ppl_threshold_low:.4g} / {stats.ppl_threshold_high:.4g}"
        )
        if cfg.ppl_apply_percentile_filter:
            lines.append("PPL：已按分位数实际删除文档")
        else:
            lines.append(
                f"PPL：仅观测分布，未删除；按当前规则约将剔除 {stats.ppl_would_remove_count or 0:,} 条"
            )
    if stats.bert_mlm_mean_nll is not None:
        lines.append(f"BERT MLM 子集平均 NLL：{stats.bert_mlm_mean_nll:.4f}")
    ta = stats.topic_audit
    if ta.ran and ta.n_topics is not None:
        lines.append(f"主题数（BERTopic）：{ta.n_topics}")
    elif ta.skipped:
        lines.append("主题审计：未启用")
    elif ta.error:
        lines.append(f"主题审计：{ta.error}")
    ax5.text(0.02, 0.95, "\n".join(lines), transform=ax5.transAxes, va="top", fontsize=12, family="monospace")
    ax5.set_title("关键标量摘要")
    p5 = out / "05_metrics_summary.png"
    fig5.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    written.append(p5)

    # 6) 文档长度分布（字符数）
    if stats.doc_length_hist_bins and stats.doc_length_hist_counts:
        edges = stats.doc_length_hist_bins
        counts_len = stats.doc_length_hist_counts
        widths = [max(edges[i + 1] - edges[i], 1e-9) for i in range(len(counts_len))]
        fig6, ax6 = plt.subplots(figsize=(9, 4.6))
        ax6.bar(
            edges[:-1],
            counts_len,
            width=widths,
            align="edge",
            color="#4c72b0",
            edgecolor="white",
            linewidth=0.4,
        )
        ax6.set_title("写出语料长度分布（len(text)）")
        ax6.set_xlabel("文档长度（字符数）")
        ax6.set_ylabel("文档条数")
        ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax6.grid(True, axis="y", linestyle="--", alpha=0.35)
        fig6.tight_layout()
        p6 = out / "06_length_distribution.png"
        fig6.savefig(p6, dpi=150)
        plt.close(fig6)
        written.append(p6)

    # 7) PPL 分布（仅在预处理计算了 PPL 时存在）
    if stats.ppl_hist_bins and stats.ppl_hist_counts:
        pedges = stats.ppl_hist_bins
        pcnts = stats.ppl_hist_counts
        pwidths = [max(pedges[i + 1] - pedges[i], 1e-9) for i in range(len(pcnts))]
        fig7, ax7 = plt.subplots(figsize=(9, 4.6))
        ax7.bar(
            pedges[:-1],
            pcnts,
            width=pwidths,
            align="edge",
            color="#55a868",
            edgecolor="white",
            linewidth=0.4,
        )
        ax7.set_title("写出语料 PPL 分布")
        ax7.set_xlabel("PPL")
        ax7.set_ylabel("文档条数")
        ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax7.grid(True, axis="y", linestyle="--", alpha=0.35)
        fig7.tight_layout()
        p7 = out / "07_ppl_distribution.png"
        fig7.savefig(p7, dpi=150)
        plt.close(fig7)
        written.append(p7)

    return written
