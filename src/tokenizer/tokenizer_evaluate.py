from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from src.ref_model import get_auto_tokenizer_local
from src.tokenizer.train_tokenizer import TokenizerTrainer


class TokenizerEvaluator:
    def __init__(self, model_dir):
        self.tokenizer = get_auto_tokenizer_local(model_dir, trust_remote_code=True, use_fast=False)

    def evaluate(self, texts):
        tokenizer = self.tokenizer

        # ========== 编码所有文本 ==========
        print("Tokenizing...")
        encodings = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
        # 如果不希望添加 bos/eos，我们使用 add_special_tokens=False
        encodings = [seq for seq in encodings if seq]
        # ========== 统计指标 ==========
        all_token_ids = [tid for seq in encodings for tid in seq]
        token_freq = Counter(all_token_ids)
        total_tokens = len(all_token_ids)
        total_chars = sum(len(t) for t in texts)

        # 序列长度（token 数）
        seq_lengths = [len(seq) for seq in encodings]
        # 压缩率（字符数 / token 数）
        compression_ratios = [len(texts[i]) / seq_len for i, seq_len in enumerate(seq_lengths)]

        # 特殊 token 使用情况（<unk> 出现次数）
        unk_token = tokenizer.unk_token
        unk_id = tokenizer.convert_tokens_to_ids(unk_token) if unk_token else None
        unk_count = token_freq.get(unk_id, 0) if unk_id is not None else 0

        print("\n=== Summary ===")
        print(f"Total samples: {len(texts)}")
        print(f"Total characters: {total_chars}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per sample: {np.mean(seq_lengths):.2f}")
        print(f"Average characters per sample: {total_chars / len(texts):.2f}")
        print(f"Average compression ratio (chars/token): {np.mean(compression_ratios):.2f}")
        print(f"<unk> usage: {unk_count} times ({unk_count / total_tokens * 100:.4f}%)")

        return {
            "token_freq": token_freq,
            "compression_ratios": compression_ratios,
            "seq_lengths": seq_lengths,
            "all_token_ids": all_token_ids,
        }

    def plot(self, evaluate_dict):
        # ========== 设置中文字体 ==========
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # ========== 绘图 ==========
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        token_freq = evaluate_dict["token_freq"]
        compression_ratios = evaluate_dict["compression_ratios"]
        seq_lengths = evaluate_dict["seq_lengths"]
        all_token_ids = evaluate_dict["all_token_ids"]

        tokenizer = self.tokenizer
        # 1. 高频 token 条形图
        ax = axes[0, 0]
        top_tokens = token_freq.most_common(20)
        top_tokens_ids, top_tokens_counts = zip(*top_tokens)
        top_tokens_str = [tokenizer.decode([tid]) for tid in top_tokens_ids]
        ax.barh(range(len(top_tokens_str)), top_tokens_counts, align="center")
        ax.set_yticks(range(len(top_tokens_str)))
        ax.set_yticklabels(top_tokens_str)
        ax.invert_yaxis()  # 高频在上
        ax.set_xlabel("Frequency")
        ax.set_title("Top 20 Most Frequent Tokens")

        # 2. 序列长度分布（直方图）
        ax = axes[0, 1]
        ax.hist(seq_lengths, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Sequence Length (tokens)")
        ax.set_ylabel("Number of Samples")
        ax.set_title("Distribution of Sequence Lengths")
        ax.axvline(np.mean(seq_lengths), color="red", linestyle="--", label=f"Mean: {np.mean(seq_lengths):.1f}")
        ax.legend()

        # 3. 压缩率分布
        ax = axes[1, 0]
        ax.hist(compression_ratios, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Compression Ratio (chars / tokens)")
        ax.set_ylabel("Number of Samples")
        ax.set_title("Compression Ratio Distribution")
        ax.axvline(np.mean(compression_ratios), color="red", linestyle="--",
                   label=f"Mean: {np.mean(compression_ratios):.2f}")
        ax.legend()

        # 4. Token 长度分布（每个 token 包含的字符数，近似）
        # 通过解码每个 token 并计算长度
        sample_tokens = all_token_ids[:50000]  # 取前 50000 个 token，避免过大
        token_lengths = [len(tokenizer.decode([tid])) for tid in sample_tokens]
        ax = axes[1, 1]
        ax.hist(token_lengths, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Token Length (characters)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Token Lengths (sample)")

        plt.tight_layout()
        plt.savefig("tokenizer_evaluation.png", dpi=150)
        print("\nFigure saved as tokenizer_evaluation.png")
        plt.show()


def main():
    from pathlib import Path
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    data_path = project_root / "data/pretrain_t2t_mini.jsonl"
    tokenizer_dir = project_root / "tokenizer/minilm"

    plain_text_path = tokenizer_dir / "train_tokenizer.txt"

    # ===== 1. 提取纯文本 =====
    # TokenizerTrainer.dump_plain_text(
    #     str(data_path),
    #     str(plain_text_path),
    #     row_range=[0, 10000]
    # )

    evaluator = TokenizerEvaluator(tokenizer_dir)

    test_texts = read_texts(tokenizer_dir / "train_tokenizer.txt")
    evaluate_dict = evaluator.evaluate(test_texts)
    evaluator.plot(evaluate_dict)


def read_texts(data_path):
    texts = []
    for i, line in enumerate(open(data_path, "r", encoding="utf-8")):
        if line:
            texts.append(line.strip())
    return texts


if __name__ == "__main__":
    main()
