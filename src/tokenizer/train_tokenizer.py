import json
import os
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

# 与 Qwen tokenizer added_tokens_decoder 151643–151664 对齐；在 BPE 学完合并词之后按序追加（不占词表前部）。
# <pad> 通过 BpeTrainer.special_tokens 注入基础词表以固定为 id=0。
# <|endoftext|> 作为普通控制符，与 <|im_start|>/<|im_end|> 对等，在词表后置追加。
POST_VOCAB_ADDED_TOKENS: list[tuple[str, bool]] = [
    ("<|endoftext|>", True),
    ("<|im_start|>", True),
    ("<|im_end|>", True),
    
    ("<|object_ref_start|>", True),
    ("<|object_ref_end|>", True),
    ("<|box_start|>", True),
    ("<|box_end|>", True),
    ("<|quad_start|>", True),
    ("<|quad_end|>", True),
    ("<|vision_start|>", True),
    ("<|vision_end|>", True),
    ("<|vision_pad|>", True),
    ("<|image_pad|>", True),
    ("<|video_pad|>", True),
 
    ("<think>", False),
    ("</think>", False),
    ("<tool_call>", False),
    ("</tool_call>", False),
    ("<tool_response>", False),
    ("</tool_response>", False),

    ("<|fim_prefix|>", False),
    ("<|fim_middle|>", False),
    ("<|fim_suffix|>", False),
    ("<|fim_pad|>", False),
    ("<|repo_name|>", False),
    ("<|file_sep|>", False),
]

QWEN_SPLIT_REGEX = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

# Hugging Face additional_special_tokens（与 Qwen 对齐；不含 pad/eos/unk 已由构造函数单独指定）
ADDITIONAL_SPECIAL_TOKENS: list[str] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]


class TokenizerTrainer:
    @staticmethod
    def _align_tokenizer_json(tokenizer_dir: str) -> None:
        tokenizer_json_path = Path(tokenizer_dir) / "tokenizer.json"
        tokenizer_json = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
        tokenizer_json["post_processor"] = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        }
        tokenizer_json_path.write_text(
            json.dumps(tokenizer_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _align_tokenizer_config(tokenizer_dir: str) -> None:
        tokenizer_json_path = Path(tokenizer_dir) / "tokenizer.json"
        tokenizer_cfg_path = Path(tokenizer_dir) / "tokenizer_config.json"

        tokenizer_json = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
        tokenizer_cfg = json.loads(tokenizer_cfg_path.read_text(encoding="utf-8"))

        target_added_tokens: list[tuple[str, bool]] = [
            ("<pad>", True),
            *POST_VOCAB_ADDED_TOKENS,
        ]
        token_attr_map = {
            t.get("content"): {
                "content": t.get("content"),
                "lstrip": bool(t.get("lstrip", False)),
                "normalized": bool(t.get("normalized", False)),
                "rstrip": bool(t.get("rstrip", False)),
                "single_word": bool(t.get("single_word", False)),
                "special": bool(t.get("special", False)),
            }
            for t in tokenizer_json.get("added_tokens", [])
        }
        token_id_map = {
            str(t.get("id")): token_attr_map[t.get("content")]
            for t in tokenizer_json.get("added_tokens", [])
            if t.get("content") in {content for content, _ in target_added_tokens}
        }

        tokenizer_cfg["add_bos_token"] = False
        tokenizer_cfg["add_prefix_space"] = False
        tokenizer_cfg["additional_special_tokens"] = ADDITIONAL_SPECIAL_TOKENS
        tokenizer_cfg["clean_up_tokenization_spaces"] = False
        tokenizer_cfg["eos_token"] = "<|im_end|>"
        tokenizer_cfg["errors"] = "replace"
        tokenizer_cfg["model_max_length"] = 8192
        tokenizer_cfg["pad_token"] = "<pad>"
        tokenizer_cfg["split_special_tokens"] = False
        tokenizer_cfg["unk_token"] = "<unk>"
        tokenizer_cfg["bos_token"] = None
        tokenizer_cfg["added_tokens_decoder"] = token_id_map

        tokenizer_cfg_path.write_text(
            json.dumps(tokenizer_cfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def dump_plain_text(data_path: str, output_path: str, row_range: list | None = None):
        """将 jsonl 转为纯文本，供 BPE 训练。"""
        with open(data_path, 'r', encoding='utf-8') as fin, \
                open(output_path, 'w', encoding='utf-8') as fout:

            for i, line in enumerate(fin):
                if row_range and i < row_range[0]:
                    continue
                if row_range and i >= row_range[1]:
                    break
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        fout.write(text.strip() + "\n")
                except Exception as e:
                    print(f"⚠️ Error parsing json: {line}", e)
                    continue

        print(f"Dumped plain text to {output_path}")

    @staticmethod
    def train_bbpe(input_path: str, tokenizer_dir: str, vocab_size: int = 16000) -> None:
        os.makedirs(tokenizer_dir, exist_ok=True)

        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=Regex(QWEN_SPLIT_REGEX),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=False,
                    trim_offsets=False,
                    use_regex=False,
                ),
            ]
        )
        tokenizer.post_processor = processors.ByteLevel(
            add_prefix_space=False,
            trim_offsets=False,
            use_regex=False,
        )
        tokenizer.decoder = decoders.ByteLevel()

        # 显式声明 pad token，确保其占据 id=0。
        # 其他控制符仍在合并后追加，避免占据词表前部。
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train([input_path], trainer)

        for content, is_special in POST_VOCAB_ADDED_TOKENS:
            tokenizer.add_tokens(
                [
                    AddedToken(
                        content,
                        special=is_special,
                        normalized=False,
                        lstrip=False,
                        rstrip=False,
                        single_word=False,
                    )
                ]
            )

        tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
        print("BBPE tokenizer training finished")

    @staticmethod
    def build_hf_tokenizer(tokenizer_dir: str, model_max_length: int = 8192) -> None:
        tokenizer_dir_path = Path(tokenizer_dir)
        chat_template = (tokenizer_dir_path / "chat_template.jinja").read_text(encoding="utf-8")
        tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = PreTrainedTokenizerFast(
            bos_token=None,
            tokenizer_file=tokenizer_json,
            eos_token="<|im_end|>",
            pad_token="<pad>",
            unk_token="<unk>",
            model_max_length=model_max_length,
            clean_up_tokenization_spaces=False,
            split_special_tokens=False,
            chat_template=chat_template,
            additional_special_tokens=ADDITIONAL_SPECIAL_TOKENS,
        )
        tokenizer.add_bos_token = False
        tokenizer.init_kwargs["add_prefix_space"] = False
        tokenizer.init_kwargs["errors"] = "replace"

        tokenizer.save_pretrained(tokenizer_dir)
        TokenizerTrainer._align_tokenizer_json(tokenizer_dir)
        TokenizerTrainer._align_tokenizer_config(tokenizer_dir)
        print("HF Fast tokenizer saved")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data/pretrain/pretrain_t2t_mini.jsonl"
    tokenizer_dir = project_root / "tokenizer/minilm"
    os.makedirs(tokenizer_dir, exist_ok=True)
    plain_text_path = tokenizer_dir / "train_tokenizer.txt"

    TokenizerTrainer.dump_plain_text(
        str(data_path),
        str(plain_text_path),
        row_range=[0, 10_000],
    )
    TokenizerTrainer.train_bbpe(
        input_path=str(plain_text_path),
        tokenizer_dir=str(tokenizer_dir),
        vocab_size=16000,
    )
    TokenizerTrainer.build_hf_tokenizer(tokenizer_dir=str(tokenizer_dir))
    print("🎉 All done!")


if __name__ == "__main__":
    main()
