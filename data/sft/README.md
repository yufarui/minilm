# SFT data

Put SFT JSONL files here.

Supported schemas per line:

1) {"prompt": "...", "response": "..."}
2) {"instruction": "...", "input": "...", "output": "..."}
3) {"text": "..."}  (fallback: prompt is empty, full text as target)

Default config paths:
- train: data/sft/train.jsonl
- val: data/sft/val.jsonl

SFT expects `pretrained_model_path` in `config/sft/data_config.json` to point to a finished pretrain export (same layout as `Trainer.save_model()`, e.g. `checkpoints/pretrain` or `checkpoints/pretrain/checkpoint-xxxx`).
