from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class TrainDataCollator:

    """预训练 / SFT 共用：动态 padding + packing 段隔离。

    - 预训练：``labels`` 与 ``input_ids`` 一致，全程参与 loss；attention_mask 为 4D 分段因果掩码。
    - SFT：仅部分位置 ``labels != ignore_index``；每个 pack 段内 prefix 全互见，监督段保持因果。
    - RoPE：在 **pack 分隔符**（默认 ``<|endoftext|>``）处将位置计数归零，
      使各文档段内为 0,1,2,…；batch 右侧对齐填充的 ``position_ids`` 为 0（与掩码一致）。
    """

    def __init__(
            self,
            tokenizer,
            ignore_index: int = -100,
    ) -> None:
        self.tokenizer = tokenizer
        pid = tokenizer.pad_token_id
        self.pad_token_id = int(pid)
        sep = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.pack_sep_token_id = int(sep)
        self.ignore_index = ignore_index

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return self._dynamic_pad(features)

    def _dynamic_pad(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("TrainDataCollator: empty batch")

        max_length = max(len(f["input_ids"]) for f in features)

        batch_input_ids: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        batch_position_ids: List[torch.Tensor] = []
        attention_masks: List[torch.Tensor] = []

        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            if not isinstance(lab, torch.Tensor):
                lab = torch.tensor(lab, dtype=torch.long)

            pos_ids = self._packed_position_ids_1d(ids)

            pad_len = max_length - len(ids)

            if pad_len > 0:
                padded_ids = torch.cat(
                    [ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)]
                )
                padded_lab = torch.cat(
                    [lab, torch.full((pad_len,), self.ignore_index, dtype=lab.dtype)]
                )
                padded_pos = torch.cat(
                    [
                        pos_ids,
                        torch.zeros((pad_len,), dtype=torch.long, device=ids.device),
                    ]
                )
            else:
                padded_ids = ids
                padded_lab = lab
                padded_pos = pos_ids

            attn_mask = self._make_attn_mask(padded_ids, padded_lab)

            batch_input_ids.append(padded_ids)
            batch_labels.append(padded_lab)
            batch_position_ids.append(padded_pos)
            attention_masks.append(attn_mask)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.stack(batch_position_ids),
            "attention_mask": torch.stack(attention_masks),
        }

    def _make_attn_mask(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[0]
        device = input_ids.device
        attn_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

        seg_start = 0
        while seg_start < seq_len:
            if input_ids[seg_start] == self.pad_token_id:
                seg_start += 1
                continue

            seg_end = seg_start
            while seg_end < seq_len and input_ids[seg_end] != self.pad_token_id:
                seg_end += 1
                if input_ids[seg_end - 1] == self.pack_sep_token_id:
                    break

            self._fill_prefix_causal_segment(attn_mask, labels, seg_start, seg_end)
            seg_start = seg_end

        return attn_mask.unsqueeze(0)

    def _fill_prefix_causal_segment(
        self,
        attn_mask: torch.Tensor,
        labels: torch.Tensor,
        seg_start: int,
        seg_end: int,
    ) -> None:
        blocks: list[tuple[int, int, str]] = []
        start = seg_start
        while start < seg_end:
            block_type = "prefix" if labels[start] == self.ignore_index else "causal"
            end = start + 1
            if block_type == "prefix":
                while end < seg_end and labels[end] == self.ignore_index:
                    end += 1
            else:
                while end < seg_end and labels[end] != self.ignore_index:
                    end += 1
            blocks.append((start, end, block_type))
            start = end

        for index, (start, end, block_type) in enumerate(blocks):
            if block_type == "causal":
                seg_len = end - start
                attn_mask[start:end, start:end] = torch.tril(
                    torch.ones((seg_len, seg_len), dtype=torch.bool, device=attn_mask.device)
                )
                continue

            rows_end = blocks[index + 1][1] if index + 1 < len(blocks) else end
            attn_mask[start:rows_end, seg_start:end] = True

    def _packed_position_ids_1d(self, input_ids: torch.Tensor) -> torch.Tensor:

        device = input_ids.device
        position_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=device)
        seq_len = input_ids.shape[0]

        current_pos = 0
        in_segment = False

        for t in range(seq_len):
            if input_ids[t] == self.pad_token_id:
                # 最后的动态填充区，会出现pad
                position_ids[t] = 0
                continue
            if not in_segment:
                in_segment = True
                current_pos = 0
            position_ids[t] = current_pos
            if input_ids[t] == self.pack_sep_token_id:
                current_pos = 0
                in_segment = False
            else:
                current_pos += 1

        return position_ids

