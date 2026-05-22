from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class TrainDataCollator:

    """预训练 / SFT 共用：动态 padding + packing 段隔离。

    - 预训练：``labels`` 与 ``input_ids`` 一致，全程参与 loss；attention_mask 为 4D 分段因果掩码。
    - SFT：连续 ``labels == ignore_index`` 的前缀段内部全互见，监督段保持因果可见。
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

    def _make_attn_mask(
            self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # 4D 掩码: [1, q_len, k_len]。先按 label 恢复 SFT prefix 语义，
        # 再屏蔽动态 padding 的 query/key。
        seq_len = input_ids.shape[0]
        non_pad = (input_ids != self.pad_token_id)
        valid_query = non_pad.unsqueeze(1)
        valid_key = non_pad.unsqueeze(0)

        attn_mask = self._packing_prefix_attn_mask(input_ids, labels)
        attn_mask = attn_mask & valid_query & valid_key
        return attn_mask.unsqueeze(0)

    def _packing_prefix_attn_mask(
            self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        seq_len = input_ids.shape[0]
        prefix_attn_mask = torch.zeros(
            seq_len, seq_len, dtype=torch.bool, device=input_ids.device
        )

        label_blocks = self.label_block(input_ids, labels)

        for index, (seg_start, seg_end, block_type) in enumerate(label_blocks):
            if block_type == "causal":
                seg_len = seg_end - seg_start
                prefix_attn_mask[seg_start: seg_end, seg_start: seg_end] = torch.tril(
                    torch.ones(seg_len, seg_len, dtype=torch.bool, device=input_ids.device),
                    diagonal=0,
                )
            elif block_type == "prefix":
                next_seg_end = seg_end
                if index + 1 < len(label_blocks):
                    next_start, candidate_end, next_type = label_blocks[index + 1]
                    if next_start == seg_end and next_type == "causal":
                        next_seg_end = candidate_end
                prefix_attn_mask[seg_start: next_seg_end, seg_start: seg_end] = True

        return prefix_attn_mask

    def label_block(self, input_ids: torch.Tensor, labels: torch.Tensor) -> list[tuple[int, int, str]]:
        labels_block: list[tuple[int, int, str]] = []
        ids = input_ids.tolist()
        lbs = labels.tolist()
        seq_len = len(ids)
        start = 0

        while start < seq_len:
            if (
                    lbs[start] == self.ignore_index
                    and ids[start] not in [self.pack_sep_token_id, self.pad_token_id]
            ):
                end = start + 1
                while (
                        end < seq_len
                        and lbs[end] == self.ignore_index
                        and ids[end] not in [self.pack_sep_token_id, self.pad_token_id]
                ):
                    end += 1
                labels_block.append((start, end, "prefix"))
                start = end
                continue

            if lbs[start] == ids[start] and ids[start] != self.pad_token_id:
                end = start + 1
                while end < seq_len and lbs[end] == ids[end] and ids[end] != self.pad_token_id:
                    if ids[end] == self.pack_sep_token_id:
                        end += 1
                        break
                    end += 1
                labels_block.append((start, end, "causal"))
                start = end
                continue

            start += 1

        return labels_block

    def _packed_position_ids_1d(self, input_ids: torch.Tensor) -> torch.Tensor:

        device = input_ids.device
        position_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=device)
        seq_len = input_ids.shape[0]

        current_pos = 0
        in_segment = False

        for t in range(seq_len):
            if input_ids[t] == self.pack_sep_token_id:
                position_ids[t] = 0
                current_pos = 0
                in_segment = False
            if input_ids[t] == self.pad_token_id:
                # 最后的动态填充区，会出现pad
                position_ids[t] = 0
            else:
                if not in_segment:
                    in_segment = True
                    current_pos = 0
                position_ids[t] = current_pos
                current_pos += 1

        return position_ids

