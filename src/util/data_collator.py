from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class TrainDataCollator:

    """预训练 / SFT 共用：动态 padding + packing 段隔离。

    - 预训练：``labels`` 与 ``input_ids`` 一致，全程参与 loss → 标准因果掩码 + packing 隔断。
    - SFT：仅部分位置 ``labels != ignore_index`` → 段内前缀全互见 + 监督段因果。
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
        # 4D 掩码: [1, q_len, k_len]
        # 1) pad query/key 全屏蔽
        # 2) 仅同一 pack 分段（由 <|endoftext|> 切分）内可见
        # 3) 预训练为段内因果；SFT 的 labels=-100 前缀块在段内全互见
        seq_len = input_ids.shape[0]
        device = input_ids.device
        labels = labels.to(device=device)

        non_pad = (input_ids != self.pad_token_id)
        segment_ids = torch.full((seq_len,), -1, dtype=torch.long, device=device)

        current_segment = 0
        for idx in range(seq_len):
            token_id = input_ids[idx]
            if token_id == self.pad_token_id:
                continue
            segment_ids[idx] = current_segment
            if token_id == self.pack_sep_token_id:
                current_segment += 1

        same_segment = (segment_ids.unsqueeze(0) == segment_ids.unsqueeze(1)) & (segment_ids.unsqueeze(0) >= 0)
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        valid_query = non_pad.unsqueeze(1)
        valid_key = non_pad.unsqueeze(0)

        attn_mask = same_segment & causal & valid_query & valid_key
        self._apply_prefix_visibility(attn_mask, input_ids, labels, segment_ids)
        return attn_mask.unsqueeze(0)

    def _apply_prefix_visibility(
        self,
        attn_mask: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> None:
        seq_len = input_ids.shape[0]
        pos = 0

        while pos < seq_len:
            segment_id = int(segment_ids[pos].item())
            if segment_id < 0:
                pos += 1
                continue

            segment_start = pos
            while pos < seq_len and int(segment_ids[pos].item()) == segment_id:
                pos += 1
            segment_end = pos

            block_start = segment_start
            while block_start < segment_end:
                if not self._is_prefix_token(input_ids[block_start], labels[block_start]):
                    block_start += 1
                    continue

                prefix_start = block_start
                prefix_end = prefix_start + 1
                while (
                    prefix_end < segment_end
                    and self._is_prefix_token(input_ids[prefix_end], labels[prefix_end])
                ):
                    prefix_end += 1

                next_block_end = prefix_end
                while (
                    next_block_end < segment_end
                    and not self._is_prefix_token(input_ids[next_block_end], labels[next_block_end])
                ):
                    next_block_end += 1

                attn_mask[prefix_start:next_block_end, segment_start:prefix_end] = True
                block_start = prefix_end

    def _is_prefix_token(self, token_id: torch.Tensor, label: torch.Tensor) -> bool:
        return (
            int(label.item()) == self.ignore_index
            and int(token_id.item()) != self.pack_sep_token_id
            and int(token_id.item()) != self.pad_token_id
        )

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

