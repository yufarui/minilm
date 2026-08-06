from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class TrainDataCollator:

    """预训练 / SFT 共用：动态 padding + 因果掩码。

    - 预训练：``labels`` 与 ``input_ids`` 一致，全程参与 loss；attention_mask 为 4D 分段因果掩码。
    - SFT：保留数据侧 ``labels=-100`` 的监督选择逻辑。SFT **不做 packing**，应关闭
      ``enable_pack_segments``，否则对话正文里字面量 ``<|endoftext|>``（``split_special_tokens=false``
      时会编成 pack 分隔符 id）会把注意力切段，使 assistant 监督位看不到用户问题。
    - 预训练 packing：attention 在每个 ``<|endoftext|>`` 分段内独立计算；RoPE 在分隔符处归零。
    - batch 右侧 padding 的 ``position_ids`` 为 0（与掩码一致）。
    """

    def __init__(
            self,
            tokenizer,
            ignore_index: int = -100,
            enable_pack_segments: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        pid = tokenizer.pad_token_id
        self.pad_token_id = int(pid)
        sep = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.pack_sep_token_id = int(sep)
        self.ignore_index = ignore_index
        self.enable_pack_segments = bool(enable_pack_segments)

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

            attn_mask = self._make_attn_mask(padded_ids)

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

    def _make_attn_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 4D 掩码: [1, q_len, k_len]
        # 1) pad query/key 全屏蔽
        # 2) 因果可见（j <= i）
        # 3) 可选：仅同一 pack 分段（由 <|endoftext|> 切分）内可见
        seq_len = input_ids.shape[0]
        device = input_ids.device

        non_pad = (input_ids != self.pad_token_id)
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        valid_query = non_pad.unsqueeze(1)
        valid_key = non_pad.unsqueeze(0)

        if not self.enable_pack_segments:
            attn_mask = causal & valid_query & valid_key
            return attn_mask.unsqueeze(0)

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
        attn_mask = same_segment & causal & valid_query & valid_key
        return attn_mask.unsqueeze(0)

    def _packed_position_ids_1d(self, input_ids: torch.Tensor) -> torch.Tensor:

        device = input_ids.device
        position_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=device)
        seq_len = input_ids.shape[0]

        if not self.enable_pack_segments:
            current_pos = 0
            for t in range(seq_len):
                if input_ids[t] == self.pad_token_id:
                    position_ids[t] = 0
                else:
                    position_ids[t] = current_pos
                    current_pos += 1
            return position_ids

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

