import torch

from src.util.data_collator import TrainDataCollator


def _make_causal_attention(input_ids: torch.Tensor):
    # 自回归模型，默认添加自回归掩码
    batch, seq_len = input_ids.shape
    mask = torch.full((batch, seq_len, seq_len),
                      1, device=input_ids.device, dtype=torch.long)
    mask.tril_(diagonal=0)

    for b in range(batch):
        pad_positions = torch.where(input_ids[b] == 0)
        print(pad_positions)
        for i in pad_positions:
            mask[b, i, :] = 0
            mask[b, :, i] = 0

    return mask.unsqueeze(1)


if __name__ == "__main__":
    input = torch.tensor([
        [10, 11, 12, 0, 13, 14, 15, 0, 17],
        [10, 11, 12, 13, 14, 15, 16, 17, 0]
    ])

    attention_mask = _make_causal_attention(input)
    print(attention_mask)
