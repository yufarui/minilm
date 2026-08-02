import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoeGate(nn.Module):
    """
    线性门控：
    - hidden → logits
    - scoring（softmax / sigmoid；默认 sigmoid，与 DeepSeek-V3 报告中 affinity 用 sigmoid 一致）
    - Top-K expert 选择
    - 返回 (topk_idx, topk_weight, aux_loss)

    与 DeepSeek-V3 技术报告（arXiv:2412.19437）的差异简述：
    - V3 主负载均衡是 **auxiliary-loss-free**：对每条 logits 加专家偏置 b_i，仅用 s+b 做 Top-K，
      真正与 expert 输出相乘的 gating 仍来自 **原始** sigmoid affinity；偏置按步更新 γ。
    - V3 **补充**序列级平衡损失 L_Bal = α Σ_i f_i P_i（整段上统计），α 很小。
    - 本实现 **未** 实现偏置 b_i 与无辅助主策略；aux 仅为 batch 内 load 与 prob 的对齐项（点积形式），
      且 Top-K 与权重均基于 **同一套** scores（无“路由用加偏置、输出用无偏置”的解耦）。
    - `seq_aux` 仅保留配置字段，当前 forward 未按序列切分计算（与 V3 的 sequence-wise 项不同）。
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sigmoid")

        self.hidden_size = config.hidden_size

        self.weight = nn.Parameter(
            torch.empty(self.n_routed_experts, self.hidden_size)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states, token_mask: torch.Tensor | None = None):
        """
        hidden_states: [B, S, H]
        token_mask: optional [B, S] bool/0-1；True/1 = 参与 aux 统计的真实 token。
            动态 padding 下 pad 位必须排除，否则（尤其 flash/SDPA 下 pad 隐状态恒为 0）
            所有 pad 会塌缩到同一 expert，严重扭曲负载均衡梯度。
        """
        B, S, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)  # [N, H]

        # ===== gating logits =====
        logits = F.linear(x, self.weight)  # [N, E]

        if self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits, )
        else:
            raise NotImplementedError

        # ===== Top-K =====
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1)  # [N, K]

        # ===== 归一化 =====
        if self.top_k > 1 and self.norm_topk_prob:
            denom = topk_weight.sum(dim=-1, keepdim=True) + 1e-9
            topk_weight = topk_weight / denom

        # ===== aux loss（负载均衡）=====
        aux_loss = torch.tensor(0.0, device=hidden_states.device)

        if self.training and self.alpha > 0:
            if token_mask is not None:
                flat_keep = token_mask.reshape(-1).bool()
            else:
                flat_keep = None

            if flat_keep is not None and not flat_keep.any():
                aux_loss = hidden_states.new_zeros(())
            else:
                idx_for_aux = topk_idx if flat_keep is None else topk_idx[flat_keep]
                scores_for_aux = scores if flat_keep is None else scores[flat_keep]

                # 每个 expert 被路由到的 token 占比（load 向量），归一化后 sum(load)=1
                one_hot = F.one_hot(idx_for_aux, num_classes=self.n_routed_experts).float()  # [M, K, E]
                load = one_hot.mean(dim=(0, 1))  # [E]

                # 每个 expert 的平均路由概率（prob 向量），归一化后 sum(prob)=1
                prob_raw = scores_for_aux
                if self.scoring_func == "sigmoid":
                    prob_raw = prob_raw / (prob_raw.sum(dim=-1, keepdim=True) + 1e-9)
                prob = prob_raw.mean(dim=0)  # [E]
                prob = prob / (prob.sum() + 1e-9)

                # load 与 prob 对齐（Switch 风格：E * dot(load, prob)）
                aux_loss = self.alpha * (load * prob).sum() * self.n_routed_experts

        return topk_idx, topk_weight, aux_loss