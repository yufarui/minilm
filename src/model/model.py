import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, GenerationMixin, Cache, DynamicCache

from src.config.model_config import MiniLMConfig
from .decode_layer import DecoderLayer
from .minilm_model_output import MiniLMModelOutputWithPast, MiniLMCausalLMOutputWithPast
from .moe import Moe
from .rms_norm import RMSNorm
from .rotary_embedding import RotaryEmbedding


class MiniLMModel(PreTrainedModel):
    config_class = MiniLMConfig

    def __init__(self, config: MiniLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config)
        self.rotary_emb = RotaryEmbedding(config)
        self.gradient_checkpointing = False

        self.post_init()

    @staticmethod
    def _prepare_autoregressive_attention_mask(
            attention_mask: torch.Tensor | None,
            batch_size: int,
            query_length: int,
            key_length: int,
            past_seen_tokens: int,
            device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            query_positions = torch.arange(query_length, device=device) + past_seen_tokens
            key_positions = torch.arange(key_length, device=device)
            return (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

        # 输入约定:
        # - 2D [batch, seq_len]：1/True 表示非 pad 可见，模型内部补齐 causal。
        # - 非 2D（如外部显式传入的 4D）：视为完整掩码，直接透传。
        if attention_mask.dim() != 2:
            return attention_mask

        if attention_mask.size(0) != batch_size:
            raise ValueError(
                f"attention_mask batch size mismatch: got {attention_mask.size(0)}, expected {batch_size}"
            )

        if attention_mask.size(1) != key_length:
            raise ValueError(
                f"attention_mask key length mismatch: got {attention_mask.size(1)}, expected {key_length}"
            )

        key_padding_mask = attention_mask.to(device=device).bool().unsqueeze(1).unsqueeze(1)

        query_positions = torch.arange(query_length, device=device) + past_seen_tokens
        key_positions = torch.arange(key_length, device=device)
        causal_mask = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

        return key_padding_mask & causal_mask

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            use_cache: bool | None = None,
            **kwargs,
    ) -> MiniLMModelOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch, seq_len, hidden_size = inputs_embeds.shape

        past_seen_tokens = (past_key_values.get_seq_length() if past_key_values is not None else 0)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            position_ids = (torch.arange(seq_len, device=inputs_embeds.device) + past_seen_tokens)
            position_ids = position_ids.unsqueeze(0)

        key_length = past_seen_tokens + seq_len
        attention_mask = self._prepare_autoregressive_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch,
            query_length=seq_len,
            key_length=key_length,
            past_seen_tokens=past_seen_tokens,
            device=inputs_embeds.device,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, attn_w = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, Moe)],
            # 和hidden_states保持相同device,dtype
            hidden_states.new_zeros(1).squeeze(),
        )

        return MiniLMModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            aux_loss=aux_loss,
        )


class MiniLmForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniLMConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: MiniLMConfig):
        super().__init__(config)
        self.model = MiniLMModel(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    def _set_gradient_checkpointing(
            self, enable: bool = True, gradient_checkpointing_func=None
    ):
        self.model.gradient_checkpointing = enable

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs,
    ) -> MiniLMCausalLMOutputWithPast:
        """
        :param input_ids: 输入 token 索引；与 inputs_embeds 二选一。
        :param attention_mask: 注意力掩码，用于屏蔽 padding 等无效位置。
        :param position_ids: 各 token 的 RoPE 位置；默认连续 arange。packing 场景下可由 ``TrainDataCollator``
            在 ``<|endoftext|>`` 分段处归零，pad 区固定为 0。
        :param past_key_values: 历史 KV 缓存，用于增量解码（生成时复用前文）。
        :param inputs_embeds: 已算好的词嵌入；与 input_ids 二选一。
        :param labels: 语言建模标签；提供时计算交叉熵 loss。
        :param use_cache: 是否返回/更新 KV 缓存，供逐步生成使用。
        :param logits_to_keep: 仅对末尾若干位置算 logits（int 为最后 k 步，Tensor 为切片索引），推理阶段使用。
        :param kwargs: 透传给底层 MiniLMModel 的其余参数。
        :return: 含 logits、可选 loss、past_key_values 等的因果 LM 输出。
        """

        outputs: MiniLMModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 与 HF CausalLM 一致：传入 num_items_in_batch 等，便于 Trainer 在梯度累积下正确缩放 loss
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs
            )
            loss = loss + outputs.aux_loss

        return MiniLMCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            aux_loss=outputs.aux_loss,
            loss=loss,
        )
