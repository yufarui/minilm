from transformers import PretrainedConfig


class MiniLMConfig(PretrainedConfig):
    model_type = "minilm"

    def __init__(
            self,
            # ===== 词表 =====
            vocab_size: int = 16027,
            pad_token_id: int = 0,

            # ===== 基础结构 =====
            hidden_size: int = 768,
            # FFN中间层大小，默认为hidden_size * 4
            intermediate_size: int = None,
            num_hidden_layers: int = 8,
            num_attention_heads: int = 8,
            num_key_value_heads: int = 2,
            rms_norm_eps: float = 1e-5,

            # 激活函数
            hidden_act: str = "silu",
            # 最大序列长度
            max_position_embeddings: int = 8192,

            # ===== dropout=====
            attention_dropout: float = 0.0,
            hidden_dropout: float = 0.0,

            # ===== RoPE =====
            rope_theta: float = 1000000.0,

            # ===== 推理优化 =====
            use_flash_attention: bool = True,
            # 长上下扩展策略
            inference_rope_scaling: bool = True,

            # Moe对齐deepseek
            moe_enable: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = "sigmoid",
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        # 词表
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        # ===== 基础 =====
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or hidden_size * 4
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps

        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings

        # ===== dropout =====
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        # ===== RoPE =====
        self.rope_theta = rope_theta

        # ===== 推理 =====
        self.use_flash_attention = use_flash_attention
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 8192
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        # Moe
        self.moe_enable = moe_enable
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家

        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        self.scoring_func = scoring_func  # 门控打分：'softmax' | 'sigmoid'


