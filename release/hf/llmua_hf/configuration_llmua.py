"""LLMUA model configuration (Hugging Face `PretrainedConfig`)."""
from transformers import PretrainedConfig


class LlmuaConfig(PretrainedConfig):
    """Config of the LLMUA decoder: pre-norm RMSNorm blocks, fused-qkv multi-head attention with optional per-head
    QK-RMSNorm (applied before RoPE), rotate-half RoPE, fused gate/up SwiGLU feed-forward, no biases, tied embeddings.

    Field names follow the Hugging Face convention where generation needs them; the rest are LLMUA's own.
    `num_attention_heads * head_dim` must equal `hidden_size` (no grouped-query attention).
    """
    model_type = "llmua"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=14,
        num_attention_heads=12,
        head_dim=64,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rms_norm_eps=1.1920928955078125e-07,
        qk_norm=True,
        use_cache=True,
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        **kwargs,
    ):
        if num_attention_heads * head_dim != hidden_size:
            raise ValueError(f"num_attention_heads * head_dim ({num_attention_heads} * {head_dim}) != hidden_size ({hidden_size})")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size          # width of each of gate and up (lin1 has 2x this many rows)
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.qk_norm = qk_norm
        self.use_cache = use_cache
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                         tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["LlmuaConfig"]
