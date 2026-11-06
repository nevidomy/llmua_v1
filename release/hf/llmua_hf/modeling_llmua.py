"""LLMUA decoder in plain PyTorch + Hugging Face plumbing (generation, caching, save/load).

Module and tensor names are those of the training code (`model/transformer.py`), so the safetensors files hold the
trained tensors unchanged:

    embed.weight                      (tied with lmh.weight, stored once)
    blocks.{i}.mha.qkv.weight         fused [3E, E]; rows are q | k | v, head h of each owns rows h*W:(h+1)*W
    blocks.{i}.mha.proj.weight
    blocks.{i}.mha.q_norm.weight      per-head RMSNorm over head_dim, applied before RoPE (qk_norm=True)
    blocks.{i}.mha.k_norm.weight
    blocks.{i}.ff.lin1.weight         fused [2F, E]; gate = first F rows, up = last F rows; silu(gate) * up
    blocks.{i}.ff.lin2.weight
    blocks.{i}.layn1.weight / layn2.weight
    layn.weight                       final RMSNorm
    lmh.weight                        output head (tied)

Depends on torch and transformers only.
"""
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .configuration_llmua import LlmuaConfig


def rope_table(width: int, head_dim: int, theta: float):
    """cos/sin tables [width, head_dim/2], built exactly like the training code (fp32, on CPU)."""
    ang = torch.arange(0, width).view(-1, 1) * torch.tensor([theta ** (-2 * i / head_dim) for i in range(head_dim // 2)])
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    """Rotate-half RoPE. x: [B, H, T, W]; cos, sin: [B, 1, T, W/2]."""
    xx, yy = x.chunk(2, -1)
    return torch.cat([xx * cos - yy * sin, xx * sin + yy * cos], -1).type_as(x)


def build_attention_mask(attention_mask, q_len: int, k_len: int, device):
    """Returns (bool mask broadcastable to [B,1,q_len,k_len] with True = attend, or None, is_causal).

    - attention_mask None, q_len == k_len: plain causal attention (is_causal fast path, same as training).
    - 2-D attention_mask [B, k_len] (1 = real token, 0 = padding): causal AND key-not-padded. Rows that would attend
      to nothing (padding queries on the left) are opened up so softmax never sees an all-masked row; their outputs
      are never used.
    - 4-D attention_mask: taken as a ready bool mask (True = attend), e.g. for packed documents.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask.bool(), False
    if attention_mask is None and q_len == k_len:
        return None, True
    if attention_mask is None and q_len == 1:
        return None, False
    past = k_len - q_len
    qi = torch.arange(q_len, device=device)[:, None] + past
    ki = torch.arange(k_len, device=device)[None, :]
    allow = (ki <= qi)[None, None]
    if attention_mask is not None:
        if attention_mask.shape[-1] != k_len:
            raise ValueError(f"attention_mask has length {attention_mask.shape[-1]}, expected past + current = {k_len}")
        allow = allow & attention_mask[:, None, None, :].bool()
        allow = allow | ~allow.any(-1, keepdim=True)
    return allow, False


class MultiheadAttention(nn.Module):
    def __init__(self, config: LlmuaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.headsize, self.headcount = config.head_dim, config.num_attention_heads
        E = config.hidden_size
        self.qkv = nn.Linear(E, 3 * E, bias=False)
        self.proj = nn.Linear(E, E, bias=False)
        self.q_norm = nn.RMSNorm(self.headsize, eps=config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = nn.RMSNorm(self.headsize, eps=config.rms_norm_eps) if config.qk_norm else None

    def forward(self, x, cos, sin, attn_mask, is_causal, past_key_values: Optional[Cache] = None):
        B, T, E = x.shape
        H, W = self.headcount, self.headsize
        q, k, v = self.qkv(x).view(B, T, 3, H, W).permute(2, 0, 3, 1, 4)          # each [B, H, T, W]
        if self.q_norm is not None:                                                # per head, over head_dim, before RoPE
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past_key_values is not None:                                            # cache holds post-RoPE k and v
            k, v = past_key_values.update(k, v, self.layer_idx)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)   # scale 1/sqrt(head_dim)
        return self.proj(a.transpose(1, 2).reshape(B, T, E))


class FeedForward(nn.Module):
    def __init__(self, config: LlmuaConfig):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)   # gate and up, fused
        self.lin2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(F.silu(gate) * up)


class AttentionBlock(nn.Module):
    def __init__(self, config: LlmuaConfig, layer_idx: int):
        super().__init__()
        self.mha = MultiheadAttention(config, layer_idx)
        self.ff = FeedForward(config)
        self.layn1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layn2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask, is_causal, past_key_values=None):
        x = x + self.mha(self.layn1(x), cos, sin, attn_mask, is_causal, past_key_values)
        return x + self.ff(self.layn2(x))


class LlmuaPreTrainedModel(PreTrainedModel):
    config_class = LlmuaConfig
    base_model_prefix = ""                     # tensors are stored without a wrapper prefix (embed.*, blocks.*, layn.*, ...)
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _no_split_modules = ["AttentionBlock"]
    _supports_sdpa = True

    def _init_weights(self, module):
        # training-time `custom_init`: N(0, 0.55 / sqrt(fan_in)) for linear and embedding weights
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.55 / math.sqrt(module.weight.shape[-1]))
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)


class _Backbone:
    """embed -> blocks -> layn, shared by LlmuaModel and LlmuaForCausalLM (which both keep the modules at top level)."""

    def _build_backbone(self, config):
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([AttentionBlock(config, i) for i in range(config.num_hidden_layers)])
        self.layn = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._rope = {}

    def _rope_for(self, device):
        key = str(device)
        if key not in self._rope:                                              # computed on CPU like training, then moved
            cos, sin = rope_table(self.config.max_position_embeddings, self.config.head_dim, self.config.rope_theta)
            self._rope[key] = (cos.to(device), sin.to(device))
        return self._rope[key]

    def _run_backbone(self, input_ids, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache, output_hidden_states):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids and inputs_embeds")
        h = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        B, T, _ = h.shape
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            if past + T > self.config.max_position_embeddings:
                raise ValueError(f"sequence length {past + T} exceeds max_position_embeddings ({self.config.max_position_embeddings})")
            position_ids = (past + torch.arange(T, device=h.device))[None].expand(B, T)
        cos_t, sin_t = self._rope_for(h.device)
        cos, sin = cos_t[position_ids].unsqueeze(1), sin_t[position_ids].unsqueeze(1)           # [B, 1, T, W/2]
        attn_mask, is_causal = build_attention_mask(attention_mask, T, past + T, h.device)
        states = [h] if output_hidden_states else None
        for block in self.blocks:
            h = block(h, cos, sin, attn_mask, is_causal, past_key_values if use_cache else None)
            if output_hidden_states:
                states.append(h)
        return self.layn(h), (past_key_values if use_cache else None), (tuple(states) if states else None)


class LlmuaModel(_Backbone, LlmuaPreTrainedModel):
    def __init__(self, config: LlmuaConfig):
        super().__init__(config)
        self._build_backbone(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                use_cache=None, output_hidden_states=None, return_dict=None, **kwargs):
        h, cache, states = self._run_backbone(input_ids, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache,
                                              output_hidden_states)
        if return_dict is False:
            return tuple(x for x in (h, cache, states) if x is not None)
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=cache, hidden_states=states)


class LlmuaForCausalLM(_Backbone, LlmuaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lmh.weight"]

    def __init__(self, config: LlmuaConfig):
        super().__init__(config)
        self._build_backbone(config)
        self.lmh = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()                                                       # also ties lmh.weight to embed.weight

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lmh

    def set_output_embeddings(self, new_embeddings):
        self.lmh = new_embeddings

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                labels=None, use_cache=None, output_hidden_states=None, return_dict=None, cache_position=None,
                logits_to_keep: Union[int, torch.Tensor] = 0, **kwargs):
        h, cache, states = self._run_backbone(input_ids, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache,
                                              output_hidden_states)
        if isinstance(logits_to_keep, int):
            keep = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            keep = logits_to_keep
        logits = self.lmh(h[:, keep, :])
        loss = None
        if labels is not None:                                                 # HF convention: labels are shifted here
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
        if return_dict is False:
            return tuple(x for x in (loss, logits, cache, states) if x is not None)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache, hidden_states=states)


__all__ = ["LlmuaPreTrainedModel", "LlmuaModel", "LlmuaForCausalLM"]
