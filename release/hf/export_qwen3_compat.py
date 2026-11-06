#!/usr/bin/env python3
"""OPTIONAL COMPATIBILITY EXPORT -- not the release format (that is convert_to_hf.py, model_type "llmua").
llmua checkpoint -> stock Hugging Face `Qwen3ForCausalLM` layout (no trust_remote_code), for tools that only know
Qwen3 (vLLM, llama.cpp). Tensors are re-split into q/k/v and gate/up, so they are NOT the trained tensors as stored.

    python export_qwen3_compat.py <ckpt_final.pt> <out_dir> [--tokenizer results/ua_bpe_32768.json]
                            [--kind auto|base|sft] [--rms-eps 1.1920928955078125e-07] [--no-check]

Why Qwen3: pre-norm decoder, RoPE (rotate-half, theta 10000), RMSNorm, SwiGLU (silu(gate) * up), no biases, tied
embeddings and per-head RMSNorm on q and k over head_dim (applied before RoPE) -- Qwen3's q_norm/k_norm exactly.

Field mapping (ours -> HF), verified against model/transformer.py; the numbers are checked in verify_hf.py:
    embed.weight / lmh.weight (tied)  -> model.embed_tokens.weight (lm_head tied, not stored twice)
    layn.weight                       -> model.norm.weight
    blocks.i.mha.qkv.weight [3E, E]   -> q_proj | k_proj | v_proj   (view(B,T,3,H,W): rows are q, then k, then v;
                                          inside each, head h owns rows h*W:(h+1)*W -- same as HF's view(B,T,H,hd))
    blocks.i.mha.proj.weight          -> o_proj
    blocks.i.mha.q_norm / k_norm      -> self_attn.q_norm / k_norm   (RMSNorm over headsize, weight copied as is)
    blocks.i.ff.lin1.weight [2F, E]   -> gate_proj | up_proj        (gate, up = lin1(x).chunk(2); silu(gate) * up)
    blocks.i.ff.lin2.weight           -> down_proj
    blocks.i.layn1 / layn2            -> input_layernorm / post_attention_layernorm
    model_arch: embeddings->hidden_size, ff_width->intermediate_size, blocks->num_hidden_layers,
                embeddings//headsize->num_attention_heads (= num_key_value_heads, no GQA), headsize->head_dim,
                width->max_position_embeddings.  rope_theta = 10000.0 (hard-coded in MultiheadAttention).
RMSNorm eps: nn.RMSNorm(eps=None) in torch 2.11 uses the fp32 epsilon (1.1920929e-07) for *all* norms, bf16 inputs
included (see eps_micro.py / eps_probe.py), so a single rms_norm_eps = 1.1920928955078125e-07 is exact.

Not representable in a stock model: doc_masking (BOS-segment attention mask used in training/val on packed
windows). A single document, which is how a HF user feeds the model, is unaffected.
"""
import argparse, json, math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_hf import (DEFAULT_TOKENIZER, FP32_EPS, PAD_ID, SFT_NAMES, load_ckpt, save_tokenizer, sft_specials)   # shared with the native exporter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ROPE_THETA = 10000.0                           # MultiheadAttention: 10000.0**(-2i/headsize)







def check_arch(arch):
    """Refuse anything the Qwen3 layout cannot reproduce."""
    need = dict(use_swiglu=True, rms_norm=True, qk_norm=True, no_biases=True, doc_rope=False)
    bad = {k: arch.get(k) for k, v in need.items() if bool(arch.get(k)) != v}
    assert not bad, f'architecture not representable as Qwen3: {bad}'
    assert arch['dropout_p'] == 0.0, 'dropout_p != 0'
    assert arch['embeddings'] % arch['headsize'] == 0


def hf_config(arch, rms_eps, bos_id, eos_id, pad_id):
    from transformers import Qwen3Config
    d, hd = arch['embeddings'], arch['headsize']
    return Qwen3Config(
        vocab_size=arch['vocab_size'], hidden_size=d, intermediate_size=arch['ff_width'],
        num_hidden_layers=arch['blocks'], num_attention_heads=d // hd, num_key_value_heads=d // hd, head_dim=hd,
        hidden_act='silu', max_position_embeddings=arch['width'], rms_norm_eps=rms_eps, rope_theta=ROPE_THETA,
        rope_scaling=None, tie_word_embeddings=True, attention_bias=False, attention_dropout=0.0,
        use_sliding_window=False, bos_token_id=bos_id, eos_token_id=eos_id, pad_token_id=pad_id, use_cache=True)


def map_state_dict(sd, arch):
    d, f, L = arch['embeddings'], arch['ff_width'], arch['blocks']
    assert torch.equal(sd['embed.weight'], sd['lmh.weight']), 'embedding and lm head are not tied'
    new = {'model.embed_tokens.weight': sd['embed.weight'], 'model.norm.weight': sd['layn.weight']}
    for i in range(L):
        p, o = f'blocks.{i}.', f'model.layers.{i}.'
        q, k, v = sd[p + 'mha.qkv.weight'].split(d, dim=0)            # rows: [3][H][W]
        new[o + 'self_attn.q_proj.weight'], new[o + 'self_attn.k_proj.weight'], new[o + 'self_attn.v_proj.weight'] = q, k, v
        new[o + 'self_attn.o_proj.weight'] = sd[p + 'mha.proj.weight']
        new[o + 'self_attn.q_norm.weight'] = sd[p + 'mha.q_norm.weight']
        new[o + 'self_attn.k_norm.weight'] = sd[p + 'mha.k_norm.weight']
        gate, up = sd[p + 'ff.lin1.weight'].split(f, dim=0)           # lin1(x).chunk(2): silu(gate) * up
        new[o + 'mlp.gate_proj.weight'], new[o + 'mlp.up_proj.weight'] = gate, up
        new[o + 'mlp.down_proj.weight'] = sd[p + 'ff.lin2.weight']
        new[o + 'input_layernorm.weight'] = sd[p + 'layn1.weight']
        new[o + 'post_attention_layernorm.weight'] = sd[p + 'layn2.weight']
    mapped = {'embed.weight', 'lmh.weight', 'layn.weight'} | {k for k in sd if k.startswith('blocks.')}
    assert set(sd) == mapped, f'unmapped checkpoint tensors: {set(sd) - mapped}'
    return new


def build_hf(sd, arch, rms_eps, bos_id, eos_id, pad_id):
    from transformers import Qwen3ForCausalLM
    check_arch(arch)
    model = Qwen3ForCausalLM(hf_config(arch, rms_eps, bos_id, eos_id, pad_id))
    res = model.load_state_dict(map_state_dict(sd, arch), strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    assert set(res.missing_keys) <= {'lm_head.weight'}, res.missing_keys      # tied, not stored
    model.tie_weights(); model.eval()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    n_ours = sum(v.numel() for k, v in sd.items() if k != 'lmh.weight')       # tied tensor counted once
    n_hf = sum(p.numel() for p in model.parameters())
    assert n_ours == n_hf, (n_ours, n_hf)
    return model








def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('out_dir')
    ap.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    ap.add_argument('--kind', default='auto', choices=['auto', 'base', 'sft'])
    ap.add_argument('--rms-eps', type=float, default=FP32_EPS)
    ap.add_argument('--no-check', action='store_true', help='skip the random-token smoke check against the native model')
    args = ap.parse_args()

    ck, sd, arch = load_ckpt(args.ckpt)
    cfg = ck.get('cfg') or {}
    sft = (args.kind == 'sft') or (args.kind == 'auto' and cfg.get('kind') == 'sft')
    specials = sft_specials(cfg) if sft else None
    bos_id = int(arch['bos_id']); eos_id = specials['end_id'] if sft else 1       # base: pretraining docs end in <|eos|>=1
    os.makedirs(args.out_dir, exist_ok=True)

    model = build_hf(sd, arch, args.rms_eps, bos_id, eos_id, PAD_ID)
    model.save_pretrained(args.out_dir, safe_serialization=True)       # fp32, tied weight stored once
    from transformers import GenerationConfig
    GenerationConfig(bos_token_id=bos_id, eos_token_id=eos_id, pad_token_id=PAD_ID).save_pretrained(args.out_dir)
    tok = save_tokenizer(args.out_dir, args.tokenizer, specials, arch['width'])
    print(f"saved {args.out_dir}: {'SFT' if sft else 'base'}  d={arch['embeddings']} L={arch['blocks']} "
          f"H={arch['embeddings'] // arch['headsize']} F={arch['ff_width']}  params {sum(p.numel() for p in model.parameters()):,}  "
          f"eps {args.rms_eps:.7e}  bos {bos_id} eos {eos_id}  vocab {len(tok)}")

    if not args.no_check:                                              # smoke check on the artifact as saved
        sys.path.insert(0, REPO)
        from model.transformer import Transformer
        from transformers import AutoModelForCausalLM
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        ref = Transformer(**arch); ref.load_state_dict({k: v for k, v in sd.items()}); ref.doc_masking = False
        ref = ref.to(dev).eval()
        hf = AutoModelForCausalLM.from_pretrained(args.out_dir, dtype=torch.float32).to(dev).eval()
        g = torch.Generator().manual_seed(0)
        x = torch.randint(2, arch['vocab_size'], (2, 512), generator=g).to(dev)      # no BOS: one causal segment
        with torch.no_grad():
            dl = (ref(x).float() - hf(x).logits.float()).abs().max().item()
        print(f'smoke check: max |dlogit| = {dl:.2e} (fp32, random tokens, single document)')
        assert dl < 1e-3, 'conversion mismatch -- check q/k/v order, gate/up order, rope_theta, eps'


if __name__ == '__main__':
    main()
