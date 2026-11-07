#!/usr/bin/env python3
"""llmua checkpoint -> Hugging Face folder with our own architecture (model_type "llmua"; see llmua_hf/).

    python convert_to_hf.py <ckpt_final.pt> <out_dir> [--tokenizer results/ua_bpe_32768.json]
                            [--kind auto|base|sft] [--rms-eps 1.1920928955078125e-07] [--no-check]

The safetensors hold the trained tensors under their training names and shapes (embed, blocks.i.mha.qkv, ...,
fused qkv and fused gate/up; lmh is tied to embed and stored once), nothing reshuffled. The folder also carries
copies of configuration_llmua.py / modeling_llmua.py and an `auto_map`, so
`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` works; with `llmua_hf.register()` it works
without trust_remote_code. Tokenizer: the `tokenizers` JSON with a BOS-only post-processor and a normalizer that maps
\r\n, \r, VT and FF to \n (the model never saw them); for SFT checkpoints the three byte tokens 201/202/203 (VT, FF, CR in
the original vocabulary) are renamed <|usr|>/<|assistant|>/<|end|> and a chat template is written. The tokenizer config
pads on the left, so batched `generate` works with the tokenizer's own padding. The generation config suppresses BOS
(`suppress_tokens: [0]`): the model should never produce it mid-text. Padding is `<|eos|>` (id 1), never BOS: with
pad = BOS, `generate(ids)` without an attention mask would mask out the prompt's own BOS.

Reproducibility: the converter writes every output file itself (weights via safetensors, the two code files copied from
llmua_hf/, and config / generation config / tokenizer files from explicit dicts), so the output does not depend on how a
transformers version would serialise them. Under transformers 4.57.x every file is byte-identical to the released ones
(except the model cards, README.md, which are written by hand). Under 5.x the only difference is the `transformers_version`
value in config.json and generation_config.json, which records the version that is actually running. Both outputs load
on both versions. (Checked with 4.57.6 and 5.22.2; see FIX_REPORT_3_2026-11-06.md.)

RMSNorm eps: nn.RMSNorm(eps=None) in torch 2.11 uses the fp32 epsilon for all norms, bf16 inputs included
(eps_micro.py, eps_probe.py), so config.rms_norm_eps = 1.1920928955078125e-07 reproduces training exactly.
Stock-Qwen3 export for vLLM / llama.cpp: export_qwen3_compat.py.
"""
import argparse, json, os, shutil, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_TOKENIZER = os.path.join(REPO, 'results', 'ua_bpe_32768.json')
FP32_EPS = 1.1920928955078125e-07              # torch.finfo(torch.float32).eps
PAD_ID = 1                                     # <|eos|>: pad must NOT be BOS (id 0): generate() without a mask would mask the prompt's BOS
SFT_NAMES = {'usr_id': '<|usr|>', 'assistant_id': '<|assistant|>', 'end_id': '<|end|>'}
# GPT-2 byte-level characters of the bytes that sft/chat.py reuses as SFT specials (ids 201/202/203 = \x0b, \x0c, \r)
SFT_RAW_BYTES = {'usr_id': 0x0B, 'assistant_id': 0x0C, 'end_id': 0x0D}

# sft/chat.py: ids = [bos, usr] + encode(text) + [assistant]; training examples continue with answer + [end].
# Only the SFT format (one user turn, optionally followed by its answer) was ever trained, so anything else raises.
CHAT_TEMPLATE = (
    "{%- if messages | length < 1 or messages | length > 2 or messages[0]['role'] != 'user'"
    " or (messages | length == 2 and messages[1]['role'] != 'assistant') -%}"
    "{{- raise_exception('LLMUA SFT was trained on single-turn chats only: [user] or [user, assistant]') -}}"
    "{%- endif -%}"
    "{{- bos_token -}}"
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}{{- '<|usr|>' + message['content'] -}}"
    "{%- else -%}{{- '<|assistant|>' + message['content'] + '<|end|>' -}}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt and messages[-1]['role'] == 'user' -%}{{- '<|assistant|>' -}}{%- endif -%}")


def bytes_to_unicode():
    """GPT-2 / tokenizers ByteLevel byte -> printable char table."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


def load_ckpt(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = {k.removeprefix('_orig_mod.'): v.float() for k, v in ck['model'].items()}
    return ck, sd, ck['model_arch']


def sft_specials(cfg):
    """Same defaults as sft/chat.py specials_from_cfg."""
    cfg = cfg or {}
    return {'usr_id': int(cfg.get('usr_id', 201)), 'assistant_id': int(cfg.get('assistant_id', 202)),
            'end_id': int(cfg.get('end_id', 203))}


def tokenizer_json(src, sft=None):
    """The `tokenizers` JSON with three edits: the post-processor adds <|bos|> only (the original also appends <|eos|>,
    wrong for prompting); a normalizer maps \\r\\n, \\r, VT and FF to \\n (so they never reach the byte tokens below);
    for SFT, ids 201-203 (the raw bytes \\x0b \\x0c \\r in the original vocab) are renamed to named specials. No BPE
    merge involves those three characters (asserted), so everything else tokenizes identically."""
    d = json.load(open(src))
    bos = next(t for t in d['added_tokens'] if t['content'] == '<|bos|>')
    pp = d['post_processor']
    procs = pp['processors'] if pp['type'] == 'Sequence' else [pp]
    tp = next(p for p in procs if p['type'] == 'TemplateProcessing')
    tp['single'] = [{'SpecialToken': {'id': bos['content'], 'type_id': 0}}, {'Sequence': {'id': 'A', 'type_id': 0}}]
    assert d['normalizer'] is None
    d['normalizer'] = {'type': 'Sequence', 'normalizers': [
        {'type': 'Replace', 'pattern': {'String': old}, 'content': '\n'} for old in ('\r\n', '\r', '\x0b', '\x0c')]}
    if sft:
        b2u, vocab = bytes_to_unicode(), d['model']['vocab']
        for key, name in SFT_NAMES.items():
            ch = b2u[SFT_RAW_BYTES[key]]
            assert vocab.get(ch) == sft[key], (key, ch, vocab.get(ch), sft[key])
            assert not any(ch in (m if isinstance(m, str) else ' '.join(m)) for m in d['model']['merges']), ch
            del vocab[ch]; vocab[name] = sft[key]
            d['added_tokens'].append(dict(id=sft[key], content=name, single_word=False, lstrip=False,
                                          rstrip=False, normalized=False, special=True))
        d['model']['vocab'] = dict(sorted(vocab.items(), key=lambda kv: kv[1]))      # Rust writes the vocab in id order
        d['added_tokens'].sort(key=lambda t: t['id'])
    return d


def write_json(path, obj, sort_keys=True, newline=True):
    """indent-2 JSON, as transformers 4.57 writes its config files."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(obj, indent=2, sort_keys=sort_keys, ensure_ascii=False) + ('\n' if newline else ''))


def tokenizer_config(d, sft, max_len):
    """tokenizer_config.json in the form transformers 4.57 writes for a PreTrainedTokenizerFast (explicit, so that
    transformers 5, which writes a different schema, gives the same file)."""
    keys = ('content', 'lstrip', 'normalized', 'rstrip', 'single_word', 'special')
    cfg = {'added_tokens_decoder': {str(t['id']): {k: t[k] for k in keys} for t in d['added_tokens']},
           'bos_token': '<|bos|>', 'eos_token': '<|end|>' if sft else '<|eos|>', 'pad_token': '<|eos|>',
           'clean_up_tokenization_spaces': False, 'extra_special_tokens': {},
           'model_input_names': ['input_ids', 'attention_mask'],       # no token_type_ids: model.generate(**tok(text)) must work
           'model_max_length': max_len, 'padding_side': 'left', 'tokenizer_class': 'PreTrainedTokenizerFast'}
    if sft:
        cfg['additional_special_tokens'] = list(SFT_NAMES.values())
    return cfg


def save_tokenizer(out_dir, src, sft, max_len):
    """Writes tokenizer.json (compact dump, as released), tokenizer_config.json, special_tokens_map.json and, for SFT,
    chat_template.jinja; then loads the folder back with transformers and checks the specials."""
    from transformers import PreTrainedTokenizerFast
    d = tokenizer_json(src, sft)
    with open(os.path.join(out_dir, 'tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(d, ensure_ascii=False))
    cfg = tokenizer_config(d, sft, max_len)
    write_json(os.path.join(out_dir, 'tokenizer_config.json'), cfg)
    smap = {k: cfg[k] for k in ('bos_token', 'eos_token', 'pad_token')}
    if sft:
        smap['additional_special_tokens'] = cfg['additional_special_tokens']
    write_json(os.path.join(out_dir, 'special_tokens_map.json'), smap)
    if sft:
        with open(os.path.join(out_dir, 'chat_template.jinja'), 'w', encoding='utf-8') as f:
            f.write(CHAT_TEMPLATE)
    tok = PreTrainedTokenizerFast.from_pretrained(out_dir)
    assert tok.bos_token_id == 0 and tok.pad_token_id == PAD_ID, (tok.bos_token_id, tok.pad_token_id)
    assert tok.eos_token_id == (sft['end_id'] if sft else 1) and tok.padding_side == 'left'
    if sft:
        for key, name in SFT_NAMES.items():
            assert tok.convert_tokens_to_ids(name) == sft[key], name
        assert tok.chat_template == CHAT_TEMPLATE
    return tok


def check_arch(arch):
    """Refuse anything the llmua_hf implementation does not cover."""
    need = dict(use_swiglu=True, rms_norm=True, no_biases=True, doc_rope=False)
    bad = {k: arch.get(k) for k, v in need.items() if bool(arch.get(k)) != v}
    assert not bad, f'architecture not supported by llmua_hf: {bad}'
    assert arch['dropout_p'] == 0.0, 'dropout_p != 0'
    assert arch['embeddings'] % arch['headsize'] == 0


def llmua_config(arch, rms_eps, bos_id, eos_id, pad_id):
    from llmua_hf import LlmuaConfig
    d, hd = arch['embeddings'], arch['headsize']
    return LlmuaConfig(
        vocab_size=arch['vocab_size'], hidden_size=d, intermediate_size=arch['ff_width'],
        num_hidden_layers=arch['blocks'], num_attention_heads=d // hd, head_dim=hd,
        max_position_embeddings=arch['width'], rope_theta=10000.0,       # hard-coded in model/transformer.py
        rms_norm_eps=rms_eps, qk_norm=bool(arch['qk_norm']), tie_word_embeddings=True,
        bos_token_id=bos_id, eos_token_id=eos_id, pad_token_id=pad_id, use_cache=True)


def build_model(sd, arch, rms_eps, bos_id, eos_id, pad_id):
    from llmua_hf import LlmuaForCausalLM
    check_arch(arch)
    assert torch.equal(sd['embed.weight'], sd['lmh.weight']), 'embedding and lm head are not tied'
    model = LlmuaForCausalLM(llmua_config(arch, rms_eps, bos_id, eos_id, pad_id))
    res = model.load_state_dict({k: v for k, v in sd.items() if k != 'lmh.weight'}, strict=False)   # our names, as trained
    assert not res.unexpected_keys, res.unexpected_keys
    assert set(res.missing_keys) <= {'lmh.weight'}, res.missing_keys
    model.tie_weights(); model.eval()
    assert model.lmh.weight.data_ptr() == model.embed.weight.data_ptr()
    n_ours = sum(v.numel() for k, v in sd.items() if k != 'lmh.weight')
    assert n_ours == sum(p.numel() for p in model.parameters()), 'parameter count mismatch'
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('out_dir')
    ap.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    ap.add_argument('--kind', default='auto', choices=['auto', 'base', 'sft'])
    ap.add_argument('--rms-eps', type=float, default=FP32_EPS)
    ap.add_argument('--no-check', action='store_true', help='skip the random-token smoke check against the native model')
    args = ap.parse_args()
    sys.path.insert(0, HERE)

    ck, sd, arch = load_ckpt(args.ckpt)
    cfg = ck.get('cfg') or {}
    sft = (args.kind == 'sft') or (args.kind == 'auto' and cfg.get('kind') == 'sft')
    specials = sft_specials(cfg) if sft else None
    bos_id = int(arch['bos_id']); eos_id = specials['end_id'] if sft else 1       # base: pretraining docs end in <|eos|>=1
    os.makedirs(args.out_dir, exist_ok=True)

    import transformers
    from safetensors.torch import save_file
    model = build_model(sd, arch, args.rms_eps, bos_id, eos_id, PAD_ID)          # sanity: tie, parameter count
    # weights: the trained tensors under their training names, fp32, tied lmh stored once (= what save_pretrained writes)
    save_file({k: v.detach().contiguous() for k, v in sd.items() if k != 'lmh.weight'},
              os.path.join(args.out_dir, 'model.safetensors'), metadata={'format': 'pt'})
    for f in ('configuration_llmua.py', 'modeling_llmua.py'):                     # the code that trust_remote_code loads
        shutil.copyfile(os.path.join(HERE, 'llmua_hf', f), os.path.join(args.out_dir, f))
    ver = transformers.__version__                                                 # honest: the version that is running
    c = llmua_config(arch, args.rms_eps, bos_id, eos_id, PAD_ID)
    config = {'architectures': ['LlmuaForCausalLM'],
              'auto_map': {'AutoConfig': 'configuration_llmua.LlmuaConfig', 'AutoModel': 'modeling_llmua.LlmuaModel',
                           'AutoModelForCausalLM': 'modeling_llmua.LlmuaForCausalLM'},
              'bos_token_id': bos_id, 'dtype': 'float32', 'eos_token_id': eos_id, 'head_dim': c.head_dim, 'hidden_size': c.hidden_size,
              'intermediate_size': c.intermediate_size, 'max_position_embeddings': c.max_position_embeddings, 'model_type': 'llmua',
              'num_attention_heads': c.num_attention_heads, 'num_hidden_layers': c.num_hidden_layers, 'pad_token_id': PAD_ID,
              'qk_norm': c.qk_norm, 'rms_norm_eps': c.rms_norm_eps, 'rope_theta': c.rope_theta, 'transformers_version': ver,
              'use_cache': True, 'vocab_size': c.vocab_size}
    assert c.model_type == 'llmua' and c.tie_word_embeddings is True               # tie_word_embeddings is the default, so not written
    write_json(os.path.join(args.out_dir, 'config.json'), config)
    gen = {'bos_token_id': bos_id, 'eos_token_id': eos_id, 'pad_token_id': PAD_ID, 'transformers_version': ver,
           'suppress_tokens': [bos_id]}                                            # never emit BOS mid-text
    write_json(os.path.join(args.out_dir, 'generation_config.json'), gen, sort_keys=False, newline=False)
    tok = save_tokenizer(args.out_dir, args.tokenizer, specials, arch['width'])
    print(f"saved {args.out_dir}: {'SFT' if sft else 'base'}  d={arch['embeddings']} L={arch['blocks']} "
          f"H={arch['embeddings'] // arch['headsize']} F={arch['ff_width']}  params {sum(p.numel() for p in model.parameters()):,}  "
          f"eps {args.rms_eps:.7e}  bos {bos_id} eos {eos_id}  vocab {len(tok)}")

    if not args.no_check:                                                         # smoke check on the artifact as saved
        sys.path.insert(0, REPO)
        from model.transformer import Transformer
        from transformers import AutoModelForCausalLM
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        ref = Transformer(**arch); ref.load_state_dict(sd); ref.doc_masking = False
        ref = ref.to(dev).eval()
        hf = AutoModelForCausalLM.from_pretrained(args.out_dir, dtype=torch.float32, trust_remote_code=True).to(dev).eval()
        g = torch.Generator().manual_seed(0)
        x = torch.randint(2, arch['vocab_size'], (2, 512), generator=g).to(dev)      # no BOS: one causal segment
        with torch.no_grad():
            dl = (ref(x).float() - hf(x).logits.float()).abs().max().item()
        print(f'smoke check: max |dlogit| = {dl:.2e} (fp32, random tokens, single document)')
        assert dl < 1e-3, 'conversion mismatch'


if __name__ == '__main__':
    main()
