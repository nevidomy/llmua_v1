#!/usr/bin/env python3
"""Checks 4-6 for the native `llmua` HF folders.   python verify_llmua_extra.py batched|roundtrip|register|noregister --hf DIR ... --out JSON
batched    left-padded batched greedy generate (4 prompts, different lengths) == per-prompt generate  (fp32, SFT folder)
roundtrip  trust_remote_code load -> save_pretrained -> load: every tensor bit-identical, and == the training checkpoint's tensors
register   after llmua_hf.register(): load with trust_remote_code=False (+ AutoModel base class loads from the same files)
noregister negative control: same load without register() must fail"""
import argparse, json, os, shutil, sys, tempfile
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, REPO)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
DEV = 'cuda'


def stage_batched(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sft.batcher import SFTSplit
    from tokenizers import Tokenizer
    assert 'llmua_hf' not in sys.modules
    hf = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.float32, trust_remote_code=True).to(DEV).eval()
    assert type(hf).__module__.startswith('transformers_modules')
    tok = AutoTokenizer.from_pretrained(a.hf); tok.padding_side = 'left'
    ref = Tokenizer.from_file(os.path.join(REPO, 'results', 'ua_bpe_32768.json'))
    val = SFTSplit(a.sft_data, 'val')
    cands = []
    for i in np.linspace(3, val.n - 3, 300).astype(int):
        e = val.example(int(i)).tolist(); j = e.index(202, 2)
        if len(e) - j >= 60:
            cands.append((j, ref.decode(e[2:j], skip_special_tokens=False)))
    cands.sort(key=lambda c: c[0])
    pick = [cands[int(q * (len(cands) - 1))] for q in (0.0, 0.33, 0.66, 1.0)]
    texts = [t for _, t in pick]
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': t}], add_generation_prompt=True, tokenize=False) for t in texts]
    singles_in = [tok(p, add_special_tokens=False, return_tensors='pt').input_ids[0] for p in prompts]
    assert len({len(x) for x in singles_in}) == 4, 'prompt lengths must differ'
    kw = dict(do_sample=False, num_beams=1, max_new_tokens=a.max_new, eos_token_id=203, pad_token_id=1, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        single = []
        for x in singles_in:
            o = hf.generate(x[None].to(DEV), attention_mask=torch.ones(1, len(x), dtype=torch.long, device=DEV), **kw)[0, len(x):].tolist()
            single.append(o)
        enc = tok(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to(DEV)
        out = hf.generate(**enc, **kw)[:, enc.input_ids.shape[1]:].tolist()
        # logits of the last prompt position: padded batch vs alone
        lb = hf(**enc).logits[:, -1].float()
        dl = max((lb[i] - hf(input_ids=singles_in[i][None].to(DEV)).logits[0, -1].float()).abs().max().item() for i in range(4))
    rows = []
    for i in range(4):
        b = out[i]
        if 203 in b: b = b[:b.index(203) + 1]
        rows.append({'prompt_tokens': len(singles_in[i]), 'pad_tokens': int((enc.attention_mask[i] == 0).sum()), 'new_tokens_single': len(single[i]),
                     'new_tokens_batched': len(b), 'equal': b == single[i], 'ends_with_end': single[i][-1] == 203})
    res = {'hf': a.hf, 'hf_class': type(hf).__module__, 'batch_shape': list(enc.input_ids.shape), 'rows': rows, 'all_equal': all(r['equal'] for r in rows),
           'max_abs_dlogit_last_prompt_position_padded_vs_alone': dl, 'max_new_tokens': a.max_new}
    json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res, indent=1))


def stage_roundtrip(a):
    from transformers import AutoModelForCausalLM
    from safetensors.torch import load_file
    assert 'llmua_hf' not in sys.modules
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)['model']
    orig = {k.removeprefix('_orig_mod.'): v for k, v in ck.items() if not k.endswith('lmh.weight')}
    m1 = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.float32, trust_remote_code=True)
    tmp = a.tmp; shutil.rmtree(tmp, ignore_errors=True)
    m1.save_pretrained(tmp)
    m2 = AutoModelForCausalLM.from_pretrained(tmp, dtype=torch.float32, trust_remote_code=True)
    s1, s2 = m1.state_dict(), m2.state_dict()
    files_orig, files_saved = load_file(os.path.join(a.hf, 'model.safetensors')), load_file(os.path.join(tmp, 'model.safetensors'))
    res = {'hf': a.hf, 'n_tensors_state_dict': len(s1), 'tensors_in_safetensors': len(files_saved),
           'state_dict_equal_after_roundtrip': set(s1) == set(s2) and all(torch.equal(s1[k], s2[k]) for k in s1),
           'saved_file_equals_original_file': set(files_orig) == set(files_saved) and all(torch.equal(files_orig[k], files_saved[k]) for k in files_orig),
           'saved_file_equals_training_ckpt_tensors': set(files_saved) == set(orig) and all(torch.equal(files_saved[k], orig[k]) for k in orig),
           'training_ckpt_tensors_not_in_file': sorted(set(ck) - set(orig)), 'lmh_tied_after_load': m2.lmh.weight.data_ptr() == m2.embed.weight.data_ptr(),
           'saved_dir_has_code_and_auto_map': all(os.path.isfile(os.path.join(tmp, f)) for f in ('configuration_llmua.py', 'modeling_llmua.py')) and 'auto_map' in json.load(open(os.path.join(tmp, 'config.json'))),
           'key_sample': sorted(files_saved)[:5]}
    res['all_pass'] = all(v for k, v in res.items() if isinstance(v, bool))
    shutil.rmtree(tmp, ignore_errors=True)
    json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res, indent=1))


def stage_register(a):
    sys.path.insert(0, os.path.join(REPO, 'release', 'hf'))
    import llmua_hf
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    llmua_hf.register()
    llmua_hf.register()                                               # idempotent
    cfg = AutoConfig.from_pretrained(a.hf, trust_remote_code=False)
    m, info = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.float32, trust_remote_code=False, output_loading_info=True)
    b, binfo = AutoModel.from_pretrained(a.hf, dtype=torch.float32, trust_remote_code=False, output_loading_info=True)
    m, b = m.to(DEV).eval(), b.to(DEV).eval()
    x = torch.randint(2, m.config.vocab_size, (2, 300), generator=torch.Generator().manual_seed(1)).to(DEV)
    with torch.no_grad():
        lg = m(x).logits; hb = b(x).last_hidden_state
        same_head = torch.equal(m.lmh(hb), lg)
        gen = m.generate(x[:1, :20], do_sample=False, max_new_tokens=10, pad_token_id=1)
    res = {'hf': a.hf, 'config_class': type(cfg).__module__ + '.' + type(cfg).__name__, 'model_class': type(m).__module__ + '.' + type(m).__name__,
           'base_class': type(b).__module__ + '.' + type(b).__name__, 'loaded_without_trust_remote_code': True,
           'causal_lm_loading_info': {k: v for k, v in info.items()}, 'base_model_loading_info': {k: v for k, v in binfo.items()},
           'base_hidden_to_head_equals_causal_logits': same_head, 'generate_new_tokens': gen.shape[1] - 20,
           'is_repo_package': type(m).__module__ == 'llmua_hf.modeling_llmua'}
    # same weights/outputs as the trust_remote_code path (separate class objects, same code)
    res['all_pass'] = res['is_repo_package'] and not any(info.values()) and not binfo['missing_keys'] and same_head
    json.dump(res, open(a.out, 'w'), indent=1, default=str); print(json.dumps(res, indent=1, default=str))


def stage_noregister(a):
    from transformers import AutoModelForCausalLM
    try:
        AutoModelForCausalLM.from_pretrained(a.hf, trust_remote_code=False); res = {'raised': False}
    except Exception as e:
        res = {'raised': True, 'error': f'{type(e).__name__}: {str(e)[:200]}'}
    json.dump(res, open(a.out, 'w'), indent=1); print(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['batched', 'roundtrip', 'register', 'noregister'])
    ap.add_argument('--hf', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt'); ap.add_argument('--tmp', default=os.path.join(tempfile.gettempdir(), 'llmua_roundtrip'))
    ap.add_argument('--max-new', type=int, default=150)
    ap.add_argument('--sft-data', default=os.path.join(os.environ.get('LLMUA_ROOT', os.path.expanduser('~/data/llmua')), 'sft_data/bins/15k'))
    a = ap.parse_args()
    {'batched': stage_batched, 'roundtrip': stage_roundtrip, 'register': stage_register, 'noregister': stage_noregister}[a.stage](a)
