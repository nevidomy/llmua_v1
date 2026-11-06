#!/usr/bin/env python3
"""Numerical proof that a converted HF folder equals the native llmua checkpoint. Read-only on the repo and data.

    python verify_hf.py logits  --hf DIR --ckpt CKPT --out JSON
    python verify_hf.py fullval --hf DIR --ckpt CKPT --out JSON [--variants a,b,cprime,native_bf16,native_fp32,hf_bf16]
    python verify_hf.py gen     --hf DIR --ckpt CKPT --out JSON [--max-new 200] [--min-answer 0]   (SFT only)
    python verify_hf.py tok     --hf BASE_DIR --hf-sft SFT_DIR --out JSON [--sft-data DIR]

fullval re-uses the repo's own trainers (same packing, windows, ignore-index, chars-per-token) and swaps only the
model forward for an adapter, so any bpc difference is the model, not the harness.
"""
import argparse, contextlib, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO); os.chdir(REPO)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
DEV = 'cuda'
EPS_F32, EPS_BF16 = torch.finfo(torch.float32).eps, torch.finfo(torch.bfloat16).eps


# --------------------------------------------------------------------------------------------- loading
def load_native(ckpt):
    from sample_generator import load_model
    return load_model(ckpt, DEV)                                   # (Transformer fp32 eval, ck)


def load_hf(path):
    from transformers import AutoModelForCausalLM
    m, info = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, output_loading_info=True, trust_remote_code=True)
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys'], info
    if m.config.model_type == 'llmua':           # must be the copy of the code shipped in the folder, not the repo package
        assert type(m).__module__.startswith('transformers_modules'), type(m).__module__
        assert 'llmua_hf' not in sys.modules
    return m.to(DEV).eval()


def set_eps(hf, qk, block):
    """Runtime patch: per-norm epsilon (q_norm/k_norm vs block+final norms). Stock config has a single value."""
    n = 0
    for name, m in hf.named_modules():
        e = qk if name.endswith(('q_norm', 'k_norm')) else block
        if type(m).__name__ == 'Qwen3RMSNorm':
            m.variance_epsilon = e; n += 1
        elif isinstance(m, nn.RMSNorm):                          # llmua_hf
            m.eps = e; n += 1
    assert n == 4 * hf.config.num_hidden_layers + 1
    return n


class HFAdapter(nn.Module):
    """Looks like llmua Transformer to the trainers. Document masking (BOS segments) is applied here as a 4D bool mask."""
    def __init__(self, hf, bos_id, vocab, width):
        super().__init__()
        self.hf, self.bos_id, self.vocab_size, self.context_width, self.doc_masking = hf, bos_id, vocab, width, False

    def forward(self, x, y=None, block_mask=None, attention_mask=None, **kw):
        from model.packed_val import packed_doc_attention_mask
        logits = self.hf(input_ids=x, attention_mask=packed_doc_attention_mask(x, self.bos_id)).logits
        return logits if y is None else (logits, self._loss(logits, y))

    def _loss(self, logits, y):
        B, T = y.shape
        return F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1), reduction='none').view(B, T).mean()


class NativeSDPAAdapter(HFAdapter):
    """The native Transformer, doc-masked through SDPA (so it can run in fp32 without flex_attention)."""
    def __init__(self, native):
        nn.Module.__init__(self)
        self.native, self.bos_id, self.vocab_size, self.context_width, self.doc_masking = \
            native, native.bos_id, native.vocab_size, native.context_width, False
        native.doc_masking = False

    def forward(self, x, y=None, block_mask=None, attention_mask=None, **kw):
        from model.packed_val import packed_doc_attention_mask
        logits = self.native(x, attention_mask=packed_doc_attention_mask(x, self.bos_id))
        return logits if y is None else (logits, self._loss(logits, y))


def make_trainer(model, ck):
    cfg = ck['cfg']
    if cfg.get('kind') == 'sft':
        from sft_training_loop import sft_eval_trainer
        return sft_eval_trainer(model, cfg, DEV)
    from model.training_loop import TrainConfig, Trainer
    kw = {k: v for k, v in cfg.items() if k in TrainConfig.__dataclass_fields__}
    kw.update(eval_only=True, log_file=None, progress=False, load_from_file=None, load_from_file_for_decay=None,
              extend_schedule=False, compile=False, device=DEV)
    return Trainer(model, TrainConfig(**kw))


def eval_rows(trainer, ck, n=8):
    """n packed rows of length T (x only), spread over the val split, built exactly as full-val builds them."""
    T = trainer.cfg.seq_len
    if ck['cfg'].get('kind') == 'sft':
        from sft.batcher import pack_rows
        val, cfg = trainer._val, trainer.cfg
        rows = pack_rows(val.lengths, np.arange(val.n), T, cfg.dense_batching)
        pick = [rows[i] for i in np.linspace(0, len(rows) - 1, n).astype(int)]
        xs = [trainer._pack_xy(np.concatenate([val.example(j) for j in r]), T)[0] for r in pick]
        return torch.from_numpy(np.stack(xs)).to(DEV)
    data = trainer._bin('val'); nw = (len(data) - 1) // T
    ix = np.linspace(0, nw - 1, n).astype(int) * T
    return torch.from_numpy(np.stack([data[i:i + T] for i in ix]).astype(np.int64)).to(DEV)


def single_docs(trainer, ck, n=8, min_len=300):
    """n single documents (BOS first, no other BOS) as 1-D id lists, <=2048 tokens, for plain-causal comparison."""
    out = []
    if ck['cfg'].get('kind') == 'sft':
        val = trainer._val
        for i in range(val.n):
            e = val.example(i)
            if len(e) >= min_len:
                out.append(e[:2048])
            if len(out) == n:
                break
        return out
    data = np.asarray(trainer._bin('val')[:3_000_000]); bos = np.flatnonzero(data == 0)
    for a, b in zip(bos[:-1], bos[1:]):
        if b - a >= min_len:
            out.append(data[a:min(b, a + 2048)].astype(np.int64))
        if len(out) == n:
            break
    return out


# --------------------------------------------------------------------------------------------- stage: logits
@torch.no_grad()
def stage_logits(a):
    native, ck = load_native(a.ckpt); hf = load_hf(a.hf)
    trainer = make_trainer(native, ck)
    from model.packed_val import packed_doc_attention_mask
    x = eval_rows(trainer, ck, 8)
    res = {'hf_class': type(hf).__module__ + '.' + type(hf).__name__, 'ckpt': a.ckpt, 'hf': a.hf, 'windows': 8, 'T': x.shape[1], 'docs_per_window': [int((r == native.bos_id).sum()) for r in x]}

    def cmp(la, lb):
        d = (la - lb).abs()
        lp = (la.log_softmax(-1) - lb.log_softmax(-1)).abs()
        return {'max_abs_dlogit': d.max().item(), 'mean_abs_dlogit': d.mean().item(), 'max_abs_dlogprob': lp.max().item(),
                'argmax_agree': (la.argmax(-1) == lb.argmax(-1)).float().mean().item(), 'logit_absmax': la.abs().max().item()}

    native.doc_masking = False
    am = packed_doc_attention_mask(x, native.bos_id)
    ours = native(x, attention_mask=am).float()
    theirs = hf(input_ids=x, attention_mask=am).logits.float()
    res['packed_docmask_sdpa_fp32'] = cmp(ours, theirs)
    try:                                                          # the flex_attention BlockMask path (as in training)
        from model.transformer import doc_block_mask
        native.doc_masking = True
        flex = native(x, block_mask=doc_block_mask(x, native.bos_id)).float()
        res['packed_docmask_flex_fp32'] = cmp(flex, theirs)
        res['native_sdpa_vs_flex_fp32'] = cmp(ours, flex)
    except Exception as e:                                        # report, don't hide
        res['flex_fp32_error'] = repr(e)[:300]
    native.doc_masking = False
    # standard HF usage: one document, default causal mask, no custom mask
    worst = {'max_abs_dlogit': 0.0, 'max_abs_dlogprob': 0.0}; lens = []
    for ids in single_docs(trainer, ck, 8):
        xi = torch.tensor(ids, device=DEV)[None]
        c = cmp(native(xi).float(), hf(input_ids=xi).logits.float()); lens.append(int(xi.shape[1]))
        worst = {k: max(worst[k], c[k]) for k in worst}
    res['single_doc_default_causal_fp32'] = dict(worst, lengths=lens)
    # weights are bit-exact copies?
    sd, hsd = native.state_dict(), hf.state_dict()
    res['embed_bit_exact'] = bool(torch.equal(sd['embed.weight'], hsd['model.embed_tokens.weight' if 'model.embed_tokens.weight' in hsd else 'embed.weight']))
    json.dump(res, open(a.out, 'w'), indent=1); print(json.dumps(res, indent=1))


# --------------------------------------------------------------------------------------------- stage: fullval
def stage_fullval(a):
    native, ck = load_native(a.ckpt); hf = load_hf(a.hf)
    trainer = make_trainer(native, ck)
    stored = ck['full_val']
    cfg_eps = hf.config.rms_norm_eps
    res_extra = {'hf_class': type(hf).__module__ + '.' + type(hf).__name__}
    bos, V, W = native.bos_id, native.vocab_size, native.context_width
    hfa, nat = HFAdapter(hf, bos, V, W), NativeSDPAAdapter(native)
    nullctx = contextlib.nullcontext()
    ac_ctx = trainer.ctx                                            # the training autocast (bf16)
    plan = {                                                        # name -> (model, ctx, (qk_eps, block_eps) | None)
        'native_bf16': (None, ac_ctx, None),                        # original flex path, original autocast: reproduces stored
        'native_fp32': (nat, nullctx, None),
        'a': (hfa, nullctx, (cfg_eps, cfg_eps)),                    # (a) one eps everywhere = fp32 eps  (= stock config)
        'b': (hfa, nullctx, (EPS_BF16, EPS_BF16)),                  # (b) the other eps everywhere = 2^-7
        'c': (hfa, nullctx, (EPS_F32, EPS_F32)),                    # (c) exact per-norm eps as measured by eps_probe.py
        'cprime': (hfa, nullctx, (EPS_BF16, EPS_F32)),              # (c') per-norm eps as hypothesised: q/k 2^-7, rest fp32
        'hf_bf16': (hfa, ac_ctx, (cfg_eps, cfg_eps)),               # converted model under the training autocast
        'hf_bf16_b': (hfa, ac_ctx, (EPS_BF16, EPS_BF16)),
    }
    res = {**res_extra, 'ckpt': a.ckpt, 'hf': a.hf, 'stored_full_val': stored, 'rms_norm_eps_in_config': cfg_eps, 'runs': {}}
    for name in a.variants.split(','):
        model, ctx, eps = plan[name]
        trainer.model = model if model is not None else native
        if model is None:
            native.doc_masking = True
        else:
            native.doc_masking = False if model is nat else native.doc_masking
        trainer.ctx = ctx
        if eps is not None:
            set_eps(hf, *eps)
        t0 = time.time()
        out = trainer.evaluate_full_val()
        nats, bpc, info = out[0], out[1], out[2]
        r = {'val_nats': float(nats), 'val_bpc': float(bpc), 'seconds': round(time.time() - t0, 1)}
        if info.get('p0_bpc') is not None:
            r.update(val_p0_nats=float(info['p0_nats']), val_p0_bpc=float(info['p0_bpc']))
        r['windows'], r['batches'], r['dropped'] = info['windows'], info['batches'], info['dropped']
        r['d_bpc_vs_stored'] = r['val_bpc'] - stored['val_bpc']
        res['runs'][name] = r
        print(name, json.dumps(r), flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)


# --------------------------------------------------------------------------------------------- stage: gen
@torch.no_grad()
def stage_gen(a):
    from sft.chat import ChatSession, specials_from_cfg, _resolve_tokenizer, format_user_turn, encode_text
    from sft.probe import load_decoder
    from transformers import AutoTokenizer
    from sft.batcher import SFTSplit
    native, ck = load_native(a.ckpt); hf = load_hf(a.hf)
    tok = AutoTokenizer.from_pretrained(a.hf)
    spec = specials_from_cfg(ck.get('cfg'), native)
    dec = load_decoder(spec['data_dir'], spec['bos_id'], spec['usr_id'], spec['assistant_id'], spec['end_id'])
    _resolve_tokenizer(dec, spec['data_dir'], native.vocab_size)
    sess = ChatSession(native, dec, spec['bos_id'], spec['usr_id'], spec['assistant_id'], spec['end_id'], temperature=0.0)
    val = SFTSplit(spec['data_dir'], 'val')
    # five prompts spread over the SFT val split: the user text is recovered from the stored ids
    picks, texts = [], []
    for i in np.linspace(3, val.n - 3, 400).astype(int):
        e = val.example(int(i)); asst = np.flatnonzero(e == spec['assistant_id'])
        if len(asst) and 4 <= asst[0] <= 120 and (len(e) - asst[0]) >= a.min_answer:
            picks.append(int(i)); texts.append(dec._hf.decode([int(t) for t in e[2:asst[0]]], skip_special_tokens=False))
        if len(picks) == 5:
            break
    assert len(picks) == 5
    rows = []
    for i, text in zip(picks, texts):
        sess.send(text, max_tokens=a.max_new)
        prefix = format_user_turn(text, bos_id=spec['bos_id'], usr_id=spec['usr_id'], assistant_id=spec['assistant_id'],
                                  encode=lambda t: encode_text(dec, t))
        ours_gen = sess.ids[len(prefix):]
        hf_prompt = tok.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True, tokenize=True)
        hf_prompt = hf_prompt['input_ids'] if isinstance(hf_prompt, dict) else list(hf_prompt)
        out = hf.generate(torch.tensor([hf_prompt], device=DEV), attention_mask=torch.ones(1, len(hf_prompt), device=DEV, dtype=torch.long),
                          do_sample=False, num_beams=1, max_new_tokens=a.max_new, eos_token_id=spec['end_id'],
                          pad_token_id=1, temperature=None, top_p=None, top_k=None)
        hf_gen = out[0, len(hf_prompt):].tolist()
        first_div = next((k for k, (p, q) in enumerate(zip(ours_gen, hf_gen)) if p != q), None)
        rows.append({'val_example': i, 'prompt_ids_equal': hf_prompt == prefix, 'n_prompt': len(prefix), 'n_ours': len(ours_gen), 'n_hf': len(hf_gen),
                     'tokens_equal': ours_gen == hf_gen, 'first_divergence': first_div, 'ended_with_end': ours_gen[-1] == spec['end_id'],
                     'text': text[:100], 'reply': tok.decode(hf_gen, skip_special_tokens=True)[:200]})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    res = {'ckpt': a.ckpt, 'hf': a.hf, 'max_new_tokens': a.max_new, 'all_equal': all(r['tokens_equal'] and r['prompt_ids_equal'] for r in rows), 'prompts': rows}
    json.dump(res, open(a.out, 'w'), indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------------------------- stage: tok
def stage_tok(a):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from sft.batcher import SFTSplit
    src = os.path.join(REPO, 'results', 'ua_bpe_32768.json')
    ref = Tokenizer.from_file(src)
    base, sft = AutoTokenizer.from_pretrained(a.hf), AutoTokenizer.from_pretrained(a.hf_sft)
    assert isinstance(base, PreTrainedTokenizerFast) and isinstance(sft, PreTrainedTokenizerFast)
    res = {'tokenizer_json': src, 'base_class': type(base).__name__, 'sft_class': type(sft).__name__}
    # 1000 pretrain val docs (spread over the split): text = decode of the stored ids
    stream = np.fromfile(os.path.join(os.environ.get('LLMUA_ROOT', os.path.expanduser('~/data/llmua')), 'bins/32768/val.bin'), dtype=np.uint16)
    bos, eos = np.flatnonzero(stream == 0), np.flatnonzero(stream == 1)
    pick = np.linspace(0, len(bos) - 1, 1000).astype(int)
    texts, orig = [], []
    for k in pick:
        s = bos[k]; e = eos[np.searchsorted(eos, s)]
        body = stream[s + 1:e].astype(int).tolist(); orig.append(body)
        texts.append(ref.decode(body, skip_special_tokens=False))
    cr = [any(c in t for c in '\x0b\x0c\r') for t in texts]
    def eq(tk, ts, add):
        return [tk(t, add_special_tokens=add).input_ids for t in ts]
    r_noadd = [ref.encode(t, add_special_tokens=False).ids for t in texts]
    r_add = [ref.encode(t, add_special_tokens=True).ids for t in texts]
    b_noadd, b_add = eq(base, texts, False), eq(base, texts, True)
    s_noadd = eq(sft, texts, False)
    res['pretrain_val_docs'] = {
        'n': len(texts), 'median_tokens': int(np.median([len(x) for x in r_noadd])), 'max_tokens': max(len(x) for x in r_noadd),
        'docs_with_VT_FF_CR': int(sum(cr)),
        'ref_roundtrip_equals_stored_ids': int(sum(a_ == b_ for a_, b_ in zip(r_noadd, orig))),
        'base_ids_equal_json(add_special_tokens=False)': int(sum(a_ == b_ for a_, b_ in zip(b_noadd, r_noadd))),
        'base_with_specials_equals_[bos]+json': int(sum(a_ == [0] + b_ for a_, b_ in zip(b_add, r_noadd))),
        'json_with_specials_equals_[bos]+ids+[eos]': int(sum(a_ == [0] + b_ + [1] for a_, b_ in zip(r_add, r_noadd))),
        'sft_tok_ids_equal_json': int(sum(a_ == b_ for a_, b_ in zip(s_noadd, r_noadd))),
        'sft_tok_mismatches_all_in_VT_FF_CR_docs': all(c for (x, y, c) in zip(s_noadd, r_noadd, cr) if x != y)}
    # SFT val: 1000 examples -> question text, answer text
    spec = {'bos': 0, 'usr': 201, 'asst': 202, 'end': 203}
    val = SFTSplit(a.sft_data, 'val'); n = min(1000, val.n)
    qs, ans, bad = [], [], 0
    for i in np.linspace(0, val.n - 1, n).astype(int):
        e = val.example(int(i)).tolist()
        try:
            j = e.index(202, 2)
            qs.append(ref.decode(e[2:j], skip_special_tokens=False)); ans.append(ref.decode(e[j + 1:-1], skip_special_tokens=False))
            assert e[0] == 0 and e[1] == 201 and e[-1] == 203
        except Exception:
            bad += 1
    enc = lambda t: ref.encode(t, add_special_tokens=False).ids
    crq = [any(c in t for c in '\x0b\x0c\r') for t in qs]; cra = [any(c in t for c in '\x0b\x0c\r') for t in ans]
    tmpl_q = [sft.apply_chat_template([{'role': 'user', 'content': t}], add_generation_prompt=True, tokenize=True) for t in qs]
    tmpl_qa = [sft.apply_chat_template([{'role': 'user', 'content': q}, {'role': 'assistant', 'content': t}], tokenize=True) for q, t in zip(qs, ans)]
    tmpl_q = [x['input_ids'] if isinstance(x, dict) else x for x in tmpl_q]; tmpl_qa = [x['input_ids'] if isinstance(x, dict) else x for x in tmpl_qa]
    want_q = [[0, 201] + enc(q) + [202] for q in qs]                              # sft/chat.py format_user_turn
    want_qa = [[0, 201] + enc(q) + [202] + enc(t) + [203] for q, t in zip(qs, ans)]
    res['sft_val_examples'] = {
        'n': len(qs), 'unparsed': bad, 'questions_with_VT_FF_CR': int(sum(crq)), 'answers_with_VT_FF_CR': int(sum(cra)),
        'sft_tok_ids_equal_json(questions)': int(sum(sft(q, add_special_tokens=False).input_ids == enc(q) for q in qs)),
        'sft_tok_ids_equal_json(answers)': int(sum(sft(t, add_special_tokens=False).input_ids == enc(t) for t in ans)),
        'chat_template(user,generation_prompt)==chat.py_prompt_ids': int(sum(x == y for x, y in zip(tmpl_q, want_q))),
        'chat_template(user,assistant)==bos,usr,q,asst,a,end': int(sum(x == y for x, y in zip(tmpl_qa, want_qa))),
        'template_mismatches_all_in_VT_FF_CR_texts': all(c for x, y, c in zip(tmpl_q, want_q, crq) if x != y),
        'template_qa_mismatches_all_in_VT_FF_CR_texts': all((c1 or c2) for x, y, c1, c2 in zip(tmpl_qa, want_qa, crq, cra) if x != y)}
    # template errors and rendering
    try:
        sft.apply_chat_template([{'role': 'system', 'content': 'x'}, {'role': 'user', 'content': 'y'}], tokenize=True); res['system_role_raises'] = False
    except Exception as e:
        res['system_role_raises'] = True
    res['rendered_example'] = sft.apply_chat_template([{'role': 'user', 'content': 'Привіт'}, {'role': 'assistant', 'content': 'Вітаю!'}], tokenize=False)
    res['special_ids_sft'] = {k: sft.convert_tokens_to_ids(k) for k in ['<|bos|>', '<|eos|>', '<|usr|>', '<|assistant|>', '<|end|>']}
    res['special_ids_base'] = {k: base.convert_tokens_to_ids(k) for k in ['<|bos|>', '<|eos|>']}
    res['eos_bos_pad_sft'] = [sft.eos_token_id, sft.bos_token_id, sft.pad_token_id]; res['eos_bos_pad_base'] = [base.eos_token_id, base.bos_token_id, base.pad_token_id]
    res['raw_VT_FF_CR_in_sft_tok'] = {'json': ref.encode('a\x0bb\x0cc\rd', add_special_tokens=False).ids, 'sft': sft('a\x0bb\x0cc\rd', add_special_tokens=False).input_ids,
                                       'base': base('a\x0bb\x0cc\rd', add_special_tokens=False).input_ids}
    json.dump(res, open(a.out, 'w'), indent=1, ensure_ascii=False); print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['logits', 'fullval', 'gen', 'tok'])
    ap.add_argument('--hf'); ap.add_argument('--ckpt'); ap.add_argument('--out', required=True)
    ap.add_argument('--variants', default='native_bf16,native_fp32,a,b,c,cprime,hf_bf16')
    ap.add_argument('--max-new', type=int, default=200); ap.add_argument('--min-answer', type=int, default=0, help='gen: only prompts whose stored answer has >= this many tokens')
    ap.add_argument('--hf-sft'); ap.add_argument('--sft-data', default=os.path.join(os.environ.get('LLMUA_ROOT', os.path.expanduser('~/data/llmua')), 'sft_data/bins/15k'))
    a = ap.parse_args()
    {'logits': stage_logits, 'fullval': stage_fullval, 'gen': stage_gen, 'tok': stage_tok}[a.stage](a)
